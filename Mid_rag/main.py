import os
import traceback
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# 랭체인 및 구글 제네레이티브 AI 관련 모듈
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from supabase import create_client, Client

# FastAPI 애플리케이션 초기화
app = FastAPI(title="도시 정책 RAG AI 서버", version="2.0")

# ==========================================
# Gemini API 키 및 환경 변수 설정
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

# 내부 라이브러리(LangChain) 호환을 위해 GOOGLE_API_KEY가 필요할 경우를 대비해 동기화 처리
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# 필수 환경 변수 검증
if not GEMINI_API_KEY:
    raise RuntimeError("[환경 변수 오류] GEMINI_API_KEY가 설정되지 않았습니다. Cloud Run 환경 변수를 확인해주세요.")
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("[환경 변수 오류] SUPABASE_URL 또는 SUPABASE_SERVICE_KEY가 누락되었습니다.")

# Supabase 클라이언트 초기화
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
except Exception as e:
    raise RuntimeError(f"[Supabase 초기화 실패] 연결 객체 생성 중 오류 발생: {str(e)}")

# Pydantic을 이용한 요청 데이터 모델 정의
class InitRAGRequest(BaseModel):
    pdf_path: str = Field(..., description="처리할 PDF 파일 경로 또는 이름")

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="사용자의 질의 내용")
    chat_history: Optional[List[Dict[str, str]]] = Field(default=[], description="이전 대화 기록 목록 (role, content)")

@app.post("/api/init_rag")
def initialize_rag(req: InitRAGRequest):
    try:
        if not os.path.exists(req.pdf_path):
            raise HTTPException(
                status_code=404, 
                detail=f"[파일 처리 오류] 지정된 경로에 PDF 파일이 존재하지 않습니다: '{req.pdf_path}'."
            )

        loader = PyPDFLoader(req.pdf_path)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(pages)

        if not docs:
            raise HTTPException(
                status_code=400, 
                detail="[PDF 파싱 오류] PDF 파일에서 추출할 수 있는 텍스트가 없습니다."
            )

        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        vector_store = SupabaseVectorStore.from_documents(
            docs,
            embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents"
        )

        return {
            "status": "success", 
            "message": "Supabase 벡터 DB 구축 및 RAG 초기화가 성공적으로 완료되었습니다.", 
            "total_chunks": len(docs)
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        error_detail = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"[RAG 초기화 내부 서버 오류] 사유: {str(e)}\n상세 스택: {error_detail}"
        )

@app.post("/api/chat")
def chat_with_rag(req: QueryRequest):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vector_store = SupabaseVectorStore(
            client=supabase, 
            embedding=embeddings, 
            table_name="documents", 
            query_name="match_documents"
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Gemini 모델 설정 (gemini-1.5-flash)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            streaming=True
        )

        formatted_history = []
        for msg in req.chat_history:
            if msg.get("role") == "user":
                formatted_history.append(HumanMessage(content=msg.get("content")))
            elif msg.get("role") == "assistant":
                formatted_history.append(AIMessage(content=msg.get("content")))

        augmentation_prompt = ChatPromptTemplate.from_messages([
            ("system", "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        query_augmentation_chain = (
            augmentation_prompt
            | llm
            | StrOutputParser()
        )

        augmented_query = query_augmentation_chain.invoke({
            "chat_history": formatted_history,
            "question": req.question
        })

        retrieved_docs = retriever.invoke(augmented_query)
        
        context_texts = []
        sources = []
        for doc in retrieved_docs:
            context_texts.append(doc.page_content)
            source_info = doc.metadata.get("source", "문서 출처 미상")
            page_info = doc.metadata.get("page", "")
            sources.append(f"{source_info} (p.{page_info})")

        combined_context = "\n\n".join(context_texts)
        unique_sources = list(set(sources))

        final_prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 문서에 기반해 답변하는 도시 정책 전문가야. 아래의 참고 문맥을 바탕으로 사용자의 질문에 정확하고 전문적으로 답변해라.\n\n[참고 문맥]\n{context}"),
            ("human", "{augmented_query}")
        ])

        final_chain = (
            {"context": lambda x: combined_context, "augmented_query": lambda x: augmented_query}
            | final_prompt
            | llm
            | StrOutputParser()
        )

        def generate():
            yield f"[증강된 질문]: {augmented_query}\n[참고 출처]: {', '.join(unique_sources)}\n------------------------------------\n\n"
            for chunk in final_chain.stream(req.question):
                yield chunk

        return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")

    except Exception as e:
        error_detail = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"[챗봇 처리 중 에러 발생]\n사유: {str(e)}\n상세 스택: {error_detail}"
        )

@app.get("/health")
def health_check():
    return {"status": "healthy", "database": "connected"}

# Cloud Run PORT 바인딩 코드
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
