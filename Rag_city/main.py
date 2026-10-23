import os
import io
import json
import asyncio
from typing import List, Dict, Any, AsyncGenerator

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import AsyncCallbackHandler

# ------------------------------------------------------------------
# 1. 환경 변수 검증
# ------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("[서버 초기화 에러] 필수 환경변수 'GEMINI_API_KEY'가 설정되지 않았습니다.")

# Chroma DB 영구 저장 경로 설정 (Cloud Run 임시/볼륨 경로)
CHROMA_PERSIST_DIRECTORY = "./chroma_db"

# Google Embeddings 설정 (models/text-embedding-004)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GEMINI_API_KEY
)

# Gemini LLM 설정 (gemini-2.5-flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3,
    streaming=True
)

app = FastAPI(title="Hyun Jin-geon RAG AI Server")

# CORS 설정 (Google Apps Script 연동 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# 2. 데이터 모델 정의
# ------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str

# ------------------------------------------------------------------
# 3. 실시간 스트리밍 핸들러 (StreamHandler)
# ------------------------------------------------------------------
class StreamHandler(AsyncCallbackHandler):
    """LLM이 생성하는 토큰을 비동기 큐에 실시간으로 적재하는 핸들러"""
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token:
            await self.queue.put(token)

    async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        await self.queue.put(None)  # 스트리밍 종료 신호

    async def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        await self.queue.put(f"\n[LLM 오류 발생: {str(error)}]\n")
        await self.queue.put(None)

# ------------------------------------------------------------------
# 4. API 엔드포인트
# ------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {"status": "ok", "message": "현진건 작가 페르소나 RAG 서버가 정상 작동 중입니다."}

@app.post("/upload-pdf")
async def upload_pdf_and_create_vector_db(file: UploadFile = File(...)):
    """PDF 파일을 업로드받아 텍스트 추출 후 Chroma DB에 인덱싱"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="[형식 오류] PDF 파일만 업로드할 수 있습니다.")

    try:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="[내용 오류] 업로드된 파일이 비어 있습니다.")

        # 임시 파일로 저장 후 PyPDFLoader로 로드
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if not documents:
            raise HTTPException(status_code=422, detail="[파싱 오류] PDF에서 텍스트를 추출할 수 없습니다.")

        # 텍스트 청크 쪼개기 (Chunk Size: 1000, Overlap: 200)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        split_docs = text_splitter.split_documents(documents)

        # 출처 파일명 메타데이터 추가
        for doc in split_docs:
            doc.metadata["source_filename"] = file.filename

        # Chroma DB에 문서 임베딩 및 저장 (영구 저장소 연동)
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )
        vector_store.persist()

        return {
            "status": "success",
            "message": f"'{file.filename}' 문서가 Chroma DB에 성공적으로 적재되었습니다.",
            "chunk_count": len(split_docs)
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[Chroma DB 생성 실패] {str(e)}")

@app.post("/query-stream")
async def query_rag_stream(req: QueryRequest):
    """
    MultiQueryRetriever와 RunnablePassthrough, Prompt Hub를 활용한 RAG 스트리밍 질의
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="[질의 오류] 질문 내용이 비어 있습니다.")

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # 1. Chroma DB 로드
            if not os.path.exists(CHROMA_PERSIST_DIRECTORY):
                raise HTTPException(status_code=400, detail="[검색 오류] 생성된 Vector DB가 없습니다. 먼저 PDF를 업로드하여 DB를 생성하세요.")

            vector_store = Chroma(
                persist_directory=CHROMA_PERSIST_DIRECTORY,
                embedding_function=embeddings
            )

            # 2. 다중 질문 검색기 (MultiQueryRetriever) 구성
            base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm
            )

            # 다중 쿼리로 문서 검색 실행
            retrieved_docs = await multi_query_retriever.ainvoke(req.query)

            # 출처 정보 가공 및 전송
            sources_info = []
            context_text = ""
            for idx, doc in enumerate(retrieved_docs):
                filename = doc.metadata.get("source_filename", "문서")
                page_num = doc.metadata.get("page", "미상")
                snippet = doc.page_content
                
                sources_info.append({
                    "id": idx + 1,
                    "filename": filename,
                    "page": page_num,
                    "content": snippet
                })
                context_text += f"\n[출처 {idx+1} - {filename} (p.{page_num})]\n{snippet}\n"

            yield f"data: {json.dumps({'type': 'sources', 'content': sources_info})}\n\n"

            # 3. LangChain Prompt Hub에서 프롬프트 가져오기 또는 기본 RAG 프롬프트 대체 활용
            # (안정적인 실행을 위해 랭체인 표준 RAG 프롬프트 구조 구성)
            from langchain_core.prompts import ChatPromptTemplate
            
            system_role_msg = (
                "당신은 소설 '운수 좋은 날'을 집필한 현진건 작가님입니다. "
                "근대 한국 문학의 사실주의 관점과 시대적 아픔을 담아, 제공된 문서와 사용자의 질문에 답변하세요."
            )
            
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", system_role_msg + "\n\n참고 문서:\n{context}"),
                ("human", "{question}")
            ])

            # 4. RunnablePassthrough를 활용한 체인 구성
            # 문서는 검색 결과 텍스트를 포맷팅하여 주입하고 질문은 그대로 전달
            def format_docs(docs):
                return "\n\n".join(d.page_content for d in docs)

            queue = asyncio.Queue()
            stream_handler = StreamHandler(queue)

            streaming_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=GEMINI_API_KEY,
                temperature=0.4,
                streaming=True,
                callbacks=[stream_handler]
            )

            rag_chain = (
                {"context": lambda x: context_text, "question": RunnablePassthrough()}
                | rag_prompt
                | streaming_llm
            )

            # 5. 비동기 실행 및 실시간 스트리밍 송신
            task = asyncio.create_task(rag_chain.ainvoke(req.query))

            while True:
                token = await queue.get()
                if token is None:
                    break
                yield f"data: {json.dumps({'type': 'answer_chunk', 'content': token})}\n\n"

            await task
            yield f"data: {json.dumps({'type': 'end'})}\n\n"

        except Exception as e:
            error_msg = f"[RAG 처리 오류] {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
