import os
import io
import json
import asyncio
from typing import List, Dict, Any, AsyncGenerator

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client, Client

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.callbacks import AsyncCallbackHandler

# ------------------------------------------------------------------
# 환경 변수 검증 (디폴트 데이터 생성 없이 에러 발생)
# ------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

missing_env = []
if not GEMINI_API_KEY: missing_env.append("GEMINI_API_KEY")
if not OPENAI_API_KEY: missing_env.append("OPENAI_API_KEY")
if not SUPABASE_URL: missing_env.append("SUPABASE_URL")
if not SUPABASE_SERVICE_ROLE_KEY: missing_env.append("SUPABASE_SERVICE_ROLE_KEY")

if missing_env:
    raise RuntimeError(f"[서버 초기화 에러] 필수 환경변수가 설정되지 않았습니다: {', '.join(missing_env)}")

# Supabase 클라이언트 설정
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# OpenAI 임베딩 모델 설정
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

# Supabase Vector Store
vector_store = SupabaseVectorStore(
    client=supabase_client,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)

# Gemini LLM 설정 (gemini-2.5-flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.2,
    streaming=True
)

app = FastAPI(title="RAG AI Server")

# CORS 설정 (Google Apps Script 호출 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# 데이터 모델 정의
# ------------------------------------------------------------------
class ChatMessageInput(BaseModel):
    role: str  # 'user' 또는 'assistant'
    content: str

class QueryRequest(BaseModel):
    query: str
    history: List[ChatMessageInput] = []

# ------------------------------------------------------------------
# 스트리밍 콜백 핸들러
# ------------------------------------------------------------------
class StreamHandler(AsyncCallbackHandler):
    """LLM 스트리밍 토큰을 큐에 전달하는 비동기 콜백 핸들러"""
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token:
            await self.queue.put(token)

    async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        await self.queue.put(None)  # 종료 신호

    async def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        await self.queue.put(f"\n[LLM 오류 발생: {str(error)}]\n")
        await self.queue.put(None)

# ------------------------------------------------------------------
# 헬퍼 함수
# ------------------------------------------------------------------
def parse_chat_history(history_data: List[ChatMessageInput]) -> List[BaseMessage]:
    """클라이언트 대화기록을 LangChain 메시지 객체 리스트로 변환"""
    messages = []
    for item in history_data:
        if item.role == "user":
            messages.append(HumanMessage(content=item.content))
        elif item.role == "assistant":
            messages.append(AIMessage(content=item.content))
    return messages

# ------------------------------------------------------------------
# API 엔드포인트
# ------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """서버 및 연동 상태 확인 엔드포인트"""
    return {"status": "ok", "message": "Google Cloud Run AI Server가 정상 작동 중입니다."}

@app.post("/upload-pdf")
async def upload_pdf_and_create_vector_db(file: UploadFile = File(...)):
    """PDF 파일 수신 -> 텍스트 변환 및 청크 쪼개기 -> Supabase Vector DB 저장"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="[파일 형식 오류] PDF 파일만 업로드할 수 있습니다.")

    try:
        # 파일 읽기
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="[파일 내용 오류] 업로드된 PDF 파일이 비어 있습니다.")

        # 임시 메모리 파일 생성 후 PDF 로딩
        pdf_file_obj = io.BytesIO(file_bytes)
        
        # 파일 로컬 저장 없이 pypdf로 가공 (메모리 내)
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
            raise HTTPException(status_code=422, detail="[PDF 파싱 오류] 문서에서 텍스트를 추출하지 못했습니다.")

        # 텍스트 청크 분할 (Chunk Size: 1000, Overlap: 200)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        split_docs = text_splitter.split_documents(documents)

        # 각 청크 메타데이터 보완
        for doc in split_docs:
            doc.metadata["source_filename"] = file.filename

        # Supabase 임베딩 벡터 저장
        vector_store.add_documents(split_docs)

        return {
            "status": "success",
            "message": f"'{file.filename}' 파일의 처리 및 벡터 DB 저장이 완료되었습니다.",
            "chunk_count": len(split_docs)
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"[Vector DB 생성 중 내부 서버 오류 발생] 세부 정보: {str(e)}"
        )

@app.post("/query-stream")
async def query_rag_stream(req: QueryRequest):
    """
    Query Augmentation 및 Supabase Vector DB 검색 후 
    Gemini 2.5 Flash Lite 모델의 스트리밍 답변 반환 (Server-Sent Events)
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="[질의 오류] 질의 내용이 비어 있습니다.")

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # 1. 질의 확장 (Query Augmentation Chain)
            aug_system_msg = (
                "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. "
                "대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라."
            )
            aug_prompt = ChatPromptTemplate.from_messages([
                ("system", aug_system_msg),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])
            aug_chain = aug_prompt | llm | StrOutputParser()

            formatted_history = parse_chat_history(req.history)
            
            # 질의 확장 실행
            augmented_query = await aug_chain.ainvoke({
                "chat_history": formatted_history,
                "question": req.query
            })

            # 증강된 질의 정보 송신
            yield f"data: {json.dumps({'type': 'augmented_query', 'content': augmented_query})}\n\n"

            # 2. Vector DB 리트리버 검색
            retriever = vector_store.as_retriever(search_kwargs={"k": 4})
            retrieved_docs = await retriever.ainvoke(augmented_query)

            # 검색된 출처 및 검색 내용 송신
            sources_info = []
            context_text = ""
            for idx, doc in enumerate(retrieved_docs):
                page_num = doc.metadata.get("page", "미상")
                filename = doc.metadata.get("source_filename", "문서")
                snippet = doc.page_content
                sources_info.append({
                    "id": idx + 1,
                    "filename": filename,
                    "page": page_num,
                    "content": snippet
                })
                context_text += f"\n[문서 {idx+1} - 파일: {filename}, 페이지: {page_num}]\n{snippet}\n"

            yield f"data: {json.dumps({'type': 'sources', 'content': sources_info})}\n\n"

            # 3. 답변 생성 메인 LLM 체인 실행
            main_system_msg = "너는 문서에 기반해 답변하는 도시 정책 전문가야."
            main_prompt = ChatPromptTemplate.from_messages([
                ("system", main_system_msg),
                ("human", "참고 문서:\n{context}\n\n질문: {question}\n\n위 참고 문서를 바탕으로 전문적이고 정확하게 답변해주세요.")
            ])

            queue = asyncio.Queue()
            stream_handler = StreamHandler(queue)

            # 스트리밍 핸들러 적용 LLM 객체 생성
            streaming_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=GEMINI_API_KEY,
                temperature=0.3,
                streaming=True,
                callbacks=[stream_handler]
            )

            main_chain = main_prompt | streaming_llm

            # 비동기로 LLM 실행 시작
            task = asyncio.create_task(
                main_chain.ainvoke({
                    "context": context_text,
                    "question": augmented_query
                })
            )

            # 토큰 발생 시 즉시 SSE로 클라이언트에 실시간 전송
            while True:
                token = await queue.get()
                if token is None:
                    break
                yield f"data: {json.dumps({'type': 'answer_chunk', 'content': token})}\n\n"

            await task  # 태스크 완료 대기
            yield f"data: {json.dumps({'type': 'end'})}\n\n"

        except Exception as e:
            error_msg = f"[RAG 처리 오류] 질의 수행 중 에러가 발생했습니다: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
