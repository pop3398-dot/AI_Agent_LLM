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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import AsyncCallbackHandler

# ------------------------------------------------------------------
# 1. 환경 변수 검증
# ------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("[서버 초기화 에러] 필수 환경변수 'GEMINI_API_KEY'가 설정되지 않았습니다.")

# Chroma DB 영구 저장 경로 설정
CHROMA_PERSIST_DIRECTORY = "./chroma_db"

# [수정 완료] 최신 표준 임베딩 모델 설정
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=GEMINI_API_KEY
)

# Gemini LLM 설정
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3,
    streaming=True
)

app = FastAPI(title="Hyun Jin-geon RAG AI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class StreamHandler(AsyncCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token:
            await self.queue.put(token)

    async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        await self.queue.put(None)

    async def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        await self.queue.put(f"\n[LLM 오류 발생: {str(error)}]\n")
        await self.queue.put(None)

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "현진건 작가 페르소나 RAG 서버가 정상 작동 중입니다."}

@app.post("/upload-pdf")
async def upload_pdf_and_create_vector_db(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="[형식 오류] PDF 파일만 업로드할 수 있습니다.")

    try:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="[내용 오류] 업로드된 파일이 비어 있습니다.")

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

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        split_docs = text_splitter.split_documents(documents)

        for doc in split_docs:
            doc.metadata["source_filename"] = file.filename

        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )
        if hasattr(vector_store, "persist"):
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
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="[질의 오류] 질문 내용이 비어 있습니다.")

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            if not os.path.exists(CHROMA_PERSIST_DIRECTORY):
                raise HTTPException(status_code=400, detail="[검색 오류] 생성된 Vector DB가 없습니다. 먼저 PDF를 업로드하여 DB를 생성하세요.")

            vector_store = Chroma(
                persist_directory=CHROMA_PERSIST_DIRECTORY,
                embedding_function=embeddings
            )

            base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm
            )

            retrieved_docs = await multi_query_retriever.ainvoke(req.query)

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

            from langchain_core.prompts import ChatPromptTemplate
            
            system_role_msg = (
                "당신은 소설 '운수 좋은 날'을 집필한 현진건 작가님입니다. "
                "근대 한국 문학의 사실주의 관점과 시대적 아픔을 담아, 제공된 문서와 사용자의 질문에 답변하세요."
            )
            
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", system_role_msg + "\n\n참고 문서:\n{context}"),
                ("human", "{question}")
            ])

            queue = asyncio.Queue()
            stream_handler = StreamHandler(queue)

            streaming_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
