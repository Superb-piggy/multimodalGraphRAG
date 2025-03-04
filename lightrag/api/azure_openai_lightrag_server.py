from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
import asyncio
import logging
import argparse
from lightrag import LightRAG, QueryParam
from lightrag.llm import (
    azure_openai_complete_if_cache,
    azure_openai_embedding,
)
from lightrag.utils import EmbeddingFunc
from typing import Optional, List
from enum import Enum
from pathlib import Path
import shutil
import aiofiles
from ascii_colors import trace_exception
import os
from dotenv import load_dotenv
import inspect
import json
from fastapi.responses import StreamingResponse

from fastapi import Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

from starlette.status import HTTP_403_FORBIDDEN

load_dotenv()

AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")


def parse_args():
    parser = argparse.ArgumentParser(
        description="LightRAG FastAPI Server with OpenAI integration"
    )

    # Server configuration
    parser.add_argument(
        "--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=9621, help="Server port (default: 9621)"
    )

    # Directory configuration
    parser.add_argument(
        "--working-dir",
        default="./rag_storage",
        help="Working directory for RAG storage (default: ./rag_storage)",
    )
    parser.add_argument(
        "--input-dir",
        default="./inputs",
        help="Directory containing input documents (default: ./inputs)",
    )

    # Model configuration
    parser.add_argument(
        "--model", default="gpt-4o", help="OpenAI model name (default: gpt-4o)"
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-large",
        help="OpenAI embedding model (default: text-embedding-3-large)",
    )

    # RAG configuration
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Maximum token size (default: 32768)",
    )
    parser.add_argument(
        "--max-embed-tokens",
        type=int,
        default=8192,
        help="Maximum embedding token size (default: 8192)",
    )
    parser.add_argument(
        "--enable-cache",
        default=True,
        help="Enable response cache (default: True)",
    )
    # Logging configuration
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--key",
        type=str,
        help="API key for authentication. This protects lightrag server against unauthorized access",
        default=None,
    )

    return parser.parse_args()


class DocumentManager:
    """Handles document operations and tracking"""

    def __init__(self, input_dir: str, supported_extensions: tuple = (".txt", ".md")):
        self.input_dir = Path(input_dir)
        self.supported_extensions = supported_extensions
        self.indexed_files = set()

        # Create input directory if it doesn't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)

    def scan_directory(self) -> List[Path]:
        """Scan input directory for new files"""
        new_files = []
        for ext in self.supported_extensions:
            for file_path in self.input_dir.rglob(f"*{ext}"):
                if file_path not in self.indexed_files:
                    new_files.append(file_path)
        return new_files

    def mark_as_indexed(self, file_path: Path):
        """Mark a file as indexed"""
        self.indexed_files.add(file_path)

    def is_supported_file(self, filename: str) -> bool:
        """Check if file type is supported"""
        return any(filename.lower().endswith(ext) for ext in self.supported_extensions)


# Pydantic models
class SearchMode(str, Enum):
    naive = "naive"
    local = "local"
    global_ = "global"
    hybrid = "hybrid"


class QueryRequest(BaseModel):
    query: str
    mode: SearchMode = SearchMode.hybrid
    only_need_context: bool = False
    # stream: bool = False


class QueryResponse(BaseModel):
    response: str


class InsertTextRequest(BaseModel):
    text: str
    description: Optional[str] = None


class InsertResponse(BaseModel):
    status: str
    message: str
    document_count: int


def get_api_key_dependency(api_key: Optional[str]):
    if not api_key:
        # If no API key is configured, return a dummy dependency that always succeeds
        async def no_auth():
            return None

        return no_auth

    # If API key is configured, use proper authentication
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def api_key_auth(api_key_header_value: str | None = Security(api_key_header)):
        if not api_key_header_value:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN, detail="API Key required"
            )
        if api_key_header_value != api_key:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN, detail="Invalid API Key"
            )
        return api_key_header_value

    return api_key_auth


async def get_embedding_dim(embedding_model: str) -> int:
    """Get embedding dimensions for the specified model"""
    test_text = ["This is a test sentence."]
    embedding = await azure_openai_embedding(test_text, model=embedding_model)
    return embedding.shape[1]


def create_app(args):
    # Setup logging
    logging.basicConfig(
        format="%(levelname)s:%(message)s", level=getattr(logging, args.log_level)
    )

    # Check if API key is provided either through env var or args
    api_key = os.getenv("LIGHTRAG_API_KEY") or args.key

    # Initialize FastAPI
    app = FastAPI(
        title="LightRAG API",
        description="API for querying text using LightRAG with separate storage and input directories"
        + "(With authentication)"
        if api_key
        else "",
        version="1.0.0",
        openapi_tags=[{"name": "api"}],
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create the optional API key dependency
    optional_api_key = get_api_key_dependency(api_key)

    # Create working directory if it doesn't exist
    Path(args.working_dir).mkdir(parents=True, exist_ok=True)

    # Initialize document manager
    doc_manager = DocumentManager(args.input_dir)

    # Get embedding dimensions
    embedding_dim = asyncio.run(get_embedding_dim(args.embedding_model))

    async def async_openai_complete(
        prompt, system_prompt=None, history_messages=[], **kwargs
    ):
        """Async wrapper for OpenAI completion"""
        kwargs.pop("keyword_extraction", None)

        return await azure_openai_complete_if_cache(
            args.model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            **kwargs,
        )

    # Initialize RAG with OpenAI configuration
    rag = LightRAG(
        enable_llm_cache=args.enable_cache,
        working_dir=args.working_dir,
        llm_model_func=async_openai_complete,
        llm_model_name=args.model,
        llm_model_max_token_size=args.max_tokens,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=args.max_embed_tokens,
            func=lambda texts: azure_openai_embedding(
                texts, model=args.embedding_model
            ),
        ),
    )

    @app.on_event("startup")
    async def startup_event():
        """Index all files in input directory during startup"""
        try:
            new_files = doc_manager.scan_directory()
            for file_path in new_files:
                try:
                    # Use async file reading
                    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                        content = await f.read()
                        # Use the async version of insert directly
                        await rag.ainsert(content)
                        doc_manager.mark_as_indexed(file_path)
                        logging.info(f"Indexed file: {file_path}")
                except Exception as e:
                    trace_exception(e)
                    logging.error(f"Error indexing file {file_path}: {str(e)}")

            logging.info(f"Indexed {len(new_files)} documents from {args.input_dir}")

        except Exception as e:
            logging.error(f"Error during startup indexing: {str(e)}")

    @app.post("/documents/scan", dependencies=[Depends(optional_api_key)])
    async def scan_for_new_documents():
        """Manually trigger scanning for new documents"""
        try:
            new_files = doc_manager.scan_directory()
            indexed_count = 0

            for file_path in new_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        await rag.ainsert(content)
                        doc_manager.mark_as_indexed(file_path)
                        indexed_count += 1
                except Exception as e:
                    logging.error(f"Error indexing file {file_path}: {str(e)}")

            return {
                "status": "success",
                "indexed_count": indexed_count,
                "total_documents": len(doc_manager.indexed_files),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/resetcache", dependencies=[Depends(optional_api_key)])
    async def reset_cache():
        """Manually reset cache"""
        try:
            cachefile = args.working_dir + "/kv_store_llm_response_cache.json"
            if os.path.exists(cachefile):
                with open(cachefile, "w") as f:
                    f.write("{}")
            return {"status": "success"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/documents/upload", dependencies=[Depends(optional_api_key)])
    async def upload_to_input_dir(file: UploadFile = File(...)):
        """Upload a file to the input directory"""
        try:
            if not doc_manager.is_supported_file(file.filename):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type. Supported types: {doc_manager.supported_extensions}",
                )

            file_path = doc_manager.input_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Immediately index the uploaded file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                await rag.ainsert(content)
                doc_manager.mark_as_indexed(file_path)

            return {
                "status": "success",
                "message": f"File uploaded and indexed: {file.filename}",
                "total_documents": len(doc_manager.indexed_files),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/query", response_model=QueryResponse, dependencies=[Depends(optional_api_key)]
    )
    async def query_text(request: QueryRequest):
        try:
            response = await rag.aquery(
                request.query,
                param=QueryParam(
                    mode=request.mode,
                    stream=False,
                    only_need_context=request.only_need_context,
                ),
            )
            return QueryResponse(response=response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/query/stream", dependencies=[Depends(optional_api_key)])
    async def query_text_stream(request: QueryRequest):
        try:
            response = await rag.aquery(
                request.query,
                param=QueryParam(
                    mode=request.mode,
                    stream=True,
                    only_need_context=request.only_need_context,
                ),
            )
            if inspect.isasyncgen(response):

                async def stream_generator():
                    async for chunk in response:
                        yield json.dumps({"data": chunk}) + "\n"

                return StreamingResponse(
                    stream_generator(), media_type="application/json"
                )
            else:
                return QueryResponse(response=response)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/documents/text",
        response_model=InsertResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def insert_text(request: InsertTextRequest):
        try:
            await rag.ainsert(request.text)
            return InsertResponse(
                status="success",
                message="Text successfully inserted",
                document_count=1,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/documents/file",
        response_model=InsertResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def insert_file(file: UploadFile = File(...), description: str = Form(None)):
        try:
            content = await file.read()

            if file.filename.endswith((".txt", ".md")):
                text = content.decode("utf-8")
                rag.insert(text)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file type. Only .txt and .md files are supported",
                )

            return InsertResponse(
                status="success",
                message=f"File '{file.filename}' successfully inserted",
                document_count=1,
            )
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File encoding not supported")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/documents/batch",
        response_model=InsertResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def insert_batch(files: List[UploadFile] = File(...)):
        try:
            inserted_count = 0
            failed_files = []

            for file in files:
                try:
                    content = await file.read()
                    if file.filename.endswith((".txt", ".md")):
                        text = content.decode("utf-8")
                        rag.insert(text)
                        inserted_count += 1
                    else:
                        failed_files.append(f"{file.filename} (unsupported type)")
                except Exception as e:
                    failed_files.append(f"{file.filename} ({str(e)})")

            status_message = f"Successfully inserted {inserted_count} documents"
            if failed_files:
                status_message += f". Failed files: {', '.join(failed_files)}"

            return InsertResponse(
                status="success" if inserted_count > 0 else "partial_success",
                message=status_message,
                document_count=len(files),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete(
        "/documents",
        response_model=InsertResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def clear_documents():
        try:
            rag.text_chunks = []
            rag.entities_vdb = None
            rag.relationships_vdb = None
            return InsertResponse(
                status="success",
                message="All documents cleared successfully",
                document_count=0,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health", dependencies=[Depends(optional_api_key)])
    async def get_status():
        """Get current system status"""
        return {
            "status": "healthy",
            "working_directory": str(args.working_dir),
            "input_directory": str(args.input_dir),
            "indexed_files": len(doc_manager.indexed_files),
            "configuration": {
                "model": args.model,
                "embedding_model": args.embedding_model,
                "max_tokens": args.max_tokens,
                "embedding_dim": embedding_dim,
            },
        }

    return app


def main():
    args = parse_args()
    import uvicorn

    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
