from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from modules.load_vectorstore import get_pinecone_index
from langchain_core.documents import Document
from langchain.schema import BaseRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import Field
from typing import List, Optional
from logger import logger

router=APIRouter()

@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"user query: {question}")

        # Get Pinecone index (lazy initialization - only happens when this endpoint is called)
        index = get_pinecone_index()
        
        # Embed model + query
        embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embedded_query = embed_model.embed_query(question)
        res = index.query(vector=embedded_query, top_k=3, include_metadata=True)

        # Handle empty results
        if not res.get("matches") or len(res["matches"]) == 0:
            return {
                "response": "I'm sorry, but I couldn't find relevant information in the uploaded documents to answer your question. Please make sure you have uploaded relevant medical documents first.",
                "sources": []
            }

        docs = [
            Document(
                page_content=match.get("metadata", {}).get("text", match.get("metadata", {}).get("page_content", "")),
                metadata=match.get("metadata", {})
            ) for match in res["matches"]
        ]

        class SimpleRetriever(BaseRetriever):
            tags: Optional[List[str]] = Field(default_factory=list)
            metadata: Optional[dict] = Field(default_factory=dict)

            def __init__(self, documents: List[Document]):
                super().__init__()
                self._docs = documents

            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self._docs

        retriever = SimpleRetriever(docs)
        chain = get_llm_chain(retriever)
        result = query_chain(chain, question)

        logger.info("query successful")
        return result

    except RuntimeError as e:
        # Handle Pinecone connection errors with user-friendly messages
        error_msg = str(e)
        logger.exception(f"Error processing question: {error_msg}")
        return JSONResponse(
            status_code=503,  # Service Unavailable for network issues
            content={"error": error_msg}
        )
    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while processing your question: {str(e)}"}
        )