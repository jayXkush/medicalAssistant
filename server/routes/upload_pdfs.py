from fastapi import APIRouter, UploadFile, File
from typing import List
from modules.load_vectorstore import load_vectorstore
from fastapi.responses import JSONResponse
from logger import logger


router=APIRouter()

@router.post("/upload_pdfs/")
async def upload_pdfs(files:List[UploadFile] = File(...)):
    try:
        logger.info("Received uploaded files")
        load_vectorstore(files)
        logger.info("Document added to vectorstore")
        return {"messages":"Files processed and vectorstore updated"}
    except RuntimeError as e:
        # Handle Pinecone connection errors with user-friendly messages
        error_msg = str(e)
        logger.exception(f"Error during PDF upload: {error_msg}")
        return JSONResponse(
            status_code=503,  # Service Unavailable for network issues
            content={"error": error_msg}
        )
    except Exception as e:
        logger.exception("Error during PDF upload")
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while processing the PDF: {str(e)}"}
        )