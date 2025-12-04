from fastapi import APIRouter
from pydantic import BaseModel
from vectorstore.store import add_document, search

router = APIRouter()


# Request body model
class AddDocumentRequest(BaseModel):
    text: str


@router.post("/add")
def add_document_to_vectordb(request: AddDocumentRequest):
    try:
        add_document(request.text)
        return {"status": "success", "message": "Document added successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/search")
def search_in_vectordb(query: str):
    try:
        results = search(query)
        # results are returned as dictionaries, adapt if needed
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}
