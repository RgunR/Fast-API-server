from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

from .rag import rag_response
from .classification import classify
app = FastAPI()


class RAGRequest(BaseModel):
    prompt: str


class RAGResponse(BaseModel):
    articles: List[Dict[str, str]]


class ClassificationRequest(BaseModel):
    text: str


class ClassificationResponse(BaseModel):
    categories: str


@app.post("/rag", response_model=RAGResponse)
async def rag_endpoint(request: RAGRequest):
    articles = rag_response(request.promt)
    return RAGResponse(articles=articles)


@app.post("/classification", response_model=ClassificationResponse)
async def classification_endpoint(request: ClassificationRequest):
    categories = classify(request.text)
    return ClassificationResponse(categories=categories)
