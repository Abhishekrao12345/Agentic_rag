
from fastapi import FastAPI
from pydantic import BaseModel

from backend import app as rag_app, ingest_urls

api = FastAPI()


class Question(BaseModel):
    question: str


class Urls(BaseModel):
    urls: list[str]


@api.post("/ask")
def ask(body: Question):
    result = rag_app.invoke({
        "question": body.question,
        "current_query": body.question,
        "retry_count": 0,
        "web_results": "",
    })
    return {
        "answer": result["answer"],
        "source_used": result["source_used"],
    }


@api.post("/ingest")
def ingest(body: Urls):
    count = ingest_urls(body.urls)
    return {"chunks_added": count}