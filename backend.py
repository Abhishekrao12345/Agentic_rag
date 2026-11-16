
import os
import hashlib
from typing import List, Literal, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from pinecone import Pinecone

load_dotenv(encoding="utf-8-sig")

groq_api = os.getenv("groq_api_key")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("industry-agentic-rag-kb")

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace="langgraph-agentic-rag",
)


DEFAULT_URLS = ["https://docs.langchain.com/oss/python/langchain/overview"]

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def ingest_urls(urls: List[str]) -> int:
    """Load pages, split, embed and store them in Pinecone. Returns the chunk count."""
    loader = WebBaseLoader(
        web_paths=urls,
        header_template={"User-Agent": "Mozilla/5.0"},
        requests_per_second=2,
        continue_on_failure=True,
    )
    docs = loader.load()
    splits = splitter.split_documents(docs)

    if not splits:
        return 0

    # ids come from the page URL, so every page has its own ids
    # and adding the same page again overwrites it instead of duplicating it
    counters = {}
    ids = []
    for s in splits:
        source = s.metadata.get("source", "unknown")
        n = counters.get(source, 0)
        counters[source] = n + 1
        ids.append(f"{hashlib.md5(source.encode()).hexdigest()[:12]}-{n}")

    vectorstore.add_documents(splits, ids=ids)
    return len(splits)


count = ingest_urls(DEFAULT_URLS)
print(f"[Ingest] {count} chunks stored")

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=groq_api,
    temperature=0,
)

tavily_tool = TavilySearch(max_results=3)

MAX_RETRIES = 1


class AgentState(TypedDict):
    question: str
    current_query: str
    kb_docs: List[Document]
    web_results: str
    kb_grade: str
    web_grade: str
    answer: str
    source_used: str
    retry_count: int


class RouteDecision(BaseModel):
    route: Literal["kb", "direct"] = Field(
        description='Use "direct" ONLY for pure greetings, thanks, or small talk '
                    'with no question. Use "kb" for every question or '
                    'information request on any topic.'
    )


class GradeDecision(BaseModel):
    relevant: Literal["yes", "no"] = Field(
        description='"yes" if the content contains enough information to answer '
                    'the question, otherwise "no".'
    )


router_llm = llm.with_structured_output(RouteDecision, method="json_mode")
grader_llm = llm.with_structured_output(GradeDecision, method="json_mode")


def route_question(state: AgentState):
    question = state["question"]

    decision = router_llm.invoke(f'''
You are a router for an Agentic RAG assistant.

Route to "direct" ONLY if the message is purely a greeting, thanks,
goodbye, or small talk, and does NOT ask for any information.
Examples of direct: "hi", "hello", "thanks!", "bye", "how are you".

Route to "kb" for EVERYTHING else, including any question or request for
information on any topic (technical, news, sports, current events, general
knowledge). If the message contains a question, choose "kb".

Question:
{question}

Return your response as valid JSON.
Example:
{{"route": "kb"}}
''')

    print("[Router]", decision.route)

    return {
        "current_query": question,
        "source_used": decision.route,
    }


def retrieve_kb(state: AgentState):
    query = state["current_query"]
    docs = retriever.invoke(query)

    print(f"[Retrieve KB] query='{query}' -> {len(docs)} docs")

    return {
        "kb_docs": docs,
        "source_used": "kb",
    }


def grade_kb_evidence(state: AgentState):
    query = state["current_query"]
    docs = state["kb_docs"]

    if not docs:
        print("[KB Grader] no documents -> no")
        return {"kb_grade": "no"}

    context = "\n\n".join(
        f"[Doc {i+1}] {d.page_content[:1000]}" for i, d in enumerate(docs)
    )

    decision = grader_llm.invoke(f'''
You are a grader checking whether retrieved documents can answer a question.

Answer "yes" if the documents contain information that directly helps answer the question.
Answer "no" if they are off-topic, too vague, or missing the key information.

Question:
{query}

Documents:
{context}

Return your response as valid JSON.
Example:
{{"relevant": "yes"}}
''')

    print("[KB Grader]", decision.relevant)

    return {"kb_grade": decision.relevant}


def search_web(state: AgentState):
    query = state["current_query"]

    response = tavily_tool.invoke({"query": query})
    results = response.get("results", [])

    web_text = "\n\n".join(
        f"[Web {i+1}] {r['title']}\n{r['content']}\nSource: {r['url']}"
        for i, r in enumerate(results)
    )

    print(f"[Web Search] query='{query}' -> {len(results)} results")

    return {
        "web_results": web_text,
        "source_used": "web",
    }


def grade_web_evidence(state: AgentState):
    query = state["current_query"]
    web_text = state["web_results"]

    if not web_text.strip():
        print("[Web Grader] no web results -> no")
        return {"web_grade": "no"}

    decision = grader_llm.invoke(f'''
You are a grader checking whether web search results can answer a question.

Answer "yes" if the results contain information that directly helps answer the question.
Answer "no" if they are off-topic, too vague, or missing the key information.

Question:
{query}

Web results:
{web_text[:4000]}

Return your response as valid JSON.
Example:
{{"relevant": "yes"}}
''')

    print("[Web Grader]", decision.relevant)

    return {"web_grade": decision.relevant}


def rewrite_query(state: AgentState):
    question = state["question"]
    failed_query = state["current_query"]
    retry_count = state.get("retry_count", 0)

    response = llm.invoke(f'''
You are a query rewriter for a retrieval system over technical documentation
(LangChain, LangGraph, RAG).

The query below did not retrieve useful documents. Rewrite it so it works
better for semantic search:
- Use specific technical keywords and standard terminology.
- Expand abbreviations and add key related terms.
- Keep the same meaning. Do not answer the question.

Original question:
{question}

Failed query:
{failed_query}

Return ONLY the rewritten query as plain text, with no quotes or explanation.
''')

    new_query = response.content.strip()

    print(f"[Rewrite] '{failed_query}' -> '{new_query}' (retry {retry_count + 1})")

    return {
        "current_query": new_query,
        "retry_count": retry_count + 1,
    }


def generate_from_kb(state: AgentState):
    question = state["question"]
    docs = state["kb_docs"]

    context = "\n\n".join(
        f"[Doc {i+1}] {d.page_content[:1000]}" for i, d in enumerate(docs)
    )

    response = llm.invoke(f'''
You are a helpful assistant answering a question using documents from a private knowledge base.

Rules:
- Use ONLY the documents below. Do not add facts from memory.
- If the documents only partly answer the question, say what is missing.
- Be clear and concise.

Question:
{question}

Documents:
{context}
''')

    print("[Generate KB] answer generated")

    return {
        "answer": response.content.strip(),
        "source_used": "kb",
    }


def generate_from_web(state: AgentState):
    question = state["question"]
    web_text = state["web_results"]

    response = llm.invoke(f'''
You are a helpful assistant answering a question using web search results.

Rules:
- Use ONLY the web results below. Do not add facts from memory.
- If the results only partly answer the question, say what is missing.
- Be clear and concise.
- End with a "Sources:" list of the URLs you used.

Question:
{question}

Web results:
{web_text[:4000]}
''')

    print("[Generate Web] answer generated")

    return {
        "answer": response.content.strip(),
        "source_used": "web",
    }


def direct_answer(state: AgentState):
    question = state["question"]

    response = llm.invoke(f'''
You are a friendly assistant for an Agentic RAG project.

The user's message is a greeting, thanks, or simple small talk.
Reply briefly and naturally in one or two sentences.
Do not make up technical facts. If they seem to want technical help,
invite them to ask a question about Agentic RAG, LangGraph, or LangChain.

User message:
{question}
''')

    print("[Direct Answer] answer generated")

    return {
        "answer": response.content.strip(),
        "source_used": "direct",
    }


def answer_insufficient(state: AgentState):
    question = state["question"]
    last_query = state["current_query"]
    retries = state.get("retry_count", 0)

    answer = (
        "I couldn't find reliable information to answer this question.\n\n"
        f"Question: {question}\n\n"
        "What I tried:\n"
        "- Searched the private knowledge base"
        + (f" (and retried with a rewritten query: \"{last_query}\")" if retries > 0 else "")
        + "\n"
        "- Searched the web with Tavily\n\n"
        "Neither source had enough relevant information. You could try "
        "rephrasing the question, adding more detail, or asking about a "
        "more specific part of the topic."
    )

    print("[Insufficient Evidence] no reliable answer found")

    return {
        "answer": answer,
        "source_used": "none",
    }


def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    if state["source_used"] == "kb":
        return "retrieve_kb"
    return "direct_answer"


def route_after_kb_grading(
    state: AgentState,
) -> Literal["generate_from_kb", "search_web"]:
    if state["kb_grade"] == "yes":
        return "generate_from_kb"
    return "search_web"


def route_after_web_grading(
    state: AgentState,
) -> Literal["generate_from_web", "rewrite_query", "answer_insufficient"]:
    if state["web_grade"] == "yes":
        return "generate_from_web"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "rewrite_query"
    return "answer_insufficient"


graph = StateGraph(AgentState)

graph.add_node("route_question", route_question)
graph.add_node("direct_answer", direct_answer)
graph.add_node("retrieve_kb", retrieve_kb)
graph.add_node("grade_kb_evidence", grade_kb_evidence)
graph.add_node("generate_from_kb", generate_from_kb)
graph.add_node("search_web", search_web)
graph.add_node("grade_web_evidence", grade_web_evidence)
graph.add_node("generate_from_web", generate_from_web)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("answer_insufficient", answer_insufficient)

graph.add_edge(START, "route_question")

graph.add_conditional_edges(
    "route_question",
    route_after_router,
    {
        "retrieve_kb": "retrieve_kb",
        "direct_answer": "direct_answer",
    },
)

graph.add_edge("retrieve_kb", "grade_kb_evidence")

graph.add_conditional_edges(
    "grade_kb_evidence",
    route_after_kb_grading,
    {
        "generate_from_kb": "generate_from_kb",
        "search_web": "search_web",
    },
)

graph.add_edge("search_web", "grade_web_evidence")

graph.add_conditional_edges(
    "grade_web_evidence",
    route_after_web_grading,
    {
        "generate_from_web": "generate_from_web",
        "rewrite_query": "rewrite_query",
        "answer_insufficient": "answer_insufficient",
    },
)

graph.add_edge("rewrite_query", "retrieve_kb")

graph.add_edge("direct_answer", END)
graph.add_edge("generate_from_kb", END)
graph.add_edge("generate_from_web", END)
graph.add_edge("answer_insufficient", END)

app = graph.compile()


def ask(q: str):
    result = app.invoke({
        "question": q,
        "current_query": q,
        "retry_count": 0,
        "web_results": "",
    })
    print("\nSource:", result["source_used"])
    print("Answer:", result["answer"])
    return result


if __name__ == "__main__":
    ask("hi")
    ask("What is agentic RAG?")
    ask("Who won the latest F1 race?")