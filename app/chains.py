"""LCEL chains for production RAG pipeline v2.

v2 changes vs v1:
  - rag_prompt: enforces [Source: filename] citations in every answer
  - citation_check_chain: verifies citations are present (new chain)

Structured output note:
  grade_chain uses .with_structured_output() — works fine on Groq for int/str fields.
  relevance_chain, hallucination_chain, citation_check_chain use JsonOutputParser
  because Groq's llama-3.1-8b-instant fails tool_use schema for boolean fields.
"""

from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from app.config import settings

llm = ChatGroq(
    model=settings.groq_model,
    temperature=0,
    api_key=settings.groq_api_key,
)


# ── 1. RAG chain — citation-enforced ─────────────────────────────────────────
# v2: prompt now requires [Source: <filename>] after every factual claim.
# The context passed in is pre-formatted with [Source: filename, Chunk N] headers
# by _format_context() in graph.py, so the LLM knows which label to use.
rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a knowledgeable assistant. Answer the question using ONLY the "
        "provided context. If the context does not contain enough information, "
        "say 'I don't have enough information to answer that.'\n\n"
        "IMPORTANT: You MUST cite your sources. After every factual claim, "
        "add a citation in the format [Source: <filename>] using the filename "
        "shown in the context headers. Every answer must contain at least one citation.\n\n"
        "Context:\n{context}",
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])
rag_chain = rag_prompt | llm | StrOutputParser()


# ── 2. Grade chain ────────────────────────────────────────────────────────────
class GradeResult(BaseModel):
    score: int = Field(description="Relevance and accuracy score 1-10", ge=1, le=10)
    reasoning: str = Field(description="One sentence explanation")

grade_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict quality evaluator. Rate the answer 1-10 for relevance "
        "and accuracy. 7+ means the answer fully addresses the question.",
    ),
    ("human", "Question: {question}\n\nAnswer: {answer}\n\nProvide score and reasoning."),
])
grade_chain = grade_prompt | llm.with_structured_output(GradeResult)


# ── 3. Rewrite chain ──────────────────────────────────────────────────────────
rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the question to be more specific and self-contained, resolving "
        "any pronouns using conversation history. Output ONLY the rewritten question.",
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Original question: {question}\n\nRewritten question:"),
])
rewrite_chain = rewrite_prompt | llm | StrOutputParser()


# ── 4. Relevance chain ────────────────────────────────────────────────────────
# Returns {"is_relevant": true/false, "reason": "..."}
relevance_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        'You are a document relevance filter. Decide if the document helps answer '
        'the question. Respond ONLY with valid JSON, no explanation outside it.\n'
        'Format: {{"is_relevant": true, "reason": "one sentence"}}',
    ),
    ("human", "Question: {question}\n\nDocument: {document}"),
])
relevance_chain = relevance_prompt | llm | JsonOutputParser()


# ── 5. Hallucination chain ────────────────────────────────────────────────────
# Returns {"grounded": true/false, "explanation": "..."}
hallucination_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        'You are a hallucination detector. Determine if every claim in the answer '
        'is directly supported by the context. If the answer contains ANY information '
        'not in the context, it is NOT grounded. '
        'Respond ONLY with valid JSON, no explanation outside it.\n'
        'Format: {{"grounded": true, "explanation": "one sentence"}}',
    ),
    (
        "human",
        "Context:\n{context}\n\nAnswer: {answer}\n\nIs the answer fully grounded?",
    ),
])
hallucination_chain = hallucination_prompt | llm | JsonOutputParser()


# ── 6. Citation check chain — v2 ─────────────────────────────────────────────
# Returns {"has_citations": true/false, "missing": "none or description"}
# Checks that the answer contains at least one [Source: <name>] citation.
citation_check_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        'You are a citation auditor. Check whether the answer contains at least '
        'one source citation in the format [Source: <filename>]. '
        'Respond ONLY with valid JSON, no explanation outside it.\n'
        'Format: {{"has_citations": true, "missing": "none or description of what is missing"}}',
    ),
    ("human", "Answer: {answer}"),
])
citation_check_chain = citation_check_prompt | llm | JsonOutputParser()
