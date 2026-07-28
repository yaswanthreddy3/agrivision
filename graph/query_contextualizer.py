import logfire
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from config.settings import settings
from config.observability import setup_logfire

setup_logfire()

CONTEXTUALIZE_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """Given a conversation history and a new user question, rewrite the question
to be fully self-contained if it depends on prior context (e.g. "what about prevention?" after
discussing a specific disease should become "what is the prevention for [disease]?").

If the new question is already self-contained and doesn't depend on prior context, return it unchanged.

Return ONLY the rewritten question, nothing else."""

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=CONTEXTUALIZE_MODEL, api_key=settings.GROQ_API_KEY, temperature=0)
    return _llm


def contextualize_query(new_query: str, message_history: list) -> str:
    """Rewrites new_query to be self-contained, using recent conversation history."""
    if not message_history:
        return new_query

    # Only use the last 4 messages (2 turns) to keep this fast and cheap
    recent = message_history[-4:]

    history_text = ""
    for msg in recent:
        if isinstance(msg, HumanMessage):
            history_text += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Assistant: {msg.content[:300]}\n"

    if not history_text:
        return new_query

    with logfire.span("contextualize_query", new_query=new_query):
        llm = get_llm()
        response = llm.invoke([
            ("system", SYSTEM_PROMPT),
            ("human", f"Conversation so far:\n{history_text}\nNew question: {new_query}")
        ])
        rewritten = response.content.strip()
        logfire.info("query_contextualized", original=new_query, rewritten=rewritten)
        return rewritten