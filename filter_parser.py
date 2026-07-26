"""
filter_parser.py
-----------------
This is the "AI stylist brain": it reads what the user TYPES
(e.g. "show me something like this but in blue, for a wedding, under 3000")
and turns it into structured JSON that vectorstore.py can use as a filter:

    {"color": "blue", "occasion": "wedding", "max_price": 3000}

Why this matters for the resume: this is LangChain's core value —
turning messy natural language into structured, machine-usable output
(a lightweight version of function calling / tool use).

FREE LLM OPTIONS (pick one, both work with the same code below):

1) Groq (recommended - free tier, cloud, fast, no GPU needed)
   - Sign up free at https://console.groq.com
   - Get an API key, put it in a .env file: GROQ_API_KEY=...
   - Uses Llama 3.1 8B - completely free tier, very generous limits

2) Ollama (fully local, zero cost, zero API key, needs ~5GB disk)
   - Install from https://ollama.com
   - Run: ollama pull llama3.1
   - No internet needed after that
"""

import json
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

SYSTEM_PROMPT = """You are a fashion search assistant. Extract structured
search filters from the user's message. Always respond with ONLY valid JSON,
no other text, no markdown formatting.

Possible fields (include ONLY fields the user actually mentioned):
- "color": string (e.g. "blue", "red", "black")
- "occasion": string (e.g. "wedding", "casual", "party", "office")
- "category": string (e.g. "saree", "kurta", "dress", "shirt")
- "max_price": number (if user mentions a budget/price limit)
- "style_note": string (any other descriptive detail, e.g. "with sleeves", "floor-length")

Example:
User: "show me something like this but in blue, for a wedding, under 3000"
Response: {{"color": "blue", "occasion": "wedding", "max_price": 3000}}

Example:
User: "more formal, and with full sleeves"
Response: {{"occasion": "formal", "style_note": "full sleeves"}}
"""


def get_llm():
    """
    Returns a chat model. Tries Groq first (if GROQ_API_KEY is set),
    falls back to local Ollama. Swap this function if you prefer a
    different free provider.
    """
    from dotenv import load_dotenv
    load_dotenv()  # reads .env file so GROQ_API_KEY is actually picked up

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=groq_key)
        except Exception as e:
            raise RuntimeError(
                f"GROQ_API_KEY is set but Groq failed to initialize: {e}. "
                "Check that your key is valid and langchain-groq is installed "
                "(pip install langchain-groq)."
            )
    else:
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(model="llama3.1", temperature=0)
        except Exception as e:
            raise RuntimeError(
                "No GROQ_API_KEY found in .env, and Ollama isn't available either. "
                "Either: (1) create backend/.env with GROQ_API_KEY=your_key, or "
                "(2) install Ollama and run 'ollama pull llama3.1'. "
                f"Original error: {e}"
            )


def parse_filters(user_message: str, conversation_history: list[dict] | None = None) -> dict:
    """
    Turns a natural language message into a filter dict.
    conversation_history (optional): list of {"role": ..., "content": ...}
    so the model can remember earlier constraints in the conversation
    (e.g. "wedding" mentioned 2 turns ago should still apply).
    """
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{message}"),
    ])

    chain = prompt | llm | StrOutputParser()
    raw_output = chain.invoke({"message": user_message})

    try:
        # Strip accidental markdown fences if the model adds them
        cleaned = raw_output.strip().removeprefix("```json").removesuffix("```").strip()
        filters = json.loads(cleaned)
    except json.JSONDecodeError:
        filters = {}

    return filters


def filters_to_chroma_where(filters: dict) -> dict | None:
    """Converts our simple filter dict into ChromaDB's `where` clause format."""
    conditions = []
    if "color" in filters:
        conditions.append({"color": filters["color"]})
    if "occasion" in filters:
        conditions.append({"occasion": filters["occasion"]})
    if "category" in filters:
        conditions.append({"category": filters["category"]})
    if "max_price" in filters:
        conditions.append({"price": {"$lte": filters["max_price"]}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


if __name__ == "__main__":
    # Manual test - requires GROQ_API_KEY or local Ollama running
    test_message = "show me something like this but in blue, for a wedding, under 3000"
    filters = parse_filters(test_message)
    print("Parsed filters:", filters)
    print("Chroma where clause:", filters_to_chroma_where(filters))
