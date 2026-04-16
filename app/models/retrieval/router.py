from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.models.rag_config import DocumentType
from app.models.utils.helpers import get_logger

logger = get_logger(__name__)


class QueryRouter:
    """
    Decides whether a query should hit the text store, code store, or both.
    """

    _PROMPT = ChatPromptTemplate.from_template(
        """You are an intelligent query router for a machine learning knowledge base.
        Decide which database(s) to query:
            - 'text'  : theory, definitions, mathematical formulas, ML concepts
            - 'code'  : implementations, algorithms in code, Python examples
            - 'both'  : question spans theory and implementation
        
        Output ONLY one of: text | code | both
        
        Question: {question}"""
    )

    def __init__(self, llm: ChatGroq):
        self._chain = self._PROMPT | llm

    def route(self, query: str) -> list[str]:
        """Returns a list of store keys: ['text'], ['code'], or ['text', 'code']."""
        response = self._chain.invoke({"question": query})
        decision = response.content.strip().lower()

        if decision == "text":
            keys = ["text"]
        elif decision == "code":
            keys = ["code"]
        else:
            keys = ["text", "code"]

        logger.info("Router decision for query '%s…': %s", query[:60], keys)
        return keys
