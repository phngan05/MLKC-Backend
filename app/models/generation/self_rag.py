from enum import Enum
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from app.models.retrieval.retriever import Retriever
from app.models.rag_config import RAGConfig, QueryTranslationMethod
from app.models.utils.helpers import get_logger

logger = get_logger(__name__)


class GradeResult(str, Enum):
    RELEVANT    = "relevant"
    IRRELEVANT  = "irrelevant"


class HallucinationResult(str, Enum):
    GROUNDED    = "grounded"
    HALLUCINATED = "hallucinated"


class AnswerResult(str, Enum):
    USEFUL   = "useful"
    NOTUSEFUL = "not_useful"


# ------------------------------------------------------------------ #
# Graders
# ------------------------------------------------------------------ #

class RelevanceGrader:
    _PROMPT = ChatPromptTemplate.from_template(
        """You are a grader assessing whether a retrieved document is relevant
        to a machine learning question.
        Document:
        {document}
        
        Question: {question}
        
        Answer with ONLY 'relevant' or 'irrelevant'."""
    )

    def __init__(self, llm: ChatGroq):
        self._chain = self._PROMPT | llm

    def grade(self, question: str, document: Document) -> GradeResult:
        resp = self._chain.invoke({
            "question": question,
            "document": document.page_content,
        })
        raw = resp.content.strip().lower()
        return GradeResult.RELEVANT if "relevant" in raw else GradeResult.IRRELEVANT


class HallucinationGrader:
    _PROMPT = ChatPromptTemplate.from_template(
        """You are a grader checking whether an answer is grounded in the provided documents and does not contain hallucinations.
        Documents:
        {documents}
        
        Answer: {answer}
        
        Answer with ONLY 'grounded' or 'hallucinated'."""
    )

    def __init__(self, llm: ChatGroq):
        self._chain = self._PROMPT | llm

    def grade(self, documents: list[Document], answer: str) -> HallucinationResult:
        doc_text = "\n\n---\n\n".join(d.page_content for d in documents)
        resp = self._chain.invoke({"documents": doc_text, "answer": answer})
        raw = resp.content.strip().lower()
        return (
            HallucinationResult.GROUNDED
            if "grounded" in raw
            else HallucinationResult.HALLUCINATED
        )


class AnswerGrader:
    _PROMPT = ChatPromptTemplate.from_template(
        """You are a grader assessing whether an answer is useful and fully addresses a machine learning question.
        Question: {question}
        
        Answer: {answer}
        
        Answer with ONLY 'useful' or 'not_useful'."""
    )

    def __init__(self, llm: ChatGroq):
        self._chain = self._PROMPT | llm

    def grade(self, question: str, answer: str) -> AnswerResult:
        resp = self._chain.invoke({"question": question, "answer": answer})
        raw = resp.content.strip().lower()
        return AnswerResult.USEFUL if "useful" in raw else AnswerResult.NOTUSEFUL


# ------------------------------------------------------------------ #
# Generator
# ------------------------------------------------------------------ #

class AnswerGenerator:
    _PROMPT = ChatPromptTemplate.from_template(
        """You are an expert machine learning assistant.
        Use the following retrieved context to answer the question.
        If context includes code, explain it clearly.
        If you don't know, say so explicitly.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
    )

    def __init__(self, llm: ChatGroq):
        self._chain = self._PROMPT | llm

    def generate(self, question: str, documents: list[Document]) -> str:
        context = "\n\n---\n\n".join(d.page_content for d in documents)
        resp = self._chain.invoke({"context": context, "question": question})
        return resp.content.strip()


# ------------------------------------------------------------------ #
# Self-RAG Orchestrator
# ------------------------------------------------------------------ #

class SelfRAG:
    """
    Implements the Self-RAG loop:

    Retrieve → Grade Relevance → Generate →
    Grade Hallucination → Grade Usefulness →
    Retry if needed (up to max_retries)
    """

    def __init__(self, cfg: RAGConfig, retriever: Retriever, llm: ChatGroq):
        self._cfg = cfg
        self._retriever = retriever
        self._generator = AnswerGenerator(llm)
        self._relevance_grader = RelevanceGrader(llm)
        self._hallucination_grader = HallucinationGrader(llm)
        self._answer_grader = AnswerGrader(llm)

    def answer(
        self,
        question: str,
        method: QueryTranslationMethod = QueryTranslationMethod.AUTO,
    ) -> dict:
        """
        Returns a dict with:
          - answer (str)
          - sources (list[Document])
          - translation_method (str)
          - attempts (int)
          - grading_log (list[dict])
        """
        grading_log = []
        translation_method = method.value

        for attempt in range(1, self._cfg.self_rag_max_retries + 2):
            logger.info("Self-RAG attempt %d/%d", attempt, self._cfg.self_rag_max_retries + 1)

            # ── 1. Retrieve ───────────────────────────────────────────
            docs = self._retriever.retrieve(question, method=method)
            if not docs:
                return self._no_docs_response(question, attempt, grading_log)

            # ── 2. Grade relevance ────────────────────────────────────
            relevant_docs = []
            for doc in docs:
                grade = self._relevance_grader.grade(question, doc)
                grading_log.append({
                    "stage": "relevance",
                    "attempt": attempt,
                    "grade": grade.value,
                    "doc_snippet": doc.page_content[:80],
                })
                if grade == GradeResult.RELEVANT:
                    relevant_docs.append(doc)

            logger.info(
                "Relevance filtering: %d/%d docs kept", len(relevant_docs), len(docs)
            )

            if not relevant_docs:
                logger.warning("No relevant docs found; retrying with broader method.")
                method = QueryTranslationMethod.STEPBACK  # broaden on retry
                continue

            # ── 3. Generate ───────────────────────────────────────────
            answer = self._generator.generate(question, relevant_docs)

            # ── 4. Grade hallucination ────────────────────────────────
            hall_grade = self._hallucination_grader.grade(relevant_docs, answer)
            grading_log.append({
                "stage": "hallucination",
                "attempt": attempt,
                "grade": hall_grade.value,
            })

            if hall_grade == HallucinationResult.HALLUCINATED:
                logger.warning("Answer appears hallucinated; regenerating.")
                continue

            # ── 5. Grade usefulness ───────────────────────────────────
            ans_grade = self._answer_grader.grade(question, answer)
            grading_log.append({
                "stage": "usefulness",
                "attempt": attempt,
                "grade": ans_grade.value,
            })

            if ans_grade == AnswerResult.USEFUL:
                return {
                    "answer": answer,
                    "sources": relevant_docs,
                    "translation_method": translation_method,
                    "attempts": attempt,
                    "grading_log": grading_log,
                }

            logger.warning("Answer not useful; retrying.")
            method = QueryTranslationMethod.RAG_FUSION   # try wider search

        # Max retries exhausted — return best effort
        logger.error("Max retries reached. Returning last generated answer.")
        return {
            "answer": answer,
            "sources": relevant_docs,
            "translation_method": translation_method,
            "attempts": self._cfg.self_rag_max_retries + 1,
            "grading_log": grading_log,
            "warning": "Max retries reached; answer quality not guaranteed.",
        }

    # ------------------------------------------------------------------ #

    def _no_docs_response(
        self, question: str, attempt: int, log: list
    ) -> dict:
        return {
            "answer": "I could not find relevant information to answer this question.",
            "sources": [],
            "translation_method": "N/A",
            "attempts": attempt,
            "grading_log": log,
        }
