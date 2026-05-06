import asyncio
import json
from typing import List

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from app.models.retrieval.retriever import Retriever
from app.models.rag_config import RAGConfig, QueryTranslationMethod
from app.models.utils.helpers import get_logger

logger = get_logger(__name__)

# --- Structured Output Models ---

class RelevanceIndices(BaseModel):
    """Indices of documents that are relevant to the question."""
    relevant_indices: List[int] = Field(description="List of 0-indexed integers representing relevant documents.")

class EvaluationResult(BaseModel):
    """Evaluation of the generated answer."""
    grounded: bool = Field(description="Is the answer supported by the provided documents?")
    useful: bool = Field(description="Does the answer directly address the user's question?")

# --- Graders ---

class RelevanceGrader:
    _PROMPT = ChatPromptTemplate.from_template(
        """You are a relevance filter. Given a question and a list of retrieved documents, 
        identify which documents contain information necessary to answer the question.

        Question: {question}

        Documents:
        {documents}

        Task: Return a JSON object containing the indices of the documents that are relevant. 
        If no documents are relevant, return an empty list [].
        """
    )

    def __init__(self, llm: ChatGroq):
        # Using structured output for faster and more reliable parsing
        self._chain = self._PROMPT | llm.with_structured_output(RelevanceIndices)

    async def grade(self, question: str, documents: List[Document]) -> List[int]:
        if not documents:
            return []
        
        # Only send first 500 characters of each doc to the grader to reduce latency
        doc_text = "\n".join(f"[{i}] {d.page_content[:500]}" for i, d in enumerate(documents))
        
        try:
            result = await self._chain.ainvoke({
                "question": question,
                "documents": doc_text
            })
            return result.relevant_indices
        except Exception as e:
            logger.error(f"Relevance grading failed: {e}")
            return list(range(len(documents))) # Fallback: treat all as relevant


class CombinedGrader:
    _PROMPT = ChatPromptTemplate.from_template(
        """You are a strict evaluator for a RAG system.

            Your task is to evaluate whether the answer is grounded in the provided context and whether it is useful.

            ---------------------
            Context:
            {context}

            Question:
            {question}

            Answer:
            {answer}
            ---------------------

            Evaluation rules:

            1. Grounded:
            - TRUE if the answer is fully supported by the context
            - FALSE if the answer contains any information NOT present in the context

            2. Useful:
            - TRUE if the answer directly and sufficiently answers the question
            - FALSE if the answer is vague, incomplete, or irrelevant

            Important:
            - Be strict. Do NOT assume facts outside the context.
            - If unsure, mark grounded = false.

            ---------------------

            Return ONLY a valid JSON object with this exact format:

            {{
            "grounded": true or false,
            "useful": true or false
            }}
            """
            )
    

    def __init__(self, llm: ChatGroq):
        self._chain = self._PROMPT | llm.with_structured_output(EvaluationResult)

    async def grade(self, question: str, documents: List[Document], answer: str) -> EvaluationResult:
        context_text = "\n\n".join(d.page_content[:1000] for d in documents) if documents else "No context."
        
        try:
            return await self._chain.ainvoke({
                "context": context_text,
                "question": question,
                "answer": answer
            })
        except Exception as e:
            logger.error(f"Combined grading failed: {e}")
            return EvaluationResult(grounded=True, useful=True) # Soft fallback


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

    async def generate(self, question: str, documents: List[Document]) -> str:
        context = "\n\n---\n\n".join(d.page_content for d in documents)
        resp = await self._chain.ainvoke({"context": context, "question": question})
        return resp.content.strip()


# --- Main Orchestrator ---

class SelfRAG:
    """
    Optimized Self-RAG Orchestrator (Async & Structured)
    """

    def __init__(self, cfg: RAGConfig, retriever: Retriever, llm: ChatGroq):
        self._cfg = cfg
        self._retriever = retriever
        self._generator = AnswerGenerator(llm)
        self._relevance_grader = RelevanceGrader(llm)
        self._combined_grader = CombinedGrader(llm)

    async def answer(
        self,
        question: str,
        method: QueryTranslationMethod = QueryTranslationMethod.AUTO,
    ) -> dict:
        """
        Executes the Self-RAG loop asynchronously.
        """
        grading_log = []
        current_method = method
        
        # Max retries loop
        for attempt in range(1, self._cfg.self_rag_max_retries + 2):
            logger.info(f"Self-RAG Attempt {attempt} | Method: {current_method.value}")

            # 1. Retrieve (Top-5 optimization)
            # Ensure your retriever implementation supports the 'k' parameter
            docs = await self._retriever.retrieve(question)
            # Ensure we only work with top 5
            docs = docs[:5]

            # 2. Relevance Filtering
            relevant_indices = await self._relevance_grader.grade(question, docs)
            relevant_docs = [docs[i] for i in relevant_indices if i < len(docs)]
            
            logger.info(f"Filtered {len(relevant_docs)}/{len(docs)} relevant documents.")
            
            # Log relevance stage
            grading_log.append({
                "stage": "relevance",
                "attempt": attempt,
                "keep_count": len(relevant_docs)
            })

            # If no docs relevant, we still try to generate or use the top 1 doc as fallback
            generation_docs = relevant_docs if relevant_docs else docs[:1]

            # 3. Generate Answer
            answer = await self._generator.generate(question, generation_docs)

            # 4. Combined Grading (Hallucination + Usefulness)
            eval_result = await self._combined_grader.grade(question, generation_docs, answer)
            
            grading_log.append({
                "stage": "evaluation",
                "attempt": attempt,
                "grounded": eval_result.grounded,
                "useful": eval_result.useful
            })

            # Check if we can exit the loop
            if eval_result.grounded and eval_result.useful:
                return {
                    "answer": answer,
                    "sources": generation_docs,
                    "translation_method": current_method.value,
                    "attempts": attempt,
                    "grading_log": grading_log,
                }

            # If not grounded or not useful, change strategy for next attempt
            if not eval_result.grounded:
                logger.warning("Hallucination detected. Re-trying...")
            elif not eval_result.useful:
                logger.warning("Answer not useful. Switching to RAG_FUSION...")
                current_method = QueryTranslationMethod.RAG_FUSION

        # If loop finishes without perfect score
        return {
            "answer": answer,
            "sources": generation_docs,
            "translation_method": current_method.value,
            "attempts": attempt,
            "grading_log": grading_log,
            "warning": "Maximum retries reached. Answer quality may be sub-optimal."
        }