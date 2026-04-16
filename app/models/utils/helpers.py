import re
import logging
from typing import Any
import os
import shutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def contains_code(text: str) -> bool:
    """Heuristic: detect code-heavy content."""
    code_patterns = [
        r"```",
        r"def\s+\w+\s*\(",
        r"class\s+\w+",
        r"import\s+\w+",
        r"from\s+\w+\s+import",
        r"for\s+\w+\s+in\s+",
        r"if\s+__name__\s*==",
        r"np\.|pd\.|torch\.|sklearn\.",
        r"^\s{4}[a-zA-Z]",          # 4-space indentation typical of Python
    ]
    score = sum(bool(re.search(p, text, re.MULTILINE)) for p in code_patterns)
    return score >= 2


def deduplicate_docs(docs: list[Any]) -> list[Any]:
    """Remove duplicate documents by page_content."""
    seen, unique = set(), []
    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique.append(doc)
    return unique


def reciprocal_rank_fusion(
    results: list[list[Any]], k: int = 60
) -> list[Any]:
    """
    Merge multiple ranked lists into one using RRF.
    Returns docs sorted by fused score (highest first).
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Any] = {}

    for ranked_list in results:
        for rank, doc in enumerate(ranked_list, start=1):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [doc_map[k] for k in sorted_keys]
def push_processed_file(src_path: str, dest_path: str):
    os.makedirs(dest_path, exist_ok=True)
    
    files = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
    for file_name in files:
        src = os.path.join(src_path, file_name)
        dest = os.path.join(dest_path, file_name)
        
        if os.path.exists(dest):
            base, extension = os.path.splitext(file_name)
            from datetime import datetime
            new_name = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{extension}"
            dest = os.path.join(dest_path, new_name)
            
        shutil.move(src, dest)
        
def format_answer(result: dict):
    answer = ""
    answer += result["answer"]
    answer += "\n\nREFERENCES:"
    src_lst = []
    i = 1
    for doc in result["sources"]:
        src = doc.metadata.get("source", "unknown")
        ftype = doc.metadata.get("file_type", "?")
        
        if src not in src_lst and ftype == "pdf":
            fixed_src = src.removesuffix(".pdf").removeprefix("data\\").removeprefix("data/")
            answer += "\n["+ str(i)+ "] " + fixed_src
            i += 1
        src_lst.append(src)
    return answer
