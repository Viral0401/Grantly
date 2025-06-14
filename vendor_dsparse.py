# vendor_dsparse.py

import json
import re
from typing import List
from langchain_community.chat_models.openai import ChatOpenAI
from langchain.schema import Document

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ----- PROMPTS -----
SECTION_PROMPT = """
You are an expert document analyst. Your task is to segment the following document into meaningful sections based on semantic content.

Instructions:
- Each section should cover a logically complete idea or topic (e.g. "Eligibility", "Budget", "Evaluation Criteria").
- Return a JSON list, where each object has:
  - a "title" (string)
  - a "start_line" (integer)
  - an "end_line" (integer)
- Use line numbers for boundary references.
- If you cannot determine a good title, return an empty string for the title.
- DO NOT explain your reasoning. Return ONLY the JSON.

Here is the document, with line numbers:
{text}
"""

CHUNK_PROMPT = """
You are an expert at preparing documents for retrieval-augmented generation (RAG).

Your task is to split the following section into coherent chunks that:
- Are each ~{chunk_size} characters in length (±100 characters)
- Have ~{overlap} characters of semantic overlap between chunks
- Maintain the flow of meaning and do NOT cut off mid-thought

Output:
Return a JSON list of clean chunk strings. Do NOT include titles, metadata, or any explanation.

Section Title: "{title}"

Section Text:
\"\"\"
{text}
\"\"\"
"""

# ----- UTILS -----
def _safe_json_parse(response_str: str):
    try:
        return json.loads(response_str)
    except:
        match = re.search(r"\[.*\]", response_str, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
    return None

def _add_line_numbers(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(f"{i}: {line}" for i, line in enumerate(lines))

# ----- SECTIONING -----
def get_sections_from_str(text: str) -> List[dict]:
    numbered_text = _add_line_numbers(text)  # ✅ line numbers only here
    ai_msg = llm.invoke(SECTION_PROMPT.format(text=numbered_text))
    result = ai_msg.content
    sections = _safe_json_parse(result)
    return sections or []

# ----- CHUNKING -----
def chunk_section(title: str, text: str, chunk_size: int, overlap: int) -> List[str]:
    ai_msg = llm.invoke(CHUNK_PROMPT.format(
        title=title,
        text=text,
        chunk_size=chunk_size,
        overlap=overlap
    ))
    result = ai_msg.content
    chunks = _safe_json_parse(result)
    return chunks or []

# ----- FULL PIPELINE -----
def parse_and_chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[Document]:
    sections = get_sections_from_str(text)
    lines = text.splitlines()
    docs: List[Document] = []

    for sec in sections:
        start = sec.get("start_line", 0)
        end = sec.get("end_line", len(lines) - 1)
        seg = "\n".join(lines[start:end + 1]).strip()  # ✅ using clean original text
        for chunk_txt in chunk_section(sec.get("title", ""), seg, chunk_size, overlap):
            metadata = {"section": sec.get("title", "")}
            docs.append(Document(page_content=chunk_txt, metadata=metadata))

    return docs
