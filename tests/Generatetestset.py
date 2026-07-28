"""
Generates a ~300-question RAGAS eval testset by sampling chunks across every
ingested PDF and asking an LLM to write one grounded question + answer per chunk.

Run once: python generate_testset.py
Output: eval/testset.json
"""
import json
import random
import time
from pathlib import Path

from langchain_groq import ChatGroq
from config.settings import settings
from ingestion.parse_docs import process_all_pdfs, DocChunk

TARGET_TOTAL = 300
OUTPUT_PATH = Path("eval/testset.json")

QUESTION_PROMPT = """You are creating a QA evaluation set for a crop-disease RAG assistant.
Given the passage below, write ONE realistic farmer question that this passage directly and
fully answers, plus a concise 1-3 sentence ground-truth answer using only this passage.

Passage (source: {source}):
\"\"\"{text}\"\"\"

Respond in exactly this format, nothing else:
Q: <question>
A: <ground truth answer>
"""


def sample_chunks_per_doc(chunks: list[DocChunk], per_doc: int) -> list[DocChunk]:
    by_doc: dict[str, list[DocChunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.source_file, []).append(c)

    sampled = []
    for doc, doc_chunks in by_doc.items():
        candidates = [c for c in doc_chunks if len(c.text) > 200]  # skip thin/boilerplate chunks
        pool = candidates if candidates else doc_chunks
        sampled.extend(random.sample(pool, min(per_doc, len(pool))))
    return sampled


def generate_testset():
    print("Parsing PDFs...")
    chunks = process_all_pdfs()
    num_docs = len({c.source_file for c in chunks})
    print(f"{len(chunks)} chunks across {num_docs} docs.")

    per_doc = max(1, TARGET_TOTAL // num_docs)
    sampled = sample_chunks_per_doc(chunks, per_doc=per_doc)
    random.shuffle(sampled)
    sampled = sampled[:TARGET_TOTAL]
    print(f"Sampled {len(sampled)} chunks for QA generation (~{per_doc}/doc).\n")

    llm = ChatGroq(model=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY, temperature=0.3)

    testset = []
    for i, chunk in enumerate(sampled, 1):
        prompt = QUESTION_PROMPT.format(source=chunk.source_file, text=chunk.text[:1500])
        for attempt in range(2):
            try:
                resp = llm.invoke(prompt).content
                q_line = next(l for l in resp.splitlines() if l.strip().startswith("Q:"))
                a_line = next(l for l in resp.splitlines() if l.strip().startswith("A:"))
                testset.append({
                    "question": q_line.split("Q:", 1)[1].strip(),
                    "ground_truth": a_line.split("A:", 1)[1].strip(),
                    "source_file": chunk.source_file,
                    "page_number": chunk.page_number,
                })
                print(f"  [{i}/{len(sampled)}] {chunk.source_file}: {testset[-1]['question'][:60]}...")
                break
            except Exception as e:
                if attempt == 0:
                    time.sleep(3)
                    continue
                print(f"  ⚠️  skipped chunk {i} ({chunk.source_file}): {e}")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(testset, indent=2))
    print(f"\n✅ Saved {len(testset)} QA pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_testset()