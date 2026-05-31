"""
Baseline answer generator.
Takes a question + retrieved evidence and generates a grounded answer.
"""
from src.utils.model_client import generate_text


ANSWER_PROMPT = """You answer HybridQA-style questions using only the provided evidence.

Rules:
- Use only facts supported by the evidence.
- Many questions require combining a table row with linked entity passages.
- Return the shortest exact answer you can.
- Do not add explanations, hedging, or extra words around the answer.
- If the answer is a number, date, name, nationality, or other short field, output just that field.
- If multiple evidence snippets appear related, prefer the one that directly answers the question.
- If the evidence does not support a confident answer, say "INSUFFICIENT EVIDENCE".

Evidence:
{evidence}

Question: {question}

Answer:"""


def generate_answer(question: str, evidence_units: list[dict]) -> dict:
    """Generate an answer given a question and retrieved evidence units."""
    # Build evidence text from retrieved units
    evidence_parts = []
    for i, unit in enumerate(evidence_units):
        prefix = f"[{unit.get('type', 'unknown')}]"
        evidence_parts.append(f"{prefix} {unit['text']}")

    evidence_text = "\n\n".join(evidence_parts)

    # Truncate if too long (stay within context limits)
    if len(evidence_text) > 6000:
        evidence_text = evidence_text[:6000] + "\n... (truncated)"

    prompt = ANSWER_PROMPT.format(evidence=evidence_text, question=question)
    answer = generate_text(prompt, max_tokens=256)

    return {
        "question": question,
        "generated_answer": answer.strip(),
        "num_evidence_units": len(evidence_units),
        "evidence_types": [u.get("type", "unknown") for u in evidence_units],
    }
