"""Quick end-to-end smoke test for the Ragas evaluation pipeline using the local backend."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import Evaluator
from src.evaluation.schemas import EvaluationExample, RetrievedContext, SystemPrediction

EXAMPLES = [
    EvaluationExample(
        question_id="q1",
        question="What is the capital of France?",
        gold_answer="Paris",
        question_type="factoid",
    ),
    EvaluationExample(
        question_id="q2",
        question="What color is the sky on a clear day?",
        gold_answer="Blue",
        question_type="factoid",
    ),
]

PREDICTIONS = [
    SystemPrediction(
        question_id="q1",
        predicted_answer="Paris is the capital of France.",
        system_name="smoke_test",
        retrieved_contexts=[
            RetrievedContext(id="ctx1", text="France is a country in Europe. Its capital city is Paris.")
        ],
    ),
    SystemPrediction(
        question_id="q2",
        predicted_answer="The sky is blue.",
        system_name="smoke_test",
        retrieved_contexts=[
            RetrievedContext(
                id="ctx2",
                text="The sky appears blue due to Rayleigh scattering of sunlight.",
            )
        ],
    ),
]


def main() -> None:
    outdir = Path(tempfile.mkdtemp())
    evaluator = Evaluator(
        dataset_name="smoke",
        system_name="smoke_test",
        model_backend="local_openai",
        generation_model="Qwen/Qwen2.5-3B-Instruct",
        embedding_model="intfloat/e5-base-v2",
        experiment_id="smoke_ragas_local",
    )
    print("Running evaluation (this may take a minute)...")
    result = evaluator.evaluate(
        examples=EXAMPLES,
        predictions=PREDICTIONS,
        output_dir=outdir,
    )

    print("\n=== Evaluation complete ===")

    # Aggregate metrics
    agg = result.get("aggregate_metrics", {}).get("overall", [{}])[0]
    ragas_keys = [
        "answer_correctness", "answer_relevancy", "semantic_similarity",
        "factual_correctness", "faithfulness", "context_precision", "context_recall",
    ]
    print("Aggregate metrics (overall):")
    for k in ragas_keys:
        v = agg.get(k)
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Per-sample results
    rows = result.get("per_question_metrics", [])
    print(f"\nPer-sample results ({len(rows)} samples):")
    for r in rows:
        m = r.get("metrics", {}) if isinstance(r, dict) else {}
        qid = r.get("question_id") if isinstance(r, dict) else r.question_id
        print(f"  [{qid}]")
        for k in ragas_keys:
            v = m.get(k)
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    print(f"\nOutput files saved to: {outdir}")
    if len(rows) == 0:
        print("WARNING: 0 samples evaluated — check that question_ids match between examples and predictions.")
        sys.exit(1)
    else:
        print("\nSmoke test PASSED — ragas pipeline is working correctly with local backend.")


if __name__ == "__main__":
    main()
