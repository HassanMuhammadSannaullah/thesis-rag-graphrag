import unittest

from src.evaluation.answer_metrics import compute_answer_metrics
from src.evaluation.hallucination_metrics import compute_hallucination_metrics
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.evaluation.schemas import RetrievedContext


class EvaluationMetricsTest(unittest.TestCase):
    def test_currency_normalization_and_numeric_match(self):
        metrics = compute_answer_metrics("$656.82 USD", ["$656.82"])
        self.assertEqual(metrics["strict_exact_match"], 0)
        self.assertEqual(metrics["normalized_exact_match"], 1)
        self.assertEqual(metrics["numeric_value_match"], 1)

    def test_retrieval_metrics_example(self):
        contexts = [
            RetrievedContext(id="transaction:T-104", rank=1),
            RetrievedContext(id="policy:P3", rank=2),
            RetrievedContext(id="irrelevant:X", rank=3),
        ]
        metrics, warnings = compute_retrieval_metrics(
            ["transaction:T-104", "policy:P3"],
            contexts,
            gold_answers=["$656.82"],
            k_values=[3],
        )
        self.assertEqual(warnings, [])
        self.assertEqual(metrics["evidence_recall_at_3"], 1.0)
        self.assertAlmostEqual(metrics["evidence_precision_at_3"], 2 / 3)
        self.assertAlmostEqual(metrics["evidence_f1_at_3"], 0.8)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_answer_support_metrics_without_gold_evidence(self):
        contexts = [
            RetrievedContext(id="context:1", text="The total amount was $656.82 USD.", rank=1),
            RetrievedContext(id="context:2", text="Another unrelated sentence.", rank=2),
        ]
        metrics, warnings = compute_retrieval_metrics(
            None,
            contexts,
            gold_answers=["$656.82"],
            k_values=[1, 3],
        )
        self.assertIn("missing_gold_evidence", warnings)
        self.assertEqual(metrics["answer_support_coverage"], 1.0)
        self.assertEqual(metrics["answer_support_hit_at_1"], 1.0)
        self.assertEqual(metrics["answer_support_hit_at_3"], 1.0)
        self.assertEqual(metrics["answer_support_mrr"], 1.0)

    def test_hallucination_metrics_detect_unsupported_prediction(self):
        contexts = [
            RetrievedContext(id="context:1", text="The approved amount was $656.82 USD.", rank=1),
        ]
        answer_metrics = compute_answer_metrics("Paris", ["London"])
        metrics = compute_hallucination_metrics(
            predicted_answer="Paris",
            gold_answers=["London"],
            retrieved_contexts=contexts,
            answer_metrics=answer_metrics,
            judge_metrics={},
        )
        self.assertEqual(metrics["prediction_in_context"], 0)
        self.assertEqual(metrics["gold_in_context"], 0)
        self.assertEqual(metrics["unsupported_prediction"], 1)
        self.assertEqual(metrics["likely_hallucination"], 1)

    def test_plus_and_and_normalization(self):
        metrics = compute_answer_metrics(
            "CFO and Board Approval",
            ["CFO + Board Approval"],
        )
        self.assertEqual(metrics["strict_exact_match"], 0)
        self.assertEqual(metrics["normalized_exact_match"], 1)
        self.assertGreaterEqual(metrics["token_f1"], 0.99)


if __name__ == "__main__":
    unittest.main()
