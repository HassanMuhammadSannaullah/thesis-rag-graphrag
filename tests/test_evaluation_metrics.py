import unittest

from src.evaluation.answer_metrics import compute_answer_metrics
from src.evaluation.hallucination_metrics import compute_hallucination_metrics
from src.evaluation.retrieval_metrics import (
    compute_answer_support_metrics,
    compute_id_retrieval_metrics,
    compute_retrieval_metrics,
)
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

    def test_proxy_evidence_metrics_are_separate_from_strict_metrics(self):
        contexts = [
            RetrievedContext(id="row::table_1::0", text="City: Paris | Mayor: Anne Hidalgo", rank=1),
            RetrievedContext(id="passage::table_1::1", text="Anne Hidalgo has served as mayor of Paris.", rank=2),
        ]
        support_metrics, support_warnings = compute_answer_support_metrics(
            contexts,
            gold_answers=["Anne Hidalgo"],
            k_values=[1, 3],
        )
        proxy_metrics, proxy_warnings = compute_id_retrieval_metrics(
            ["row::table_1::0", "passage::table_1::1"],
            contexts,
            k_values=[1, 3],
            prefix="proxy_evidence",
            missing_warning="missing_proxy_evidence",
        )
        strict_metrics, strict_warnings = compute_id_retrieval_metrics(
            None,
            contexts,
            k_values=[1, 3],
            prefix="evidence",
            missing_warning="missing_gold_evidence",
        )

        self.assertEqual(support_warnings, [])
        self.assertEqual(proxy_warnings, [])
        self.assertIn("missing_gold_evidence", strict_warnings)
        self.assertEqual(proxy_metrics["proxy_evidence_coverage"], 1.0)
        self.assertEqual(proxy_metrics["proxy_evidence_mrr"], 1.0)
        self.assertEqual(proxy_metrics["proxy_evidence_recall_at_1"], 0.5)
        self.assertEqual(proxy_metrics["proxy_evidence_recall_at_3"], 1.0)
        self.assertEqual(support_metrics["answer_support_hit_at_1"], 1.0)
        self.assertEqual(strict_metrics, {})

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
