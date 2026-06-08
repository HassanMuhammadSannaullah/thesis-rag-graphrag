import unittest

from src.data_pipeline.hybridqa_parser import (
    PARSER_SCHEMA_VERSION,
    attach_passage_provenance,
    build_hybridqa_record,
    build_linked_passages,
    parse_table,
)


class HybridQAParserTest(unittest.TestCase):
    def setUp(self):
        self.table_id = "table_123"
        self.table_json = {
            "title": "City Leaders",
            "section_title": "Administration",
            "section_text": "Section text",
            "intro": "Intro text",
            "header": [
                ["City"],
                ["Mayor"],
                ["Founded"],
            ],
            "data": [
                [
                    ["Paris", ["/wiki/Paris"]],
                    ["Anne Hidalgo", ["/wiki/Anne_Hidalgo", "/wiki/Anne_Hidalgo"]],
                    ["1999", ["/wiki/1999"]],
                ],
                [
                    ["Lyon", ["/wiki/Lyon"]],
                    ["Gregory Doucet", ["/wiki/Gregory_Doucet"]],
                    ["43 BC", []],
                ],
            ],
        }
        self.passages_json = {
            "/wiki/Paris": "Paris is the capital city of France.",
            "/wiki/Anne_Hidalgo": "Anne Hidalgo has served as mayor of Paris.",
            "/wiki/1999": "The year 1999 had several notable events.",
            "/wiki/Lyon": "Lyon is a major city in France.",
        }

    def test_parse_table_preserves_row_and_cell_metadata(self):
        table = parse_table(self.table_json, self.table_id)

        self.assertEqual(table["headers"], ["City", "Mayor", "Founded"])
        self.assertEqual(table["num_rows"], 2)
        self.assertEqual(table["rows"][0]["City"], "Paris")
        self.assertEqual(
            table["rows"][0]["_links"],
            ["/wiki/Paris", "/wiki/Anne_Hidalgo", "/wiki/1999"],
        )
        self.assertEqual(
            table["all_links"],
            ["/wiki/Paris", "/wiki/Anne_Hidalgo", "/wiki/1999", "/wiki/Lyon", "/wiki/Gregory_Doucet"],
        )

        first_row_meta = table["row_metadata"][0]
        self.assertEqual(first_row_meta["row_id"], "row::table_123::0")
        self.assertEqual(first_row_meta["cell_ids"][0], "cell::table_123::0::0")

        first_cell_meta = table["cell_metadata"][1]
        self.assertEqual(first_cell_meta["column_name"], "Mayor")
        self.assertEqual(first_cell_meta["links"], ["/wiki/Anne_Hidalgo"])

    def test_build_linked_passages_keeps_all_text_backed_links_and_inventory(self):
        table = parse_table(self.table_json, self.table_id)
        linked_passages, inventory = build_linked_passages(
            self.table_id,
            table["all_links"],
            self.passages_json,
        )

        self.assertEqual(len(linked_passages), 4)
        self.assertEqual(linked_passages[0]["passage_id"], "passage::table_123::0")
        self.assertEqual(linked_passages[0]["link"], "/wiki/Paris")
        self.assertEqual(linked_passages[2]["link"], "/wiki/Lyon")
        self.assertEqual(linked_passages[3]["link"], "/wiki/1999")
        self.assertEqual(linked_passages[3]["link_group"], "generic")
        self.assertEqual(len(inventory), 5)

        missing_text_entry = next(item for item in inventory if item["link"] == "/wiki/Gregory_Doucet")
        self.assertFalse(missing_text_entry["has_text"])
        self.assertIsNone(missing_text_entry["passage_id"])

    def test_attach_passage_provenance_maps_rows_and_cells(self):
        table = parse_table(self.table_json, self.table_id)
        linked_passages, _ = build_linked_passages(
            self.table_id,
            table["all_links"],
            self.passages_json,
        )

        attach_passage_provenance(table, linked_passages)

        self.assertEqual(
            table["row_metadata"][0]["row_link_passage_ids"],
            ["passage::table_123::0", "passage::table_123::1", "passage::table_123::3"],
        )
        lyon_cell = next(
            cell for cell in table["cell_metadata"] if cell["cell_id"] == "cell::table_123::1::0"
        )
        self.assertEqual(lyon_cell["linked_passage_ids"], ["passage::table_123::2"])

    def test_build_hybridqa_record_preserves_backward_compatible_fields(self):
        record = build_hybridqa_record(
            question_payload={
                "question_id": "q1",
                "question": "Who is the mayor of Paris?",
                "answer-text": "Anne Hidalgo",
                "table_id": self.table_id,
            },
            table_json=self.table_json,
            passages_json=self.passages_json,
            split="dev",
        )

        self.assertEqual(record["answer"], "Anne Hidalgo")
        self.assertIsNone(record["gold_evidence"])
        self.assertEqual(record["proxy_evidence"], ["row::table_123::0", "passage::table_123::1"])
        self.assertEqual(record["table"]["rows"][0]["Mayor"], "Anne Hidalgo")
        self.assertEqual(record["num_linked_passages"], 4)
        self.assertEqual(record["linked_passage_ids"][1], "passage::table_123::1")
        self.assertEqual(record["parser_metadata"]["schema_version"], PARSER_SCHEMA_VERSION)
        self.assertEqual(record["parser_metadata"]["linked_passage_strategy"], "all_table_links_with_text")
        self.assertEqual(record["evidence_alignment"]["label_mode"], "proxy_answer_anchored_v1")
        self.assertEqual(record["evidence_alignment"]["proxy_kind"], "table_and_passage")
        self.assertEqual(
            record["table"]["row_metadata"][0]["row_link_passage_ids"],
            ["passage::table_123::0", "passage::table_123::1", "passage::table_123::3"],
        )
        self.assertEqual(
            record["source_metadata"]["table_path"],
            "WikiTables-WithLinks-master/tables_tok/table_123.json",
        )


if __name__ == "__main__":
    unittest.main()
