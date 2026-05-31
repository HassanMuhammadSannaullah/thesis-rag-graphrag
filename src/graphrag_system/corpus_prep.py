"""
GraphRAG corpus preparation.

Converts our structured datasets (HybridQA or compliance) into text documents
suitable for Microsoft GraphRAG indexing.

GraphRAG expects a directory of text files or a CSV with text content.
"""
import json
from pathlib import Path


def hybridqa_record_to_text(record: dict, max_passages_per_record: int = 10) -> str:
    """Render a parsed HybridQA record as a single source document."""
    table = record["table"]
    parts = []

    parts.append(f"# Table: {table['title']}")
    if table.get("section_title"):
        parts.append(f"Section: {table['section_title']}")
    if table.get("intro"):
        parts.append(f"\n{table['intro']}")

    parts.append("\n## Table Data")
    for j, row in enumerate(table["rows"]):
        # Create natural language description from table row
        row_items = {k: v for k, v in row.items() if k != "_links"}
        if row_items:
            # Generate a narrative sentence for relationship extraction
            items_str = ", ".join(f"{k} is {v}" for k, v in row_items.items())
            parts.append(f"Row {j+1}: {items_str}.")
        else:
            parts.append(f"Row {j+1}: (empty row)")

    if rec := record.get("linked_passages"):
        parts.append("\n## Related Entities and Context")
        for lp in rec[:max_passages_per_record]:
            entity_name = lp["link"].split("/")[-1].replace("_", " ")
            text = lp["text"]
            if len(text) > 800:
                text = text[:800] + "..."
            # Add explicit relationship context
            parts.append(f"\n### {entity_name}")
            parts.append(f"This is the Wikipedia article for {entity_name}. {text}")

    return "\n".join(parts)


def hybridqa_to_graphrag_docs(records: list[dict], output_dir: Path,
                               max_passages_per_record: int = 10):
    """Convert HybridQA parsed records into text documents for GraphRAG."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, rec in enumerate(records):
        doc_text = hybridqa_record_to_text(
            rec,
            max_passages_per_record=max_passages_per_record,
        )
        doc_path = output_dir / f"hybridqa_doc_{i:04d}.txt"
        doc_path.write_text(doc_text, encoding="utf-8")

    print(f"  Created {len(records)} GraphRAG documents in {output_dir}")


def compliance_to_graphrag_docs(transactions_path: Path, policies_path: Path,
                                 output_dir: Path):
    """Convert compliance dataset into text documents for GraphRAG."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Policy documents (one file per policy)
    with open(policies_path, encoding="utf-8") as f:
        policies = json.load(f)

    for pol in policies:
        doc_text = (
            f"# {pol['title']}\n"
            f"Policy ID: {pol['policy_id']}\n"
            f"Effective Date: {pol['effective_date']}\n"
            f"Category: {pol['category']}\n\n"
            f"{pol['content']}"
        )
        doc_path = output_dir / f"policy_{pol['policy_id']}.txt"
        doc_path.write_text(doc_text, encoding="utf-8")

    # Transaction summaries (grouped by department)
    with open(transactions_path, encoding="utf-8") as f:
        transactions = json.load(f)

    dept_groups: dict[str, list] = {}
    for t in transactions:
        dept = t["department"]
        if dept not in dept_groups:
            dept_groups[dept] = []
        dept_groups[dept].append(t)

    for dept, txns in dept_groups.items():
        parts = [f"# {dept} Department Transactions\n"]
        for t in txns:
            parts.append(
                f"- {t['transaction_id']}: {t['date']}, {t['vendor']}, "
                f"{t['category']}, ${t['amount']:,.2f}, "
                f"Status: {t['status']}, Approved by: {t['actual_approver']}, "
                f"Compliant: {t['compliant']}"
            )
        doc_text = "\n".join(parts)
        doc_path = output_dir / f"transactions_{dept.lower()}.txt"
        doc_path.write_text(doc_text, encoding="utf-8")

    # Also create a combined document with all transaction details
    parts = ["# All Transactions Summary\n"]
    for t in transactions:
        parts.append(
            f"Transaction {t['transaction_id']}: Date {t['date']}, "
            f"Department {t['department']}, Vendor {t['vendor']}, "
            f"Category {t['category']}, Amount ${t['amount']:,.2f} {t['currency']}, "
            f"Status {t['status']}, Required Approval {t['required_approval']}, "
            f"Actual Approver {t['actual_approver']}, "
            f"Compliant {t['compliant']}, Invoice {t['invoice_ref']}"
        )
    doc_text = "\n".join(parts)
    (output_dir / "transactions_all.txt").write_text(doc_text, encoding="utf-8")

    total_files = len(policies) + len(dept_groups) + 1
    print(f"  Created {total_files} GraphRAG documents in {output_dir}")
