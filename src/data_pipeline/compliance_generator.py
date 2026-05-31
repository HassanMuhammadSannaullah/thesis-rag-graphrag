"""
Synthetic compliance/policy QA dataset generator.

Creates:
  - Structured data: transactions, invoices, entries
  - Unstructured data: policy documents, rules, clauses
  - Questions: labeled by type (simple_lookup, hybrid_lookup, multi_hop, compliance_reasoning)
  - Gold answers with supporting evidence references

This is the final thesis dataset for comparative evaluation.
"""
import json, random
from pathlib import Path
from datetime import datetime, timedelta

random.seed(42)

# ── Company/Domain Setup ─────────────────────────────────────────────
DEPARTMENTS = ["Finance", "IT", "HR", "Operations", "Marketing", "Legal"]
VENDORS = ["Acme Corp", "TechSupply Inc", "Office Pro Ltd", "CloudNet Systems",
           "DataWorks AG", "SecureIT Solutions"]
EXPENSE_CATEGORIES = ["Software License", "Hardware", "Consulting", "Travel",
                      "Office Supplies", "Training", "Cloud Services"]
CURRENCIES = ["USD", "EUR"]
APPROVAL_LEVELS = {
    "under_1000": "Department Manager",
    "1000_to_5000": "Department Head",
    "5000_to_25000": "Finance Director",
    "above_25000": "CFO + Board Approval",
}


def generate_transactions(n: int = 50) -> list[dict]:
    """Generate synthetic transaction records."""
    transactions = []
    base_date = datetime(2024, 1, 1)
    for i in range(n):
        amount = round(random.choice([
            random.uniform(50, 999),
            random.uniform(1000, 4999),
            random.uniform(5000, 24999),
            random.uniform(25000, 100000),
        ]), 2)
        dept = random.choice(DEPARTMENTS)
        vendor = random.choice(VENDORS)
        category = random.choice(EXPENSE_CATEGORIES)
        date = base_date + timedelta(days=random.randint(0, 365))
        approved = random.choice([True, True, True, False])  # 75% approved
        currency = random.choice(CURRENCIES)

        # Determine required approval level
        if amount < 1000:
            required_approval = "Department Manager"
        elif amount < 5000:
            required_approval = "Department Head"
        elif amount < 25000:
            required_approval = "Finance Director"
        else:
            required_approval = "CFO + Board Approval"

        # Sometimes create compliance violations
        actual_approver = required_approval
        compliant = True
        if random.random() < 0.15:  # 15% violations
            # Lower approval than required
            lower_levels = ["Department Manager", "Department Head"]
            actual_approver = random.choice(lower_levels)
            if actual_approver != required_approval:
                compliant = False

        transactions.append({
            "transaction_id": f"TXN-{2024}-{i+1:04d}",
            "date": date.strftime("%Y-%m-%d"),
            "department": dept,
            "vendor": vendor,
            "category": category,
            "amount": amount,
            "currency": currency,
            "status": "Approved" if approved else "Pending",
            "required_approval": required_approval,
            "actual_approver": actual_approver,
            "compliant": compliant,
            "invoice_ref": f"INV-{vendor[:3].upper()}-{i+1:04d}",
            "description": f"{category} from {vendor} for {dept} department",
        })
    return transactions


def generate_policies() -> list[dict]:
    """Generate synthetic compliance policy documents."""
    policies = [
        {
            "policy_id": "POL-001",
            "title": "Expense Approval Policy",
            "effective_date": "2024-01-01",
            "content": (
                "All business expenses must be approved according to the following thresholds:\n"
                "- Expenses under $1,000: Approval by Department Manager\n"
                "- Expenses between $1,000 and $5,000: Approval by Department Head\n"
                "- Expenses between $5,000 and $25,000: Approval by Finance Director\n"
                "- Expenses above $25,000: Approval by CFO with Board notification\n\n"
                "All expenses must be submitted within 30 days of the transaction date. "
                "Expenses submitted after 30 days require additional justification and "
                "Finance Director approval regardless of amount.\n\n"
                "Travel expenses must include original receipts. Digital copies are "
                "accepted only if the original is lost, with a signed declaration."
            ),
            "category": "finance",
        },
        {
            "policy_id": "POL-002",
            "title": "Vendor Management Policy",
            "effective_date": "2024-01-01",
            "content": (
                "All vendors must be registered in the approved vendor list before any "
                "purchase order is issued. New vendors require:\n"
                "- Completed vendor registration form\n"
                "- Tax identification verification\n"
                "- Compliance certificate for IT vendors handling data\n\n"
                "Vendor contracts exceeding $10,000 annually must be reviewed by Legal. "
                "All IT service vendors must provide SOC 2 Type II certification or equivalent. "
                "Vendors processing personal data must sign a Data Processing Agreement (DPA).\n\n"
                "Annual vendor reviews are mandatory for all vendors with cumulative "
                "spending above $50,000."
            ),
            "category": "procurement",
        },
        {
            "policy_id": "POL-003",
            "title": "Data Protection and Privacy Policy",
            "effective_date": "2024-01-01",
            "content": (
                "All personal data processing must comply with GDPR requirements. "
                "Each department must maintain a Record of Processing Activities (ROPA).\n\n"
                "Data retention periods:\n"
                "- Financial records: 7 years\n"
                "- Employee records: Duration of employment + 3 years\n"
                "- Customer data: Duration of relationship + 5 years\n"
                "- Marketing consent records: Until withdrawal + 1 year\n\n"
                "Data breaches must be reported to the Data Protection Officer within "
                "24 hours of discovery. If the breach affects personal data of EU residents, "
                "the supervisory authority must be notified within 72 hours.\n\n"
                "All cloud services storing personal data must be hosted within the EU "
                "or in countries with an adequacy decision."
            ),
            "category": "data_protection",
        },
        {
            "policy_id": "POL-004",
            "title": "IT Security and Access Control Policy",
            "effective_date": "2024-01-01",
            "content": (
                "Access to financial systems requires multi-factor authentication (MFA). "
                "Passwords must be at least 12 characters with complexity requirements.\n\n"
                "Access reviews must be conducted quarterly for critical systems. "
                "Privileged access requires approval from IT Security and the system owner.\n\n"
                "All software purchases must be approved by IT department. "
                "Shadow IT (unauthorized software) is prohibited. "
                "Cloud services must be approved through the Cloud Governance Board.\n\n"
                "Security incidents must be reported to IT Security within 1 hour. "
                "Annual security awareness training is mandatory for all employees."
            ),
            "category": "it_security",
        },
        {
            "policy_id": "POL-005",
            "title": "Travel and Entertainment Policy",
            "effective_date": "2024-01-01",
            "content": (
                "Business travel must be pre-approved by the department head. "
                "International travel requires VP-level approval.\n\n"
                "Accommodation limits per night:\n"
                "- Domestic travel: $200\n"
                "- European travel: EUR 250\n"
                "- International travel: $300\n\n"
                "Flight bookings must be economy class for flights under 6 hours. "
                "Business class is permitted for flights over 6 hours with department head approval.\n\n"
                "Entertainment expenses for client meetings are limited to $150 per person. "
                "All entertainment expenses must include the business purpose, "
                "attendee names, and their company affiliations."
            ),
            "category": "travel",
        },
    ]
    return policies


def generate_questions(transactions: list[dict], policies: list[dict]) -> list[dict]:
    """Generate labeled QA pairs requiring structured, unstructured, or hybrid reasoning."""
    questions = []

    # ── Simple Lookup (structured only) ──────────────────────────────
    # Q1: Find specific transaction
    txn = transactions[0]
    questions.append({
        "question_id": "CQ-001",
        "question": f"What is the amount of transaction {txn['transaction_id']}?",
        "answer": f"${txn['amount']:,.2f}",
        "question_type": "simple_lookup",
        "evidence_type": "structured",
        "evidence_refs": [txn["transaction_id"]],
    })

    # Q2: Count transactions
    it_txns = [t for t in transactions if t["department"] == "IT"]
    questions.append({
        "question_id": "CQ-002",
        "question": "How many transactions were made by the IT department?",
        "answer": str(len(it_txns)),
        "question_type": "simple_lookup",
        "evidence_type": "structured",
        "evidence_refs": ["transactions_table"],
    })

    # Q3: Total spending by vendor
    acme_txns = [t for t in transactions if t["vendor"] == "Acme Corp"]
    acme_total = sum(t["amount"] for t in acme_txns)
    questions.append({
        "question_id": "CQ-003",
        "question": "What is the total amount spent with Acme Corp?",
        "answer": f"${acme_total:,.2f}",
        "question_type": "simple_lookup",
        "evidence_type": "structured",
        "evidence_refs": ["transactions_table"],
    })

    # ── Simple Lookup (unstructured only) ────────────────────────────
    questions.append({
        "question_id": "CQ-004",
        "question": "What is the accommodation limit per night for domestic travel?",
        "answer": "$200",
        "question_type": "simple_lookup",
        "evidence_type": "unstructured",
        "evidence_refs": ["POL-005"],
    })

    questions.append({
        "question_id": "CQ-005",
        "question": "How long must financial records be retained?",
        "answer": "7 years",
        "question_type": "simple_lookup",
        "evidence_type": "unstructured",
        "evidence_refs": ["POL-003"],
    })

    # ── Hybrid Lookup (structured + unstructured) ────────────────────
    # Find a large transaction and check its compliance
    large_txns = [t for t in transactions if t["amount"] > 25000]
    if large_txns:
        lt = large_txns[0]
        questions.append({
            "question_id": "CQ-006",
            "question": f"What approval level is required for transaction {lt['transaction_id']} "
                        f"which has an amount of ${lt['amount']:,.2f}?",
            "answer": "CFO + Board Approval",
            "question_type": "hybrid_lookup",
            "evidence_type": "both",
            "evidence_refs": [lt["transaction_id"], "POL-001"],
        })

    # Find non-compliant transactions
    violations = [t for t in transactions if not t["compliant"]]
    if violations:
        v = violations[0]
        questions.append({
            "question_id": "CQ-007",
            "question": f"Is transaction {v['transaction_id']} (${v['amount']:,.2f}, "
                        f"approved by {v['actual_approver']}) compliant with the "
                        f"Expense Approval Policy?",
            "answer": (f"No. The transaction amount of ${v['amount']:,.2f} requires "
                       f"{v['required_approval']} approval, but it was only approved by "
                       f"{v['actual_approver']}."),
            "question_type": "hybrid_lookup",
            "evidence_type": "both",
            "evidence_refs": [v["transaction_id"], "POL-001"],
        })

    # ── Multi-hop Reasoning ──────────────────────────────────────────
    # Which IT vendor purchases require legal review?
    it_vendor_txns = [t for t in transactions
                      if t["department"] == "IT" and t["amount"] > 10000]
    if it_vendor_txns:
        vendors_needing_review = list(set(t["vendor"] for t in it_vendor_txns))
        questions.append({
            "question_id": "CQ-008",
            "question": ("Which vendors that supplied the IT department have contracts "
                         "that require Legal review according to the Vendor Management Policy?"),
            "answer": ", ".join(sorted(vendors_needing_review)),
            "question_type": "multi_hop",
            "evidence_type": "both",
            "evidence_refs": ["transactions_table", "POL-002"],
        })

    # Cross-reference data retention with actual data
    questions.append({
        "question_id": "CQ-009",
        "question": ("A customer relationship ended in 2020. According to the Data Protection "
                     "Policy, until what year must their data be retained?"),
        "answer": "2025",
        "question_type": "multi_hop",
        "evidence_type": "unstructured",
        "evidence_refs": ["POL-003"],
    })

    # ── Compliance Reasoning ─────────────────────────────────────────
    cloud_txns = [t for t in transactions if t["category"] == "Cloud Services"]
    if cloud_txns:
        ct = cloud_txns[0]
        questions.append({
            "question_id": "CQ-010",
            "question": (f"Transaction {ct['transaction_id']} is a Cloud Services purchase "
                         f"from {ct['vendor']}. What compliance requirements apply to this "
                         f"purchase based on company policies?"),
            "answer": ("The purchase must be approved by IT department (IT Security Policy). "
                       "If the vendor handles personal data, they must provide SOC 2 Type II "
                       "certification and sign a DPA (Vendor Management Policy). "
                       "Cloud services storing personal data must be hosted in the EU or "
                       "countries with adequacy decisions (Data Protection Policy)."),
            "question_type": "compliance_reasoning",
            "evidence_type": "both",
            "evidence_refs": [ct["transaction_id"], "POL-002", "POL-003", "POL-004"],
        })

    # Overall compliance check
    total_violations = len(violations)
    questions.append({
        "question_id": "CQ-011",
        "question": "How many transactions in the dataset violate the Expense Approval Policy?",
        "answer": str(total_violations),
        "question_type": "compliance_reasoning",
        "evidence_type": "both",
        "evidence_refs": ["transactions_table", "POL-001"],
    })

    # Security compliance question
    questions.append({
        "question_id": "CQ-012",
        "question": ("If a data breach affecting EU residents is discovered at 3 PM on Monday, "
                     "what is the latest time the supervisory authority must be notified?"),
        "answer": "By 3 PM on Thursday (within 72 hours)",
        "question_type": "compliance_reasoning",
        "evidence_type": "unstructured",
        "evidence_refs": ["POL-003"],
    })

    return questions


def generate_compliance_dataset(output_dir: Path, n_transactions: int = 50):
    """Generate the full compliance dataset and save to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic transactions ...")
    transactions = generate_transactions(n_transactions)
    with open(output_dir / "transactions.json", "w", encoding="utf-8") as f:
        json.dump(transactions, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(transactions)} transactions")

    print("Generating policy documents ...")
    policies = generate_policies()
    with open(output_dir / "policies.json", "w", encoding="utf-8") as f:
        json.dump(policies, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(policies)} policies")

    print("Generating QA pairs ...")
    questions = generate_questions(transactions, policies)
    with open(output_dir / "questions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(questions)} questions")

    # Summary
    type_counts = {}
    for q in questions:
        qt = q["question_type"]
        type_counts[qt] = type_counts.get(qt, 0) + 1
    print(f"\n  Question types: {type_counts}")

    return transactions, policies, questions


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.config import settings as cfg
    generate_compliance_dataset(cfg.COMPLIANCE_DIR)
