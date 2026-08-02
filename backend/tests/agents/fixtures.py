FAKE_EXTRACT = {
    "product_name": {"value": "Amoxicillin 500mg", "confidence": 0.9},
    "batch_lot_number": {"value": "B99999", "confidence": 0.9},
    "customer_name": {"value": "Jane Doe", "confidence": 0.8},
    "customer_contact": {"value": "jane@example.com", "confidence": 0.8},
    "date_received": {"value": "2026-08-02", "confidence": 0.8},
    "description": {"value": "Tablets are cracked in several blister cells.", "confidence": 0.9},
    "category": {"value": "Quality", "confidence": 0.8},
    "source": {"value": "email", "confidence": 0.7},
}
FAKE_RISK = {"classification": "Major", "rationale": "Cracked tablets may affect dosing."}
FAKE_REGULATORY = {"reportable": False, "rationale": "No patient harm reported."}
FAKE_ROOT_CAUSE = {"category": "Machine", "explanation": "Likely tableting press fault."}
FAKE_CAPA = {"corrective": "Inspect press tooling.", "preventive": "Add in-process cracked-tablet detection."}


async def fake_stream(*_args, **_kwargs):
    for token in ["Summary ", "of ", "the ", "complaint."]:
        yield token
