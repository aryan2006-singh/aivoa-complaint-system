# backend/scripts/seed.py
import asyncio
import uuid

from app.db.models import AIAssessment, Complaint
from app.db.models import Base
from app.db.session import SessionLocal, engine
from app.services.embeddings import get_embedding

DEMO_COMPLAINTS = [
    dict(product_name="Amoxicillin 500mg", batch_lot_number="B10021", category="Quality",
         description="Several tablets in the blister pack show yellow discoloration and an unusual odor.",
         severity="Major"),
    dict(product_name="Amoxicillin 500mg", batch_lot_number="B10021", category="Quality",
         description="Customer reports yellow-tinted tablets with an off smell from the same batch.",
         severity="Major"),  # deliberate near-duplicate of the row above
    dict(product_name="Metformin 1000mg", batch_lot_number="B20044", category="Packaging",
         description="Outer carton was crushed during shipping; blister strips intact.",
         severity="Minor"),
    dict(product_name="Insulin Glargine", batch_lot_number="B30099", category="Adverse Event",
         description="Patient reported injection site reaction and reduced efficacy; possible cold-chain excursion.",
         severity="Critical"),
    dict(product_name="Ibuprofen 200mg", batch_lot_number="B40012", category="Quality",
         description="Bottle seal was broken on arrival; tablet count appears correct.",
         severity="Minor"),
    dict(product_name="Losartan 50mg", batch_lot_number="B50077", category="Efficacy",
         description="Patient reports no change in blood pressure after two weeks on this batch.",
         severity="Major"),
]


async def main() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with SessionLocal() as db:
        for i, row in enumerate(DEMO_COMPLAINTS, start=1):
            complaint = Complaint(
                id=uuid.uuid4(),
                complaint_number=f"CMPL-2026-{i:06d}",
                product_name=row["product_name"],
                batch_lot_number=row["batch_lot_number"],
                source="manual",
                description=row["description"],
                category=row["category"],
                severity=row["severity"],
                status="New",
            )
            db.add(complaint)
            await db.flush()
            db.add(
                AIAssessment(
                    complaint_id=complaint.id,
                    extracted_fields={},
                    completeness_score=1.0,
                    missing_fields=[],
                    embedding=get_embedding(row["description"]),
                    risk_classification=row["severity"],
                    summary=row["description"][:120],
                    model_used="seed-script",
                )
            )
        await db.commit()
    print(f"Seeded {len(DEMO_COMPLAINTS)} demo complaints.")


if __name__ == "__main__":
    asyncio.run(main())
