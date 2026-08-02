import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.agents.graph import build_graph
from app.services.file_parsing import extract_text_from_eml, extract_text_from_pdf

router = APIRouter(prefix="/api/complaints", tags=["intake"])
_graph = build_graph()


async def _resolve_raw_text(text: str | None, file: UploadFile | None) -> str:
    if file is not None:
        data = await file.read()
        if file.filename and file.filename.lower().endswith(".pdf"):
            return extract_text_from_pdf(data)
        if file.filename and file.filename.lower().endswith(".eml"):
            return extract_text_from_eml(data)
        raise HTTPException(status_code=400, detail="Unsupported file type; use .pdf or .eml")
    if text:
        return text
    raise HTTPException(status_code=400, detail="Provide either 'text' or 'file'")


@router.post("/intake")
async def intake(text: str | None = Form(default=None), file: UploadFile | None = File(default=None)):
    raw_text = await _resolve_raw_text(text, file)

    async def event_stream():
        async for event in _graph.astream(
            {"raw_text": raw_text, "trace": [], "errors": []}, stream_mode="custom"
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
