from unittest.mock import patch

import pytest

from app.agents.nodes.summarize import summarize


async def _fake_stream(_system_prompt, user_prompt):
    # Surface the prompt actually sent so the test can assert the raw_text
    # fallback was used for the description.
    for token in [f"[{user_prompt}]"]:
        yield token


@pytest.mark.asyncio
async def test_summarize_does_not_crash_when_extracted_fields_missing() -> None:
    """Regression test: `summarize` used to do `state["extracted_fields"]` (a bare
    dict subscript). If an upstream node like `extract_fields` fails, `traced_node`
    swallows the exception and never writes the `extracted_fields` key at all
    (state is `total=False`) -- so `summarize`, which is deliberately NOT wrapped
    in `traced_node`, would raise an uncaught KeyError and crash the whole graph
    run before ever reaching `finalize`. `summarize` must degrade gracefully and
    fall back to `raw_text` for the description, same as every other node.
    """
    state = {
        "raw_text": "Tablets cracked in blister, no fields could be extracted.",
        "trace": [],
        "errors": ["extract_fields: groq outage"],
        # note: no "extracted_fields" key at all, simulating extract_fields
        # having failed and traced_node swallowing the exception.
    }
    with (
        patch("app.agents.nodes.summarize.get_stream_writer", return_value=lambda *_: None),
        patch("app.agents.nodes.summarize.call_groq_stream", new=_fake_stream),
    ):
        result = await summarize(state)

    assert "Tablets cracked in blister, no fields could be extracted." in result["summary"]
