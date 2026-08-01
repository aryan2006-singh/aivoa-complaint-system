import json
from collections.abc import AsyncIterator

from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

_client = AsyncGroq(api_key=settings.groq_api_key)


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, max=4))
async def call_groq_json(system_prompt: str, user_prompt: str) -> dict:
    response = await _client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        repair = await _client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": "Fix the following into valid JSON only, no prose."},
                {"role": "user", "content": raw},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(repair.choices[0].message.content)


async def call_groq_stream(system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
    stream = await _client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
