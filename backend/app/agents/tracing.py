import functools
from collections.abc import Awaitable, Callable

from langgraph.config import get_stream_writer

from app.agents.state import ComplaintAgentState


def traced_node(name: str):
    def decorator(fn: Callable[[ComplaintAgentState], Awaitable[dict]]):
        @functools.wraps(fn)
        async def wrapper(state: ComplaintAgentState) -> dict:
            writer = get_stream_writer()
            writer({"step": name, "status": "started"})
            try:
                result = await fn(state)
            except Exception as exc:  # noqa: BLE001 - node failures degrade, not crash
                writer({"step": name, "status": "error", "data": str(exc)})
                return {"errors": [f"{name}: {exc}"], "trace": [{"step": name, "status": "error"}]}
            writer({"step": name, "status": "completed", "data": result})
            return {**result, "trace": [{"step": name, "status": "completed"}]}

        return wrapper

    return decorator
