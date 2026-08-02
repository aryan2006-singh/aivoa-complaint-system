from langgraph.graph import END, StateGraph

from app.agents.nodes.check_completeness import check_completeness
from app.agents.nodes.check_duplicates import check_duplicates
from app.agents.nodes.classify_risk import classify_risk
from app.agents.nodes.extract_fields import extract_fields
from app.agents.nodes.finalize import finalize, finalize_duplicate
from app.agents.nodes.recommend_capa import recommend_capa
from app.agents.nodes.regulatory_reportability import regulatory_reportability
from app.agents.nodes.suggest_root_cause import suggest_root_cause
from app.agents.nodes.summarize import summarize
from app.agents.state import ComplaintAgentState


async def merge_checks(state: ComplaintAgentState) -> dict:
    return {}


def route_after_duplicate_check(state: ComplaintAgentState) -> list[str]:
    """Fan out (or short-circuit) after `merge_checks` joins `check_completeness`
    and `check_duplicates`.

    This is attached to `merge_checks`'s outgoing edge rather than directly to
    `check_duplicates`'s outgoing edge. LangGraph's fan-in join fires as soon as
    *any* of its incoming edges delivers a write in a superstep -- it does not
    require *all* of them to. `check_completeness -> merge_checks` is a plain,
    unconditional edge, so if the duplicate/continue branch lived on
    `check_duplicates`'s edge instead, `merge_checks` (and therefore
    `classify_risk`/`suggest_root_cause`) would still fire on the duplicate path
    purely because of `check_completeness`'s edge, breaking the short-circuit.
    Making both `check_completeness -> merge_checks` and
    `check_duplicates -> merge_checks` plain edges (a true join of both
    branches) and deciding where to go *after* that join is the only way to
    guarantee `classify_risk`/`suggest_root_cause` are genuinely skipped when
    `is_duplicate` is true.
    """
    if state.get("is_duplicate"):
        return ["finalize_duplicate"]
    return ["classify_risk", "suggest_root_cause"]


def build_graph():
    graph = StateGraph(ComplaintAgentState)

    graph.add_node("extract_fields", extract_fields)
    graph.add_node("check_completeness", check_completeness)
    graph.add_node("check_duplicates", check_duplicates)
    graph.add_node("merge_checks", merge_checks)
    graph.add_node("classify_risk", classify_risk)
    graph.add_node("regulatory_reportability", regulatory_reportability)
    graph.add_node("suggest_root_cause", suggest_root_cause)
    graph.add_node("recommend_capa", recommend_capa)
    graph.add_node("summarize", summarize)
    graph.add_node("finalize", finalize)
    graph.add_node("finalize_duplicate", finalize_duplicate)

    graph.set_entry_point("extract_fields")
    graph.add_edge("extract_fields", "check_completeness")
    graph.add_edge("extract_fields", "check_duplicates")
    graph.add_edge("check_completeness", "merge_checks")
    graph.add_edge("check_duplicates", "merge_checks")
    graph.add_conditional_edges(
        "merge_checks",
        route_after_duplicate_check,
        ["finalize_duplicate", "classify_risk", "suggest_root_cause"],
    )
    graph.add_edge("classify_risk", "regulatory_reportability")
    graph.add_edge("suggest_root_cause", "recommend_capa")
    graph.add_edge("regulatory_reportability", "summarize")
    graph.add_edge("recommend_capa", "summarize")
    graph.add_edge("summarize", "finalize")
    graph.add_edge("finalize", END)
    graph.add_edge("finalize_duplicate", END)

    return graph.compile()
