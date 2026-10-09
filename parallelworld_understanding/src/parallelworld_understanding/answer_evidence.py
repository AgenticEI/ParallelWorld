from __future__ import annotations

import copy
from typing import Any


ALL = "all"
TOP1_ROUTE = "top1-route"


def mode(config) -> str:
    value = str(getattr(config, "answer_evidence_mode", TOP1_ROUTE) or TOP1_ROUTE).strip().lower()
    return value if value in {ALL, TOP1_ROUTE} else TOP1_ROUTE


def _top1_route(selected_groups: list[list[Any]]) -> list[Any]:
    if not selected_groups or not selected_groups[-1]:
        return []
    route = []
    seen: set[int] = set()
    node = selected_groups[-1][0]
    while int(getattr(node, "depth", 0) or 0) > 0:
        identity = id(node)
        if identity in seen:
            raise ValueError("Top-1 answer route contains a parent cycle.")
        seen.add(identity)
        route.append(node)
        node = getattr(node, "parent_node", None)
        if node is None and int(getattr(route[-1], "depth", 0) or 0) > 1:
            raise ValueError("Top-1 answer route is missing an ancestor node.")
    route.reverse()
    depths = [int(getattr(item, "depth", 0) or 0) for item in route]
    if depths != list(range(1, len(route) + 1)):
        raise ValueError(f"Top-1 answer route has non-contiguous depths: {depths}")
    return route


def select(
    config,
    selected_groups: list[list[Any]],
    history: list[dict[str, Any]],
) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    if mode(config) == ALL:
        return [list(group) for group in selected_groups], [dict(item) for item in history]

    route = _top1_route(selected_groups)
    route_groups: list[list[Any]] = []
    branch_by_depth: dict[int, str] = {}
    for node in route:
        route_node = copy.copy(node)
        # Preserve every action frame attached to this selected route node.
        # Unselected sibling branches are still excluded because the route is
        # reconstructed only from the final rank-1 node's ancestor chain.
        route_groups.append([route_node])
        branch_by_depth[int(getattr(node, "depth", 0) or 0)] = str(getattr(node, "branch_id", ""))

    route_history = []
    for item in history:
        step = int(item.get("step") or 0)
        if step not in branch_by_depth:
            continue
        route_history.append(
            {
                **item,
                "selected_count": 1,
                "selected_branch_ids": [branch_by_depth[step]],
            }
        )
    return route_groups, route_history
