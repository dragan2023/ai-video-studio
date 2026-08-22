"""ComfyUI UI-workflow to API-prompt conversion helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class WorkflowFormatError(ValueError):
    """Raised when a ComfyUI UI workflow cannot be converted."""


def load_ui_workflow(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("nodes"), list):
        raise WorkflowFormatError(f"not a ComfyUI UI workflow: {path}")
    return payload


def ui_workflow_to_api(workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert the exported UI graph into the graph accepted by POST /prompt."""
    links: dict[int, tuple[int, int]] = {}
    for raw in workflow.get("links", []):
        if len(raw) >= 3:
            links[int(raw[0])] = (int(raw[1]), int(raw[2]))

    prompt: dict[str, dict[str, Any]] = {}
    for node in workflow.get("nodes", []):
        node_id = str(node["id"])
        class_type = node.get("type")
        if not class_type:
            continue
        widgets = node.get("widgets_values", [])
        widget_index = 0
        inputs: dict[str, Any] = {}
        for input_def in node.get("inputs", []):
            name = input_def.get("name")
            if not name:
                continue
            link_id = input_def.get("link")
            if link_id is not None:
                source = links.get(int(link_id))
                if source is not None:
                    inputs[name] = [str(source[0]), source[1]]
                continue
            widget = input_def.get("widget")
            if widget is None:
                continue
            widget_name = widget.get("name", name) if isinstance(widget, dict) else name
            if isinstance(widgets, dict):
                if widget_name in widgets:
                    inputs[name] = copy.deepcopy(widgets[widget_name])
                elif name in widgets:
                    inputs[name] = copy.deepcopy(widgets[name])
            elif isinstance(widgets, list) and widget_index < len(widgets):
                inputs[name] = copy.deepcopy(widgets[widget_index])
            widget_index += 1
        prompt[node_id] = {"class_type": class_type, "inputs": inputs}
    return prompt


def patch_inputs(
    graph: dict[str, dict[str, Any]],
    node_id: int | str,
    **changes: Any,
) -> None:
    key = str(node_id)
    if key not in graph:
        raise WorkflowFormatError(f"node {node_id} is not present in API graph")
    graph[key].setdefault("inputs", {}).update(changes)


def node_ids_by_class(prompt: dict[str, dict[str, Any]], class_type: str) -> list[str]:
    return [node_id for node_id, node in prompt.items() if node.get("class_type") == class_type]
