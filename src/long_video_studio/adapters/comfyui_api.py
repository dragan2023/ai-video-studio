"""ComfyUI UI-workflow to API-prompt conversion helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


NON_EXECUTABLE_TYPES = frozenset({
    "MarkdownNote",
    "Fast Groups Bypasser (rgthree)",
})


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
        if not class_type or class_type in NON_EXECUTABLE_TYPES:
            continue
        widgets = node.get("widgets_values", [])
        widget_index = 0
        inputs: dict[str, Any] = {}
        for input_def in node.get("inputs", []):
            name = input_def.get("name")
            if not name:
                continue
            link_id = input_def.get("link")
            widget = input_def.get("widget")
            if link_id is not None:
                source = links.get(int(link_id))
                if source is not None:
                    inputs[name] = [str(source[0]), source[1]]
                # Linked widgets still occupy a slot in widgets_values.
                if widget is not None:
                    widget_index += 1
                continue
            if widget is None:
                continue
            widget_name = widget.get("name", name) if isinstance(widget, dict) else name
            if isinstance(widgets, dict):
                if widget_name in widgets:
                    inputs[name] = copy.deepcopy(widgets[widget_name])
                elif name in widgets:
                    inputs[name] = copy.deepcopy(widgets[name])
            elif isinstance(widgets, list):
                value_index = widget_index
                if class_type == "ReservedVRAMSetter":
                    # The seed control widget is stored between seed and
                    # auto_max_reserved but is not an API input.
                    value_index += 1 if widget_name in {"auto_max_reserved", "clean_gpu_before"} else 0
                if value_index < len(widgets):
                    inputs[name] = copy.deepcopy(widgets[value_index])
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


def prune_to_output(graph: dict[str, dict[str, Any]], output_node: int | str) -> dict[str, dict[str, Any]]:
    """Keep only the dependency closure of one output node.

    Work-Fisher stores 5/10/15-second branches in one UI file. ComfyUI treats
    every output node in an API prompt as executable, so submitting the whole
    graph would render all branches. This closure preserves only the selected
    FL2VA or Ref2VA branch and its shared model dependencies.
    """
    keep: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in keep or node_id not in graph:
            return
        keep.add(node_id)
        for value in graph[node_id].get("inputs", {}).values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                visit(value[0])

    visit(str(output_node))
    return {node_id: graph[node_id] for node_id in keep}
