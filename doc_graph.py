from __future__ import annotations

from typing import Any, Dict, List

CHILD_KEYS = ("sections", "children", "parts", "subsections", "items", "clauses")
TEXT_FIELDS = (
    "content",
    "text",
    "summary",
    "summary_text",
    "body",
    "description",
    "details",
    "abstract",
    "notes",
)
META_KEYS = {"title", "heading", "name", "id"}


def build_document_graph(document: Any) -> Dict[str, Any]:
    """Normalize an arbitrary nested document JSON payload into graph primitives."""
    if document is None:
        raise ValueError("Document payload is empty.")

    normalized = _normalize_node(document, "Document")

    node_counter = 0
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    child_counts: Dict[int, int] = {}
    max_depth = 0

    def walk(section: Dict[str, Any], parent_id: int | None, level: int, trail: List[str]) -> Dict[str, Any]:
        nonlocal node_counter, max_depth
        if not isinstance(section, dict):
            section = {"title": str(section), "children": []}
        node_counter += 1
        title = section.get("title") or f"Section {node_counter}"
        content = section.get("content") or ""
        snippet = _snippet(content)
        path_tokens = trail + [title]
        node_id = node_counter
        max_depth = max(max_depth, level)

        nodes.append(
            {
                "id": node_id,
                "title": title,
                "snippet": snippet,
                "level": level,
                "path": " / ".join(path_tokens),
            }
        )
        if parent_id is not None:
            edges.append(
                {
                    "from": parent_id,
                    "to": node_id,
                    "type": "hierarchy",
                    "reason": f"{title} nests under {trail[-1]}",
                }
            )
            child_counts[parent_id] = child_counts.get(parent_id, 0) + 1

        children = section.get("children") or []
        child_entries = []
        for child in children:
            child_entries.append(walk(child, node_id, level + 1, path_tokens))

        return {"title": title, "snippet": snippet, "children": child_entries}

    hierarchy = [walk(normalized, parent_id=None, level=0, trail=[])]

    leaf_nodes = sum(1 for node in nodes if child_counts.get(node["id"], 0) == 0)
    avg_branch = (
        round(sum(child_counts.values()) / len(child_counts), 2)
        if child_counts
        else 0.0
    )

    return {
        "graph": {"nodes": nodes, "edges": edges},
        "hierarchy": hierarchy,
        "stats": {
            "total_nodes": len(nodes),
            "leaf_nodes": leaf_nodes,
            "max_depth": max_depth,
            "average_branching": avg_branch,
            "root_title": hierarchy[0]["title"] if hierarchy else "Document",
        },
    }


def _normalize_node(value: Any, title_hint: str) -> Dict[str, Any]:
    """Convert arbitrary JSON value into a normalized {title, content, children} mapping."""
    if isinstance(value, dict):
        title = str(
            value.get("title")
            or value.get("heading")
            or value.get("name")
            or value.get("id")
            or title_hint
        )
        content = _extract_content(value)
        children: List[Dict[str, Any]] = []
        visited_keys = set()

        # Preferred child keys (sections, children, etc.) flatten directly.
        for child_key in CHILD_KEYS:
            data = value.get(child_key)
            if isinstance(data, list) and data:
                visited_keys.add(child_key)
                for idx, child in enumerate(data, 1):
                    hint = _humanize(child_key, idx)
                    children.append(_normalize_node(child, hint))

        for key, data in value.items():
            if key in visited_keys or key in META_KEYS or key in TEXT_FIELDS:
                continue
            if data in (None, "", []):
                continue
            label = _humanize(key)
            if isinstance(data, list):
                if not data:
                    continue
                list_children = []
                for idx, item in enumerate(data, 1):
                    hint = _humanize(key, idx)
                    list_children.append(_normalize_node(item, hint))
                children.append({"title": label, "content": "", "children": list_children})
            elif isinstance(data, dict):
                children.append(_normalize_node(data, label))
            else:
                children.append({"title": label, "content": str(data), "children": []})

        if not content:
            content = _stringify_fallback(value)

        return {"title": title, "content": content, "children": children}

    if isinstance(value, list):
        list_children = []
        for idx, item in enumerate(value, 1):
            hint = _humanize(title_hint, idx)
            list_children.append(_normalize_node(item, hint))
        return {"title": title_hint, "content": "", "children": list_children}

    return {"title": title_hint, "content": str(value), "children": []}


def _humanize(key: str, idx: int | None = None) -> str:
    base = key.replace("_", " ").strip().title() or "Section"
    if idx is not None and idx > 0:
        return f"{base} {idx}"
    if base.lower().endswith("s") and idx is None:
        return base
    return base


def _extract_content(section: Dict[str, Any]) -> str:
    for key in TEXT_FIELDS:
        val = section.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _stringify_fallback(section: Dict[str, Any], limit: int = 4) -> str:
    parts: List[str] = []
    for key, val in section.items():
        if key in META_KEYS or key in CHILD_KEYS or key in TEXT_FIELDS:
            continue
        if isinstance(val, (str, int, float)) and str(val).strip():
            parts.append(f"{_humanize(key)}: {val}")
        elif isinstance(val, bool):
            parts.append(f"{_humanize(key)}: {val}")
        if len(parts) >= limit:
            break
    return " | ".join(parts)


def _snippet(content: str, limit: int = 160) -> str:
    if not content:
        return ""
    content = " ".join(content.split())
    return content if len(content) <= limit else f"{content[:limit].rstrip()}…"
