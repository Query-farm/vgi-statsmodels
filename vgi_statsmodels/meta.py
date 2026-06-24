"""Shared discovery / description metadata helpers for the statsmodels worker.

Centralizes the per-object tags that the ``vgi-lint`` strict profile expects on
**every** function and table, plus the canonical GitHub ``source_url`` builder.

Each function surfaces these in its ``Meta.tags``:

- ``vgi.title`` (VGI124)        — human-friendly display name
- ``vgi.doc_llm`` (VGI112)      — Markdown narrative aimed at LLMs / agents
- ``vgi.doc_md`` (VGI113)       — Markdown narrative aimed at human docs
- ``vgi.keywords`` (VGI126)     — comma-separated search terms / synonyms
- ``vgi.source_url`` (VGI128)   — link to the implementing source file

``vgi.title`` must not normalize-equal the machine name, and ``vgi.doc_llm`` /
``vgi.doc_md`` must carry distinct content.
"""

from __future__ import annotations

#: Base GitHub blob URL for source files in this repo (pinned to ``main``).
SOURCE_BASE = "https://github.com/Query-farm/vgi-statsmodels/blob/main"


def source_url(relative_path: str) -> str:
    """Build the implementation ``vgi.source_url`` for a repo-relative file.

    Args:
        relative_path: Path under the repo root, e.g. ``vgi_statsmodels/tables.py``.

    Returns:
        The canonical GitHub blob URL for the file (pinned to ``main``).
    """
    return f"{SOURCE_BASE}/{relative_path}"


def object_tags(
    *,
    title: str,
    doc_llm: str,
    doc_md: str,
    keywords: str,
    relative_path: str,
) -> dict[str, str]:
    """Build the five standard per-object discovery / description tags.

    Args:
        title: Human display name (must add a word beyond the machine name).
        doc_llm: Markdown narrative for an LLM / agent audience.
        doc_md: Markdown narrative for human docs (distinct from ``doc_llm``).
        keywords: Comma-separated search terms / synonyms.
        relative_path: Implementing file, relative to the repo root.

    Returns:
        A tag dict ready to merge into a function's ``Meta.tags``.
    """
    return {
        "vgi.title": title,
        "vgi.doc_llm": doc_llm,
        "vgi.doc_md": doc_md,
        "vgi.keywords": keywords,
        "vgi.source_url": source_url(relative_path),
    }
