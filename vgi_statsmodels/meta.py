"""Shared discovery / description metadata helpers for the statsmodels worker.

Centralizes the per-object tags that the ``vgi-lint`` strict profile expects on
**every** function and table.

Each function surfaces these in its ``Meta.tags``:

- ``vgi.title`` (VGI124)        — human-friendly display name
- ``vgi.doc_llm`` (VGI112)      — Markdown narrative aimed at LLMs / agents
- ``vgi.doc_md`` (VGI113)       — Markdown narrative aimed at human docs
- ``vgi.keywords`` (VGI126)     — JSON array of search terms / synonyms

``vgi.title`` must not normalize-equal the machine name, and ``vgi.doc_llm`` /
``vgi.doc_md`` must carry distinct content. ``vgi.keywords`` must be a JSON array
of strings (VGI138), not a comma-separated string. Per-object ``vgi.source_url``
is intentionally omitted: the canonical ``source_url`` lives only on the catalog
object (VGI139).
"""

from __future__ import annotations

import json
from collections.abc import Sequence


def object_tags(
    *,
    title: str,
    doc_llm: str,
    doc_md: str,
    keywords: Sequence[str],
) -> dict[str, str]:
    """Build the four standard per-object discovery / description tags.

    Args:
        title: Human display name (must add a word beyond the machine name).
        doc_llm: Markdown narrative for an LLM / agent audience.
        doc_md: Markdown narrative for human docs (distinct from ``doc_llm``).
        keywords: Search terms / synonyms; serialized to a JSON array of strings.

    Returns:
        A tag dict ready to merge into a function's ``Meta.tags``.
    """
    return {
        "vgi.title": title,
        "vgi.doc_llm": doc_llm,
        "vgi.doc_md": doc_md,
        "vgi.keywords": json.dumps(list(keywords)),
    }
