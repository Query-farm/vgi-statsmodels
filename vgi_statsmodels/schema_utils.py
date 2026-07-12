"""Shared Arrow-schema helper for the statsmodels worker.

Keeps column-comment plumbing in one place so every function exposes
consistent, documented output schemas to DuckDB.
"""

from __future__ import annotations

import json

import pyarrow as pa

# Arrow -> DuckDB SQL type names for the column types this worker emits. Used to
# render `vgi.result_columns_schema` (VGI307/VGI321/VGI322) straight from the
# authoritative output `pa.schema`, so the declared result schema can never
# drift from what the function actually returns (VGI910 checks the two match).
_ARROW_TO_DUCKDB: dict[pa.DataType, str] = {
    pa.string(): "VARCHAR",
    pa.float64(): "DOUBLE",
    pa.int32(): "INTEGER",
    pa.int64(): "BIGINT",
    pa.bool_(): "BOOLEAN",
}


def result_columns_schema(schema: pa.Schema) -> str:
    """Render an output ``pa.schema`` as a ``vgi.result_columns_schema`` JSON string.

    Emits a JSON array of ``{name, type, description}`` objects -- one per
    returned column -- pulling ``type`` from :data:`_ARROW_TO_DUCKDB` and
    ``description`` from each field's ``comment`` metadata. Deriving it from the
    real output schema keeps the declared result schema and the actual returned
    columns in lockstep.

    Args:
        schema: The function's output Arrow schema (fields carry ``comment``
            metadata via :func:`field`).

    Returns:
        A JSON string suitable for the ``vgi.result_columns_schema`` tag.
    """
    columns = []
    for f in schema:
        duckdb_type = _ARROW_TO_DUCKDB.get(f.type)
        if duckdb_type is None:  # pragma: no cover - guards against a new column type
            raise ValueError(f"no DuckDB type mapping for Arrow type {f.type!r} (column {f.name!r})")
        comment = (f.metadata or {}).get(b"comment", b"").decode("utf-8")
        columns.append({"name": f.name, "type": duckdb_type, "description": comment})
    return json.dumps(columns)


def field(
    name: str,
    type: pa.DataType,
    comment: str,
    *,
    nullable: bool = True,
) -> pa.Field:
    """Build a ``pa.Field`` carrying a column comment in its metadata.

    The ``comment`` metadata key is the framework's transport for column
    comments -- DuckDB surfaces it via ``duckdb_columns()`` and ``DESCRIBE``.

    Args:
        name: Column name.
        type: Arrow data type.
        comment: Human-readable column comment.
        nullable: Whether the column is nullable.

    Returns:
        A pyarrow Field with the comment attached as metadata.
    """
    return pa.field(
        name,
        type,
        nullable=nullable,
        metadata={b"comment": comment.encode("utf-8")},
    )
