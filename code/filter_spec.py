"""Filter specification for the build_filter tool: validation and execution.

A filter spec is a JSON object; every key is optional:
  {
    "acn": [1884087],                         # look up specific reports
    "year_min": 2022, "year_max": 2024,       # inclusive incident-year bounds
    "where": {"aircraft_family": ["B737"],    # coded fields: OR within a field, AND across fields;
              "human_factors": ["Fatigue"]},  #   list-valued fields match if any value overlaps
    "text_any": ["handoff", "turnover"],      # some narrative contains one of these (case-insensitive)
    "group_by": "anomaly"                     # return counts per value instead of matching ACNs
  }

Values in "where" must be ones observed in the corpus (data/profile/filter_values.json),
so an invented category is rejected rather than silently matching nothing.
"""

import json

import duckdb

from preprocess import DB_PATH, ROOT

FILTER_VALUES = json.loads((ROOT / "data/profile/filter_values.json").read_text())
YEAR_RANGE = FILTER_VALUES["year"]
FIELDS = {f: set(v) for f, v in FILTER_VALUES.items() if f != "year"}
KEYS = {"acn", "year_min", "year_max", "where", "text_any", "group_by"}


def validate(spec):
    """Return a list of human-readable problems; an empty list means the spec is valid."""
    if not isinstance(spec, dict):
        return ["filter spec must be a JSON object"]
    errors = [f"unknown key '{k}'" for k in spec if k not in KEYS]

    acns = spec.get("acn", [])
    if not (isinstance(acns, list) and all(isinstance(a, int) for a in acns)):
        errors.append("'acn' must be a list of integers")

    years = {}
    for key in ("year_min", "year_max"):
        year = spec.get(key)
        if isinstance(year, int) and YEAR_RANGE["min"] <= year <= YEAR_RANGE["max"]:
            years[key] = year
        elif year is not None:
            errors.append(f"'{key}' must be an integer in {YEAR_RANGE['min']}-{YEAR_RANGE['max']}, got {year!r}")
    if len(years) == 2 and years["year_min"] > years["year_max"]:
        errors.append("'year_min' is after 'year_max'")

    # Specs come from an LLM tool call, so malformed shapes must produce errors, not exceptions.
    where = spec.get("where", {})
    if not isinstance(where, dict):
        errors.append("'where' must be an object mapping field names to lists of values")
        where = {}
    for field, values in where.items():
        if field not in FIELDS:
            errors.append(f"unknown field '{field}'")
        elif not (isinstance(values, list) and values):
            errors.append(f"'where.{field}' must be a non-empty list")
        else:
            errors += [f"'{v}' is not a value of '{field}'" for v in values
                       if not (isinstance(v, str) and v in FIELDS[field])]

    terms = spec.get("text_any", [])
    if not (isinstance(terms, list) and all(isinstance(t, str) and t.strip() for t in terms)):
        errors.append("'text_any' must be a list of non-empty strings")

    group_by = spec.get("group_by")
    if group_by is not None and not (isinstance(group_by, str) and group_by in FIELDS):
        errors.append(f"cannot group by unknown field '{group_by}'")
    return errors


def _is_list_column(con, field):
    dtype = con.execute(
        "SELECT data_type FROM information_schema.columns WHERE table_name = 'reports' AND column_name = ?",
        [field],
    ).fetchone()[0]
    return dtype.endswith("[]")


def to_sql(con, spec):
    """Build a parameterized WHERE clause over `reports r`."""
    clauses, params = [], []
    if spec.get("acn"):
        clauses.append("r.acn IN (SELECT unnest(?))")
        params.append(spec["acn"])
    if spec.get("year_min") is not None:
        clauses.append("r.year >= ?")
        params.append(spec["year_min"])
    if spec.get("year_max") is not None:
        clauses.append("r.year <= ?")
        params.append(spec["year_max"])
    for field, values in spec.get("where", {}).items():
        # Field names are checked against FIELDS by validate() before reaching here.
        if _is_list_column(con, field):
            clauses.append(f"list_has_any(r.{field}, ?::VARCHAR[])")
        else:
            clauses.append(f"r.{field} IN (SELECT unnest(?::VARCHAR[]))")
        params.append(values)
    if spec.get("text_any"):
        likes = " OR ".join("n.text ILIKE ?" for _ in spec["text_any"])
        clauses.append(f"EXISTS (SELECT 1 FROM narratives n WHERE n.acn = r.acn AND ({likes}))")
        params += [f"%{t}%" for t in spec["text_any"]]
    return (" AND ".join(clauses) or "TRUE"), params


def run(spec, con=None):
    """Validate and execute a spec. Returns {"count", "acns"}, plus "groups" (value -> count) with group_by."""
    errors = validate(spec)
    if errors:
        raise ValueError("; ".join(errors))
    con = con or duckdb.connect(str(DB_PATH), read_only=True)
    where, params = to_sql(con, spec)
    acns = [a for (a,) in con.execute(f"SELECT r.acn FROM reports r WHERE {where} ORDER BY r.acn", params).fetchall()]
    result = {"count": len(acns), "acns": acns}

    field = spec.get("group_by")
    if field is not None:
        value = f"unnest(r.{field})" if _is_list_column(con, field) else f"r.{field}"
        rows = con.execute(
            f"SELECT v, count(*) AS n FROM (SELECT {value} AS v FROM reports r WHERE {where}) "
            f"WHERE v IS NOT NULL GROUP BY v ORDER BY n DESC, v",
            params,
        ).fetchall()
        result["groups"] = dict(rows)
    return result
