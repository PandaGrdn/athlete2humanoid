"""JSON Schema helpers for Muse's strict structured output.

Schemas are plain dicts in the subset Muse accepts with strict=true. `validate` checks a response
locally (and every fixture), so we don't depend on the jsonschema package. Numeric ranges are
deliberately not in the schemas (strict mode may not enforce them); each caller checks its own.
"""

NUM = {"type": "number"}
INT = {"type": "integer"}
STR = {"type": "string"}
BOOL = {"type": "boolean"}


def obj(**props):
    """Strict-mode object: every property required, nothing else allowed."""
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def arr(items):
    return {"type": "array", "items": items}


def enum(values):
    return {"type": "string", "enum": list(values)}


_PY_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool}


def _is_type(value, t):
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, float) and value.is_integer())
    return isinstance(value, _PY_TYPES[t])


def validate(value, schema, path="$"):
    """Problems with `value` against `schema`, as 'path: problem' strings; empty when it matches."""
    t = schema["type"]
    if not _is_type(value, t):
        return [f"{path}: expected {t}, got {type(value).__name__}"]
    errors = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if t == "object":
        props = schema.get("properties", {})
        errors += [f"{path}: missing {k!r}" for k in schema.get("required", []) if k not in value]
        if schema.get("additionalProperties") is False:
            errors += [f"{path}: unexpected {k!r}" for k in value if k not in props]
        for k, v in value.items():
            if k in props:
                errors += validate(v, props[k], f"{path}.{k}")
    elif t == "array":
        for i, v in enumerate(value):
            errors += validate(v, schema["items"], f"{path}[{i}]")
    return errors


def strict_errors(schema, path="$"):
    """Why Muse would reject `schema` with strict=true (HTTP 400): objects must list every property
    as required and set additionalProperties false; no allOf/oneOf; plain object at the root."""
    errors = []
    if path == "$" and (schema.get("type") != "object" or {"anyOf", "oneOf", "allOf", "enum", "not"} & set(schema)):
        errors.append("$: root must be a plain object")
    if {"allOf", "oneOf"} & set(schema):
        errors.append(f"{path}: allOf/oneOf not allowed")
    if schema.get("type") == "object":
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is not False:
            errors.append(f"{path}: additionalProperties must be false")
        if sorted(schema.get("required", [])) != sorted(props):
            errors.append(f"{path}: required must list every property")
        for k, v in props.items():
            errors += strict_errors(v, f"{path}.{k}")
    elif schema.get("type") == "array":
        errors += strict_errors(schema["items"], f"{path}[]")
    return errors
