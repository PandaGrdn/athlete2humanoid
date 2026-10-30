"""task_spec.json's text fields: what each stage is called, for people. The reward never reads them.

Torch-free (the server and muse/ import it). A stage's display label is, in order:

    muse_label   Muse Spark's plain-English name for the key moment ("step left, turn left"), written by
                 muse/moment_namer.py. The stable key for front ends: present only on named pose and flight
                 stages.
    label        the skill reader's phase name (skill.task.attach_skill), when a skill spec was attached
    name         the compiler's own name (skill/keypose.py name_moment: "right foot up, right foot forward"),
                 or Muse's once the spec is named, with the compiler's kept as `measured`
    kind         "pose", "flight", "settle"

Named specs also carry, per pose and flight stage, `description` (one sentence) and `measured`, and at the top
`action` (snake_case), `action_description` and `named_by` ({model, mode, created, ...}).
"""

LABEL_KEY = "muse_label"
NAMED_KINDS = ("pose", "flight")
MAX_LABEL_CHARS, MAX_DESCRIPTION_CHARS = 40, 240

_STR, _OBJ = {"type": "string"}, {"type": "object"}
# Optional fields: absent on an unnamed spec, of this type when present.
STAGE_TEXT = {LABEL_KEY: _STR, "description": _STR, "measured": _STR, "label": _STR, "name": _STR}
SPEC_TEXT = {"action": _STR, "action_description": _STR, "named_by": _OBJ, "summary": _STR}


def stage_label(st):
    """The stage's label for display: Muse's, else the skill reader's, else the compiler's."""
    return st.get(LABEL_KEY) or st.get("label") or st.get("name") or st["kind"]


def _type_ok(value, t):
    return isinstance(value, {"string": str, "object": dict}[t["type"]])


def text_errors(spec):
    """Problems with the spec's text fields -> ['path: problem']; empty when they are well formed."""
    errors = [f"$.{k}: expected {t['type']}" for k, t in SPEC_TEXT.items() if k in spec and not _type_ok(spec[k], t)]
    for i, st in enumerate(spec.get("stages", [])):
        path = f"$.stages[{i}]"
        errors += [f"{path}.{k}: expected {t['type']}" for k, t in STAGE_TEXT.items() if k in st and not _type_ok(st[k], t)]
        if isinstance(st.get("description"), str) and len(st["description"]) > MAX_DESCRIPTION_CHARS:
            errors.append(f"{path}.description: at most {MAX_DESCRIPTION_CHARS} characters")
        label = st.get(LABEL_KEY)
        if label is None:
            continue
        if st.get("kind") not in NAMED_KINDS:
            errors.append(f"{path}.{LABEL_KEY}: only pose and flight stages are named (this one is {st.get('kind')!r})")
        elif isinstance(label, str) and not (0 < len(label.strip()) <= MAX_LABEL_CHARS and label == label.strip()):
            errors.append(f"{path}.{LABEL_KEY}: 1-{MAX_LABEL_CHARS} characters, no surrounding spaces")
    return errors
