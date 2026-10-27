"""Muse Spark client for schema-constrained calls, with an offline fixture mode.

    live     calls the Meta Model API (OpenAI-compatible Chat Completions, strict json_schema).
             Key: MUSE_API_KEY via config.get_secret (the older name META_API_KEY also works).
    fixture  no network and no key: responses come from a recorded file. The request is still
             built exactly as in live mode (prompt, media, schema) and can be saved for inspection.

A live client with record_dir saves each exchange as {"request", "responses"}, which is itself a
fixture file, so the hand-written fixtures in muse/fixtures/ can be replaced with real ones.
"""

import base64
import copy
import json
import time
from pathlib import Path

import config
from muse.schema import validate

INLINE_LIMIT = 45 * 2**20  # Meta caps inline media at 50 MB (Files API: 1 GiB)
FIXTURES = Path(__file__).resolve().parent / "fixtures"


class MuseInvalid(RuntimeError):
    """Muse's output failed the schema or the caller's checks, after the repair attempt."""

    def __init__(self, errors, text):
        super().__init__("Muse returned invalid output:\n  " + "\n  ".join(errors[:20]))
        self.errors, self.text = errors, text


def api_key():
    for name in (config.MUSE_API_KEY_ENV, config.META_API_KEY_ENV):
        key = config.get_secret(name, required=False)
        if key:
            return key
    raise RuntimeError(f"Secret {config.MUSE_API_KEY_ENV} not found. Put it in "
                       f"{config.SECRETS_DIR / config.MUSE_API_KEY_ENV} as {config.MUSE_API_KEY_ENV} = \"...\", "
                       f"set the environment variable, or run with --dry-run.")


def data_url(data, mime):
    """data: a path or raw bytes."""
    raw = data if isinstance(data, bytes) else Path(data).read_bytes()
    if len(raw) > INLINE_LIMIT:
        raise ValueError(f"{len(raw) / 2**20:.0f} MB is over the inline media limit; downscale it first")
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def image_part(data, mime="image/png"):
    return {"type": "image_url", "image_url": {"url": data_url(data, mime)}}


def video_part(path, fps):
    """Muse samples the video at `fps` frames per second."""
    return {"type": "video_url", "video_url": {"url": data_url(path, "video/mp4"), "fps": fps}}


def text_part(text):
    return {"type": "text", "text": text}


def load_fixture(path):
    """A recording ({"request", "responses"}), one chat completion, or a list of them."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "responses" in data:
        return list(data["responses"])
    return list(data) if isinstance(data, list) else [data]


def completion(content, model="muse-spark-1.3"):
    """A chat completion carrying `content` (dict -> JSON), shaped like the API's: for fixtures."""
    text = content if isinstance(content, str) else json.dumps(content)
    return {"object": "chat.completion", "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}]}


def redact(request):
    """The request with inline media replaced by a size note (for saved requests and recordings)."""
    def walk(x):
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, str) and x.startswith("data:") and len(x) > 256:
            mime = x[5:x.index(";")]
            return f"<{mime}, {len(x) * 3 // 4} bytes base64>"
        return x
    return walk(request)


class MuseClient:
    def __init__(self, fixture=None, model=None, record_dir=None):
        self.model = model or config.MUSE_MODEL
        self.fixture = Path(fixture) if fixture else None
        self._responses = load_fixture(self.fixture) if self.fixture else None
        self.record_dir = Path(record_dir) if record_dir else None
        self.requests, self._raw, self._client = [], [], None

    @property
    def live(self):
        return self.fixture is None

    @property
    def mode(self):
        return "live" if self.live else f"fixture:{self.fixture.name}"

    def complete(self, messages, schema, name, effort):
        """One request -> the raw chat completion as a dict; None once a fixture is used up."""
        req = {"model": self.model, "messages": copy.deepcopy(messages), "reasoning_effort": effort,
               "response_format": {"type": "json_schema",
                                   "json_schema": {"name": name, "schema": schema, "strict": True}}}
        self.requests.append(req)
        if not self.live:
            return self._responses.pop(0) if self._responses else None
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key(), base_url=config.META_API_BASE)
        raw = self._client.chat.completions.create(**req).model_dump()
        self._raw.append(raw)
        return raw

    def ask_json(self, name, system, content, schema, check=None, effort="medium", retries=1):
        """Schema-constrained call -> (value, info). The value is validated against `schema`, then
        `check(value) -> [errors]`; on failure Muse is shown the errors and asked again (`retries`
        times). info: model, mode, attempts, usage."""
        messages = [{"role": "developer", "content": system}, {"role": "user", "content": content}]
        errors, text, usage = ["no response"], None, []
        try:
            for attempt in range(1, retries + 2):
                raw = self.complete(messages, schema, name, effort)
                if raw is None:
                    break
                text = raw["choices"][0]["message"]["content"]
                usage.append(raw.get("usage"))
                try:
                    value = json.loads(text)
                except json.JSONDecodeError as e:
                    errors = [f"$: not valid JSON ({e})"]
                else:
                    errors = validate(value, schema) or (check(value) if check else [])
                    if not errors:
                        return value, {"model": raw.get("model", self.model), "mode": self.mode,
                                       "attempts": attempt, "usage": usage}
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": "That JSON has problems:\n- " + "\n- ".join(errors)
                                                         + "\nReturn the corrected JSON only."}]
            raise MuseInvalid(errors, text)
        finally:
            self._record(name)

    def save_requests(self, path):
        Path(path).write_text(json.dumps([redact(r) for r in self.requests], indent=1))

    def _record(self, name):
        if self.record_dir and self._raw:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            f = self.record_dir / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
            f.write_text(json.dumps({"request": redact(self.requests[-1]), "responses": self._raw}, indent=1))
            self._raw = []


def make_client(dry_run=False, fixture=None, default_fixture=None, model=None, record_dir=None):
    """CLI helper: --dry-run uses the bundled fixture, --fixture a given one, else live."""
    if fixture or dry_run:
        return MuseClient(fixture=fixture or FIXTURES / default_fixture, model=model)
    return MuseClient(model=model, record_dir=record_dir)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=1))


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")
