"""Abuse limits and response hardening, as plain ASGI middleware (safe around streaming responses).

- client_ip: the socket peer, or the entry A2H_PROXY_HOPS from the right of X-Forwarded-For (the one the
  trusted proxy appended; entries further left are client-controlled).
- RateLimiter: token buckets per (client, bucket), in memory. One process, so that is the whole truth.
- BodyLimit: 413 when Content-Length is over the limit, and stops reading a body (chunked or lying about
  its length) the moment it passes the limit, before Starlette spools it to disk.
- Hardening: request ids, security headers and one JSON access-log line per request.
"""

import hashlib
import json
import logging
import threading
import time
import uuid

log = logging.getLogger("a2h.http")

API_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"strict-transport-security", b"max-age=31536000"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
    # JSON and media need no scripts, styles or frames: a stray HTML response can't run anything.
    (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'; sandbox"),
]


def client_ip(scope, hops=0):
    if hops > 0:
        for name, value in scope.get("headers") or []:
            if name == b"x-forwarded-for":
                parts = [p.strip() for p in value.decode("latin-1").split(",") if p.strip()]
                if parts:
                    return parts[-min(hops, len(parts))]
    peer = scope.get("client")
    return peer[0] if peer else "unknown"


class RateLimiter:
    """Token buckets: `rate` tokens per `per` seconds, bursting up to `burst`."""

    def __init__(self, clock=time.monotonic):
        self.clock, self.lock, self.buckets = clock, threading.Lock(), {}

    def hit(self, key, rate, per, burst=None):
        """(allowed, seconds until the next token)."""
        cap = burst or rate
        now = self.clock()
        with self.lock:
            tokens, last = self.buckets.get(key, (cap, now))
            tokens = min(cap, tokens + (now - last) * rate / per)
            if tokens >= 1:
                self.buckets[key] = (tokens - 1, now)
                return True, 0.0
            self.buckets[key] = (tokens, now)
            if len(self.buckets) > 50_000:  # forget idle clients rather than grow forever
                self.buckets = {k: v for k, v in self.buckets.items() if now - v[1] < per}
            return False, (1 - tokens) * per / rate


async def _send_json(send, status, body, headers=()):
    data = json.dumps(body).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode()),
                            *headers]})
    await send({"type": "http.response.body", "body": data})


class _TooLarge(Exception):
    pass


class BodyLimit:
    """Cap request bodies at `limit` bytes; only POST /api/jobs may send up to `upload_limit`."""

    def __init__(self, app, limit, upload_limit, upload_paths=("/api/jobs",)):
        self.app, self.limit, self.upload_limit, self.upload_paths = app, limit, upload_limit, upload_paths

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        cap = self.upload_limit if scope["path"] in self.upload_paths and scope["method"] == "POST" else self.limit
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None:
            try:
                too_big = int(declared) > cap
            except ValueError:
                return await _send_json(send, 400, {"detail": "bad Content-Length"})
            if too_big:
                return await _send_json(send, 413, {"detail": f"request body is larger than {cap / 1048576:g} MB"})
        seen, over, replied = 0, False, False
        too_big = {"detail": f"request body is larger than {cap / 1048576:g} MB"}

        async def limited_receive():
            nonlocal seen, over
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > cap:
                    over = True
                    raise _TooLarge  # stop reading; whatever the app makes of that, the reply is a 413
            return msg

        async def guarded_send(msg):
            nonlocal replied
            if over:  # replace the app's reaction (a 400 from the form parser, a 500) with the 413
                if not replied:
                    replied = True
                    await _send_json(send, 413, too_big)
                return
            replied = replied or msg["type"] == "http.response.start"
            await send(msg)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _TooLarge:
            pass
        if over and not replied:
            await _send_json(send, 413, too_big)


class RateLimit:
    """Per-client request budget on /api/*, plus a tighter one on job creation."""

    def __init__(self, app, limiter, settings):
        self.app, self.limiter, self.s = app, limiter, settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/api/") or scope["method"] == "OPTIONS":
            return await self.app(scope, receive, send)
        ip = client_ip(scope, self.s.proxy_hops)
        checks = [("req", self.s.rate_requests_per_min, 60.0, None)]
        if scope["method"] == "POST" and scope["path"] == "/api/jobs":
            # a few in a row are fine (retrying an upload); sustained, it is rate_jobs_per_hour
            checks.append(("job", self.s.rate_jobs_per_hour, 3600.0, min(5.0, self.s.rate_jobs_per_hour)))
        for bucket, rate, per, burst in checks:
            ok, wait = self.limiter.hit((bucket, ip), rate, per, burst)
            if not ok:
                log.warning("rate limited", extra={"fields": {"bucket": bucket, "client": anon(ip), "path": scope["path"]}})
                return await _send_json(send, 429, {"detail": "too many requests; slow down and try again shortly"},
                                        [(b"retry-after", str(max(1, int(wait + 0.999))).encode())])
        await self.app(scope, receive, send)


def anon(ip):
    """Logs keep a stable pseudonym, not the address."""
    return hashlib.sha256(f"a2h:{ip}".encode()).hexdigest()[:10]


class Hardening:
    """Request id, security headers, and an access-log line per request."""

    def __init__(self, app, proxy_hops=0):
        self.app, self.hops = app, proxy_hops

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        rid = uuid.uuid4().hex[:16]
        scope.setdefault("state", {})["request_id"] = rid
        t0, status = time.perf_counter(), [0]
        # the SPA (A2H_STATIC_DIR) and Swagger UI (/docs) get everything but the API's script-free CSP
        is_api = scope["path"].startswith("/api/") or scope["path"] in ("/healthz", "/readyz")

        async def wrapped(msg):
            if msg["type"] == "http.response.start":
                status[0] = msg["status"]
                headers = [(k, v) for k, v in msg.get("headers", []) if k.lower() != b"server"]
                have = {k.lower() for k, _ in headers}
                extra = API_HEADERS if is_api else API_HEADERS[:-1]
                headers += [(k, v) for k, v in extra if k not in have]
                headers.append((b"x-request-id", rid.encode()))
                msg = {**msg, "headers": headers}
            await send(msg)

        try:
            await self.app(scope, receive, wrapped)
        finally:
            if scope["path"] not in ("/healthz", "/readyz"):
                log.info("request", extra={"fields": {
                    "request_id": rid, "method": scope["method"], "path": scope["path"], "status": status[0],
                    "ms": round((time.perf_counter() - t0) * 1000, 1), "client": anon(client_ip(scope, self.hops))}})
