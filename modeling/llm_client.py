"""
LLMClient: Provider-Agnostic LLM Completion Clients

The minimal `LLMClient` interface that LLM-driven searches depend on, with six implementations
(the Python counterpart of ScalaTion's `modeling/LLMClient.scala`):

    OpenAIClient     -- OpenAI chat-completions endpoint (cloud, needs a key: OPENAI_API_KEY)
    AnthropicClient  -- Anthropic messages endpoint (cloud, needs a key: ANTHROPIC_API_KEY)
    LocalClient      -- ANY OpenAI-chat-completions-COMPATIBLE local server (Ollama, vLLM,
                        llama.cpp's llama-server, LM Studio): a genuine LLM, but OFFLINE -- no
                        network egress, no API key, no per-token cost, no rate limit
    CachedClient     -- disk cache wrapper keyed by SHA-256 of (tag, system, user); gives free
                        re-runs, exact reproducibility (ship the cache directory as a paper
                        artifact) and a token cost ledger.  NOTE: with temperature > 0 the FIRST
                        population of the cache is still stochastic -- the cache freezes a run,
                        it does not make the API deterministic
    CountingClient   -- counts consultations, estimated tokens (characters / 4) and cost so
                        experiments can report LLM cost per run; wrap it OUTSIDE `CachedClient`
                        to count calls the cache answered too
    MockLLMClient    -- canned responses for offline UNIT TESTS (fixed strings, not a model;
                        the tests must never need a network, an API key or a local server)

The requests go through the standard library (`urllib`), so no third-party dependency is added;
the bodies are rendered and the replies read by the small `Json` object (the same field-reading
contract as ScalaTion's, on top of `json`).  API keys are read from environment variables --
NEVER hard-code keys in source or commit them.

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class LLMClient:
    """Abstracts a large language model completion service, so the search loop is independent of
    the provider (OpenAI, Anthropic, local, cached, or mocked)."""

    def complete(self, system: str, user: str) -> str:
        """
        The model's completion for the given prompts.

        Parameters
        ----------
        system : str
            The system prompt (task contract, e.g., the formula grammar).
        user : str
            The user prompt (dataset schema, scored population, request).
        """
        raise NotImplementedError


class Json:
    """Renders the request bodies the chat endpoints expect and reads one string field out of
    their replies (ScalaTion's `Json` object, on top of the standard `json` module)."""

    @staticmethod
    def str_(s: str) -> str:
        """A string as a JSON string literal (quoted, escaped)."""
        return json.dumps(s, ensure_ascii=False)

    @staticmethod
    def num(d: float) -> str:
        return repr(float(d))

    @staticmethod
    def chatBody(model: str, temp: float, system: str, user: str, **extra: Any) -> str:
        """A chat-completions body (OpenAI and compatible local servers)."""
        body = {"model": model, "temperature": float(temp),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        body.update(extra)
        return json.dumps(body, ensure_ascii=False)

    @staticmethod
    def string(text: str, key: str) -> str:
        """
        The value of the FIRST string field with the given key in a JSON text (e.g., "content" for
        choices[0].message.content of a chat reply, "text" for content[0].text of a messages reply);
        raises when absent.  A key inside a string value is not matched.
        """
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Json.string: reply is not JSON: {text[:200]}") from e

        def find(o: Any) -> Optional[str]:
            if isinstance(o, dict):
                for k, v in o.items():
                    if k == key and isinstance(v, str):
                        return v
                for v in o.values():
                    r = find(v)
                    if r is not None:
                        return r
            elif isinstance(o, list):
                for v in o:
                    r = find(v)
                    if r is not None:
                        return r
            return None

        out = find(obj)
        if out is None:
            raise RuntimeError(f"Json.string: no string field '{key}' in reply: {text[:200]}")
        return out


class LLMClient_:
    """Shared HTTP plumbing for the API clients (ScalaTion's `LLMClient_` object)."""

    MAX_TRIES = 3                                                      # attempts per request
    PAUSE = 5.0                                                        # seconds, times the attempt

    @classmethod
    def sendWithRetry(cls, url: str, body: str, headers: Dict[str, str], timeout: float) -> str:
        """
        Posts the body, retrying (up to `MAX_TRIES` attempts, with a growing pause) on a dropped
        connection, a rate-limit (429) or a server (5xx) error; raises on any other non-200 status
        or when the attempts run out.
        """
        attempt = 0
        data = body.encode("utf-8")
        while True:
            attempt += 1
            req = urllib.request.Request(url, data=data, method="POST",
                                         headers={"Content-Type": "application/json", **headers})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read().decode("utf-8")
            except urllib.error.HTTPError as e:
                code = e.code
                text = e.read().decode("utf-8", errors="replace") if e.fp is not None else ""
                if (code == 429 or code >= 500) and attempt < cls.MAX_TRIES:
                    print(f"LLMClient_.sendWithRetry: status {code}; retrying in {cls.PAUSE * attempt:.0f}s")
                    time.sleep(cls.PAUSE * attempt)
                    continue
                raise RuntimeError(f"LLM HTTP {code}: {text}") from e
            except (urllib.error.URLError, OSError) as e:
                if attempt >= cls.MAX_TRIES:
                    raise
                print(f"LLMClient_.sendWithRetry: {e}; retrying in {cls.PAUSE * attempt:.0f}s")
                time.sleep(cls.PAUSE * attempt)


class OpenAIClient(LLMClient):
    """
    Calls the OpenAI chat-completions endpoint (the key from OPENAI_API_KEY).

    Parameters
    ----------
    model : str, default="gpt-4o-mini"
        The model name.
    temp : float, default=0.8
        The sampling temperature (higher => more diverse proposals).
    """

    URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model: str = "gpt-4o-mini", temp: float = 0.8) -> None:
        self.model, self.temp = model, float(temp)
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise RuntimeError("environment variable OPENAI_API_KEY not set")

    def complete(self, system: str, user: str) -> str:
        body = Json.chatBody(self.model, self.temp, system, user)
        reply = LLMClient_.sendWithRetry(self.URL, body, {"Authorization": f"Bearer {self.key}"}, 120.0)
        return Json.string(reply, "content")                             # choices[0].message.content


class AnthropicClient(LLMClient):
    """
    Calls the Anthropic messages endpoint (the key from ANTHROPIC_API_KEY).

    Parameters
    ----------
    model : str, default="claude-sonnet-4-5"
        The model name.
    temp : float, default=0.8
        The sampling temperature.
    maxTokens : int, default=2048
        The response token cap (required by the API).
    """

    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str = "claude-sonnet-4-5", temp: float = 0.8, maxTokens: int = 2048) -> None:
        self.model, self.temp, self.maxTokens = model, float(temp), int(maxTokens)
        self.key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise RuntimeError("environment variable ANTHROPIC_API_KEY not set")

    def complete(self, system: str, user: str) -> str:
        body = json.dumps({"model": self.model, "temperature": self.temp, "max_tokens": self.maxTokens,
                           "system": system, "messages": [{"role": "user", "content": user}]}, ensure_ascii=False)
        reply = LLMClient_.sendWithRetry(self.URL, body, {"x-api-key": self.key, "anthropic-version": "2023-06-01"}, 120.0)
        return Json.string(reply, "text")                                # content[0].text


class LocalClient(LLMClient):
    """
    Calls ANY locally-hosted server that speaks the OpenAI chat-completions wire format -- Ollama,
    vLLM, llama.cpp's llama-server, LM Studio.  A REAL model (unlike `MockLLMClient`) that never
    leaves the machine: no network egress beyond localhost, no API key, no billing, no rate limit.

    Parameters
    ----------
    endpoint : str
        The chat-completions URL of the local server (Ollama's by default).
    model : str, default="llama3.1"
        The model name as registered with that server.
    temp : float, default=0.8
        The sampling temperature.
    apiKeyEnv : str, default=""
        The ENV VAR holding an optional local API key ("" => no Authorization header).
    timeoutMs : int, default=180000
        The per-request timeout (local inference can be slow on CPU-only hardware).
    """

    def __init__(self, endpoint: str = "http://localhost:11434/v1/chat/completions", model: str = "llama3.1",
                 temp: float = 0.8, apiKeyEnv: str = "", timeoutMs: int = 180000) -> None:
        self.endpoint, self.model, self.temp = endpoint, model, float(temp)
        self.key = os.environ.get(apiKeyEnv) if apiKeyEnv else None
        self.timeoutMs = int(timeoutMs)

    def reachable(self, probeMs: int = 500) -> bool:
        """Whether a server is listening at `endpoint` (any HTTP response, including a 404/405 on a
        bare GET, counts -- only a connection failure or timeout means "not reachable")."""
        try:
            req = urllib.request.Request(self.endpoint, method="GET")
            with urllib.request.urlopen(req, timeout=probeMs / 1000.0):
                return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            return False

    def complete(self, system: str, user: str) -> str:
        body = Json.chatBody(self.model, self.temp, system, user)
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        reply = LLMClient_.sendWithRetry(self.endpoint, body, headers, self.timeoutMs / 1000.0)
        return Json.string(reply, "content")


class CachedClient(LLMClient):
    """
    Wraps any `LLMClient` with a disk cache keyed by a SHA-256 hash of (tag, system, user): a hit
    reads from disk (no network); a miss calls through and persists.  The `tag` MUST identify the
    model (and sampling temperature) behind `inner`: the prompts alone do not.  An empty tag hashes
    exactly as ScalaTion's untagged key, so a cache directory written by the Scala runs is valid.

    Parameters
    ----------
    inner : LLMClient
        The wrapped client (called only on cache misses).
    dir : str, default="llm_cache"
        The cache directory (created if absent).
    tag : str, default=""
        The model/temperature identity folded into the cache key (e.g., "gpt-4o-mini@0.8").
    """

    def __init__(self, inner: LLMClient, dir: str = "llm_cache", tag: str = "") -> None:
        self.inner, self.dir, self.tag = inner, Path(dir), tag

    def keyOf(self, system: str, user: str) -> str:
        """The cache key: SHA-256 of tag, the length-prefixed system prompt and the user prompt."""
        return hashlib.sha256(f"{self.tag}{len(system)}:{system}{user}".encode("utf-8")).hexdigest()

    def complete(self, system: str, user: str) -> str:
        file = self.dir / f"{self.keyOf(system, user)}.txt"
        if file.exists():
            return file.read_text(encoding="utf-8")
        out = self.inner.complete(system, user)
        self.dir.mkdir(parents=True, exist_ok=True)
        file.write_text(out, encoding="utf-8")
        return out


class CountingClient(LLMClient):
    """
    Wraps any `LLMClient` and counts consultations, prompt and reply characters (tokens are
    estimated as characters / 4) and the resulting cost.  Wrap it OUTSIDE `CachedClient`, as in
    CountingClient (CachedClient (...)), to count the consultations whether or not the cache
    answered them.

    Parameters
    ----------
    inner : LLMClient
        The wrapped client.
    priceIn, priceOut : float
        The price of prompt / reply tokens in USD per million.
    """

    def __init__(self, inner: LLMClient, priceIn: float = 0.0, priceOut: float = 0.0) -> None:
        self.inner, self.priceIn, self.priceOut = inner, float(priceIn), float(priceOut)
        self._calls = 0
        self._inChars = 0
        self._outChars = 0

    @property
    def calls(self) -> int:
        """The number of completions requested so far."""
        return self._calls

    @property
    def chars(self) -> int:
        """The total prompt + reply characters so far."""
        return self._inChars + self._outChars

    @property
    def tokens(self) -> int:
        """The estimated total tokens so far (characters / 4)."""
        return self.chars // 4

    @property
    def cost(self) -> float:
        """The estimated cost so far in USD."""
        return (self._inChars / 4.0) * self.priceIn / 1e6 + (self._outChars / 4.0) * self.priceOut / 1e6

    def complete(self, system: str, user: str) -> str:
        self._calls += 1
        self._inChars += len(system) + len(user)
        out = self.inner.complete(system, user)
        self._outChars += len(out)
        return out


class MockLLMClient(LLMClient):
    """
    Returns canned responses in order (cycling), for offline unit tests of the parser and the
    search loop.

    Parameters
    ----------
    responses : sequence of str
        The canned responses to return, in order.
    """

    def __init__(self, responses: Sequence[str]) -> None:
        self.responses: List[str] = list(responses)
        if not self.responses:
            raise ValueError("MockLLMClient: at least one response is needed")
        self.i = 0

    def complete(self, system: str, user: str) -> str:
        out = self.responses[self.i % len(self.responses)]
        self.i += 1
        return out
