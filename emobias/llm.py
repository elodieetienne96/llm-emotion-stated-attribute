"""Calls to the models through the OpenRouter API, with a disk cache.

Every request is one user message. Recognition uses temperature 0 and a fixed seed where
the endpoint accepts one; generation uses temperature 1 and one seed per synthetic actor.
The key is read from the environment variable OPENROUTER_API_KEY or from a `.env` file at
the root of the repository. An identical request (same model, prompt and parameters) is
never sent twice: answers are kept in `cache/<model>.jsonl`.
The model `mock` answers without network and serves to test the pipeline.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from .corpus import ROOT

CACHE_DIR = ROOT / "cache"
API_URL = "https://openrouter.ai/api/v1"
REASONING_MODELS = re.compile(r"gemini-3|grok|deepseek-(v4|r)|thinking|qwen3|gpt-oss|/o[1-9]", re.I)


def api_key() -> str | None:
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))
    return os.environ.get("OPENROUTER_API_KEY")


class Cache:
    def __init__(self, model: str):
        CACHE_DIR.mkdir(exist_ok=True)
        self.path = CACHE_DIR / (re.sub(r"[^A-Za-z0-9_.-]", "_", model) + ".jsonl")
        self._d: dict[str, dict] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    self._d[rec["key"]] = rec
                except json.JSONDecodeError:
                    continue

    @staticmethod
    def key(model: str, prompt: str, params: dict) -> str:
        return hashlib.sha256(json.dumps([model, prompt, params], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:32]

    def get(self, key: str) -> dict | None:
        return self._d.get(key)

    def put(self, rec: dict) -> None:
        with self._lock:
            self._d[rec["key"]] = rec
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class FatalAPIError(RuntimeError):
    """Invalid key or exhausted credit: retrying will not help."""


class LLM:
    def __init__(self, model: str, *, temperature: float = 0.0, max_tokens: int = 60, seed: int | None = 0,
                 workers: int = 4, timeout: int = 60, max_retries: int = 5):
        self.model = model
        self.params = {"temperature": temperature, "max_tokens": max_tokens, "seed": seed}
        self.workers, self.timeout, self.max_retries = workers, timeout, max_retries
        self.cache = Cache(model)
        self.usage = {"calls": 0, "cached": 0, "errors": 0, "cost": 0.0}
        self.fatal: str | None = None
        self._lock = threading.Lock()

    def _call_api(self, prompt: str) -> dict:
        key = api_key()
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is missing (environment variable or .env file)")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": self.params["temperature"], "max_tokens": self.params["max_tokens"],
                "usage": {"include": True}}
        if self.params["seed"] is not None:
            body["seed"] = self.params["seed"]
        if REASONING_MODELS.search(self.model):
            # models that reason by default: minimal effort, reasoning not returned
            body["reasoning"] = {"effort": "low", "exclude": True}
        delay, last = 2.0, None
        for _ in range(self.max_retries):
            try:
                r = requests.post(f"{API_URL}/chat/completions", headers=headers, json=body, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 524):
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                    time.sleep(delay + random.random()); delay = min(delay * 2, 120)
                    continue
                if r.status_code == 400 and "reasoning" in body and "reasoning" in r.text.lower():
                    body.pop("reasoning"); continue
                if r.status_code in (401, 402, 403):
                    raise FatalAPIError(f"HTTP {r.status_code}: {r.text[:120]}")
                r.raise_for_status()
                data = r.json()
                if "error" in data:
                    raise RuntimeError(data["error"])
                choice = data["choices"][0]
                usage = data.get("usage") or {}
                return {"text": choice["message"]["content"], "finish": choice.get("finish_reason"),
                        "cost": float(usage.get("cost") or 0.0), "provider": data.get("provider")}
            except requests.RequestException as e:
                last = str(e)
                time.sleep(delay + random.random()); delay = min(delay * 2, 60)
        raise RuntimeError(f"failed after {self.max_retries} attempts: {last}")

    def _call_mock(self, prompt: str) -> dict:
        rnd = random.Random(hashlib.md5(f"{self.params['seed']}|{prompt}".encode()).hexdigest())
        m = re.search(r"Choose exactly ONE of: (.*?)\.\n", prompt)
        if m:
            opts = [o.strip() for o in m.group(1).split(",")]
            emo = rnd.choice(opts)
            if "smile" in prompt and "happiness" in opts and rnd.random() < 0.6:
                emo = "happiness"
            elif "lowered brows" in prompt and "anger" in opts and rnd.random() < 0.5:
                emo = "anger"
            if "The woman" in prompt and "sadness" in opts and rnd.random() < 0.15:
                emo = "sadness"
            text = json.dumps({"emotion": emo, "intensity": rnd.choice(["low", "high"])})
        else:
            from .corpus import CUES, VOCAB
            lv = {f: rnd.choice(VOCAB[f]) if rnd.random() < 0.3 else VOCAB[f][1 if f in ("speech_rate", "pitch", "loudness") else 0] for f in CUES}
            text = json.dumps(lv)
        return {"text": text, "finish": "stop", "cost": 0.0, "provider": "mock"}

    def complete(self, prompt: str) -> dict:
        key = Cache.key(self.model, prompt, self.params)
        hit = self.cache.get(key)
        if hit and (hit.get("text") or "").strip():
            with self._lock:
                self.usage["cached"] += 1
            return hit
        if self.fatal:
            return {"key": key, "text": "", "error": self.fatal}
        try:
            rec = self._call_mock(prompt) if self.model == "mock" else self._call_api(prompt)
        except FatalAPIError as e:
            self.fatal = str(e)
            return {"key": key, "text": "", "error": str(e)}
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self.usage["errors"] += 1
            return {"key": key, "text": "", "error": str(e)}
        rec.update(key=key, model=self.model, params=self.params, ts=time.time())
        with self._lock:
            self.usage["calls"] += 1
            self.usage["cost"] += rec.get("cost", 0.0)
        self.cache.put(rec)
        return rec

    def complete_many(self, prompts: list[str], progress=None) -> list[dict]:
        out: list[dict | None] = [None] * len(prompts)
        done = 0
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(self.complete, p): i for i, p in enumerate(prompts)}
            for fut in as_completed(futs):
                out[futs[fut]] = fut.result()
                done += 1
                if progress and done % 200 == 0:
                    progress(done, len(prompts))
        return out  # type: ignore[return-value]
