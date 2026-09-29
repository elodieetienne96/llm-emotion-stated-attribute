"""Recognition task: run the conditions of a configuration and store one answer per clip.

    python -m emobias plan      configs/recognition_attributes.yaml
    python -m emobias recognise configs/recognition_attributes.yaml [--models m1,m2] [--conditions c1,c2]

A configuration gives the corpus, the models, the conditions, the prompt template and the
input. The answers go to results/recognition/<model>/<group>.csv, one row per clip and
condition, with the parsed emotion, the intensity and the raw answer. A condition already
present in that file is replaced.
"""
from __future__ import annotations

import time

import pandas as pd
import yaml

from .corpus import CORPORA, RESULTS, load_clips, transcript_without_words, transcript_words_only
from .llm import LLM
from .prompts import CONTROL, REPEAT, all_conditions, parse_recognition_answer, recognition_prompt

MODEL_IDS = {  # short name used in results/ -> identifier at the API
    "gpt-5.4": "openai/gpt-5.4", "claude-sonnet-4.6": "anthropic/claude-sonnet-4.6",
    "gemini-3.5-flash": "google/gemini-3.5-flash", "grok-4.20": "x-ai/grok-4.20",
    "mistral-small-3": "mistralai/mistral-small-2603", "qwen-2.5-72b": "qwen/qwen-2.5-72b-instruct",
    "llama-4-maverick": "meta-llama/llama-4-maverick", "mock": "mock"}
DEFAULT = {"corpus": "eve", "template": "recognition", "input": "transcript", "conditions": ["control"],
           "models": ["mock"], "group": None, "llm": {"temperature": 0.0, "max_tokens": 60, "seed": 0, "workers": 8}}


def load_config(path: str) -> dict:
    cfg = {**DEFAULT, **yaml.safe_load(open(path, encoding="utf-8"))}
    cfg["llm"] = {**DEFAULT["llm"], **(cfg.get("llm") or {})}
    groups = all_conditions()
    conds = []
    for c in cfg["conditions"]:
        conds += groups.get(c, [c])
    cfg["conditions"] = list(dict.fromkeys(conds))
    cfg["group"] = cfg["group"] or "run"
    return cfg


def prompts_for(cfg: dict, condition: str) -> tuple[pd.DataFrame, list[str]]:
    clips = load_clips(cfg["corpus"])
    emotions = CORPORA[cfg["corpus"]]["emotions"]
    texts = clips["transcript"].astype(str)
    if cfg["input"] == "words_only":
        texts = texts.map(transcript_words_only)
    elif cfg["input"] == "behaviour_only":
        texts = texts.map(transcript_without_words)
    cond = CONTROL if condition == REPEAT else condition
    return clips, [recognition_prompt(t, cond, emotions, cfg["template"]) for t in texts]


def plan(cfg: dict) -> dict:
    clips, ex = prompts_for(cfg, cfg["conditions"][0])
    return {"corpus": cfg["corpus"], "clips": len(clips), "models": cfg["models"], "conditions": len(cfg["conditions"]),
            "calls_per_model": len(clips) * len(cfg["conditions"]), "example_prompt": ex[0]}


def run(cfg: dict, models: list[str] | None = None, conditions: list[str] | None = None, log=print) -> None:
    emotions = CORPORA[cfg["corpus"]]["emotions"]
    for model in models or cfg["models"]:
        out = RESULTS / "recognition" / model
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{cfg['group']}.csv"
        done = pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame(columns=["clip", "condition", "emotion", "intensity", "raw"])
        for condition in conditions or cfg["conditions"]:
            clips, prompts = prompts_for(cfg, condition)
            p = cfg["llm"]
            # the repeat control is the same prompt sent again: the cache must not serve the first answer
            seed = p["seed"] if condition != REPEAT else (p["seed"] or 0) + 1
            llm = LLM(MODEL_IDS.get(model, model), temperature=p["temperature"], max_tokens=p["max_tokens"],
                      seed=seed, workers=p["workers"])
            t0 = time.time()
            log(f"{model} | {condition}: {len(prompts)} prompts")
            recs = llm.complete_many(prompts, progress=lambda d, n: log(f"  {d}/{n}"))
            if llm.fatal:
                raise RuntimeError(f"stopped: {llm.fatal}. Answers received so far are cached.")
            rows = []
            for clip, rec in zip(clips["clip"], recs):
                emo, inten = parse_recognition_answer(rec.get("text", ""), emotions)
                rows.append({"clip": clip, "condition": condition, "emotion": emo, "intensity": inten,
                             "raw": (rec.get("text") or "").replace("\n", " ")})
            done = pd.concat([done[done["condition"] != condition], pd.DataFrame(rows)], ignore_index=True)
            done.sort_values(["condition", "clip"]).to_csv(path, index=False)
            log(f"  done in {time.time() - t0:.0f} s, calls {llm.usage['calls']}, cached {llm.usage['cached']}, "
                f"errors {llm.usage['errors']}, cost {llm.usage['cost']:.2f} USD -> {path.relative_to(RESULTS.parent)}")
