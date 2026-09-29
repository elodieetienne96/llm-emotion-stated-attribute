"""Generation task: the LLM writes the transcripts from the instructions given to the actors.

    python -m emobias generate configs/generation.yaml [--models m1,m2] [--conditions c1,c2]

A synthetic actor is a sampling seed at temperature 1. Ten synthetic actors are drawn, each
writing the full design of one real actor: the ten neutral sentences, then every emotion,
sentence and intensity (2 100 takes per condition, one take each). An answer outside the
coding scheme is asked again once, then rejected. The valid takes go to
results/generation/<model>/<condition>.csv with the 22 cues and the rendered transcript.
"""
from __future__ import annotations

import time

import pandas as pd
import yaml

from .corpus import CUES, EMOTIONS, RESULTS, SENTENCES, render_transcript
from .llm import LLM
from .prompts import GEN_RETRY, generation_prompt, parse_generation_answer
from .recognise import MODEL_IDS

DEFAULT = {"models": ["mock"], "conditions": ["control"], "n_actors": 10, "sentences": list(range(1, 11)),
           "intensities": [1, 2], "llm": {"temperature": 1.0, "max_tokens": 300, "workers": 8}}


def load_config(path: str) -> dict:
    cfg = {**DEFAULT, **yaml.safe_load(open(path, encoding="utf-8"))}
    cfg["llm"] = {**DEFAULT["llm"], **(cfg.get("llm") or {})}
    return cfg


def takes(cfg: dict) -> pd.DataFrame:
    """One row per take to write: the design of one real actor, for each synthetic actor."""
    rows = []
    for actor in range(1, int(cfg["n_actors"]) + 1):
        for s in cfg["sentences"]:
            rows.append((actor, "neutral", s, 0))
        for e in EMOTIONS[1:]:
            for s in cfg["sentences"]:
                for i in cfg["intensities"]:
                    rows.append((actor, e, s, int(i)))
    df = pd.DataFrame(rows, columns=["actor", "emotion", "sentence", "intensity"])
    df["text"] = df["sentence"].map(lambda s: SENTENCES[int(s) - 1])
    df["take_id"] = [f"S_{a:02d}_{EMOTIONS.index(e):02d}_{s:02d}_{i}" for a, e, s, i in zip(df.actor, df.emotion, df.sentence, df.intensity)]
    return df


def plan(cfg: dict) -> dict:
    t = takes(cfg)
    return {"models": cfg["models"], "conditions": cfg["conditions"], "takes_per_condition": len(t),
            "calls_per_model": len(t) * len(cfg["conditions"]),
            "example_prompt": generation_prompt(t.text[0], t.emotion[0], int(t.intensity[0]), cfg["conditions"][0])}


def run(cfg: dict, models: list[str] | None = None, conditions: list[str] | None = None, log=print) -> None:
    t = takes(cfg)
    p = cfg["llm"]
    for model in models or cfg["models"]:
        out = RESULTS / "generation" / model
        out.mkdir(parents=True, exist_ok=True)
        for condition in conditions or cfg["conditions"]:
            t0 = time.time()
            prompts = [generation_prompt(r.text, r.emotion, int(r.intensity), condition) for r in t.itertuples()]
            rows, rejected, calls, cost = [], 0, 0, 0.0
            for actor, idx in t.groupby("actor").indices.items():
                llm = LLM(MODEL_IDS.get(model, model), temperature=p["temperature"], max_tokens=p["max_tokens"],
                          seed=int(actor), workers=p["workers"])
                recs = llm.complete_many([prompts[i] for i in idx])
                if llm.fatal:
                    raise RuntimeError(f"stopped: {llm.fatal}. Answers received so far are cached.")
                for i, rec in zip(idx, recs):
                    cues, why = parse_generation_answer(rec.get("text", ""))
                    if cues is None:
                        rec = llm.complete(prompts[i] + GEN_RETRY.format(reason=why))
                        cues, why = parse_generation_answer(rec.get("text", ""))
                    if cues is None:
                        rejected += 1
                        continue
                    r = t.iloc[i]
                    row = {"take_id": r.take_id, "actor": f"{int(r.actor):02d}", "emotion": r.emotion, "sentence": int(r.sentence),
                           "intensity": int(r.intensity), "take": 1, "text": r.text, **cues}
                    row["transcript"] = render_transcript(cues, f"Actor {int(r.actor):02d}", r.text)
                    rows.append(row)
                calls += llm.usage["calls"]; cost += llm.usage["cost"]
            df = pd.DataFrame(rows, columns=["take_id", "actor", "emotion", "sentence", "intensity", "take", "text"] + CUES + ["transcript"])
            df.to_csv(out / f"{condition}.csv", index=False)
            log(f"{model} | {condition}: {len(df)} valid takes, {rejected} rejected, calls {calls}, "
                f"cost {cost:.2f} USD, {time.time() - t0:.0f} s")
