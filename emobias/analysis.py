"""All the numbers of the paper, recomputed from results/ and data/.

    python -m emobias analyse        -> results/measures/*.json and docs/data/*.json (the site)

Sections follow the paper: the corpus check (words alone, behaviour alone, both), the
reference point (Table 2), the effect of a stated attribute in recognition (Tables 3 and 4,
Figure 1) and in generation (Table 5). Everything is also written in the compact form
read by the interactive site in docs/.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .corpus import AU_CODES, CORPORA, CUES, EMOTIONS, RESULTS, ROOT, load_clips, rated
from .measures import (accuracy, benjamini_hochberg, between_actor_divergence, bootstrap_actors, compare_conditions,
                       cue_tvd, fleiss_kappa, macro_f1, shares, value_share_diff)
from .prompts import ATTRIBUTES, COMBINED_GENDERS, CONTROL, REPEAT, attribute_sentence, speaker_phrase

MEASURES = RESULTS / "measures"
SITE = ROOT / "docs" / "data"
MODELS = {"gpt-5.4": {"name": "GPT-5.4", "provider": "OpenAI", "weights": "commercial"},
          "claude-sonnet-4.6": {"name": "Claude Sonnet 4.6", "provider": "Anthropic", "weights": "commercial"},
          "gemini-3.5-flash": {"name": "Gemini 3.5 Flash", "provider": "Google", "weights": "commercial"},
          "grok-4.20": {"name": "Grok 4.20", "provider": "xAI", "weights": "commercial"},
          "mistral-small-3": {"name": "Mistral Small 3", "provider": "Mistral AI", "weights": "open"},
          "qwen-2.5-72b": {"name": "Qwen 2.5 72B", "provider": "Alibaba", "weights": "open"},
          "llama-4-maverick": {"name": "Llama 4 Maverick", "provider": "Meta", "weights": "open"}}
FAMILY = {"speech_rate": "prosody", "pitch": "prosody", "loudness": "prosody", "gaze": "gaze", "head": "head",
          **{au: "face" for au in AU_CODES}}
SINGLE = [a for a in ATTRIBUTES if a not in ("trans_woman", "trans_man")]      # the twenty attributes
OTHERS = [a for a in SINGLE if ATTRIBUTES[a]["group"] != "gender"]              # the fifteen non-gender attributes
FIG_EMOTIONS = ["happiness", "sadness", "disgust", "confidence", "confusion", "contempt"]


def r2(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 2)


def load_predictions(model: str, group: str) -> pd.DataFrame | None:
    p = RESULTS / "recognition" / model / f"{group}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, usecols=["clip", "condition", "emotion", "intensity"], low_memory=False)
    return d


def predictions(model: str, groups=("control", "attributes", "combinations")) -> pd.DataFrame:
    parts = [d for g in groups if (d := load_predictions(model, g)) is not None]
    if not parts:
        return pd.DataFrame(columns=["clip", "condition", "emotion", "intensity"])
    return pd.concat(parts, ignore_index=True).drop_duplicates(["condition", "clip"])


# ---------------------------------------------------------------- reference point
def reference_point(clips: pd.DataFrame, log) -> dict:
    ref = clips.set_index("clip")["majority_label"]
    actors = clips.set_index("clip")["actor"]
    out = {"annotators": {"shares": {}, "n_clips": int(len(clips))}, "models": {}}
    maj = shares(ref, EMOTIONS)
    out["annotators"]["shares"] = {e: r2(v) for e, v in maj.items()}
    # a random annotator against the majority label: share of the votes that go to the majority label
    votes = clips[[f"votes_{e}" for e in EMOTIONS[1:]]].to_numpy(float)
    out["annotators"]["single_annotator_vs_majority"] = r2(100 * float((votes.max(axis=1) / votes.sum(axis=1)).mean()))
    out["annotators"]["most_frequent_class"] = max(maj, key=maj.get)
    out["annotators"]["most_frequent_class_accuracy"] = r2(max(maj.values()))
    out["annotators"]["random_accuracy"] = r2(100 / len(EMOTIONS))
    pivot = {}
    for m in MODELS:
        d = load_predictions(m, "control")
        if d is None:
            continue
        c = d[d.condition == CONTROL].set_index("clip")["emotion"].reindex(ref.index)
        c = c[c.notna()]
        if len(c) < 100:
            continue
        rr, aa = ref.loc[c.index], actors.loc[c.index]
        df = pd.DataFrame({"pred": c.to_numpy(), "ref": rr.to_numpy()})
        acc = 100 * accuracy(df.pred, df.ref)
        lo, hi = bootstrap_actors(df, lambda s: 100 * accuracy(s.pred, s.ref), aa)
        # macro-F1 over the ten emotions the annotators could choose (they had no neutral option)
        f1 = 100 * macro_f1(df.pred, df.ref, EMOTIONS[1:])
        flo, fhi = bootstrap_actors(df, lambda s: 100 * macro_f1(s.pred, s.ref, EMOTIONS[1:]), aa, n_boot=500)
        # intensity answered (low / high / none) and its agreement with the recorded intensity of the clip
        ii = d[d.condition == CONTROL].set_index("clip")["intensity"].reindex(c.index).fillna("?").astype(str).str.lower()
        rec = clips.set_index("clip")["intensity"].reindex(c.index).map({1: "low", 2: "high", 1.0: "low", 2.0: "high"})
        ok = rec.notna()
        intensity = {"shares": {k: r2(100 * float((ii == k).mean())) for k in ("low", "high", "none")},
                     "accuracy_vs_recorded": r2(100 * float((ii[ok] == rec[ok]).mean())), "n_recorded": int(ok.sum())}
        out["models"][m] = {"n": int(len(c)), "shares": {e: r2(v) for e, v in shares(c, EMOTIONS).items()}, "intensity": intensity,
                            "accuracy": r2(acc), "accuracy_ci": [r2(lo), r2(hi)], "macro_f1": r2(f1), "macro_f1_ci": [r2(flo), r2(fhi)],
                            "unparsed": r2(100 * float(d[d.condition == CONTROL]["emotion"].isna().mean()))}
        pivot[m] = c
        log(f"  {m}: accuracy {acc:.1f} [{lo:.1f}, {hi:.1f}], macro-F1 {f1:.1f}")
    # agreement between the models on the clips they all answered
    P = pd.DataFrame(pivot).dropna()
    if len(P.columns) > 1:
        table = np.stack([(P.to_numpy() == e).sum(axis=1) for e in EMOTIONS], axis=1)
        out["models_fleiss_kappa"] = r2(fleiss_kappa(table))
        out["models_fleiss_n_clips"] = int(len(P))
    vt = clips[[f"votes_{e}" for e in EMOTIONS[1:]]].to_numpy(float)
    out["annotators"]["fleiss_kappa"] = r2(fleiss_kappa(vt[vt.sum(axis=1) == vt.sum(axis=1).max()]))
    return out


# ---------------------------------------------------------------- effect of a stated attribute
def attribute_effects(clips: pd.DataFrame, log) -> dict:
    """For every model and condition: comparison with the control on the same clips."""
    ref = clips.set_index("clip")
    out = {}
    for m in MODELS:
        d = predictions(m)
        if d.empty or CONTROL not in set(d.condition):
            continue
        piv = d.pivot_table(index="clip", columns="condition", values="emotion", aggfunc="first").reindex(ref.index)
        pin = d.assign(intensity=d["intensity"].fillna("?").astype(str).str.lower()).pivot_table(index="clip", columns="condition", values="intensity", aggfunc="first").reindex(ref.index)
        ctrl = piv[CONTROL]
        actors = ref["actor"]
        res = {}
        conds = [c for c in piv.columns if c != CONTROL]
        log(f"  {m}: {len(conds)} conditions")
        for c in conds:
            sub = pd.DataFrame({"a": ctrl, "b": piv[c], "actor": actors}).dropna()
            if len(sub) < 500:
                continue
            res[c] = compare_conditions(sub.a, sub.b, EMOTIONS, sub.actor)
            ia, ib = pin[CONTROL].reindex(sub.index), pin[c].reindex(sub.index)
            res[c]["intensity"] = {"high_control": r2(100 * float((ia == "high").mean())), "high_condition": r2(100 * float((ib == "high").mean())),
                                   "low_control": r2(100 * float((ia == "low").mean())), "low_condition": r2(100 * float((ib == "low").mean())),
                                   "flip": r2(100 * float((ia != ib).mean()))}
        # Benjamini-Hochberg over the 20 attributes x 11 classes shifts of the model, and over the combinations
        for family, keys in (("attributes", [c for c in res if c in SINGLE]), ("combinations", [c for c in res if "+" in c]),
                             ("trans", [c for c in res if c in ("trans_woman", "trans_man")])):
            ps, where = [], []
            for c in keys:
                for e in EMOTIONS:
                    ps.append(res[c]["p_shift"][e]); where.append((c, e))
            for (c, e), ok in zip(where, benjamini_hochberg(ps)):
                res[c].setdefault("bh", {})[e] = bool(ok)
            ps = [res[c]["p_tvd"] for c in keys]
            for c, ok in zip(keys, benjamini_hochberg(ps)):
                res[c]["bh_tvd"] = bool(ok)
        # the repeat control as a floor: a shift counts only if it exceeds the shift between two runs of the control
        rep = res.get(REPEAT)
        for c, r in res.items():
            if rep and c != REPEAT:
                r["above_repeat"] = {e: bool(abs(r["shift"][e]) > abs(rep["shift"][e])) for e in EMOTIONS}
                r["tvd_above_repeat"] = bool(r["tvd"] > rep["tvd"])
        out[m] = res
    return out


def summary_tables(effects: dict, ctrl_shares: dict) -> dict:
    """Tables 3 and 4 and the combination analysis."""
    t3, t4, additivity = {}, {}, {}
    for m, res in effects.items():
        single = {c: r for c, r in res.items() if c in SINGLE}
        if not single:
            continue
        flips = [r["flip"] for r in single.values()]
        tvds = sorted(r["tvd"] for r in single.values())
        big = sorted(((abs(r["shift"][e]), c, e, r["shift"][e]) for c, r in single.items() for e in EMOTIONS), reverse=True)
        reported = [b for b in big if r_ok(single[b[1]], b[2])]
        t3[m] = {"n_attributes": len(single), "flip_min": min(flips), "flip_max": max(flips),
                 "tvd_median": r2(float(np.median(tvds))), "tvd_max": tvds[-1],
                 "repeat": {k: res[REPEAT][k] for k in ("flip", "tvd")} if REPEAT in res else None,
                 "largest_shifts": [{"condition": c, "emotion": e, "shift": s} for _, c, e, s in reported[:2]]}
        # Table 4: where the clips of the largest shift come from
        if reported:
            _, c, e, s = reported[0]
            tab = np.array(single[c]["transfers"])
            k = EMOTIONS.index(e)
            gains = [(int(tab[i, k]), EMOTIONS[i]) for i in range(len(EMOTIONS)) if i != k]
            gains.sort(reverse=True)
            n_ctrl = tab.sum(axis=1)
            t4[m] = {"condition": c, "emotion": e, "shift": s,
                     "transfers": [{"from": src, "to": e, "clips": n, "share_of_control_class": r2(100 * n / n_ctrl[EMOTIONS.index(src)]) if n_ctrl[EMOTIONS.index(src)] else None}
                                   for n, src in gains[:2]]}
        # combinations: distance of a combination against the sum of the distances of its two attributes
        xs, ys = [], []
        for c, r in res.items():
            if "+" not in c:
                continue
            g, a = c.split("+")
            if g in res and a in res:
                xs.append(res[g]["tvd"] + res[a]["tvd"]); ys.append(r["tvd"])
        if len(xs) > 5:
            x, y = np.array(xs), np.array(ys)
            slope = float((x * y).sum() / (x * x).sum())
            additivity[m] = {"n": len(xs), "slope_through_origin": r2(slope),
                             "mean_ratio": r2(float(np.mean(y / x))), "points": [[r2(a), r2(b)] for a, b in zip(xs, ys)]}
    return {"table3": t3, "table4": t4, "additivity": additivity}


def r_ok(r: dict, e: str) -> bool:
    """A shift is reported if it passes the correction and exceeds the repeat control (when available)."""
    return bool(r.get("bh", {}).get(e, True)) and bool(r.get("above_repeat", {}).get(e, True))


def figure_matrices(effects: dict, ctrl_shares: dict) -> dict:
    """Figure 1: for each emotion, gender (rows) x second attribute (columns), relative change of the
    share of the emotion against the control, per model and averaged over the models with combinations."""
    out = {"emotions": FIG_EMOTIONS, "rows": COMBINED_GENDERS, "cols": OTHERS, "models": {}}
    models = [m for m, res in effects.items() if sum("+" in c for c in res) >= 60]
    for m in models:
        res, base = effects[m], ctrl_shares[m]
        mats = {}
        for e in FIG_EMOTIONS:
            b = base[e] or 1e-9
            cell = [[r2(100 * res[f"{g}+{a}"]["shift"][e] / b) if f"{g}+{a}" in res else None for a in OTHERS] for g in COMBINED_GENDERS]
            alone_row = [r2(100 * res[g]["shift"][e] / b) if g in res else None for g in COMBINED_GENDERS]
            alone_col = [r2(100 * res[a]["shift"][e] / b) if a in res else None for a in OTHERS]
            mats[e] = {"cells": cell, "gender_alone": alone_row, "attribute_alone": alone_col, "control_share": r2(base[e])}
        out["models"][m] = mats
    if models:
        mean = {}
        for e in FIG_EMOTIONS:
            def avg(vals):
                v = [x for x in vals if x is not None]
                return r2(np.mean(v)) if v else None
            mean[e] = {"cells": [[avg([out["models"][m][e]["cells"][i][j] for m in models]) for j in range(len(OTHERS))] for i in range(len(COMBINED_GENDERS))],
                       "gender_alone": [avg([out["models"][m][e]["gender_alone"][i] for m in models]) for i in range(len(COMBINED_GENDERS))],
                       "attribute_alone": [avg([out["models"][m][e]["attribute_alone"][j] for m in models]) for j in range(len(OTHERS))],
                       "control_share": avg([out["models"][m][e]["control_share"] for m in models])}
        out["mean"] = mean
        out["mean_of"] = models
    return out


# ---------------------------------------------------------------- corpus check (4.1)
def corpus_check(log) -> dict:
    out = {}
    for corpus, spec in CORPORA.items():
        clips = load_clips(corpus)
        refcol = spec["reference"]
        clips = clips[clips[refcol].notna() & (clips[refcol].astype(str) != "")]
        ref = clips.set_index("clip")[refcol].astype(str)
        actors = clips.set_index("clip")["actor"].astype(str)
        out[corpus] = {"n_clips": int(len(ref)), "n_classes": len(spec["emotions"]), "reference": refcol,
                       "chance": r2(100 / len(spec["emotions"])), "models": {}}
        for m in MODELS:
            d = load_predictions(m, f"check_{corpus}")
            if d is None:
                continue
            row = {}
            for inp in ("words_only", "behaviour_only", "both"):
                c = d[d.condition == inp].set_index("clip")["emotion"].reindex(ref.index)
                c = c[c.index.isin(d[d.condition == inp]["clip"])]     # the clips answered in this condition
                if len(c) < 100:
                    continue
                df = pd.DataFrame({"pred": c.fillna("?").to_numpy(), "ref": ref.loc[c.index].to_numpy()})
                acc = 100 * accuracy(df.pred, df.ref)
                lo, hi = bootstrap_actors(df, lambda s: 100 * accuracy(s.pred, s.ref), actors.loc[c.index], n_boot=1000)
                row[inp] = {"accuracy": r2(acc), "ci": [r2(lo), r2(hi)], "n": int(len(c))}
            if row:
                out[corpus]["models"][m] = row
                log(f"  {corpus} {m}: " + ", ".join(f"{k} {v['accuracy']}" for k, v in row.items()))
    return out


# ---------------------------------------------------------------- generation (Table 5)
def generation(clips: pd.DataFrame, log) -> dict:
    real = clips[clips["take"].notna() | (clips["intended_emotion"] == "neutral")].copy()
    real["emotion"] = real["intended_emotion"]
    out = {"real": {}, "models": {}, "cues": CUES, "family": FAMILY}
    div = between_actor_divergence(real, CUES)
    out["real"]["divergence"] = {c: round(v, 4) for c, v in div.items()}
    out["real"]["divergence_family"] = family_means(div)
    women, men = real[real.actor_sex == "F"], real[real.actor_sex == "M"]
    tv = cue_tvd(women, men, CUES)
    out["real"]["women_vs_men_tvd"] = {c: r2(v) for c, v in tv.items()}
    out["real"]["women_vs_men_tvd_mean"] = r2(np.mean(list(tv.values())))
    out["real"]["women_vs_men_value_diff"] = r2(value_share_diff(women, men, CUES))
    out["real"]["shares"] = cue_shares(real)
    out["real"]["shares_by_emotion"] = {e: cue_shares(real[real.emotion == e]) for e in EMOTIONS}
    out["real"]["n_takes"] = int(len(real))
    for m in MODELS:
        d = RESULTS / "generation" / m
        if not (d / "control.csv").exists():
            continue
        ctrl = pd.read_csv(d / "control.csv", dtype={"actor": str})
        div = between_actor_divergence(ctrl, CUES)
        row = {"control": {"n_takes": int(len(ctrl)), "n_actors": int(ctrl.actor.nunique()),
                           "divergence": {c: round(v, 4) for c, v in div.items()}, "divergence_family": family_means(div),
                           "shares": cue_shares(ctrl), "shares_by_emotion": {e: cue_shares(ctrl[ctrl.emotion == e]) for e in EMOTIONS},
                           "identical_across_actors": identical_share(ctrl)},
               "conditions": {}}
        for f in sorted(d.glob("*.csv")):
            cond = f.stem
            if cond == "control":
                continue
            g = pd.read_csv(f, dtype={"actor": str})
            j = ctrl.merge(g, on="take_id", suffixes=("_c", "_p"))
            if len(j) < 100:
                continue
            cells, tvs = {}, []
            for c in CUES:
                a, b = j[f"{c}_c"].astype(str), j[f"{c}_p"].astype(str)
                labs = sorted(set(a) | set(b))
                if len(labs) < 2:
                    cells[c] = {"tvd": 0.0, "p": 1.0}; tvs.append(0.0); continue
                r = compare_conditions(a, b, labs, j["actor_c"].astype(str), n_perm=2000, n_boot=200)
                cells[c] = {"tvd": r["tvd"], "p": r["p_tvd"], "ci": r.get("tvd_ci")}
                tvs.append(r["tvd"])
            row["conditions"][cond] = {"n_pairs": int(len(j)), "n_takes": int(len(g)), "tvd_mean": r2(np.mean(tvs)), "tvd_max": r2(max(tvs)),
                                       "cues": cells, "shares": cue_shares(g)}
        # woman against man, compared with the real actresses against the real actors
        if "woman" in row["conditions"] and "man" in row["conditions"]:
            w = pd.read_csv(d / "woman.csv", dtype={"actor": str}); mn = pd.read_csv(d / "man.csv", dtype={"actor": str})
            tvw = cue_tvd(w, mn, CUES)
            real_v = np.array([tv[c] for c in CUES]); synth_v = np.array([tvw[c] for c in CUES])
            row["woman_vs_man"] = {"tvd": {c: r2(v) for c, v in tvw.items()}, "tvd_mean": r2(synth_v.mean()),
                                   "value_diff": r2(value_share_diff(w, mn, CUES)), "real_value_diff": r2(value_share_diff(women, men, CUES)),
                                   "real_tvd_mean": r2(real_v.mean()),
                                   "correlation_with_real": r2(np.corrcoef(real_v, synth_v)[0, 1]) if synth_v.std() > 0 else None}
        conds = [v["tvd_mean"] for v in row["conditions"].values()]
        row["summary"] = {"n_conditions": len(conds), "tvd_median": r2(np.median(conds)) if conds else None,
                          "tvd_min": r2(min(conds)) if conds else None, "tvd_max": r2(max(conds)) if conds else None}
        out["models"][m] = row
        log(f"  {m}: {len(conds)} conditions, median TVD {row['summary']['tvd_median']}, divergence all {row['control']['divergence_family']['all']}")
    return out


def family_means(div: dict) -> dict:
    fam = {}
    for c, v in div.items():
        fam.setdefault(FAMILY[c], []).append(v)
    out = {k: round(float(np.mean(v)), 3) for k, v in fam.items()}
    out["all"] = round(float(np.mean(list(div.values()))), 3)
    return out


def cue_shares(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    return {c: {k: r2(100 * v) for k, v in df[c].astype(str).value_counts(normalize=True).items()} for c in CUES}


def identical_share(df: pd.DataFrame) -> float | None:
    key = df[CUES].astype(str).agg("|".join, axis=1)
    g = pd.DataFrame({"cell": df["emotion"].astype(str) + "/" + df["sentence"].astype(str) + "/" + df["intensity"].astype(str), "vec": key})
    per = g.groupby("cell")["vec"].nunique()
    return r2(100 * float((per == 1).mean()))


# ---------------------------------------------------------------- site data
def site_clips(clips: pd.DataFrame) -> dict:
    code = {c: {v: i for i, v in enumerate(vals)} for c, vals in __import__("emobias.corpus", fromlist=["VOCAB"]).VOCAB.items()}
    rows = []
    for r in clips.itertuples():
        cues = "".join(str(code[c].get(str(getattr(r, c)), 0)) for c in CUES)
        votes = [int(getattr(r, f"votes_{e}")) if str(getattr(r, f"votes_{e}")) not in ("", "nan") else 0 for e in EMOTIONS[1:]]
        rows.append([r.clip, str(r.actor), r.actor_sex, r.intended_emotion, "" if pd.isna(r.intensity) else int(r.intensity),
                     int(r.sentence), r.majority_label if isinstance(r.majority_label, str) else "", votes, cues, r.transcript])
    return {"columns": ["clip", "actor", "actor_sex", "intended_emotion", "intensity", "sentence", "majority_label", "votes", "cues", "transcript"],
            "vote_emotions": EMOTIONS[1:], "cue_codes": {c: list(v) for c, v in __import__("emobias.corpus", fromlist=["VOCAB"]).VOCAB.items()}, "rows": rows}


def site_predictions(clips: pd.DataFrame) -> dict:
    order = clips["clip"].tolist()
    idx = {e: str(i) for i, e in enumerate(EMOTIONS)}
    out = {"clips": order, "emotions": EMOTIONS, "models": {}, "intensity": {}}
    icode = {"low": "l", "high": "h", "none": "n"}
    for m in MODELS:
        d = predictions(m)
        if d.empty:
            continue
        piv = d.pivot_table(index="clip", columns="condition", values="emotion", aggfunc="first").reindex(order)
        out["models"][m] = {c: "".join(idx.get(v, "-") if isinstance(v, str) else "-" for v in piv[c]) for c in piv.columns}
        pin = d.assign(intensity=d["intensity"].fillna("?").astype(str).str.lower()).pivot_table(index="clip", columns="condition", values="intensity", aggfunc="first").reindex(order)
        out["intensity"][m] = {c: "".join(icode.get(v, "-") if isinstance(v, str) else "-" for v in pin[c]) for c in pin.columns}
    return out


def site_check(corpus: str) -> dict:
    clips = load_clips(corpus)
    order = clips["clip"].tolist()
    out = {"corpus": corpus, "emotions": CORPORA[corpus]["emotions"], "reference": CORPORA[corpus]["reference"],
           "rows": [[r.clip, str(r.actor), r.intended_emotion, r.majority_label if "majority_label" in clips.columns and isinstance(r.majority_label, str) else "", r.transcript]
                    for r in clips.itertuples()],
           "columns": ["clip", "actor", "intended_emotion", "majority_label", "transcript"], "models": {}}
    idx = {e: str(i) for i, e in enumerate(out["emotions"])}
    for m in MODELS:
        d = load_predictions(m, f"check_{corpus}")
        if d is None:
            continue
        piv = d.pivot_table(index="clip", columns="condition", values="emotion", aggfunc="first").reindex(order)
        out["models"][m] = {c: "".join(idx.get(v, "-") if isinstance(v, str) else "-" for v in piv[c]) for c in piv.columns}
    return out


def site_generation_takes() -> dict:
    from .corpus import VOCAB
    code = {c: {v: i for i, v in enumerate(vals)} for c, vals in VOCAB.items()}
    out = {"cues": CUES, "models": {}}
    for m in MODELS:
        d = RESULTS / "generation" / m
        if not d.exists():
            continue
        out["models"][m] = {}
        for f in sorted(d.glob("*.csv")):
            g = pd.read_csv(f, dtype={"actor": str})
            out["models"][m][f.stem] = [[r.take_id, r.actor, r.emotion, int(r.sentence), int(r.intensity),
                                         "".join(str(code[c].get(str(getattr(r, c)), 0)) for c in CUES)] for r in g.itertuples()]
    return out


def site_prompts(clips: pd.DataFrame) -> dict:
    from .prompts import generation_prompt, load_template, recognition_prompt
    ex = clips[clips.majority_label.notna()].iloc[0]
    conds = {c: speaker_phrase(c) for c in [CONTROL] + list(ATTRIBUTES)}
    conds.update({f"{g}+{a}": speaker_phrase(f"{g}+{a}") for g in COMBINED_GENDERS for a in OTHERS})
    ex = {k: ex[k] for k in ("clip", "transcript", "text", "intended_emotion", "intensity")}
    return {"templates": {n: load_template(n) for n in ("recognition", "answer_block", "generation", "actors_brief")},
            "attributes": {a: {**ATTRIBUTES[a], "phrase": speaker_phrase(a), "sentence": attribute_sentence(a).strip()} for a in ATTRIBUTES},
            "conditions": conds,
            "generation_sentences": {c: attribute_sentence(c).strip() for c in [CONTROL] + list(ATTRIBUTES)},
            "example": {"clip": ex["clip"], "control": recognition_prompt(ex["transcript"], CONTROL),
                        "woman": recognition_prompt(ex["transcript"], "woman"),
                        "generation_control": generation_prompt(ex["text"], ex["intended_emotion"], int(ex["intensity"]), "control"),
                        "generation_woman": generation_prompt(ex["text"], ex["intended_emotion"], int(ex["intensity"]), "woman")}}


def site_code() -> dict:
    files = {}
    for p in sorted((ROOT / "emobias").glob("*.py")) + sorted((ROOT / "configs").glob("*.yaml")) + [ROOT / "prompts" / "attributes.yaml", ROOT / "README.md"]:
        files[str(p.relative_to(ROOT))] = p.read_text(encoding="utf-8")
    return files


def run(log=print) -> None:
    MEASURES.mkdir(parents=True, exist_ok=True)
    SITE.mkdir(parents=True, exist_ok=True)
    clips_all = load_clips("eve")
    clips = rated(clips_all).reset_index(drop=True)

    log("reference point")
    ref = reference_point(clips, log)
    log("stated attribute in recognition")
    effects = attribute_effects(clips, log)
    ctrl_shares = {m: ref["models"][m]["shares"] for m in effects if m in ref["models"]}
    tables = summary_tables(effects, ctrl_shares)
    fig = figure_matrices(effects, ctrl_shares)
    log("corpus check")
    check = corpus_check(log)
    log("generation")
    gen = generation(clips_all, log)

    meta = {"emotions": EMOTIONS, "models": MODELS, "attributes": {a: {"group": ATTRIBUTES[a]["group"], "phrase": speaker_phrase(a)} for a in ATTRIBUTES},
            "single": SINGLE, "others": OTHERS, "combined_genders": COMBINED_GENDERS, "control": CONTROL, "repeat": REPEAT,
            "n_rated_clips": int(len(clips))}
    measures = {"meta": meta, "reference": ref, "effects": effects, **tables, "figure1": fig, "check": check, "generation": gen}
    for name, obj in (("reference", ref), ("effects", effects), ("tables", tables), ("figure1", fig), ("check", check), ("generation", gen)):
        (MEASURES / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    (SITE / "measures.json").write_text(json.dumps(measures, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (SITE / "clips.json").write_text(json.dumps(site_clips(clips_all), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (SITE / "predictions.json").write_text(json.dumps(site_predictions(clips_all), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    for corpus in ("eve", "iemocap", "enterface"):
        (SITE / f"check_{corpus}.json").write_text(json.dumps(site_check(corpus), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (SITE / "generation_takes.json").write_text(json.dumps(site_generation_takes(), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (SITE / "prompts.json").write_text(json.dumps(site_prompts(clips), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (SITE / "code.json").write_text(json.dumps(site_code(), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    write_markdown(measures)
    log(f"written {MEASURES} and {SITE}")


def write_markdown(M: dict) -> None:
    """results/measures/summary.md: the tables of the paper in text form."""
    L = ["# Measures (generated by `python -m emobias analyse`)", ""]
    ref = M["reference"]
    L += ["## Reference point (control condition, rated clips of EVE)", "",
          "| Answers given by | " + " | ".join(EMOTIONS) + " | Accuracy | Macro-F1 |", "|" + "---|" * (len(EMOTIONS) + 3)]
    L.append("| The annotators | " + " | ".join("--" if e == "neutral" else f"{ref['annotators']['shares'][e]:.1f}" for e in EMOTIONS) + " | -- | -- |")
    for m, r in ref["models"].items():
        L.append(f"| {MODELS[m]['name']} | " + " | ".join(f"{r['shares'][e]:.1f}" for e in EMOTIONS) + f" | {r['accuracy']:.1f} [{r['accuracy_ci'][0]:.1f}, {r['accuracy_ci'][1]:.1f}] | {r['macro_f1']:.1f} |")
    L += ["", f"Fleiss' kappa between the models: {ref.get('models_fleiss_kappa')} on {ref.get('models_fleiss_n_clips')} clips; between the annotators: {ref['annotators']['fleiss_kappa']}.", ""]
    L += ["## Effect of one stated attribute (Table 3)", "", "| Model | Attributes | Clips whose emotion changes (%) | TVD median / largest (points) | Repeat control flip / TVD | Two largest shifts (points) |", "|---|---|---|---|---|---|"]
    for m, t in M["table3"].items():
        rep = f"{t['repeat']['flip']} / {t['repeat']['tvd']}" if t["repeat"] else "--"
        L.append(f"| {MODELS[m]['name']} | {t['n_attributes']} | {t['flip_min']} to {t['flip_max']} | {t['tvd_median']} / {t['tvd_max']} | {rep} | " +
                 "; ".join(f"{s['condition']} -> {s['emotion']} {s['shift']:+.1f}" for s in t["largest_shifts"]) + " |")
    L += ["", "## Origin of the clips whose emotion changes (Table 4)", ""]
    for m, t in M["table4"].items():
        L.append(f"- {MODELS[m]['name']}: {t['condition']} -> {t['emotion']} {t['shift']:+.1f} points; " +
                 "; ".join(f"{x['from']} -> {x['to']}: {x['clips']} clips ({x['share_of_control_class']} % of the control class)" for x in t["transfers"]))
    L += ["", "## Combinations", ""]
    for m, a in M["additivity"].items():
        L.append(f"- {MODELS[m]['name']}: slope of the distance of a combination on the sum of the distances of its two attributes: {a['slope_through_origin']} ({a['n']} combinations)")
    L += ["", "## Corpus check (words alone, behaviour alone, both)", ""]
    for corpus, c in M["check"].items():
        L.append(f"- {corpus} ({c['n_classes']} classes, chance {c['chance']} %, reference {c['reference']}, {c['n_clips']} clips): " +
                 "; ".join(f"{MODELS[m]['name']}: " + ", ".join(f"{k} {v['accuracy']}" for k, v in r.items()) for m, r in c["models"].items()))
    L += ["", "## Generation (Table 5)", "", "| Written by | Prosody | Head | Face | Gaze | All | Attribute TVD median (min-max, n) |", "|---|---|---|---|---|---|---|"]
    g = M["generation"]
    fr = g["real"]["divergence_family"]
    L.append(f"| Real actors of EVE | {fr['prosody']} | {fr['head']} | {fr['face']} | {fr['gaze']} | {fr['all']} | -- |")
    for m, r in g["models"].items():
        f = r["control"]["divergence_family"]; s = r["summary"]
        L.append(f"| {MODELS[m]['name']} | {f['prosody']} | {f['head']} | {f['face']} | {f['gaze']} | {f['all']} | {s['tvd_median']} ({s['tvd_min']}-{s['tvd_max']}, {s['n_conditions']}) |")
    L.append("")
    L.append(f"Real actresses against real actors of EVE: mean TVD over the cues {g['real']['women_vs_men_tvd_mean']} points, "
             f"mean difference of the share of a cue value {g['real']['women_vs_men_value_diff']} points.")
    for m, r in g["models"].items():
        if "woman_vs_man" in r:
            w = r["woman_vs_man"]
            L.append(f"- {MODELS[m]['name']}: woman against man, TVD {w['tvd_mean']} points, share of a cue value {w['value_diff']} points, correlation with the real difference {w['correlation_with_real']}")
    (MEASURES / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
