"""The measures of the paper and their statistical tests.

Point of reference (not the object of the study): accuracy and macro-F1 of a model against
the human label, Fleiss' kappa between models.

Object of the study, for one model, one attribute condition against the control on the
same clips:
  - flip rate: share of clips whose recognised emotion changes, in percent of clips;
  - shift: change in the share of each emotion, in percentage points;
  - total variation distance (TVD): half the sum of the absolute shifts, the share of
    answers that would have to move for the two conditions to coincide.
Tests: the unit of observation is the actor, so confidence intervals are obtained by
bootstrap over actors; a paired permutation test tells whether the difference between
two conditions could arise by chance; Benjamini-Hochberg controls the number of tests.
Generation: the same TVD on each cue, and the Jensen-Shannon divergence between the cue
distributions of two actors.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- reference point
def accuracy(pred: pd.Series, ref: pd.Series) -> float:
    return float((pred.to_numpy() == ref.to_numpy()).mean())


def macro_f1(pred: pd.Series, ref: pd.Series, labels: list[str]) -> float:
    f1s = []
    p, r = pred.to_numpy(), ref.to_numpy()
    for lab in labels:
        tp = float(((p == lab) & (r == lab)).sum())
        fp = float(((p == lab) & (r != lab)).sum())
        fn = float(((p != lab) & (r == lab)).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def shares(pred: pd.Series, labels: list[str]) -> dict[str, float]:
    """Share of each label in the answers, in percent (unparsed answers count in the denominator)."""
    n = len(pred)
    vc = pred.value_counts()
    return {lab: 100 * float(vc.get(lab, 0)) / n for lab in labels}


def fleiss_kappa(table: np.ndarray) -> float:
    """Fleiss' kappa from a (subjects x categories) table of counts, each row summing to n raters."""
    table = np.asarray(table, dtype=float)
    n = table.sum(axis=1)
    if len(set(n)) != 1:
        raise ValueError("every subject must have the same number of raters")
    n = n[0]
    p_j = table.sum(axis=0) / table.sum()
    P_i = ((table ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else 1.0


def bootstrap_actors(values: pd.DataFrame, stat, actors: pd.Series, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95 % confidence interval of stat(subset) by resampling the actors with replacement."""
    rng = np.random.default_rng(seed)
    av = actors.to_numpy()
    groups = [np.nonzero(av == u)[0] for u in np.unique(av)]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        boots[b] = stat(values.iloc[idx])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---------------------------------------------------------------- effect of an attribute
def _codes(a: pd.Series, b: pd.Series, labels: list[str]):
    idx = {lab: i for i, lab in enumerate(labels)}
    ca, cb = a.map(idx).to_numpy(), b.map(idx).to_numpy()
    ok = ~(pd.isna(ca) | pd.isna(cb))
    return ca[ok].astype(int), cb[ok].astype(int), ok


def tvd_from_counts(na: np.ndarray, nb: np.ndarray) -> float:
    n = na.sum()
    return float(50.0 * np.abs(na / n - nb / n).sum()) if n else float("nan")


def compare_conditions(control: pd.Series, condition: pd.Series, labels: list[str], actors: pd.Series | None = None,
                       n_perm: int = 4000, n_boot: int = 400, seed: int = 0) -> dict:
    """Flip rate, shifts, TVD, their tests, and the transfer table between two conditions on the same clips.

    control, condition: the emotion answered on each clip (aligned). The paired permutation test
    swaps, clip by clip, the two answers of the discordant pairs; it is computed on the (K x K)
    table of pairs. Per-class permutation p-values test each shift. The bootstrap resamples actors."""
    rng = np.random.default_rng(seed)
    ca, cb, ok = _codes(control, condition, labels)
    K = len(labels)
    if not len(ca):
        return {"n": 0}
    na = np.bincount(ca, minlength=K).astype(float)
    nb = np.bincount(cb, minlength=K).astype(float)
    n = len(ca)
    obs_tvd = tvd_from_counts(na, nb)
    obs_shift = 100 * (nb - na) / n
    tab = np.zeros((K, K), dtype=int)
    np.add.at(tab, (ca, cb), 1)
    di, dj = np.nonzero(tab - np.diag(np.diag(tab)))
    cnt = tab[di, dj]
    hits_tvd, hits_shift = 0, np.zeros(K)
    for _ in range(n_perm):
        sw = rng.binomial(cnt, 0.5)
        d = np.zeros(K)
        np.add.at(d, dj, sw.astype(float))
        np.add.at(d, di, -sw.astype(float))
        if tvd_from_counts(na + d, nb - d) >= obs_tvd:
            hits_tvd += 1
        hits_shift += np.abs(100 * ((nb - d) - (na + d)) / n) >= np.abs(obs_shift) - 1e-12
    out = {"n": int(n), "flip": round(100 * float((ca != cb).mean()), 2), "tvd": round(obs_tvd, 2),
           "p_tvd": round((hits_tvd + 1) / (n_perm + 1), 5),
           "shift": {lab: round(float(s), 2) for lab, s in zip(labels, obs_shift)},
           "p_shift": {lab: round(float((h + 1) / (n_perm + 1)), 5) for lab, h in zip(labels, hits_shift)},
           "transfers": tab.tolist()}
    if actors is not None:
        av = actors.to_numpy()[ok]
        groups = [np.nonzero(av == u)[0] for u in np.unique(av)]
        bt, bf = np.empty(n_boot), np.empty(n_boot)
        bs = np.empty((n_boot, K))
        for b in range(n_boot):
            pick = rng.integers(0, len(groups), len(groups))
            idx = np.concatenate([groups[i] for i in pick])
            xa, xb = np.bincount(ca[idx], minlength=K).astype(float), np.bincount(cb[idx], minlength=K).astype(float)
            bt[b] = tvd_from_counts(xa, xb)
            bf[b] = 100 * float((ca[idx] != cb[idx]).mean())
            bs[b] = 100 * (xb - xa) / len(idx)
        out["tvd_ci"] = [round(float(np.percentile(bt, 2.5)), 2), round(float(np.percentile(bt, 97.5)), 2)]
        out["flip_ci"] = [round(float(np.percentile(bf, 2.5)), 2), round(float(np.percentile(bf, 97.5)), 2)]
        out["shift_ci"] = {lab: [round(float(np.percentile(bs[:, k], 2.5)), 2), round(float(np.percentile(bs[:, k], 97.5)), 2)]
                           for k, lab in enumerate(labels)}
    return out


def benjamini_hochberg(pvals: list[float], q: float = 0.05) -> list[bool]:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, n + 1) / n
    k = int(np.nonzero(passed)[0].max() + 1) if passed.any() else 0
    out = np.zeros(n, dtype=bool)
    out[order[:k]] = True
    return out.tolist()


# ---------------------------------------------------------------- generation
def js_divergence(p: pd.Series, q: pd.Series) -> float:
    """Jensen-Shannon divergence (base 2, between 0 and 1) of two value distributions."""
    idx = sorted(set(p.index) | set(q.index))
    a = p.reindex(idx).fillna(0).to_numpy(float)
    b = q.reindex(idx).fillna(0).to_numpy(float)
    a = a / a.sum() if a.sum() else a
    b = b / b.sum() if b.sum() else b
    m = (a + b) / 2

    def kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def between_actor_divergence(df: pd.DataFrame, cues: list[str], by: str = "emotion") -> dict[str, float]:
    """Mean JS divergence between two actors, per cue: for each emotion and cue, the divergence between
    the value distributions of two actors, averaged over the pairs of actors and the emotions."""
    actors = sorted(df["actor"].unique())
    out = {}
    for c in cues:
        vals = []
        for _, t in df.groupby(by):
            dists = {a: t[t["actor"] == a][c].astype(str).value_counts(normalize=True) for a in actors}
            vals += [js_divergence(dists[a], dists[b]) for a, b in itertools.combinations(actors, 2)
                     if len(dists[a]) and len(dists[b])]
        out[c] = float(np.mean(vals)) if vals else float("nan")
    return out


def cue_tvd(a: pd.DataFrame, b: pd.DataFrame, cues: list[str]) -> dict[str, float]:
    """TVD, in points, between the value distributions of each cue in two sets of takes."""
    out = {}
    for c in cues:
        pa = a[c].astype(str).value_counts(normalize=True)
        pb = b[c].astype(str).value_counts(normalize=True)
        idx = sorted(set(pa.index) | set(pb.index))
        out[c] = float(50 * np.abs(pa.reindex(idx).fillna(0) - pb.reindex(idx).fillna(0)).sum())
    return out


def value_share_diff(a: pd.DataFrame, b: pd.DataFrame, cues: list[str]) -> float:
    """Mean absolute difference, in points, of the share of each cue value between two sets of takes
    (every value of every cue counts once)."""
    diffs = []
    for c in cues:
        pa = a[c].astype(str).value_counts(normalize=True)
        pb = b[c].astype(str).value_counts(normalize=True)
        idx = sorted(set(pa.index) | set(pb.index))
        diffs += list(100 * np.abs(pa.reindex(idx).fillna(0) - pb.reindex(idx).fillna(0)))
    return float(np.mean(diffs)) if diffs else float("nan")
