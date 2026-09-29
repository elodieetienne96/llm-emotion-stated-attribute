"""Corpora, cue vocabulary and the enriched multimodal transcript.

Every clip is described by 22 cues: three prosodic levels relative to the speaker's own
neutral recordings, gaze, head gesture and seventeen facial action units. The enriched
multimodal transcript is one sentence rendered from these cues (`render_transcript`).
The same renderer writes the transcripts of the generation task, so a generated
transcript has exactly the form of a transcript extracted from a video.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PROMPTS = ROOT / "prompts"

EMOTIONS = ["neutral", "fear", "anger", "happiness", "sadness", "disgust", "surprise",
            "confidence", "confusion", "contempt", "empathy"]

CORPORA = {
    "eve": {"file": "eve/clips.csv", "emotions": EMOTIONS, "reference": "majority_label"},
    "iemocap": {"file": "iemocap/clips.csv", "emotions": ["neutral", "anger", "sadness", "happiness"], "reference": "majority_label"},
    "enterface": {"file": "enterface/clips.csv", "emotions": ["anger", "disgust", "fear", "happiness", "sadness", "surprise"], "reference": "intended_emotion"},
}

AU_CODES = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU12", "AU14",
            "AU15", "AU17", "AU20", "AU23", "AU25", "AU26", "AU45"]
CUES = ["speech_rate", "pitch", "loudness", "gaze", "head"] + AU_CODES
VOCAB = {"speech_rate": ["slow", "normal", "fast"],
         "pitch": ["lower", "normal", "higher"],
         "loudness": ["quieter", "normal", "louder"],
         "gaze": ["front", "left", "right", "up", "down"],
         "head": ["none", "nodding", "shaking", "tilting"],
         **{au: ["none", "weak", "moderate", "strong"] for au in AU_CODES}}

AU_TEXT = {"AU01": "raised inner eyebrows", "AU02": "raised outer eyebrows", "AU04": "lowered brows",
           "AU05": "raised upper eyelids", "AU06": "raised cheeks", "AU07": "tightened eyelids",
           "AU09": "wrinkled nose", "AU10": "raised upper lip", "AU12": "a smile", "AU14": "a tight smile",
           "AU15": "downturned mouth corners", "AU17": "raised chin", "AU20": "stretched lips",
           "AU23": "tightened lips", "AU25": "parted lips", "AU26": "a dropped jaw", "AU45": "blinking"}
AU_FACS = {"AU01": "inner brow raiser", "AU02": "outer brow raiser", "AU04": "brow lowerer",
           "AU05": "upper lid raiser", "AU06": "cheek raiser", "AU07": "lid tightener",
           "AU09": "nose wrinkler", "AU10": "upper lip raiser", "AU12": "lip corner puller",
           "AU14": "dimpler", "AU15": "lip corner depressor", "AU17": "chin raiser",
           "AU20": "lip stretcher", "AU23": "lip tightener", "AU25": "lips part", "AU26": "jaw drop",
           "AU45": "blink"}
HEAD_TEXT = {"nodding": "while *nodding*", "shaking": "while *shaking* their **head**",
             "tilting": "while *tilting* their **head**"}
GAZE_DIRECTION = {"left": "to their left", "right": "to their right", "up": "upwards", "down": "downwards"}

# The ten phonetically balanced sentences of the corpus, as given to the actors.
SENTENCES = ["The birch canoe slid on the smooth planks.",
             "Glue the sheet to the dark blue background.",
             "It's easy to tell the depth of a well.",
             "These days a chicken leg is a rare dish.",
             "Rice is often served in round bowls.",
             "The juice of lemons makes fine punch.",
             "The box was thrown beside the parked truck.",
             "The hogs were fed chopped corn and garbage.",
             "Four hours of steady work faced us.",
             "A large size in stockings is hard to sell."]


def load_clips(corpus: str = "eve") -> pd.DataFrame:
    """The clips of a corpus: one row per clip, with its cues, transcript and human labels."""
    df = pd.read_csv(DATA / CORPORA[corpus]["file"], low_memory=False, dtype={"actor": str})
    df["corpus"] = corpus
    return df


def rated(df: pd.DataFrame) -> pd.DataFrame:
    """The clips that have a human label (for EVE, the 2 000 emotional clips rated by the annotators)."""
    return df[df["majority_label"].notna() & (df["majority_label"].astype(str) != "")]


def _join(parts: list[str]) -> str:
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return "".join(parts)
    if len(parts) == 2:
        return parts[0] + " and " + parts[1]
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def render_transcript(cues: dict, speaker: str, text: str | None) -> str:
    """The enriched multimodal transcript, from the 22 cue values.

    `speaker` is what the sentence starts with ("The speaker", "The woman"). With `text`
    None, the quoted sentence is left out ("The speaker spoke, with ...")."""
    mods = [f"with *{cues['speech_rate']}* **speech rate**", f"with *{cues['pitch']}* **pitch**",
            f"with *{cues['loudness']}* **loudness**"]
    gaze = str(cues.get("gaze") or "front").lower()
    mods.append("while **looking at the camera**" if gaze in ("front", "nan", "none", "")
                else f"with **gaze** *averted* ({GAZE_DIRECTION.get(gaze, gaze)})")
    head = str(cues.get("head", "none"))
    if head in HEAD_TEXT:
        mods.append(HEAD_TEXT[head])
    aus = [f"*{cues[au]}* **{AU_TEXT[au]}**" for au in AU_CODES if cues.get(au) in ("weak", "moderate", "strong")]
    if aus:
        mods.append("with " + _join(aus))
    head_part = f"{speaker} said “{text}”" if text is not None else f"{speaker} spoke"
    rest = [m[5:] if m.startswith("with ") else m for m in mods[1:]]
    return f"{head_part}, {_join([mods[0]] + rest)}."


def transcript_without_words(transcript: str) -> str:
    """Behaviour alone: the quoted sentence is removed ("The speaker spoke, with ...")."""
    import re
    return re.sub(r"\s+said\s+[“\"«]\s?.*?\s?[”\"»],?", " spoke,", transcript, count=1)


def transcript_words_only(transcript: str) -> str:
    """Words alone: only the quoted sentence is kept ("The speaker said “...”.")."""
    cut = transcript.find(", with ")
    head = transcript if cut < 0 else transcript[:cut]
    return head if head.endswith(".") else head + "."
