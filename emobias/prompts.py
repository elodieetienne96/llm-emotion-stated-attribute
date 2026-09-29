"""Prompts of the two tasks.

Recognition: the LLM receives the enriched multimodal transcript and names the emotion.
In the control condition the transcript begins with "The speaker said". In an attribute
condition "The speaker" is replaced by the attribute phrase ("The woman said"). Nothing
else changes. A combination states a gender attribute and one attribute of another group
("The old woman said").

Generation: the LLM receives the actors' brief, a sentence, an emotion and an intensity,
and writes the 22 cues. In an attribute condition one sentence before the brief names the
attribute ("You are a woman."), in the same form as in the recognition task.
"""
from __future__ import annotations

import re

import yaml

from .corpus import AU_CODES, AU_FACS, CUES, EMOTIONS, PROMPTS, VOCAB

CONTROL = "speaker"
REPEAT = "speaker_repeat"        # the control prompt sent a second time (repeat control)
SPEAKER_RE = re.compile(r"^(The speaker|Actor \d+|[A-Za-z]+ ?\d+[FM]?|S\d+|Speaker[_ ]?\w+)\b")

_SPEC = yaml.safe_load((PROMPTS / "attributes.yaml").read_text(encoding="utf-8"))
GROUPS: dict[str, dict] = _SPEC["groups"]
ATTRIBUTES: dict[str, dict] = {name: {**spec, "group": group} for group, d in GROUPS.items() for name, spec in d.items()}
COMBINED_GENDERS: list[str] = _SPEC["combined_genders"]


def load_template(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text(encoding="utf-8").strip("\n")


def parse_condition(condition: str) -> list[str]:
    """'speaker' -> [], 'woman' -> ['woman'], 'woman+old' -> ['woman', 'old'] (checked)."""
    if condition in (CONTROL, REPEAT):
        return []
    parts = condition.split("+")
    groups = []
    for p in parts:
        if p not in ATTRIBUTES:
            raise ValueError(f"unknown attribute: {p} (see prompts/attributes.yaml)")
        if ATTRIBUTES[p]["group"] in groups:
            raise ValueError(f"two attributes of the same group in {condition}")
        groups.append(ATTRIBUTES[p]["group"])
    return parts


def noun_phrase(condition: str) -> str:
    """The noun phrase of a condition, without article: 'speaker', 'woman', 'older adult', 'old man',
    'person of Western European descent', 'neurotic transgender person'."""
    parts = parse_condition(condition)
    if not parts:
        return "speaker"
    if len(parts) == 1 and "alone" in ATTRIBUTES[parts[0]]:
        return ATTRIBUTES[parts[0]]["alone"]
    noun, before, after = "person", [], []
    for p in parts:
        spec = ATTRIBUTES[p]
        if "noun" in spec:
            noun = spec["noun"]
        if "before" in spec:
            before.append(spec["before"])
        if "after" in spec:
            after.append(spec["after"])
    return " ".join(before + [noun] + ([" and ".join(after)] if after else []))


def speaker_phrase(condition: str) -> str:
    """What the transcript begins with: 'The speaker', 'The woman', 'The old woman'."""
    return "The " + noun_phrase(condition)


def attribute_sentence(condition: str) -> str:
    """Generation task: 'You are a woman.', 'You are an old woman.'; empty in the control condition."""
    if condition in ("control", CONTROL, REPEAT, "", None):
        return ""
    np_ = noun_phrase(condition)
    article = "an" if np_[:1].lower() in "aeiou" else "a"
    return f" You are {article} {np_}."


def all_conditions() -> dict[str, list[str]]:
    """The conditions of the recognition task: control, repeat, the attributes, the 75 combinations."""
    singles = list(ATTRIBUTES)
    others = [a for a in singles if ATTRIBUTES[a]["group"] != "gender"]
    combos = [f"{g}+{a}" for g in COMBINED_GENDERS for a in others]
    return {"control": [CONTROL, REPEAT], "attributes": singles, "combinations": combos}


# ---------------------------------------------------------------- recognition
def set_speaker(transcript: str, condition: str) -> str:
    """Replace the speaker identifier at the start of a transcript by the phrase of the condition."""
    return SPEAKER_RE.sub(speaker_phrase(condition), transcript, count=1)


def recognition_prompt(transcript: str, condition: str = CONTROL, emotions: list[str] = EMOTIONS,
                       template: str = "recognition") -> str:
    body = load_template(template)
    return body.format(emotions=", ".join(emotions), transcript=set_speaker(transcript, condition),
                       answer_block=load_template("answer_block"))


def parse_recognition_answer(text: str, emotions: list[str] = EMOTIONS) -> tuple[str | None, str | None]:
    """The emotion and the intensity of a JSON answer; (None, None) if no listed emotion is found."""
    import json
    if not text:
        return None, None
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    i, j = s.find("{"), s.rfind("}")
    obj = None
    if i >= 0 and j > i:
        try:
            obj = json.loads(s[i:j + 1])
        except json.JSONDecodeError:
            obj = dict(re.findall(r'"([^"]+)"\s*:\s*"([^"]*)"', s[i:j + 1])) or None
    if isinstance(obj, dict):
        emo = str(obj.get("emotion", "")).strip().lower()
        inten = str(obj.get("intensity", "")).strip().lower() or None
        if emo in emotions:
            return emo, inten
    found = [e for e in emotions if re.search(rf"\b{e}\b", s, re.I)]
    return (found[0], None) if len(found) == 1 else (None, None)


# ---------------------------------------------------------------- generation
SCHEME = ("Coding scheme. Speech rate, pitch and loudness are relative to your own neutral delivery "
          "of these sentences.\n"
          "- speech_rate: slow | normal | fast\n"
          "- pitch: lower | normal | higher\n"
          "- loudness: quieter | normal | louder\n"
          "- gaze: front | left | right | up | down   (front = looking at the camera)\n"
          "- head: none | nodding | shaking | tilting\n"
          "- facial action units (FACS), one value each: none | weak | moderate | strong\n"
          "  " + ", ".join(f"{au} {AU_FACS[au]}" + (" (smile)" if au == "AU12" else "") for au in AU_CODES))
GEN_ANSWER = ("Answer with a single JSON object and nothing else (no explanation, no code fence), with exactly "
              f"these {len(CUES)} keys in this order:\n" + "{" + ", ".join(f'"{f}": ""' for f in CUES) + "}")
GEN_RETRY = "\n\nYour previous answer was not valid ({reason}). Answer again with only the JSON object, using only the listed values."
BRIEF_NAME = {"neutral": "Neutral", "fear": "Fear", "anger": "Anger", "happiness": "Happiness",
              "sadness": "Sadness", "disgust": "Disgust", "surprise": "Surprise",
              "confidence": "Self-Confidence", "confusion": "Confusion", "contempt": "Contempt",
              "empathy": "Sympathy"}
INTENSITY_TEXT = {1: "Low intensity (see the sheet)", 2: "High intensity (see the sheet)",
                  0: "Not applicable: this is one of your neutral recordings, the reference for all the "
                     "others. Say the sentence in a neutral way, without emotion."}


def generation_prompt(sentence: str, emotion: str, intensity: int, condition: str = "control") -> str:
    return load_template("generation").format(attribute_sentence=attribute_sentence(condition),
                                              brief=load_template("actors_brief"), scheme=SCHEME,
                                              sentence=sentence, emotion=BRIEF_NAME[emotion],
                                              intensity=INTENSITY_TEXT[int(intensity)], answer_block=GEN_ANSWER)


_KEY_ALIASES = {"speech rate": "speech_rate", "rate": "speech_rate", "speechrate": "speech_rate",
                "head_gesture": "head", "gaze_direction": "gaze"}


def parse_generation_answer(text: str) -> tuple[dict | None, str]:
    """The 22 cues of a JSON answer, or (None, reason) if the answer leaves the scheme."""
    import json
    if not text or not text.strip():
        return None, "empty answer"
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None, "no JSON object"
    try:
        obj = json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        obj = dict(re.findall(r'"([^"]+)"\s*:\s*"([^"]*)"', s[i:j + 1]))
        if not obj:
            return None, "unparsable JSON"
    if not isinstance(obj, dict):
        return None, "JSON is not an object"
    norm = {}
    for k, v in obj.items():
        key = str(k).strip()
        m = re.match(r"^au[\s_-]?0*(\d+)$", key, re.I)
        key = f"AU{int(m.group(1)):02d}" if m else _KEY_ALIASES.get(key.lower(), key.lower())
        norm[key] = str(v).strip().lower() if v is not None else ""
    missing = [f for f in CUES if f not in norm]
    if missing:
        return None, "missing keys: " + ", ".join(missing[:5])
    bad = [f"{f}={norm[f]!r}" for f in CUES if norm[f] not in VOCAB[f]]
    if bad:
        return None, "values outside the scheme: " + ", ".join(bad[:5])
    return {f: norm[f] for f in CUES}, ""
