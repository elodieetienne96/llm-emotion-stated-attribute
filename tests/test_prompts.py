"""python -m pytest tests/ (or python tests/test_prompts.py)"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from emobias.corpus import render_transcript, transcript_without_words, transcript_words_only, CUES
from emobias.prompts import speaker_phrase, attribute_sentence, all_conditions, set_speaker, parse_recognition_answer, parse_generation_answer


def test_phrases():
    assert speaker_phrase("speaker") == "The speaker"
    assert speaker_phrase("woman") == "The woman"
    assert speaker_phrase("old") == "The older adult"
    assert speaker_phrase("young") == "The young adult"
    assert speaker_phrase("other_gender") == "The person of another gender identity"
    assert speaker_phrase("man+old") == "The old man"
    assert speaker_phrase("woman+old") == "The old woman"
    assert speaker_phrase("transgender+neurotic") == "The neurotic transgender person"
    assert speaker_phrase("man+east_asian") == "The man of East Asian descent"
    assert attribute_sentence("control") == ""
    assert attribute_sentence("old") == " You are an older adult."
    c = all_conditions()
    assert len(c["attributes"]) == 22 and len(c["combinations"]) == 75


def test_transcript():
    t = "Actor 03 said “Hello.”, with *slow* **speech rate**, *normal* **pitch**, *louder* **loudness**, while **looking at the camera**, and *weak* **a smile**."
    assert set_speaker(t, "woman").startswith("The woman said")
    assert set_speaker(t, "speaker").startswith("The speaker said")
    assert transcript_without_words(t).startswith("Actor 03 spoke, with")
    assert transcript_words_only(t) == "Actor 03 said “Hello.”."
    cues = {c: "none" for c in CUES}
    cues.update(speech_rate="slow", pitch="normal", loudness="louder", gaze="front", head="none", AU12="weak")
    assert render_transcript(cues, "Actor 03", "Hello.") == t


def test_answers():
    assert parse_recognition_answer('{"emotion": "anger", "intensity": "high"}') == ("anger", "high")
    assert parse_recognition_answer("I think it is sadness.") == ("sadness", None)
    cues, why = parse_generation_answer('{"speech_rate": "slow", "pitch": "normal", "loudness": "louder", "gaze": "front", "head": "none", '
                                        + ", ".join(f'"{c}": "none"' for c in CUES if c.startswith("AU")) + "}")
    assert cues is not None and cues["speech_rate"] == "slow"
    assert parse_generation_answer('{"speech_rate": "very slow"}')[0] is None


if __name__ == "__main__":
    test_phrases(); test_transcript(); test_answers(); print("ok")
