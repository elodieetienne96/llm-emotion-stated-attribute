# Gender Bias in LLM Emotion Recognition from Verbalised Multimodal Transcripts

Code, prompts, data and raw answers of the paper. The repository is anonymised for review.

The paper proposes a methodology to measure the effect of a **stated attribute** (gender, age,
descent, personality) on what an LLM does with emotional behaviour. The behaviour is fixed in an
**enriched multimodal transcript**, one sentence that says what the speaker said and how
(prosody, facial action units, gaze, head). The same transcript is presented in a control
condition, *The speaker said …*, and in an attribute condition, *The woman said …*. Only the
stated attribute changes. The method covers two tasks: **recognition** (the LLM names the
emotion) and **generation** (the LLM writes the transcript from the instructions given to the
actors).

An interactive site, https://elodieetienne96.github.io/llm-emotion-stated-attribute/ (`docs/`, served with GitHub Pages), shows the transcripts, the emotions
perceived by the annotators and by each model, the effect of every attribute, the generated
transcripts, the measures, the prompts and the code.

## Layout

```
prompts/        the prompts, verbatim
  recognition.txt      recognition task (control and attribute conditions)
  check.txt            corpus check: words alone, described behaviour alone, both
  answer_block.txt     the JSON answer asked in recognition
  attributes.yaml      the stated attributes (Table 1) and how they are phrased
  generation.txt       generation task
  actors_brief.txt     the brief given to the actors of the corpus, reproduced verbatim
emobias/        the code (Python 3.10+, numpy, pandas, pyyaml, requests)
  corpus.py            corpora, cue vocabulary, rendering of the enriched multimodal transcript
  prompts.py           prompt construction, attribute phrasing, answer parsing
  llm.py               calls to the models (OpenRouter), disk cache
  recognise.py         recognition task
  generate.py          generation task
  measures.py          accuracy, macro-F1, Fleiss' kappa, flip rate, shifts, TVD, tests
  analysis.py          every number of the paper, and the data of the site
configs/        one YAML per experiment of the paper
data/           the corpora: one row per clip, 22 cues, transcript, human labels
results/
  recognition/<model>/<group>.csv    one answer per clip and condition (parsed emotion, intensity, raw answer)
  generation/<model>/<condition>.csv one row per generated take (22 cues, transcript)
  measures/                          the computed measures (JSON) and summary.md
docs/           the interactive site (static HTML + JSON)
```

## Data

`data/eve/clips.csv`: the 2 100 clips of the English part of the EVE corpus used in the paper
(the rated take of the ten actors: ten neutral sentences and 2 000 emotional clips). Columns:
`clip`, `actor`, `actor_sex`, `intended_emotion`, `intensity`, `sentence`, `take`, `text`,
`majority_label` (the emotion chosen by the majority of the fifteen annotators, empty for the
neutral clips, which were not rated), `n_annotators`, `votes_<emotion>` (number of annotators
who chose each emotion), the 22 cues (`speech_rate`, `pitch`, `loudness`, `gaze`, `head`,
`AU01` … `AU45`) and `transcript`, the enriched multimodal transcript.

`data/iemocap/clips.csv` and `data/enterface/clips.csv`: the two corpora of the corpus check
(Section 4.1), same cue columns and transcript; the reference is `majority_label` for IEMOCAP
and `intended_emotion` for eNTERFACE'05.

The transcripts begin with the speaker identifier of the corpus (`Actor 01`). The code replaces
it by the phrase of the condition before the transcript enters the prompt.

## Conditions

| Condition id | Transcript begins with | Generation sentence |
|---|---|---|
| `speaker` (control) | The speaker said | (nothing) |
| `speaker_repeat` | The speaker said, sent a second time | |
| `woman`, `man`, `non_binary`, `transgender`, `other_gender`, `trans_woman`, `trans_man` | The woman said, … | You are a woman. |
| `young`, `middle_aged`, `old` | The young person said, … | You are a young person. |
| `western_european`, … `latin_american` | The person of Western European descent said, … | You are a person of … descent. |
| `extraverted`, `agreeable`, `conscientious`, `neurotic`, `open` | The extraverted person said, … | You are an extraverted person. |
| `woman+old`, `transgender+neurotic`, … (75) | The old woman said, The neurotic transgender person said | You are an old woman. |

The full list is produced by `python -c "from emobias.prompts import *; print(all_conditions())"`.

## Running

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # put an OpenRouter key in .env

python -m emobias plan      configs/recognition_attributes.yaml     # calls and example prompt
python -m emobias recognise configs/recognition_control.yaml
python -m emobias recognise configs/recognition_attributes.yaml
python -m emobias recognise configs/recognition_combinations.yaml
python -m emobias recognise configs/check_eve.yaml                 # then input: words_only / behaviour_only
python -m emobias generate  configs/generation.yaml
python -m emobias analyse                                          # results/measures/ and docs/data/
python -m emobias recognise configs/smoke_mock.yaml                 # no network: tests the pipeline
```

Recognition calls use temperature 0 and seed 0 (where the endpoint accepts a seed); the repeat
control is the same prompt with seed 1, so that the cache does not serve the first answer.
Generation uses temperature 1; a synthetic actor is a seed (1 to 10). Every answer is cached in
`cache/`, so an interrupted run resumes without paying twice.

## Measures

`python -m emobias analyse` recomputes from `results/` and `data/`:

- the corpus check: accuracy from the words alone, the described behaviour alone, and both;
- the reference point: share of each emotion in the answers of each model, accuracy and macro-F1
  against the majority label with 95 % confidence intervals (bootstrap over actors), Fleiss' kappa
  between models and between annotators;
- the effect of a stated attribute: for every model and condition, the share of clips whose
  emotion changes (flip), the shift of each emotion in points, the total variation distance,
  the transfers between emotions, a paired permutation test (4 000 draws), actor-level bootstrap
  intervals, Benjamini-Hochberg correction at 5 % over the attribute × class shifts of a model, and
  the repeat control as a floor;
- the combinations: the distance of each combination against the sum of the distances of its
  two attributes;
- the generation task: the total variation distance of each written cue between an attribute
  condition and the control, the difference between *woman* and *man* against the real
  difference between the actresses and the actors of the corpus, and the Jensen-Shannon
  divergence between two actors (real or synthetic).

`results/measures/summary.md` gives the tables of the paper in text form.

## Site

Online: https://elodieetienne96.github.io/llm-emotion-stated-attribute/

Open `docs/index.html` (it reads `docs/data/*.json`; serve the folder with any static server,
for example `python -m http.server -d docs`, or enable GitHub Pages on `docs/`).
