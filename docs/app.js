/* Interactive site: reads docs/data/*.json written by `python -m emobias analyse`. No dependency. */
"use strict";
const REPO = "";   // e.g. "https://github.com/<user>/<repo>/blob/main/" to link the data files; empty = no links
const TABS = [["overview", "Overview"], ["clips", "Clips"], ["recognition", "Recognised emotions"], ["attributes", "Stated attribute"],
              ["combinations", "Combinations"], ["generation", "Generation"], ["check", "Corpus check"], ["prompts", "Prompts"], ["code", "Code"]];
const COLORS = {neutral: "#9a9a9a", fear: "#7b4fa0", anger: "#c0392b", happiness: "#e0a020", sadness: "#3b6fb6", disgust: "#5f8a2f",
                surprise: "#e07a3a", confidence: "#1f8f8f", confusion: "#b78bc8", contempt: "#7a4b2a", empathy: "#d64d8a"};
const D = {};                 // loaded data
const S = {tab: "overview", clip: null, model: null, cond: "woman"};
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const md = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/\*(.+?)\*/g, "<i>$1</i>");
const f1 = (x, d = 1) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : Number(x).toFixed(d);
const sgn = (x, d = 1) => (x > 0 ? "+" : "") + f1(x, d);
const name = (m) => (D.M.meta.models[m] || {name: m}).name;
const cap = (s) => s.replace(/_/g, " ");
const condLabel = (c) => c === "speaker" ? "The speaker (control)" : c === "speaker_repeat" ? "The speaker, repeat" : (D.P.conditions[c] || cap(c));

async function load(name) {
  if (D[name]) return D[name];
  const r = await fetch(`data/${name}.json`);
  D[name] = await r.json();
  return D[name];
}

async function main() {
  const nav = $("nav");
  TABS.forEach(([id, label]) => {
    const b = document.createElement("button"); b.textContent = label; b.dataset.tab = id;
    b.onclick = () => show(id); nav.appendChild(b);
  });
  [D.M, D.C, D.PR, D.P] = await Promise.all([load("measures"), load("clips"), load("predictions"), load("prompts")]);
  D.EMO = D.M.meta.emotions;
  // models that ran every condition first, then partial designs, then the control-only models
  D.models = Object.keys(D.M.reference.models)
    .sort((x, y) => Object.keys((D.PR.models || {})[y] || {}).length - Object.keys((D.PR.models || {})[x] || {}).length);
  // models that ran every condition first; a model with a partial design (fewer conditions) comes last
  D.attrModels = Object.keys(D.M.effects).filter((m) => Object.keys(D.M.effects[m]).length > 0)
    .sort((x, y) => Object.keys(D.M.effects[y]).length - Object.keys(D.M.effects[x]).length);
  S.model = D.attrModels[0];
  renderOverview(); renderClips(); renderRecognition(); renderAttributes(); renderCombinations(); renderGeneration(); renderCheck(); renderPrompts(); renderCode();
  show(location.hash.slice(1) || "overview");
}

function show(id) {
  if (!TABS.some((t) => t[0] === id)) id = "overview";
  S.tab = id; location.hash = id;
  document.querySelectorAll("section").forEach((s) => s.classList.toggle("on", s.id === id));
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("on", b.dataset.tab === id));
}

// ---------------------------------------------------------------- helpers: charts
function barRows(rows, max, opts = {}) {
  // rows: [{label, values:[{v, cls, title}], text}]
  let h = '<div class="bars">';
  rows.forEach((r) => {
    h += `<div>${esc(r.label)}</div><div class="stack">`;
    r.values.forEach((x) => { h += `<div class="bar ${x.cls || ""}" style="width:${Math.max(0, 100 * x.v / max)}%;${x.color ? "background:" + x.color : ""}" title="${esc(x.title || "")}"></div>`; });
    h += `</div><div class="small">${esc(r.text)}</div>`;
  });
  return h + "</div>";
}
function emoDot(e) { return `<span class="emo" style="background:${COLORS[e]}"></span>`; }
function heatColor(v, scale, sign = true) {
  if (v === null || v === undefined) return "#f4f2ee";
  const t = Math.min(1, Math.abs(v) / scale);
  if (!sign) return `rgba(47,95,138,${0.08 + 0.85 * t})`;
  return v >= 0 ? `rgba(47,125,90,${0.08 + 0.85 * t})` : `rgba(178,58,58,${0.08 + 0.85 * t})`;
}
function heatText(v, scale) { return (v !== null && v !== undefined && Math.abs(v) / scale > 0.55) ? "color:#fff" : ""; }
function heatmap(rowsLabels, colsLabels, cells, scale, fmt = (v) => sgn(v, 0), sign = true, marks = null) {
  let h = `<div class="heat" style="grid-template-columns:auto repeat(${colsLabels.length},1fr)">`;
  h += `<div></div>` + colsLabels.map((c) => `<div class="hc">${esc(c)}</div>`).join("");
  rowsLabels.forEach((r, i) => {
    h += `<div class="hd">${esc(r)}</div>`;
    colsLabels.forEach((c, j) => {
      const v = cells[i][j];
      const m = marks && marks[i][j];
      h += `<div style="background:${heatColor(v, scale, sign)};${heatText(v, scale)}" class="${m ? "sig" : ""}" title="${esc(r)} × ${esc(c)}: ${v === null || v === undefined ? "no data" : fmt(v)}">${v === null || v === undefined ? "" : fmt(v)}${m ? "*" : ""}</div>`;
    });
  });
  return h + "</div>";
}
function selectHtml(id, options, value, labels = null) {
  return `<select id="${id}">` + options.map((o) => `<option value="${esc(o)}" ${o === value ? "selected" : ""}>${esc(labels ? labels(o) : o)}</option>`).join("") + "</select>";
}
function svgScatter(points, xlab, ylab, w = 420, h = 320, extra = null) {
  const pad = {l: 44, r: 12, t: 12, b: 38};
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  const mx = Math.max(1, ...xs, ...ys) * 1.05;
  const sx = (x) => pad.l + (w - pad.l - pad.r) * x / mx, sy = (y) => h - pad.b - (h - pad.t - pad.b) * y / mx;
  let s = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" style="max-width:100%">`;
  s += `<line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(mx)}" y2="${sy(mx)}" stroke="#bbb" stroke-dasharray="4 3"/>`;
  if (extra) s += `<line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(mx)}" y2="${sy(mx * extra)}" stroke="#8a2f1e" stroke-width="1.5"/>`;
  s += `<line x1="${pad.l}" y1="${sy(0)}" x2="${w - pad.r}" y2="${sy(0)}" stroke="#333"/><line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${sy(0)}" stroke="#333"/>`;
  for (let t = 0; t <= mx; t += Math.max(5, Math.round(mx / 6 / 5) * 5)) {
    s += `<text x="${sx(t)}" y="${h - pad.b + 14}" font-size="11" text-anchor="middle">${t}</text><text x="${pad.l - 6}" y="${sy(t) + 4}" font-size="11" text-anchor="end">${t}</text>`;
  }
  points.forEach((p) => { s += `<circle cx="${sx(p[0])}" cy="${sy(p[1])}" r="3.5" fill="#2f5f8a" fill-opacity=".7"><title>${esc(p[2] || "")}: ${f1(p[0])} → ${f1(p[1])}</title></circle>`; });
  s += `<text x="${(w + pad.l) / 2}" y="${h - 4}" font-size="12" text-anchor="middle">${esc(xlab)}</text>`;
  s += `<text x="12" y="${h / 2}" font-size="12" text-anchor="middle" transform="rotate(-90 12 ${h / 2})">${esc(ylab)}</text></svg>`;
  return s;
}

// ---------------------------------------------------------------- clips
const CUE_NAMES = {speech_rate: "speech rate", pitch: "pitch", loudness: "loudness", gaze: "gaze", head: "head",
  AU01: "AU01 inner brow raiser", AU02: "AU02 outer brow raiser", AU04: "AU04 brow lowerer", AU05: "AU05 upper lid raiser", AU06: "AU06 cheek raiser",
  AU07: "AU07 lid tightener", AU09: "AU09 nose wrinkler", AU10: "AU10 upper lip raiser", AU12: "AU12 lip corner puller (smile)", AU14: "AU14 dimpler",
  AU15: "AU15 lip corner depressor", AU17: "AU17 chin raiser", AU20: "AU20 lip stretcher", AU23: "AU23 lip tightener", AU25: "AU25 lips part", AU26: "AU26 jaw drop", AU45: "AU45 blink"};
const CUES = Object.keys(CUE_NAMES);
function decodeCues(code) { const o = {}; CUES.forEach((c, i) => { o[c] = D.C.cue_codes[c][+code[i]]; }); return o; }
const AU_TEXT = {AU01: "raised inner eyebrows", AU02: "raised outer eyebrows", AU04: "lowered brows", AU05: "raised upper eyelids", AU06: "raised cheeks",
  AU07: "tightened eyelids", AU09: "wrinkled nose", AU10: "raised upper lip", AU12: "a smile", AU14: "a tight smile", AU15: "downturned mouth corners",
  AU17: "raised chin", AU20: "stretched lips", AU23: "tightened lips", AU25: "parted lips", AU26: "a dropped jaw", AU45: "blinking"};
function joinNice(p) { if (p.length <= 1) return p.join(""); if (p.length === 2) return p[0] + " and " + p[1]; return p.slice(0, -1).join(", ") + ", and " + p[p.length - 1]; }
function renderTranscript(cues, speaker, text) {
  const mods = [`with *${cues.speech_rate}* **speech rate**`, `with *${cues.pitch}* **pitch**`, `with *${cues.loudness}* **loudness**`];
  const g = cues.gaze || "front";
  mods.push(g === "front" ? "while **looking at the camera**" : `with **gaze** *averted* (${{left: "to their left", right: "to their right", up: "upwards", down: "downwards"}[g] || g})`);
  const hd = {nodding: "while *nodding*", shaking: "while *shaking* their **head**", tilting: "while *tilting* their **head**"}[cues.head];
  if (hd) mods.push(hd);
  const aus = CUES.filter((c) => c.startsWith("AU") && ["weak", "moderate", "strong"].includes(cues[c])).map((c) => `*${cues[c]}* **${AU_TEXT[c]}**`);
  if (aus.length) mods.push("with " + joinNice(aus));
  const head = text !== null ? `${speaker} said “${text}”` : `${speaker} spoke`;
  const rest = mods.slice(1).map((m) => m.startsWith("with ") ? m.slice(5) : m);
  return `${head}, ${joinNice([mods[0], ...rest])}.`;
}
function withSpeaker(transcript, cond) {
  const phrase = cond === "speaker" || cond === "speaker_repeat" ? "The speaker" : D.P.conditions[cond] || "The speaker";
  return transcript.replace(/^Actor \d+/, phrase);
}
// condition picker: group 1 -> attribute 1, then optional group 2 -> attribute 2
const GROUPS = ["gender", "age", "descent", "personality"];
function groupAttrs(g) { return Object.keys(D.M.meta.attributes).filter((a) => D.M.meta.attributes[a].group === g); }
function condPickerHtml(prefix) {
  return `<label>Condition ${selectHtml(prefix + "-g1", ["control", "repeat", ...GROUPS], "gender", (g) => g === "control" ? "The speaker (control)" : g === "repeat" ? "The speaker, repeat" : g)}</label>
          <label>Attribute ${selectHtml(prefix + "-a1", groupAttrs("gender"), "woman", (a) => D.M.meta.attributes[a].phrase)}</label>
          <label>Second attribute (optional) ${selectHtml(prefix + "-g2", ["none", ...GROUPS], "none", (g) => g === "none" ? "none" : g)}</label>
          <label>&nbsp;${selectHtml(prefix + "-a2", [], "")}</label>`;
}
function condPickerBind(prefix, onchange) {
  const g1 = $(prefix + "-g1"), a1 = $(prefix + "-a1"), g2 = $(prefix + "-g2"), a2 = $(prefix + "-a2");
  const fill = (sel, g, keep) => { const opts = g === "none" || g === "control" || g === "repeat" ? [] : groupAttrs(g); sel.innerHTML = opts.map((a) => `<option value="${a}" ${a === keep ? "selected" : ""}>${esc(D.M.meta.attributes[a].phrase)}</option>`).join(""); sel.parentElement.style.display = opts.length ? "" : "none"; };
  fill(a1, g1.value, a1.value); fill(a2, g2.value, "");
  g1.onchange = () => { fill(a1, g1.value, ""); const two = !["control", "repeat"].includes(g1.value); g2.parentElement.style.display = two ? "" : "none"; if (!two) { g2.value = "none"; fill(a2, "none", ""); } onchange(); };
  g2.onchange = () => { fill(a2, g2.value, ""); onchange(); };
  a1.onchange = onchange; a2.onchange = onchange;
}
function condPickerValue(prefix) {
  const g1 = $(prefix + "-g1").value;
  if (g1 === "control") return "speaker";
  if (g1 === "repeat") return "speaker_repeat";
  const a1 = $(prefix + "-a1").value, g2 = $(prefix + "-g2").value, a2 = g2 === "none" ? "" : $(prefix + "-a2").value;
  if (!a2) return a1;
  const [g, o] = D.M.meta.attributes[a1].group === "gender" ? [a1, a2] : [a2, a1];
  return `${g}+${o}`;
}
function orderConds(cs) {
  const rank = (c) => c === "speaker" ? 0 : c === "speaker_repeat" ? 1 : c.includes("+") ? 3 : 2;
  return [...cs].sort((a, b) => rank(a) - rank(b) || cs.indexOf(a) - cs.indexOf(b));
}
function clipRow(i) { const r = D.C.rows[i]; const o = {}; D.C.columns.forEach((c, j) => { o[c] = r[j]; }); o.i = i; return o; }
function pred(model, cond, i) { const s = D.PR.models[model] && D.PR.models[model][cond]; if (!s) return null; const ch = s[i]; return ch === "-" ? null : D.EMO[+ch]; }
function predInt(model, cond, i) { const s = D.PR.intensity && D.PR.intensity[model] && D.PR.intensity[model][cond]; if (!s) return ""; return {l: " low", h: " high", n: " none"}[s[i]] || ""; }

function renderClips() {
  const actors = [...new Set(D.C.rows.map((r) => r[1]))].sort();
  const conds = orderConds(Object.keys(D.PR.models[S.model] || {}));
  $("clips-body").innerHTML = `
  <p class="note">The 2 100 clips of the corpus used in the paper: ten actors, the ten neutral sentences and 2 000 emotional clips rated by fifteen annotators. Click a clip to see its transcript, the votes of the annotators and the answer of every model in the control condition and in the condition chosen below.</p>
  <div class="card row">
    <label>Actor ${selectHtml("f-actor", ["all", ...actors], "all")}</label>
    <label>Intended emotion ${selectHtml("f-int", ["all", ...D.EMO], "all")}</label>
    <label>Majority label ${selectHtml("f-maj", ["all", ...D.EMO.slice(1)], "all")}</label>
    <label>Model for the condition column ${selectHtml("f-model", D.models, S.model, name)}</label>
    ${condPickerHtml("f")}
    <label>Search in transcript <input type="search" id="f-q" placeholder="e.g. smile"></label>
    <span class="small" id="f-n"></span>
  </div>
  <div class="clips-grid"><div class="tbl" style="max-height:75vh"><table id="clips-table"></table></div><div id="clip-detail" class="card">Select a clip.</div></div>`;
  const refresh = () => { S.model = $("f-model").value; clipsTable(); if (S.clip !== null) clipDetail(); };
  ["f-actor", "f-int", "f-maj", "f-model", "f-q"].forEach((id) => { $(id).oninput = refresh; });
  condPickerBind("f", refresh);
  clipsTable();
}
function clipsTable() {
  const a = $("f-actor").value, it = $("f-int").value, mj = $("f-maj").value, q = $("f-q").value.toLowerCase(), m = $("f-model").value, c = condPickerValue("f");
  S.cond = c;
  const has = !!(D.PR.models[m] && D.PR.models[m][c]);
  const rows = [];
  D.C.rows.forEach((r, i) => {
    const o = clipRow(i);
    if (a !== "all" && o.actor !== a) return; if (it !== "all" && o.intended_emotion !== it) return; if (mj !== "all" && o.majority_label !== mj) return;
    if (q && !o.transcript.toLowerCase().includes(q)) return;
    rows.push(o);
  });
  $("f-n").textContent = `${rows.length} clips` + (has ? "" : ` (${name(m)} did not run the condition ${condLabel(c)})`);
  let h = `<tr><th class="l">Clip</th><th>Actor</th><th class="l">Intended</th><th class="l">Majority</th><th class="l">${esc(name(m))}: control</th><th class="l">${esc(name(m))}: ${esc(c === "speaker" ? "control" : condLabel(c))}</th></tr>`;
  rows.slice(0, 600).forEach((o) => {
    const pc = pred(m, "speaker", o.i), pk = pred(m, c, o.i);
    h += `<tr class="click ${o.i === S.clip ? "sel" : ""}" data-i="${o.i}"><td class="l">${o.clip}</td><td>${o.actor} (${o.actor_sex})</td><td class="l">${emoDot(o.intended_emotion)}${o.intended_emotion}${o.intensity ? " " + (o.intensity == 1 ? "low" : "high") : ""}</td><td class="l">${o.majority_label ? emoDot(o.majority_label) + o.majority_label : "<span class=small>not rated</span>"}</td><td class="l">${pc ? emoDot(pc) + pc : "–"}</td><td class="l ${pk && pc && pk !== pc ? "sig" : ""}">${pk ? emoDot(pk) + pk : "–"}</td></tr>`;
  });
  if (rows.length > 600) h += `<tr><td colspan="6" class="small">Only the first 600 clips are listed; narrow the filters.</td></tr>`;
  $("clips-table").innerHTML = h;
  $("clips-table").querySelectorAll("tr.click").forEach((tr) => { tr.onclick = () => { S.clip = +tr.dataset.i; clipsTable(); clipDetail(); }; });
}
function clipDetail() {
  const o = clipRow(S.clip); const cues = decodeCues(o.cues); const c = condPickerValue("f");
  let h = `<h3 style="margin-top:0">${o.clip}</h3><div class="small">Actor ${o.actor} (${o.actor_sex === "F" ? "woman" : "man"}), sentence ${o.sentence}, intended ${o.intended_emotion}${o.intensity ? ", " + (o.intensity == 1 ? "low" : "high") + " intensity" : ""}</div>`;
  h += `<p><b>Enriched multimodal transcript</b> (as given to the models in the control condition)</p><div class="transcript">${md(withSpeaker(o.transcript, "speaker"))}</div>`;
  h += c === "speaker" ? "" : `<p><b>Condition ${esc(condLabel(c))}</b>: the transcript begins with <i>${esc(withSpeaker("Actor 00", c))} said</i>; nothing else changes.</p>`;
  if (o.majority_label) {
    const tot = o.votes.reduce((a, b) => a + b, 0);
    h += `<p><b>Emotion perceived by the ${tot} annotators</b> (majority: ${emoDot(o.majority_label)}${o.majority_label})</p>`;
    h += barRows(D.C.vote_emotions.map((e, k) => ({label: e, values: [{v: o.votes[k], color: COLORS[e]}], text: `${o.votes[k]}`})).filter((r) => r.values[0].v > 0), tot);
  } else h += `<p class="small">Neutral clip: not rated by the annotators.</p>`;
  h += `<p><b>Emotion recognised by each model</b> (with the intensity answered)</p><table><tr><th class="l">Model</th><th class="l">Control</th><th class="l">${esc(condLabel(c))}</th><th class="l">Repeat control</th></tr>`;
  D.models.forEach((m) => {
    const pc = pred(m, "speaker", o.i), pk = pred(m, c, o.i), pr = pred(m, "speaker_repeat", o.i);
    h += `<tr><td class="l">${esc(name(m))}</td><td class="l">${pc ? emoDot(pc) + pc + `<span class="small">${predInt(m, "speaker", o.i)}</span>` : "–"}</td><td class="l ${pk && pc && pk !== pc ? "sig" : ""}">${pk ? emoDot(pk) + pk + `<span class="small">${predInt(m, c, o.i)}</span>` : "–"}</td><td class="l">${pr ? emoDot(pr) + pr + `<span class="small">${predInt(m, "speaker_repeat", o.i)}</span>` : "–"}</td></tr>`;
  });
  h += `</table>`;
  const m = $("f-model").value; const conds = Object.keys(D.PR.models[m] || {}).filter((k) => k !== "speaker");
  const pc = pred(m, "speaker", o.i);
  const changed = conds.filter((k) => { const p = pred(m, k, o.i); return p && p !== pc; });
  h += `<p><b>${esc(name(m))}</b>: the answer differs from the control (${pc || "–"}) in ${changed.length} of ${conds.length} conditions</p><div>` +
       changed.map((k) => `<span class="chip">${esc(condLabel(k))} → ${pred(m, k, o.i)}</span>`).join("") + `</div>`;
  h += `<p><b>Cues</b></p><table>` + CUES.filter((k) => cues[k] !== "none" || !k.startsWith("AU")).map((k) => `<tr><td class="l">${CUE_NAMES[k]}</td><td class="l">${cues[k]}</td></tr>`).join("") + `</table>`;
  $("clip-detail").innerHTML = h;
}

// ---------------------------------------------------------------- overview
function renderOverview() {
  const R = D.M.reference, T = D.M.table3, G = D.M.generation;
  const flips = Object.values(T).flatMap((t) => [t.flip_min, t.flip_max]);
  let h = `<p>This site accompanies the paper. Everything shown here is computed from the released answers (<code>results/</code>) by <code>python -m emobias analyse</code>. The method compares, on the same 2 000 rated clips, what an LLM answers when the speaker is not described (<i>The speaker said …</i>) and when one attribute is stated (<i>The woman said …</i>). Only the stated attribute changes.</p>`;
  h += `<div class="kpi"><div><b>${R.annotators.n_clips}</b><span>rated clips, 15 annotators each</span></div><div><b>${D.models.length}</b><span>models in the control condition</span></div>` +
       `<div><b>${D.M.meta.single.length}</b><span>attributes stated alone</span></div><div><b>${D.M.meta.combined_genders.length * D.M.meta.others.length}</b><span>combinations of two attributes</span></div>` +
       `<div><b>${f1(Math.min(...flips), 0)} to ${f1(Math.max(...flips), 0)} %</b><span>clips whose emotion changes when one attribute is stated</span></div>` +
       `<div><b>${Object.keys(G.models).length}</b><span>models writing transcripts</span></div></div>`;
  h += `<h3>What the tabs show</h3><p><b>Clips</b>: every transcript, the votes of the annotators and the answer of each model per condition. <b>Recognised emotions</b>: the share of each emotion in the answers of each model against the annotators, accuracy and macro-F1 (reference point). <b>Stated attribute</b>: for each model and attribute, the clips whose emotion changes, the shift of each emotion, the total variation distance, the tests, and where the moved clips come from. <b>Combinations</b>: a gender attribute stated with an attribute of another group. <b>Generation</b>: the transcripts written by the models, with and without a stated attribute. <b>Corpus check</b>: the words alone, the described behaviour alone, and both, on three corpora. <b>Prompts</b> and <b>Code</b>: the exact text sent to the models and the source files.</p>`;
  h += `<h3>Reading the measures</h3><p><b>Flip</b>: share of clips whose recognised emotion differs between the attribute condition and the control, in percent of clips. <b>Shift</b>: change in the share of an emotion, in percentage points. <b>TVD</b>: total variation distance, half the sum of the absolute shifts: the share of answers that would have to move for the two conditions to coincide. Confidence intervals are obtained by bootstrap over the ten actors. The p-value comes from a paired permutation test that swaps the two answers of a clip. A star marks a shift that passes the Benjamini-Hochberg correction at 5 % over the attribute × class shifts of a model and exceeds the shift observed between two runs of the same control prompt (repeat control).</p>`;
  $("overview-body").innerHTML = h;
}

// ---------------------------------------------------------------- recognised emotions (Table 2)
function renderRecognition() {
  const R = D.M.reference;
  let h = `<p class="note">Answers of the models in the control condition on the ${R.annotators.n_clips} rated clips. Share of each emotion in the answers, in percent, next to the share of each emotion in the majority labels of the annotators (who had no neutral option). Accuracy and macro-F1 against the majority label, with 95 % intervals over actors. A random answer is correct on ${f1(R.annotators.random_accuracy)} %; always answering the most frequent class (${R.annotators.most_frequent_class}) on ${f1(R.annotators.most_frequent_class_accuracy)} %.</p>`;
  h += `<div class="tbl"><table><tr><th class="l">Answers given by</th>` + D.EMO.map((e) => `<th>${emoDot(e)}${e}</th>`).join("") + `<th>Accuracy (95 % CI)</th><th>Macro-F1</th><th>Intensity: high</th><th>Intensity agrees with the recording</th><th>Unparsed</th></tr>`;
  h += `<tr><td class="l">The annotators</td>` + D.EMO.map((e) => `<td>${e === "neutral" ? "–" : f1(R.annotators.shares[e])}</td>`).join("") + `<td>–</td><td>–</td><td></td><td></td><td></td></tr>`;
  const mean = {}; D.EMO.forEach((e) => { mean[e] = D.models.reduce((a, m) => a + R.models[m].shares[e], 0) / D.models.length; });
  D.models.forEach((m) => { const r = R.models[m]; h += `<tr><td class="l">${esc(name(m))}</td>` + D.EMO.map((e) => `<td>${f1(r.shares[e])}</td>`).join("") + `<td>${f1(r.accuracy)} [${f1(r.accuracy_ci[0])}, ${f1(r.accuracy_ci[1])}]</td><td>${f1(r.macro_f1)}</td><td>${f1(r.intensity.shares.high)} %</td><td>${f1(r.intensity.accuracy_vs_recorded)} %</td><td>${f1(r.unparsed)} %</td></tr>`; });
  h += `<tr><td class="l"><i>Mean of the models</i></td>` + D.EMO.map((e) => `<td><i>${f1(mean[e])}</i></td>`).join("") + `<td><i>${f1(D.models.reduce((a, m) => a + R.models[m].accuracy, 0) / D.models.length)}</i></td><td><i>${f1(D.models.reduce((a, m) => a + R.models[m].macro_f1, 0) / D.models.length)}</i></td><td><i>${f1(D.models.reduce((a, m) => a + R.models[m].intensity.shares.high, 0) / D.models.length)} %</i></td><td><i>${f1(D.models.reduce((a, m) => a + R.models[m].intensity.accuracy_vs_recorded, 0) / D.models.length)} %</i></td><td></td></tr></table></div>`;
  h += `<p class="small">Fleiss' kappa between the models on the ${R.models_fleiss_n_clips} clips they all answered: ${f1(R.models_fleiss_kappa, 2)}; between the fifteen annotators: ${f1(R.annotators.fleiss_kappa, 2)}. A single annotator's vote agrees with the majority label on ${f1(R.annotators.single_annotator_vs_majority)} % of the votes. Intensity: the models also answer <i>low</i> or <i>high</i>; the last two columns give the share of <i>high</i> and how often the answer matches the intensity the actor was asked to play (chance 50 %).</p>`;
  h += `<h3>Share of each emotion: annotators, each model, mean of the models</h3><div class="card"><div class="row"><label>Show ${selectHtml("rec-sel", ["mean", ...D.models], "mean", (m) => m === "mean" ? "Mean of the models" : name(m))}</label></div><div id="rec-bars"></div></div>`;
  $("recognition-body").innerHTML = h;
  const draw = () => {
    const sel = $("rec-sel").value;
    const sh = sel === "mean" ? mean : R.models[sel].shares;
    const max = Math.max(...D.EMO.map((e) => Math.max(sh[e], R.annotators.shares[e] || 0)));
    $("rec-bars").innerHTML = `<div class="legend"><span><span class="emo" style="background:#b9c7d6"></span>annotators</span><span><span class="emo" style="background:#2f5f8a"></span>${sel === "mean" ? "mean of the models" : esc(name(sel))}</span></div>` +
      barRows(D.EMO.map((e) => ({label: e, values: [{v: R.annotators.shares[e] || 0, cls: "ctrl", title: "annotators"}, {v: sh[e], title: "model"}], text: `${f1(R.annotators.shares[e] || 0)} / ${f1(sh[e])}`})), max);
  };
  $("rec-sel").onchange = draw; draw();
}

// ---------------------------------------------------------------- stated attribute
function effectsFor(model) { return D.M.effects[model] || {}; }
function meanEffect(cond, models) {
  // mean over models of flip, tvd and shifts (models that have the condition)
  const ms = models.filter((m) => effectsFor(m)[cond]);
  if (!ms.length) return null;
  const avg = (f) => ms.reduce((a, m) => a + f(effectsFor(m)[cond]), 0) / ms.length;
  const shift = {}; D.EMO.forEach((e) => { shift[e] = avg((r) => r.shift[e]); });
  return {n: ms.length, flip: avg((r) => r.flip), tvd: avg((r) => r.tvd), shift, models: ms};
}
function renderAttributes() {
  const single = D.M.meta.single, all = [...single, "trans_woman", "trans_man"];
  let h = `<p class="note">For each model, the answers with one stated attribute are compared with the control on the same clips. Choose a model (or the mean of the models) and an attribute. Bold values with a star pass the Benjamini-Hochberg correction and exceed the repeat control.</p>`;
  h += `<div class="card"><div class="row"><label>Model ${selectHtml("at-model", ["mean", ...D.attrModels], D.attrModels[0], (m) => m === "mean" ? "Mean of the models" : name(m))}</label>${condPickerHtml("at")}</div><div id="at-detail"></div></div>`;
  h += `<h3>Overview: flip and total variation distance of every attribute</h3><div id="at-table"></div>`;
  h += `<h3>Shift of each emotion, in points, for every attribute</h3><div class="card"><div class="row"><label>Model ${selectHtml("at-heat-model", ["mean", ...D.attrModels], "mean", (m) => m === "mean" ? "Mean of the models" : name(m))}</label></div><div id="at-heat"></div></div>`;
  $("attributes-body").innerHTML = h;
  $("at-model").onchange = attrDetail; condPickerBind("at", attrDetail); $("at-heat-model").onchange = attrHeat;
  attrDetail(); attrTable(); attrHeat();
}
function attrDetail() {
  const m = $("at-model").value, c = condPickerValue("at"); const R = D.M.reference;
  if (c === "speaker") { $("at-detail").innerHTML = "<p>Choose an attribute (or the repeat control) to compare with the control condition.</p>"; return; }
  let h = "";
  if (m === "mean") {
    const full = D.attrModels.filter((x) => D.M.meta.single.every((k) => effectsFor(x)[k]));
    const e = meanEffect(c, full);
    if (!e) { $("at-detail").innerHTML = "<p>No model has this condition.</p>"; return; }
    h += `<div class="kpi"><div><b>${f1(e.flip)} %</b><span>clips whose emotion changes (mean of ${e.n} models: ${e.models.map(name).join(", ")})</span></div><div><b>${f1(e.tvd)} pts</b><span>total variation distance</span></div></div>`;
    const ctrl = {}; D.EMO.forEach((k) => { ctrl[k] = e.models.reduce((a, mm) => a + R.models[mm].shares[k], 0) / e.n; });
    h += shiftBars(ctrl, e.shift, null, null);
  } else {
    const r = effectsFor(m)[c];
    if (!r) { $("at-detail").innerHTML = "<p>This model did not run this condition.</p>"; return; }
    const rep = effectsFor(m).speaker_repeat;
    h += `<div class="kpi"><div><b>${f1(r.flip)} %</b><span>clips whose emotion changes [${f1(r.flip_ci[0])}, ${f1(r.flip_ci[1])}]${rep && c !== "speaker_repeat" ? `; repeat control ${f1(rep.flip)} %` : ""}</span></div>` +
         `<div><b>${f1(r.tvd)} pts</b><span>total variation distance [${f1(r.tvd_ci[0])}, ${f1(r.tvd_ci[1])}]${rep && c !== "speaker_repeat" ? `; repeat control ${f1(rep.tvd)}` : ""}</span></div>` +
         `<div><b>p = ${r.p_tvd < 0.001 ? "< 0.001" : f1(r.p_tvd, 3)}</b><span>paired permutation test on the TVD${r.bh_tvd !== undefined ? (r.bh_tvd ? ", passes BH" : ", fails BH") : ""}</span></div><div><b>${r.n}</b><span>clips compared</span></div>` +
         (r.intensity ? `<div><b>${f1(r.intensity.high_control)} → ${f1(r.intensity.high_condition)} %</b><span>answers with high intensity, control → condition; intensity changes on ${f1(r.intensity.flip)} % of clips</span></div>` : "") + `</div>`;
    const ctrl = R.models[m].shares;
    h += shiftBars(ctrl, r.shift, r, rep);
    h += `<h3>Transfers: where the answers go</h3><p class="note">Rows: emotion recognised in the control condition; columns: emotion recognised with the attribute. Off-diagonal cells are the clips that change. Colour: share of the row.</p>`;
    const tab = r.transfers; const rowsum = tab.map((row) => row.reduce((a, b) => a + b, 0));
    const cells = tab.map((row, i) => row.map((v, j) => i === j ? null : (rowsum[i] ? 100 * v / rowsum[i] : null)));
    h += heatmap(D.EMO.map((e) => `${e} (${rowsum[D.EMO.indexOf(e)]})`), D.EMO, cells, 30, (v) => f1(v, 0) + "%", false);
  }
  $("at-detail").innerHTML = h;
}
function shiftBars(ctrl, shift, r, rep) {
  const max = Math.max(...D.EMO.map((e) => Math.max(ctrl[e] || 0, (ctrl[e] || 0) + shift[e])));
  let h = `<p><b>Share of each emotion</b>: control (grey) against the condition (blue), in percent of clips, and the shift in points</p>`;
  h += barRows(D.EMO.map((e) => {
    const mark = r && r.bh && r.bh[e] && (!r.above_repeat || r.above_repeat[e]);
    const ci = r && r.shift_ci ? ` [${sgn(r.shift_ci[e][0])}, ${sgn(r.shift_ci[e][1])}]` : "";
    return {label: e, values: [{v: ctrl[e] || 0, cls: "ctrl", title: "control"}, {v: Math.max(0, (ctrl[e] || 0) + shift[e]), title: "condition"}],
            text: `${f1(ctrl[e])} → ${f1((ctrl[e] || 0) + shift[e])} (${sgn(shift[e])}${ci})${mark ? " *" : ""}`};
  }), max || 1);
  return h.replace('<div class="bars">', '<div class="bars" style="grid-template-columns:110px 1fr 230px">');
}
function attrTable() {
  const all = [...D.M.meta.single, "trans_woman", "trans_man"];
  const full = D.attrModels.filter((x) => D.M.meta.single.every((c) => effectsFor(x)[c]));
  let h = `<div class="tbl"><table><tr><th class="l">Attribute</th><th class="l">Group</th><th colspan="2">Mean of the models</th>` + D.attrModels.map((m) => `<th colspan="2">${esc(name(m))}</th>`).join("") + `</tr><tr><th></th><th></th><th>flip %</th><th>TVD</th>` + D.attrModels.map(() => `<th>flip %</th><th>TVD</th>`).join("") + `</tr>`;
  ["speaker_repeat", ...all].forEach((c) => {
    h += `<tr><td class="l">${esc(condLabel(c))}</td><td class="l">${c === "speaker_repeat" ? "control" : D.M.meta.attributes[c].group}</td>`;
    const e = meanEffect(c, full);
    h += e ? `<td><i>${f1(e.flip)}</i></td><td><i>${f1(e.tvd)}</i></td>` : `<td>–</td><td>–</td>`;
    D.attrModels.forEach((m) => { const r = effectsFor(m)[c]; h += r ? `<td>${f1(r.flip)}</td><td class="${r.bh_tvd && r.tvd_above_repeat ? "sig" : ""}">${f1(r.tvd)}${r.bh_tvd && r.tvd_above_repeat ? "*" : ""}</td>` : `<td>–</td><td>–</td>`; });
    h += `</tr>`;
  });
  h += `</table></div><p class="small">TVD in points; * : passes the Benjamini-Hochberg correction over the attributes of the model and exceeds the repeat control. Mean over the models that ran every attribute (${full.map(name).join(", ")}).</p>`;
  $("at-table").innerHTML = h;
}
function attrHeat() {
  const m = $("at-heat-model").value;
  const all = ["speaker_repeat", ...D.M.meta.single, "trans_woman", "trans_man"];
  let conds, cells, marks, note;
  if (m === "mean") {
    // mean over the models that ran the full set of attributes (a partial design is left out)
    const full = D.attrModels.filter((x) => D.M.meta.single.every((c) => effectsFor(x)[c]));
    conds = all.filter((c) => full.some((x) => effectsFor(x)[c]));
    cells = conds.map((c) => D.EMO.map((e) => { const v = full.filter((x) => effectsFor(x)[c]).map((x) => effectsFor(x)[c].shift[e]); return v.reduce((s, y) => s + y, 0) / v.length; }));
    marks = null;
    note = `Mean over ${full.map(name).join(", ")}.`;
  } else {
    const E = effectsFor(m);
    conds = all.filter((c) => E[c]);
    cells = conds.map((c) => D.EMO.map((e) => E[c].shift[e]));
    marks = conds.map((c) => D.EMO.map((e) => E[c].bh && E[c].bh[e] && (!E[c].above_repeat || E[c].above_repeat[e])));
    note = "* : significant after correction and above the repeat control.";
  }
  const scale = Math.max(5, ...cells.flat().map(Math.abs));
  $("at-heat").innerHTML = heatmap(conds.map(condLabel), D.EMO, cells, scale, (v) => sgn(v, 1), true, marks) + `<p class="small">Green: the emotion is answered more often with the attribute; red: less often. ${note}</p>`;
}

// ---------------------------------------------------------------- combinations (Figure 1)
function renderCombinations() {
  const F = D.M.figure1; const A = D.M.additivity;
  if (!F.mean) { $("combinations-body").innerHTML = "<p>No model ran the combinations.</p>"; return; }
  let h = `<p class="note">A gender attribute (rows) stated together with a second attribute (columns). Each cell gives the change in the share of the emotion relative to the control condition, in percent of that share (a relative change, not a shift in points). <i>Alone</i> is the attribute stated by itself. Choose an emotion and a model, or the mean over the models that ran the combinations (${F.mean_of.map(name).join(", ")}).</p>`;
  h += `<div class="card"><div class="row"><label>Emotion ${selectHtml("cb-emo", F.emotions, F.emotions[0])}</label><label>Model ${selectHtml("cb-model", ["mean", ...Object.keys(F.models)], "mean", (m) => m === "mean" ? "Mean of the models" : name(m))}</label><label>Value ${selectHtml("cb-val", ["relative", "points"], "relative", (v) => v === "relative" ? "relative change (%)" : "shift in points")}</label></div><div id="cb-heat"></div></div>`;
  h += `<h3>Does a combination add the effects of its two attributes?</h3><p class="note">Each point is a combination: the sum of the distances of its two attributes stated alone (x) against the distance of the combination (y), in points. The dashed line is equality; the red line is the fitted slope through the origin.</p><div class="row">`;
  Object.entries(A).forEach(([m, a]) => { h += `<div class="card"><b>${esc(name(m))}</b>: slope ${f1(a.slope_through_origin, 2)}, mean ratio ${f1(a.mean_ratio, 2)} (${a.n} combinations)<br>${svgScatter(a.points, "TVD(gender) + TVD(attribute)", "TVD(combination)", 380, 300, a.slope_through_origin)}</div>`; });
  h += `</div><h3>Every combination: flip and distance</h3><div id="cb-table"></div>`;
  $("combinations-body").innerHTML = h;
  const draw = () => {
    const e = $("cb-emo").value, m = $("cb-model").value, val = $("cb-val").value;
    const M = m === "mean" ? F.mean[e] : F.models[m][e];
    const rows = [...F.rows, "mean", "alone"], cols = [...F.cols, "mean", "alone"];
    const rel = (v) => v; const conv = val === "relative" ? rel : (v) => v === null ? null : v * M.control_share / 100;
    const cells = rows.map((g, i) => cols.map((a, j) => {
      let v;
      if (g === "alone" && a === "alone") v = null;
      else if (g === "alone") v = j < F.cols.length ? M.attribute_alone[j] : null;
      else if (a === "alone") v = i < F.rows.length ? M.gender_alone[i] : null;
      else if (g === "mean" && a === "mean") { const all = M.cells.flat().filter((x) => x !== null); v = all.length ? all.reduce((s, x) => s + x, 0) / all.length : null; }
      else if (g === "mean") { const col = M.cells.map((r) => r[j]).filter((x) => x !== null); v = col.length ? col.reduce((s, x) => s + x, 0) / col.length : null; }
      else if (a === "mean") { const row = M.cells[i].filter((x) => x !== null); v = row.length ? row.reduce((s, x) => s + x, 0) / row.length : null; }
      else v = M.cells[i][j];
      return conv(v);
    }));
    const scale = Math.max(1, ...cells.flat().filter((x) => x !== null).map(Math.abs));
    $("cb-heat").innerHTML = `<p><b>${e}</b>: ${f1(M.control_share)} % of the answers in the control condition</p>` + heatmap(rows.map(cap), cols.map(cap), cells, scale, (v) => val === "relative" ? sgn(v, 0) + "%" : sgn(v, 1));
  };
  ["cb-emo", "cb-model", "cb-val"].forEach((id) => { $(id).onchange = draw; }); draw();
  // table of all combinations
  const models = Object.keys(F.models);
  let t = `<div class="tbl" style="max-height:60vh"><table><tr><th class="l">Combination</th>` + models.map((m) => `<th colspan="2">${esc(name(m))}</th>`).join("") + `</tr><tr><th></th>` + models.map(() => `<th>flip %</th><th>TVD</th>`).join("") + `</tr>`;
  F.rows.forEach((g) => F.cols.forEach((a) => { const c = `${g}+${a}`; t += `<tr><td class="l">${esc(condLabel(c))}</td>` + models.map((m) => { const r = effectsFor(m)[c]; return r ? `<td>${f1(r.flip)}</td><td class="${r.bh_tvd ? "sig" : ""}">${f1(r.tvd)}</td>` : "<td>–</td><td>–</td>"; }).join("") + `</tr>`; }));
  $("cb-table").innerHTML = t + `</table></div>`;
}

// ---------------------------------------------------------------- generation
function renderGeneration() {
  const G = D.M.generation; const models = Object.keys(G.models);
  let h = `<p class="note">The models wrote transcripts from the actors' brief: ten synthetic actors (sampling seeds), the same sentences, emotions and intensities as one real actor, in the control condition (nothing said about the actor) and with one stated attribute (<i>You are a woman.</i>). The answer of the model is the set of 22 cues; it is rendered here as an enriched multimodal transcript, with the same renderer as the real recordings. Choose a model, a condition, an emotion, a sentence and an intensity to read what each synthetic actor wrote.</p>`;
  h += `<div class="card"><div class="row"><label>Model ${selectHtml("g-t-model", models, models[0], name)}</label><label>Condition ${selectHtml("g-t-cond", [], "")}</label><label>Emotion ${selectHtml("g-t-emo", D.EMO, "anger")}</label><label>Sentence ${selectHtml("g-t-sent", [1,2,3,4,5,6,7,8,9,10].map(String), "1")}</label><label>Intensity ${selectHtml("g-t-int", ["1", "2"], "2", (v) => v === "1" ? "low" : "high")}</label></div><div id="g-takes">Loading…</div></div>`;
  $("generation-body").innerHTML = h;
  const fillConds = (m) => { const cs = ["control", ...Object.keys(G.models[m].conditions)]; $("g-t-cond").innerHTML = cs.map((c) => `<option value="${c}">${esc(c === "control" ? "Control (nothing said about the actor)" : condLabel(c))}</option>`).join(""); };
  fillConds(models[0]);
  $("g-t-model").onchange = () => { fillConds($("g-t-model").value); genTakes(); };
  ["g-t-cond", "g-t-emo", "g-t-sent", "g-t-int"].forEach((id) => { $(id).onchange = genTakes; });
  genTakes();
}
async function genTakes() {
  const T = await load("generation_takes");
  const m = $("g-t-model").value, c = $("g-t-cond").value, e = $("g-t-emo").value, s = +$("g-t-sent").value, it = e === "neutral" ? 0 : +$("g-t-int").value;
  const takes = (T.models[m] && T.models[m][c] || []).filter((t) => t[2] === e && t[3] === s && t[4] === it);
  const sentence = SENTENCES[s - 1];
  let h = `<p><b>${esc(name(m))}</b>, ${esc(c === "control" ? "control" : condLabel(c))}: ${takes.length} synthetic actors, sentence ${s}, ${e}${e === "neutral" ? "" : ", " + (it === 1 ? "low" : "high") + " intensity"}</p>`;
  takes.forEach((t) => { const cues = {}; CUES.forEach((k, i) => { cues[k] = D.C.cue_codes[k][+t[5][i]]; }); h += `<div class="transcript" style="margin:4px 0">${md(renderTranscript(cues, "Actor " + t[1], sentence))}</div>`; });
  if (!takes.length) h += `<p class="small">No take for this combination (this condition was generated on a subset of sentences or actors).</p>`;
  $("g-takes").innerHTML = h;
}
const SENTENCES = ["The birch canoe slid on the smooth planks.", "Glue the sheet to the dark blue background.", "It's easy to tell the depth of a well.",
  "These days a chicken leg is a rare dish.", "Rice is often served in round bowls.", "The juice of lemons makes fine punch.", "The box was thrown beside the parked truck.",
  "The hogs were fed chopped corn and garbage.", "Four hours of steady work faced us.", "A large size in stockings is hard to sell."];

// ---------------------------------------------------------------- corpus check
function renderCheck() {
  const K = D.M.check;
  let h = `<p class="note">The property required of the corpus: the sentences carry no emotion by themselves. The three models that do the attribute conditions recognised the emotion from three inputs: the words alone, the described behaviour alone, and both, with the same instruction. Accuracy against the reference of the corpus, in percent of clips, with 95 % intervals over actors.</p>`;
  h += `<div class="tbl"><table><tr><th class="l">Corpus</th><th>Classes</th><th>Chance</th><th class="l">Reference</th><th class="l">Model</th><th>Words alone</th><th>Described behaviour alone</th><th>Both</th><th>Clips</th></tr>`;
  Object.entries(K).forEach(([corpus, c]) => { Object.entries(c.models).forEach(([m, r], i) => {
    h += `<tr>${i === 0 ? `<td class="l" rowspan="${Object.keys(c.models).length}">${corpus.toUpperCase()}</td><td rowspan="${Object.keys(c.models).length}">${c.n_classes}</td><td rowspan="${Object.keys(c.models).length}">${f1(c.chance)} %</td><td class="l" rowspan="${Object.keys(c.models).length}">${cap(c.reference)}</td>` : ""}<td class="l">${esc(name(m))}</td>` +
         ["words_only", "behaviour_only", "both"].map((k) => r[k] ? `<td>${f1(r[k].accuracy)} [${f1(r[k].ci[0])}, ${f1(r[k].ci[1])}]</td>` : "<td>–</td>").join("") + `<td>${r.both ? r.both.n : (r.words_only || {}).n || ""}</td></tr>`; }); });
  h += `</table></div>`;
  h += `<h3>Browse the clips of the check corpora</h3><div class="card"><div class="row"><label>Corpus ${selectHtml("ck-corpus", Object.keys(K), "eve", (c) => c.toUpperCase())}</label><label>Model ${selectHtml("ck-model", Object.keys(K.eve.models), Object.keys(K.eve.models)[0], name)}</label><label>Search <input type="search" id="ck-q"></label><span class="small" id="ck-n"></span></div><div class="tbl" style="max-height:55vh"><table id="ck-table"></table></div></div>`;
  $("check-body").innerHTML = h;
  ["ck-corpus", "ck-model", "ck-q"].forEach((id) => { $(id).oninput = checkTable; }); checkTable();
}
async function checkTable() {
  const corpus = $("ck-corpus").value, m = $("ck-model").value, q = $("ck-q").value.toLowerCase();
  const X = await load(`check_${corpus}`);
  const P = X.models[m] || {};
  let h = `<tr><th class="l">Clip</th><th class="l">Reference</th><th class="l">Words alone</th><th class="l">Behaviour alone</th><th class="l">Both</th><th class="l">Transcript</th></tr>`;
  let n = 0;
  X.rows.forEach((r, i) => {
    const ref = X.reference === "majority_label" ? r[3] : r[2];
    if (q && !r[4].toLowerCase().includes(q)) return;
    if (!P.both || P.both[i] === "-") return;
    n++; if (n > 400) return;
    const g = (k) => { const s = P[k]; if (!s || s[i] === "-") return "–"; const e = X.emotions[+s[i]]; return `<span class="${e === ref ? "sig" : ""}">${e}</span>`; };
    h += `<tr><td class="l">${r[0]}</td><td class="l">${ref}</td><td class="l">${g("words_only")}</td><td class="l">${g("behaviour_only")}</td><td class="l">${g("both")}</td><td class="l" style="white-space:normal;min-width:420px">${md(r[4])}</td></tr>`;
  });
  $("ck-n").textContent = `${n} clips answered by ${name(m)}${n > 400 ? " (first 400 shown)" : ""}`;
  $("ck-table").innerHTML = h;
}

// ---------------------------------------------------------------- prompts
function renderPrompts() {
  const P = D.P;
  let h = `<p class="note">The exact text sent to the models. Each call is one user message, no system message. Recognition: temperature 0, seed 0, 60 output tokens. Generation: temperature 1, one seed per synthetic actor, 300 output tokens.</p>`;
  h += `<h3>Recognition task</h3><div class="grid2"><div><p>Template (<code>prompts/recognition.txt</code>); <code>{emotions}</code> is the closed list of the eleven classes in a fixed order, <code>{transcript}</code> the enriched multimodal transcript, <code>{answer_block}</code> the JSON answer asked.</p><pre class="light">${esc(P.templates.recognition)}</pre><pre class="light">${esc(P.templates.answer_block)}</pre></div>`;
  h += `<div><p>Build the prompt for a clip and a condition:</p><div class="row"><label>Clip <input type="text" id="pp-clip" value="${esc(P.example.clip)}" list="pp-clips"><datalist id="pp-clips">${D.C.rows.slice(0, 2100).map((r) => `<option value="${r[0]}">`).join("")}</datalist></label><label>Condition ${selectHtml("pp-cond", Object.keys(P.conditions), "woman", condLabel)}</label></div><pre class="light" id="pp-out"></pre></div></div>`;
  h += `<h3>The stated attributes (Table 1)</h3><div class="tbl"><table><tr><th class="l">Group</th><th class="l">Attribute id</th><th class="l">Recognition: the transcript begins with</th><th class="l">Generation: sentence before the brief</th></tr>`;
  Object.entries(P.attributes).forEach(([a, s]) => { h += `<tr><td class="l">${s.group}</td><td class="l">${a}</td><td class="l">${esc(s.phrase)} said …</td><td class="l">${esc(s.sentence)}</td></tr>`; });
  h += `</table></div><p class="small">Control: <i>The speaker said …</i>. Combinations: <i>The old woman said</i>, <i>The neurotic transgender person said</i>, <i>The man of East Asian descent said</i> (see the condition list in the prompt builder above).</p>`;
  h += `<h3>Corpus check</h3><p>Same template and same answer block; only the input changes. Words alone: <i>The speaker said “…”.</i>; described behaviour alone: <i>The speaker spoke, with …</i>.</p>`;
  h += `<h3>Generation task</h3><p>Template <code>prompts/generation.txt</code>. <code>{attribute_sentence}</code> is empty in the control condition. The brief is the sheet given to the real actors, reproduced verbatim. Example with <i>You are a woman.</i>:</p><pre class="light" style="max-height:60vh">${esc(P.example.generation_woman)}</pre>`;
  $("prompts-body").innerHTML = h;
  const build = () => {
    const id = $("pp-clip").value.trim(), c = $("pp-cond").value;
    const i = D.C.rows.findIndex((r) => r[0] === id);
    if (i < 0) { $("pp-out").textContent = "Unknown clip id."; return; }
    const o = clipRow(i);
    const t = withSpeaker(o.transcript, c);
    $("pp-out").textContent = P.templates.recognition.replace("{emotions}", D.EMO.join(", ")).replace("{transcript}", t).replace("{answer_block}", P.templates.answer_block);
  };
  $("pp-clip").oninput = build; $("pp-cond").onchange = build; build();
}

// ---------------------------------------------------------------- code
async function renderCode() {
  const C = await load("code");
  const files = Object.keys(C);
  let h = `<p class="note">The source files of the repository. The pipeline runs with <code>python -m emobias &lt;plan|recognise|generate|analyse&gt; configs/&lt;config&gt;.yaml</code>.</p><div>` + files.map((f) => `<span class="file" data-f="${esc(f)}">${esc(f)}</span>`).join("") + `</div><pre id="code-view" style="max-height:75vh"></pre>`;
  $("code-body").innerHTML = h;
  document.querySelectorAll(".file").forEach((el) => { el.onclick = () => { $("code-view").textContent = C[el.dataset.f]; }; });
  $("code-view").textContent = C["README.md"] || C[files[0]];
}

main().catch((e) => { $("overview-body").innerHTML = `<div class="warn">Could not load the data files (${esc(e.message)}). Serve the docs/ folder with a static server, for example <code>python -m http.server -d docs</code>.</div>`; console.error(e); });
