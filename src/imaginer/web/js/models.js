// The model library and one model card.
//
// #/models: a card per image model imported for sd-server, with its family (an sd-server
// recipe: SDXL, FLUX.1, Qwen Image 2.1...), whether every file it needs is here, and which
// engines run it. Below them, every component file on disk with the cards that use it, so a
// FLUX VAE or a T5 encoder is reused rather than downloaded again.
//
// #/models/<id> (design "Model Cards" 2a): readiness as the headline with one next action,
// then a section rail: its files (a missing one opens as a fix-it row), sampling as values
// beside their family defaults, presets as a matrix of what each changes, and prompting,
// launch arguments, notes and cover folded below. Every edit saves as you leave its field.
//
// Importing (2c, 1g, 2b) takes three steps: where the model is (dropped, a path, or the Hub),
// which family it is (ranked guesses), and what else it needs, drawn as its pipeline with
// each file matched to one already here, to download, or to leave for later.

import { api, setImage, setImageUrl } from './api.js';
import { formatArgv, parseArgv } from './argv.js';
import { connectionStrip, dialog, menu, modal, statusLine, toast } from './chrome.js';
import { append, corners, h, icon, iconButton, clear } from './dom.js';
import { ago, plural } from './format.js';
import { hubDialog, looksLikeRef, modelTarget } from './hub.js';
import { lastPlace, loadBackends, loadModels } from './store.js';

const VAE_FORMATS = ['', 'auto', 'flux', 'sd3', 'flux2', 'wan'];
const SAMPLERS = ['euler', 'euler_a', 'heun', 'dpm2', 'dpm++2s_a', 'dpm++2m', 'dpm++2mv2', 'ipndm', 'ipndm_v', 'lcm',
  'ddim_trailing', 'tcd', 'res_multistep', 'res_2s', 'er_sde'];
const SCHEDULERS = ['discrete', 'karras', 'exponential', 'ays', 'gits', 'sgm_uniform', 'simple', 'smoothstep',
  'kl_optimal', 'lcm', 'bong_tangent', 'flux', 'beta'];
const MAIN_SLOTS = ['model', 'diffusion_model', 'uncond_diffusion_model'];
const ASPECTS = [[1, 1], [4, 3], [3, 4], [16, 9], [9, 16]];
const CAPS = { t2i: 'text to image', edit: 'editing', multi_ref: 'several references', rgba: 'transparency', lora: 'LoRA' };
const WORDS = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];
const MODEL_FILE = /\.(gguf|safetensors|sft|ckpt|pt|pth|bin)$/i;

/** A Hub source as a button says it: its file (or folder) when it names one, else its repo. */
function sourceLabel(ref) {
  const [path, quant] = ref.replace(/^hf:\/\//, '').split(':');
  const parts = path.split('/');
  return parts.length > 2 ? `${parts[0]}/…/${parts[parts.length - 1]}` : quant ? `${parts[1]}:${quant}` : path;
}

/** A Hub reference's repo, quant and path within the repo (mirrors hub/refs.py loosely). */
function refParts(ref) {
  let v = ref.trim().replace(/^hf:\/\//, '').replace(/^https?:\/\/(www\.)?(huggingface\.co|hf\.co)\//, '').replace(/^models\//, '');
  const parts = v.split('/');
  let quant = null;
  if (parts[1]?.includes(':')) [parts[1], quant] = parts[1].split(':');
  let path = parts.slice(2);
  if (['resolve', 'blob', 'tree'].includes(path[0])) path = path.slice(2);
  v = parts.slice(0, 2).join('/');
  return { repo: v, quant, path: path.join('/') };
}

/** The variant of a repo a reference names: by its path, else by its quant. */
function variantFor(r, ref) {
  const { quant, path } = refParts(ref);
  return r.variants.find((v) => path && (v.id === path || v.files?.some((f) => f.path === path)))
    ?? r.variants.find((v) => path && v.id.startsWith(`${path}/`))
    ?? r.variants.find((v) => quant && v.quant && v.quant.toLowerCase() === quant.toLowerCase())
    ?? null;
}

/** What a reference downloads, matched to a card's slot when the card is given. */
async function hubPick(ref, { model = null, slot = null } = {}) {
  const r = await api.hubRepo(ref, null, model);
  let v = null;
  if (model && slot) {
    const role = r.roles.find((x) => x.key === slot);
    const id = role?.suggested ?? role?.candidates?.[0];
    v = r.variants.find((x) => x.id === id) ?? null;
  }
  v ??= variantFor(r, ref);
  return v && { repo: r.id, sha: r.sha ?? null, variant: v.id, name: v.name, size_bytes: v.size_bytes ?? null, local: v.local?.path ?? null };
}

const bytes = (n) => (n == null ? '' : n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB`
  : n >= 1e6 ? `${Math.round(n / 1e6)} MB` : `${Math.max(1, Math.round(n / 1e3))} KB`);
const fileName = (path) => (path ? path.split(/[\\/]/).pop() : '');
const enter = (e) => { if (e.key === 'Enter' && e.target.tagName === 'INPUT') e.target.blur(); };
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);
const pct = (score) => `${Math.round(score * 100)}%`;
const article = (word) => (/^[aeiou]/i.test(word) ? 'an' : 'a');
const isDecoder = (slot) => /vae|tae/.test(slot);
const hasFiles = (e) => [...(e.dataTransfer?.types ?? [])].includes('Files');
const jobPct = (job) => (job.state === 'queued' ? 'queued' : `↓ ${Math.floor((100 * job.done_bytes) / (job.total_bytes || 1))}%`);

/** A slot as the page names it: {kicker, short, heading}. c: a component or a family slot. */
function slotName(c, familyName = null) {
  const slot = c.slot ?? c.id;
  const what = c.what ?? slot;
  if (MAIN_SLOTS.includes(slot)) return { kicker: cap(what), short: familyName ?? cap(what), heading: cap(what) };
  if (isDecoder(slot)) {
    const tiny = /tiny/i.test(what);
    return { kicker: 'Decoder', short: tiny ? 'TAE' : 'VAE', heading: tiny ? 'Tiny autoencoder' : 'VAE' };
  }
  const m = /^(.*\S)\s+text encoder$/i.exec(what);
  if (m) return { kicker: 'Text encoder', short: m[1], heading: `Text encoder · ${m[1]}` };
  if (/text encoder/i.test(what)) {
    const short = (c.label ?? '').replace(/\s*text encoder\s*$/i, '').trim() || 'LLM';
    return { kicker: 'Text encoder', short, heading: `Text encoder · ${short}` };
  }
  return { kicker: cap(what), short: cap(what), heading: cap(what) };
}

/** What a family must have, as units: a required slot alone, or a group ("vae or tae"). */
function needUnits(slots, groups, main = null) {
  const grouped = new Set(groups.flat());
  const out = [];
  for (const c of slots) if ((c.required || (c.slot ?? c.id) === main) && !grouped.has(c.slot ?? c.id)) out.push([c]);
  for (const g of groups) {
    const unit = slots.filter((c) => g.includes(c.slot ?? c.id));
    if (unit.length) out.push(unit);
  }
  return out;
}

/** Files already here that can fill any of these slots, its kin's first: cards of the same
 * family group (FLUX.1 dev and schnell share their encoders and VAE). */
function onDisk(lib, slots, { except = null, family = null } = {}) {
  if (slots.some((s) => MAIN_SLOTS.includes(s))) return [];
  const cards = new Map((lib?.items ?? []).map((i) => [i.id, i]));
  const fams = new Map((lib?.families ?? []).map((f) => [f.id, f]));
  // Kin share the file a slot holds: the same family, or one of its group that labels the
  // slot alike. Qwen-Image's text encoder ("Qwen 2.5 VL 7B") is not Qwen-Image-2.1's.
  const kin = (other, slot) => {
    if (other == null || family == null) return false;
    if (other === family) return true;
    const a = fams.get(other);
    const b = fams.get(family);
    if (a?.group == null || a.group !== b?.group) return false;
    const key = slot === 'llm_vision' ? 'llm' : slot; // a vision projector goes with its encoder
    const la = a.slot_labels?.[key];
    const lb = b.slot_labels?.[key];
    return la != null && la === lb;
  };
  return (lib?.components ?? [])
    .filter((s) => s.exists && s.path !== except && s.slots.some((x) => slots.includes(x)))
    .map((s) => {
      const slot = slots.find((x) => s.slots.includes(x));
      return {
        ...s,
        slot,
        names: s.cards.map((id) => cards.get(id)?.name ?? id),
        same: s.cards.some((id) => kin(cards.get(id)?.family, slot)),
      };
    })
    .sort((a, b) => (b.same - a.same) || (b.cards.length - a.cards.length));
}

const same = (a, b) => (a ?? null) === (b ?? null)
  || (typeof a === 'number' && typeof b === 'number' && Math.abs(a - b) < 1e-9)
  || (a != null && b != null && JSON.stringify(a) === JSON.stringify(b));

/** Rebuild root with build(), keeping focus (and the caret) on the field with the same data-key. */
function keepFocus(root, build) {
  const active = document.activeElement;
  const key = root.contains(active) ? active?.dataset?.key : null;
  const caret = key && typeof active.selectionStart === 'number' ? [active.selectionStart, active.selectionEnd] : null;
  build();
  if (!key) return;
  const el = root.querySelector(`[data-key="${CSS.escape(key)}"]`);
  if (!el) return;
  el.focus({ preventScroll: true });
  if (caret) try { el.setSelectionRange(...caret); } catch { /* not a text field */ }
}

function stateChip(item) {
  if (item.ready) return h('span.state.done', 'READY');
  if (item.problems.length) return h('span.state.fail', { title: item.problems.join('\n') }, 'CAN\'T LOAD');
  return h('span.state.wait', `NEEDS ${item.missing.join(' · ').toUpperCase()}`);
}

/** Families for a picker, grouped (optgroups), the current one included even if hidden. */
function familyOptions(families, current) {
  const groups = new Map();
  for (const f of families) {
    if (f.hidden && f.id !== current) continue;
    const key = f.group ?? 'Other';
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(f);
  }
  return [...groups].sort(([a], [b]) => a.localeCompare(b)).map(([group, list]) => h('optgroup', { label: group },
    list.map((f) => h('option', { value: f.id, selected: f.id === current }, f.name))));
}

// --- the library -----------------------------------------------------------------------------

export function modelsScreen({ navigate }) {
  let lib = null;
  let error = null;
  let query = '';
  let gone = false;

  const status = statusLine();
  const meta = h('span.topbar-meta');
  const search = h('input.search-input', {
    type: 'search', placeholder: 'Search names, families, files', 'aria-label': 'Search models',
    oninput: () => { query = search.value.trim().toLowerCase(); render(); },
    onkeydown: (e) => { if (e.key === 'Escape') { search.value = ''; query = ''; render(); search.blur(); } },
  });
  const add = h('button.btn.btn-primary.blueprint.engine-go', { type: 'button', onclick: () => importWizard(lib, navigate) },
    corners(), icon('plus', 14), h('span.only-wide', 'Import a model'));
  const header = h('header.topbar.sub',
    iconButton('back', 'Back', () => navigate(lastPlace())),
    h('div.topbar-heading', h('span.topbar-title', 'Models'), meta),
    h('span.spacer'),
    h('label.search.lib-search', icon('search', 15), search),
    h('div.only-wide', status.el),
    add);
  const body = h('div.lib-body');
  const scroller = h('div.gallery-scroll.lib-scroll', body);
  const strip = connectionStrip();
  const el = h('div.screen.loras.models', header, strip.el, scroller);

  async function refresh() {
    try {
      lib = await loadModels();
      if (!lib) throw new Error('the model library could not be loaded');
      error = null;
    } catch (err) {
      error = err.message;
    }
    if (!gone) render();
  }

  const searched = (item) => !query || [item.name, item.family_name, item.group, item.main_file, ...item.tags]
    .filter(Boolean).join(' ').toLowerCase().includes(query);

  function render() {
    clear(body);
    if (!lib) {
      body.append(h('div.notice', error ?? 'Reading the model cards…'));
      return;
    }
    const ready = lib.items.filter((i) => i.ready).length;
    meta.textContent = [plural(lib.items.length, 'model'), ready !== lib.items.length && `${ready} ready`,
      plural(lib.components.length, 'file')].filter(Boolean).join(' · ');
    if (!lib.items.length) {
      body.append(h('div.notice.lib-empty',
        h('span.eyebrow.accent', 'No models yet'),
        h('span', 'Import a model file from this machine or Hugging Face: its family is detected, and the card lists the files it still needs.'),
        h('div.lib-empty-actions', h('button.btn-line', { type: 'button', onclick: () => importWizard(lib, navigate) }, icon('plus', 15), 'Import a model'))));
      return;
    }
    const list = lib.items.filter(searched);
    append(body, [h('div.model-tiles', list.map(tile)),
      !list.length && h('div.notice', `Nothing matches “${query}”.`),
      componentsTable()]);
  }

  function tile(item) {
    let img;
    if (item.cover) {
      img = h('img.tile-pic', { alt: item.name, loading: 'lazy', decoding: 'async' });
      setImageUrl(img, item.cover.url);
    } else {
      img = h('span.tile-desc', item.family_name ?? 'no family yet');
    }
    const engines = item.engines.map((u) => `${u.running ? '●' : '○'} ${u.engine}${u.role === 'edit' ? ' (edits)' : ''}`).join('  ');
    const line = [item.family_name, bytes(item.size_bytes), item.uses ? plural(item.uses, 'image') : 'unused',
      item.last_used_at && ago(item.last_used_at)].filter(Boolean).join(' · ');
    return h('a.tile.result.lib-tile.model-tile', { href: `#/models/${item.id}`, 'aria-label': item.name },
      h('div.tile-img', { style: { aspectRatio: item.cover?.width ? `${item.cover.width} / ${item.cover.height}` : '4 / 3' } }, img),
      h('div.tile-foot.lib-foot',
        h('div.tile-foot-text',
          h('span.tile-title', item.pinned ? [icon('pin', 12), ' ', item.name] : item.name),
          h('span.mono-meta', line),
          h('span.model-tile-state', stateChip(item), engines && h('span.mono-meta', engines)))));
  }

  function componentsTable() {
    if (!lib.components.length) return null;
    const names = new Map(lib.items.map((i) => [i.id, i.name]));
    const rows = lib.components.map((c) => h('div.model-file-row',
      h('span.mono-meta.strong.wrap', { title: c.path }, fileName(c.path)),
      h('span.mono-meta', c.slots.join(', ')),
      h('span.mono-meta', c.exists ? bytes(c.size_bytes) : 'missing'),
      h('span.mono-meta.wrap', c.cards.map((id) => names.get(id) ?? id).join(', '))));
    return h('section.model-files',
      h('div.lib-usage-head', h('span.search-count', 'Files on disk'),
        h('span.mono-meta', 'each component file, and the cards that use it · a card can reuse any of them')),
      h('div.model-file-row.lib-head', h('span.eyebrow', 'File'), h('span.eyebrow', 'Slot'), h('span.eyebrow', 'Size'), h('span.eyebrow', 'Used by')),
      rows);
  }

  refresh();
  return {
    el,
    destroy() {
      gone = true;
      status.destroy();
      strip.destroy();
    },
  };
}

// --- importing: source, family, files ----------------------------------------------------------

/** Make a card in three steps: where the model is, which family it is, what else it needs. */
function importWizard(lib, navigate) {
  // Open at once (a slow library read must not invite a second click); the families and the
  // library arrive while the source is picked.
  let families = [];
  let famById = new Map();
  const ready = Promise.all([api.modelFamilies(), lib ?? loadModels()]).then(([list, library]) => {
    families = list;
    famById = new Map(list.map((f) => [f.id, f]));
    lib = library;
  });
  ready.catch((err) => {
    toast(err.message, { error: true });
    box?.close();
  });
  const st = {
    step: 1,
    text: '', // the path field
    source: null, // {kind: 'path', path, name, size_bytes} | {kind: 'ref', ref, name, repo?, sha?, variant?, size_bytes?}
    note: null, // {text, warn} under the drop zone
    found: [], // paths a dropped file's name matched
    busy: false,
    detected: null,
    family: '',
    browse: false, // every family in a picker, not only the guesses
    fam: null, // the chosen family in full
    main: null,
    units: [],
    choices: {}, // unit key -> {kind: 'file'|'hub'|'none', slot, path?, ref?, pick?, mine?, auto?, picked?}
    sel: null,
    name: '',
    named: false,
    hub: new Map(), // ref -> what it downloads (null: asking; false: unknown)
  };
  let box;
  modal((close) => {
    box = h('div.dialog.blueprint.mi', { role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Import a model' });
    box.close = close;
    render();
    return box;
  });

  const famName = () => famById.get(st.family)?.name ?? st.fam?.name ?? st.family;
  const unitKey = (u) => u[0].slot;
  const isGap = (u) => {
    const c = st.choices[unitKey(u)];
    return !(c.mine || c.auto || c.picked);
  };
  const disk = (u) => onDisk(lib, u.map((c) => c.slot), { family: st.family });
  const hubInfo = (c) => (c.kind === 'hub' ? (c.pick ?? st.hub.get(c.ref) ?? null) || null : null);

  function render() {
    if (!box) return;
    keepFocus(box, () => {
      clear(box).append(...corners());
      const body = st.step === 1 ? sourceStep() : st.step === 2 ? familyStep() : filesStep();
      append(box, [steps(), h('div.mi-main', body), foot()]);
    });
  }

  // --- the step bar ------------------------------------------------------------------------

  function steps() {
    const guess = st.detected?.guesses.find((g) => g.family === st.family);
    const found = st.units.filter((u) => !isGap(u) && st.choices[unitKey(u)].kind !== 'none').length;
    const cells = [
      { word: 'Source', ask: 'Where is the model?', done: st.source?.name ?? 'No file yet', mono: !!st.source },
      { word: 'Family', ask: 'Which model is it?', done: st.family ? `${famName()}${guess ? ` · ${pct(guess.score)}` : ''}` : '' },
      { word: 'Files', ask: st.step === 3 ? `${found} of ${st.units.length} found` : 'What else it needs' },
    ];
    return h('div.mi-steps', cells.map((c, i) => {
      const n = i + 1;
      const now = st.step === n;
      const past = st.step > n;
      return h(past ? 'button.mi-step.done' : 'div.mi-step', {
        class: now ? 'now' : '', type: past ? 'button' : null, 'aria-current': now ? 'step' : null,
        title: past ? `Back to ${c.word.toLowerCase()}` : null, onclick: past ? () => { st.step = n; render(); } : null,
      },
      h('span.mi-step-n', `0${n}${past ? ' ✓' : ''} ${c.word.toUpperCase()}`),
      h(past && c.mono ? 'span.mi-step-v.mono' : 'span.mi-step-v', past ? c.done : c.ask));
    }));
  }

  function foot() {
    const cancel = h('button.btn.btn-ghost.mi-btn', { type: 'button', onclick: () => box.close() }, 'Cancel');
    const back = (n) => h('button.btn.btn-ghost.mi-btn', { type: 'button', onclick: () => { st.step = n; render(); } }, 'Back');
    const primary = (label, onclick, disabled = false) => h('button.btn.btn-primary.blueprint.mi-btn.mi-go', {
      type: 'button', disabled: disabled || st.busy, onclick,
    }, corners(), label);
    if (st.step === 1) {
      return h('div.mi-foot',
        h('span.mi-foot-text', 'Or ', h('button.link-btn', { type: 'button', onclick: fromFamily }, 'start from a family'), ' and add files later.'),
        h('span.spacer'), cancel,
        primary(st.busy ? 'Reading…' : 'Next: family', () => next(), !st.text.trim()));
    }
    if (st.step === 2) {
      return h('div.mi-foot', back(1), h('span.spacer'), cancel,
        primary(st.busy ? 'Reading…' : 'Next: files', () => goFiles(), !st.family));
    }
    const s = summary();
    const name = h('input.seed-input.text', {
      value: st.name, placeholder: famName(), 'aria-label': 'Name', spellcheck: false, dataset: { key: 'mi-name' },
      oninput: () => { st.name = name.value; st.named = true; },
      onkeydown: (e) => { if (e.key === 'Enter') { e.preventDefault(); create(); } },
    });
    return h('div.mi-foot',
      back(2),
      h('label.mi-name', h('span.eyebrow', 'Name'), h('span.seed-field', name)),
      h('span.mi-foot-text.mi-status', { class: s.ready ? 'ok' : '' }, s.text),
      primary(st.busy ? 'Making it…' : s.button, () => create()));
  }

  // --- 01 source (2c) ------------------------------------------------------------------------

  function sourceStep() {
    const picker = h('input', {
      type: 'file', accept: '.gguf,.safetensors,.sft,.ckpt', hidden: true,
      onchange: () => { if (picker.files[0]) dropped(picker.files[0]); },
    });
    const zone = h('div.mi-drop', {
      tabindex: '0', role: 'button', 'aria-label': 'Drop a model file here, or choose one',
      class: st.busy ? 'busy' : '',
      ondragenter: (e) => { if (hasFiles(e)) { e.preventDefault(); zone.classList.add('over'); } },
      ondragover: (e) => { if (hasFiles(e)) { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; } },
      ondragleave: (e) => { if (!zone.contains(e.relatedTarget)) zone.classList.remove('over'); },
      ondrop: (e) => {
        e.preventDefault();
        zone.classList.remove('over');
        const file = e.dataTransfer.files[0];
        if (file) dropped(file);
      },
      onclick: () => picker.click(),
      onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); picker.click(); } },
    },
    icon('upload', 28),
    h('span.mi-drop-title', 'Drop a .gguf or .safetensors'),
    h('span.mi-drop-sub', 'Its family is read from the weights and the name.'));
    const path = h('input.seed-input.path-input', {
      type: 'text', value: st.text, spellcheck: false, placeholder: 'or paste a path — D:\\models\\…', 'aria-label': 'A model file on this machine',
      dataset: { key: 'mi-path' },
      oninput: () => { st.text = path.value; st.found = []; syncNext(); },
      onkeydown: (e) => { if (e.key === 'Enter') { e.preventDefault(); next(); } },
    });
    const found = st.found.length > 1 && h('div.mi-found', { role: 'radiogroup', 'aria-label': 'Which file' },
      st.found.map((p) => h('button.mi-found-row', {
        type: 'button', role: 'radio', 'aria-checked': String(st.text === p), class: st.text === p ? 'on' : '',
        onclick: () => { st.text = p; render(); },
      }, h('span.mi-radio'), h('span.mono-meta.wrap', p))));
    return h('div.mi-source',
      h('div.mi-source-here',
        h('span.mi-kicker', 'On this machine'),
        zone,
        st.note && h('span.mi-note', { class: st.note.warn ? 'warn' : '', title: st.note.title ?? null }, st.note.text),
        found,
        h('div.mi-path', h('label.seed-field', icon('folder', 13), path),
          h('button.btn.btn-secondary.mi-btn', { type: 'button', onclick: () => picker.click() }, 'Browse…'), picker)),
      h('div.mi-source-hub',
        h('span.mi-kicker', 'Not downloaded yet'),
        h('span.mi-copy', 'Find it on Hugging Face. The file downloads into the new card, and the family comes from the repo.'),
        h('button.btn.btn-secondary.mi-btn.mi-hub-btn', { type: 'button', onclick: searchHub }, icon('search', 14), 'Search Hugging Face'),
        h('span.spacer'),
        h('span.mi-small', 'Opens the same browser as the Engines screen.')));
  }

  /** The Next button follows the path field without rebuilding it while typing. */
  function syncNext() {
    const go = box.querySelector('.mi-go');
    if (go) go.disabled = !st.text.trim() || st.busy;
  }

  async function dropped(file) {
    if (!MODEL_FILE.test(file.name)) {
      st.note = { warn: true, text: `${file.name} isn't a model file: drop a .gguf or .safetensors.` };
      render();
      return;
    }
    // A browser tells a dropped file's name and size, not its folder: the server finds it.
    st.busy = true;
    st.found = [];
    st.note = { text: `Looking for ${file.name} on this machine…` };
    render();
    let res = null;
    try {
      res = await api.locateModel({ name: file.name, size_bytes: file.size });
    } catch (err) {
      st.note = { warn: true, text: err.message };
    }
    st.busy = false;
    if (res?.paths.length === 1) {
      st.text = res.paths[0];
      st.note = null;
      st.source = { kind: 'path', path: res.paths[0], name: file.name, size_bytes: file.size, text: res.paths[0] };
      next();
      return;
    }
    if (res) {
      st.found = res.paths;
      if (res.paths.length) st.text = res.paths[0];
      st.note = res.paths.length
        ? { text: `${res.paths.length} files here are called ${file.name}. Pick one, then Next.` }
        : {
          warn: true,
          text: `${file.name} isn't in the model folders here. Paste its full path below.`,
          title: `Looked in:\n${res.searched.join('\n')}`,
        };
    }
    render();
    if (!res?.paths.length) box.querySelector('[data-key="mi-path"]')?.focus();
  }

  function searchHub() {
    hubDialog({ name: 'new-model', label: 'New model', type: 'sdcpp' }, {
      onPick: (p) => {
        if (p.local) {
          st.text = p.local;
          st.source = { kind: 'path', path: p.local, name: fileName(p.local), size_bytes: p.size_bytes, text: p.local };
        } else {
          st.text = p.ref;
          st.source = { kind: 'ref', ref: p.ref, name: p.name, repo: p.repo, sha: p.sha, variant: p.variant, size_bytes: p.size_bytes, text: p.ref };
        }
        next();
      },
    });
  }

  async function fromFamily() {
    try {
      await ready;
    } catch {
      return;
    }
    st.source = null;
    st.detected = null;
    st.text = '';
    st.browse = true;
    st.step = 2;
    render();
  }

  const isRef = (value) => looksLikeRef(value)
    || (/^[\w.-]+\/[\w.-]+(\/.*)?$/.test(value) && !/^[A-Za-z]:/.test(value) && !value.startsWith('/'));

  async function next() {
    const value = st.text.trim();
    if (!value || st.busy) return;
    const ref = isRef(value);
    if (st.source?.text !== value) {
      st.source = ref ? { kind: 'ref', ref: value, name: sourceLabel(value), text: value }
        : { kind: 'path', path: value, name: fileName(value), text: value };
    }
    st.busy = true;
    render();
    try {
      [st.detected] = await Promise.all([api.detectModel(ref ? { ref: value } : { path: value }), ready]);
      const best = st.detected.guesses[0];
      st.family = best && best.score >= 0.5 ? best.family : '';
      st.browse = !st.detected.guesses.length;
      if (!st.named) st.name = st.detected.name ?? '';
      st.step = 2;
    } catch (err) {
      toast(err.message, { error: true });
    }
    st.busy = false;
    render();
  }

  // --- 02 family (1g) ------------------------------------------------------------------------

  function familyStep() {
    const guesses = (st.detected?.guesses ?? []).slice(0, 4);
    const rows = guesses.map((g, i) => {
      const on = st.family === g.family;
      const group = famById.get(g.family)?.group;
      return h('button.mi-guess', {
        type: 'button', role: 'radio', 'aria-checked': String(on), class: [on && 'on', i === 0 && 'best'].filter(Boolean).join(' '),
        onclick: () => { st.family = g.family; if (!st.named) st.name = st.detected?.name ?? ''; render(); },
      },
      h('span.mi-radio'),
      h('span.mi-guess-text',
        h('span.mi-guess-name', h('b', g.name), group && h('span', group)),
        h('span.mi-guess-why', g.reasons.join(' · ') || 'Nothing points elsewhere')),
      h('span.mi-conf', h('b', pct(g.score)), h('span.mi-bar', h('i', { style: { width: pct(g.score) } }))));
    });
    const pick = h('select.seed-input', {
      'aria-label': 'Family', dataset: { key: 'mi-family' },
      onchange: () => { st.family = pick.value; render(); },
    }, h('option', { value: '' }, 'Pick a family…'), familyOptions(families, st.family));
    return h('div.mi-family',
      guesses.length
        ? h('div.mi-guesses', { role: 'radiogroup', 'aria-label': 'Family' }, rows)
        : h('span.mi-copy', st.source ? 'Nothing about it says which family it is: pick one.'
          : 'Pick its family: it says which files the card needs, and how to drive it.'),
      guesses.length > 0 && !st.browse && h('button.link-btn.mi-more', { type: 'button', onclick: () => { st.browse = true; render(); } },
        'None of these — browse all families'),
      st.browse && h('label.seed-field.mi-all', pick));
  }

  async function goFiles() {
    if (!st.family || st.busy) return;
    st.busy = true;
    render();
    try {
      if (st.fam?.id !== st.family) {
        st.fam = await api.modelFamily(st.family);
        st.choices = {};
      }
    } catch (err) {
      toast(err.message, { error: true });
      st.busy = false;
      render();
      return;
    }
    plan();
    st.step = 3;
    st.busy = false;
    render();
    sizes();
  }

  // --- 03 files (2b) -------------------------------------------------------------------------

  /** The family's units, each with a first choice: matched on disk, to download, or none. */
  function plan() {
    const fam = st.fam;
    const slots = fam.slots.map((s) => ({ slot: s.id, flag: s.flag, what: s.what, label: s.label, required: s.required, sources: s.sources }));
    st.main = slots.find((s) => MAIN_SLOTS.includes(s.slot))?.slot ?? 'diffusion_model';
    const units = needUnits(slots, fam.required_any ?? [], st.main);
    units.sort((a, b) => order(a) - order(b));
    const keep = st.choices;
    st.units = units;
    st.choices = {};
    for (const u of units) {
      const old = keep[unitKey(u)];
      st.choices[unitKey(u)] = old && !old.mine ? old : firstChoice(u);
    }
    st.sel = unitKey(units.find(isGap) ?? units[0]);
  }

  const order = (u) => (MAIN_SLOTS.includes(u[0].slot) ? 1 : isDecoder(u[0].slot) ? 2 : 0);

  function firstChoice(u) {
    const slot = unitKey(u);
    if (slot === st.main && st.source) {
      return st.source.kind === 'path'
        ? { kind: 'file', slot, path: st.source.path, mine: true }
        : { kind: 'hub', slot, ref: st.source.ref, mine: true,
          pick: st.source.variant ? { repo: st.source.repo, sha: st.source.sha, variant: st.source.variant, size_bytes: st.source.size_bytes } : null };
    }
    const found = disk(u);
    if (found[0]) return { kind: 'file', slot: found[0].slot, path: found[0].path, auto: found[0].same };
    const withSource = u.find((c) => c.sources.length);
    return withSource ? { kind: 'hub', slot: withSource.slot, ref: withSource.sources[0] } : { kind: 'none', slot };
  }

  /** Sizes of what each source downloads, and whether it is in the Hugging Face cache. */
  function sizes() {
    const refs = st.units.flatMap((u) => u.flatMap((c) => c.sources.slice(0, 3)));
    if (st.source?.kind === 'ref' && !st.source.variant) refs.push(st.source.ref);
    for (const ref of refs) {
      if (st.hub.has(ref)) continue;
      st.hub.set(ref, null);
      hubPick(ref).then((info) => { st.hub.set(ref, info ?? false); }, () => { st.hub.set(ref, false); })
        .finally(() => { if (st.step === 3) render(); });
    }
  }

  function options(u) {
    const slot = unitKey(u);
    const out = [];
    if (slot === st.main && st.source) {
      const c = firstChoice(u);
      const info = hubInfo(c);
      out.push({ choice: c, title: st.source.kind === 'path' ? `This file · ${st.source.name}` : `Download ${st.source.name}`,
        sub: st.source.kind === 'path' ? st.source.path : refParts(st.source.ref).repo, size: st.source.size_bytes ?? info?.size_bytes });
    }
    const norm = (p) => p.replace(/\\/g, '/').toLowerCase();
    const listed = new Set();
    for (const f of disk(u)) {
      listed.add(norm(f.path));
      out.push({ choice: { kind: 'file', slot: f.slot, path: f.path }, title: `Reuse ${fileName(f.path)}`,
        sub: `On disk · ${f.names.join(', ')} ${f.names.length > 1 ? 'use' : 'uses'} it`, size: f.size_bytes, path: f.path });
    }
    for (const c of u) {
      for (const ref of c.sources.slice(0, 3)) {
        const info = st.hub.get(ref);
        if (info?.local && listed.has(norm(info.local))) continue; // a card already uses that copy
        if (info?.local) {
          out.push({ choice: { kind: 'file', slot: c.slot, path: info.local }, title: `Use ${info.name}`,
            sub: 'On disk · in the Hugging Face cache', size: info.size_bytes, path: info.local });
        } else {
          out.push({ choice: { kind: 'hub', slot: c.slot, ref, pick: info || null },
            title: `Download ${info?.name ?? sourceLabel(ref)}${u.length > 1 ? ` · ${slotName(c).short}` : ''}`,
            sub: refParts(ref).repo, size: info?.size_bytes, asking: info === null });
        }
      }
    }
    return out;
  }

  const sameChoice = (a, b) => a.kind === b.kind && a.slot === b.slot && (a.path ?? a.ref ?? null) === (b.path ?? b.ref ?? null);

  function choose(u, choice) {
    st.choices[unitKey(u)] = { ...choice, picked: true };
    render();
  }

  async function choosePath(u) {
    const value = (await dialog({
      title: `${cap(u[0].what)} file`, body: 'A path on this machine.',
      input: { placeholder: 'D:\\models\\…' }, confirm: 'Use it',
    }))?.trim();
    if (value) choose(u, { kind: 'file', slot: unitKey(u), path: value });
  }

  function filesStep() {
    const total = st.units.length;
    const gaps = st.units.filter(isGap).length;
    const later = st.units.filter((u) => !isGap(u) && st.choices[unitKey(u)].kind === 'none').length;
    const matched = total - gaps - later;
    const word = (n) => WORDS[n] ?? String(n);
    const lede = [`${famName()} loads ${word(total)} ${total === 1 ? 'file' : 'files'}.`,
      matched && gaps ? ` ${matched === 1 ? 'One was' : `${cap(word(matched))} were`} matched to files you already have; pick ${gaps === 1 ? 'the last one' : `the other ${word(gaps)}`}.`
        : gaps ? ` Pick ${gaps === 1 ? 'it' : 'each'} below.`
          : later ? ` ${cap(word(matched))} ${matched === 1 ? 'is' : 'are'} set; ${later === 1 ? 'one waits' : `${word(later)} wait`} for the card.`
            : ' Each is matched: check them, or make the card.'].join('');
    const u = st.units.find((x) => unitKey(x) === st.sel) ?? st.units[0];
    return h('div.mi-files',
      h('span.mi-copy.muted', lede),
      pipeline(),
      u && choiceList(u));
  }

  function choiceList(u) {
    const current = st.choices[unitKey(u)];
    const c = u.find((x) => x.slot === current.slot) ?? u[0];
    const opts = options(u);
    const rows = opts.map((o, i) => {
      const on = sameChoice(o.choice, current);
      return h('button.mi-choice', {
        type: 'button', role: 'radio', 'aria-checked': String(on), class: on ? 'on' : '',
        onclick: () => choose(u, o.choice),
      },
      h('span.mi-radio'),
      h('span.mi-choice-text', h('b', { title: o.path ?? null }, o.title), h('span', o.sub)),
      h('span.mi-choice-size', o.asking ? '…' : bytes(o.size)),
      i === 0 && unitKey(u) !== st.main && o.choice.kind === 'file' ? h('span.tag.tag-accent.mi-best', 'BEST') : h('span'));
    });
    const none = { kind: 'none', slot: unitKey(u) };
    const isNone = current.kind === 'none';
    return h('div.mi-choices',
      h('div.mi-choices-head', h('b', u.length > 1 ? u.map((x) => slotName(x).heading).join(' or ') : cap(c.what)),
        h('span.mono-meta', u.map((x) => x.flag).join(' · '))),
      h('div.mi-choice-list', { role: 'radiogroup', 'aria-label': cap(c.what) }, rows,
        h('button.mi-choice.quiet', { type: 'button', onclick: () => choosePath(u) },
          h('span.mi-radio.square'), h('span.mi-choice-text', h('b', 'Another file on this machine…'), h('span', 'A path')), h('span'), h('span'))),
      unitKey(u) !== st.main || !st.source
        ? h('button.link-btn.mi-later', { type: 'button', class: isNone ? 'on' : '', 'aria-pressed': String(isNone), onclick: () => choose(u, none) },
          isNone ? 'Left empty: pick it later from the card.' : 'Or leave it empty and pick later from the card.')
        : null);
  }

  /** The family drawn as its pipeline: prompt → encoders → model → decoder → image. */
  function pipeline() {
    const enc = st.units.filter((u) => order(u) === 0);
    const main = st.units.find((u) => order(u) === 1);
    const dec = st.units.filter((u) => order(u) === 2);
    const W = { prompt: 64, enc: 140, main: 164, dec: 110, img: 86 };
    const GAP = 32;
    const stack = (n, bh) => n * bh + Math.max(0, n - 1) * (n > 2 ? 20 : 40);
    const H = Math.max(230, stack(enc.length, 70) + 40, stack(dec.length, 70) + 40);
    const cy = H / 2;
    let x = 0;
    const col = (w) => { const at = x; x += w + GAP; return at; };
    const xPrompt = col(W.prompt);
    const xEnc = enc.length ? col(W.enc) : null;
    const xMain = col(W.main);
    const xDec = dec.length ? col(W.dec) : null;
    const xImg = col(W.img);
    const width = xImg + W.img;
    const tops = (n, bh) => { const start = (H - stack(n, bh)) / 2; const step = bh + (n > 2 ? 20 : 40); return Array.from({ length: n }, (_, i) => start + i * step); };
    const paths = [];
    const line = (d, gap) => paths.push(`<path d="${d}" stroke="var(${gap ? '--color-accent' : '--color-accent-700'})"${gap ? ' stroke-dasharray="4 4"' : ''}/>`);
    const blocks = [];
    const encTops = tops(enc.length, 70);
    const mainGap = main && isGap(main);
    enc.forEach((u, i) => {
      const y = encTops[i] + 35;
      const gap = isGap(u) || st.choices[unitKey(u)].kind === 'none';
      line(`M${W.prompt} ${cy} H${xPrompt + W.prompt + GAP / 2} V${y} H${xEnc}`, gap);
      line(`M${xEnc + W.enc} ${y} H${xEnc + W.enc + GAP / 2} V${cy} H${xMain}`, gap);
      blocks.push(block(u, xEnc, encTops[i], W.enc, 70));
    });
    if (!enc.length) line(`M${W.prompt} ${cy} H${xMain}`, false);
    if (main) blocks.push(block(main, xMain, cy - 50, W.main, 100, true));
    const decTops = tops(dec.length, 70);
    dec.forEach((u, i) => {
      const y = decTops[i] + 35;
      const gap = isGap(u) || st.choices[unitKey(u)].kind === 'none';
      line(`M${xMain + W.main} ${cy} H${xMain + W.main + GAP / 2} V${y} H${xDec}`, gap || mainGap);
      line(`M${xDec + W.dec} ${y} H${xDec + W.dec + GAP / 2} V${cy} H${xImg}`, gap);
      blocks.push(block(u, xDec, decTops[i], W.dec, 70));
    });
    if (!dec.length) line(`M${xMain + W.main} ${cy} H${xImg}`, mainGap);
    const svg = h('span.mi-pipe-lines');
    svg.innerHTML = `<svg width="${width}" height="${H}" fill="none" stroke-width="1.5" aria-hidden="true">${paths.join('')}</svg>`;
    const fw = st.fam.defaults?.width;
    const fh = st.fam.defaults?.height;
    const size = fw && fh ? (fw === fh ? `${fw}²` : `${fw}×${fh}`) : '';
    return h('div.mi-pipe.blueprint', corners(),
      h('div.mi-pipe-scroll',
        h('div.mi-pipe-canvas', { style: { width: `${width}px`, height: `${H}px` } },
          svg,
          h('div.mi-node.mi-io', { style: { left: `${xPrompt}px`, top: `${cy - 25}px`, width: `${W.prompt}px`, height: '50px' } }, 'Prompt'),
          blocks,
          h('div.mi-node.mi-io.mi-image', { style: { left: `${xImg}px`, top: `${cy - 35}px`, width: `${W.img}px`, height: '70px' } },
            h('b', 'Image'), size && h('span', size)))));
  }

  function block(u, left, top, width, height, isMain = false) {
    const k = unitKey(u);
    const ch = st.choices[k];
    const c = u.find((x) => x.slot === ch.slot) ?? u[0];
    const nm = slotName(c, famName());
    const gap = isGap(u);
    let kicker;
    let sub;
    let cls = '';
    if (gap) {
      kicker = 'Pick one';
      const n = options(u).length;
      sub = `${n} option${n === 1 ? '' : 's'} below`;
      cls = 'gap';
    } else if (ch.kind === 'none') {
      kicker = 'Later';
      sub = 'Not set';
      cls = 'later';
    } else if (ch.mine) {
      const info = hubInfo(ch);
      kicker = ch.kind === 'hub' ? '↓ Downloads' : '✓ This file';
      sub = [st.source.name.match(/(Q\d[\w_]*|BF16|F16|FP8|FP16|fp8\w*|bf16)/i)?.[0], bytes(st.source.size_bytes ?? info?.size_bytes)].filter(Boolean).join(' · ') || st.source.name;
    } else if (ch.kind === 'hub') {
      const info = hubInfo(ch);
      kicker = '↓ Download';
      sub = info?.size_bytes ? bytes(info.size_bytes) : refParts(ch.ref).repo.split('/')[0];
    } else {
      const f = disk(u).find((x) => x.path === ch.path);
      kicker = f ? '✓ Reused' : '✓ On disk';
      sub = f ? `from ${f.names[0]}` : fileName(ch.path);
    }
    return h('button.mi-node.mi-block', {
      type: 'button', class: [cls, isMain && 'main', st.sel === k && 'sel'].filter(Boolean).join(' '),
      style: { left: `${left}px`, top: `${top}px`, width: `${width}px`, height: `${height}px` },
      'aria-pressed': String(st.sel === k), title: `${nm.heading} · ${c.flag}`,
      onclick: () => { st.sel = k; render(); },
    },
    h('span.mi-node-kicker', kicker),
    h('b', isMain ? famName() : nm.short),
    h('span.mi-node-sub', sub));
  }

  function summary() {
    const chosen = st.units.map((u) => st.choices[unitKey(u)]);
    const downloads = chosen.filter((c) => c.kind === 'hub');
    const empty = st.units.filter((u) => st.choices[unitKey(u)].kind === 'none');
    const sized = downloads.map((c) => (c.mine ? st.source.size_bytes ?? hubInfo(c)?.size_bytes : hubInfo(c)?.size_bytes));
    const known = sized.every((n) => n != null);
    const total = sized.reduce((n, s) => n + (s ?? 0), 0);
    const parts = [];
    if (empty.length) parts.push(`Still needs ${empty.map((u) => slotName(u[0]).short).join(', ')}: pick ${empty.length === 1 ? 'it' : 'them'} later`);
    if (downloads.length) parts.push(`${known ? bytes(total) : plural(downloads.length, 'file')} to download once it's made`);
    return {
      ready: !empty.length && !downloads.length,
      text: parts.length ? parts.join(' · ') : 'Ready to run · nothing to download',
      button: downloads.length ? (known ? `Make it · get ${bytes(total)}` : `Make it · download ${downloads.length}`) : 'Make the card',
    };
  }

  async function create() {
    if (st.busy) return;
    st.busy = true;
    render();
    const components = {};
    for (const u of st.units) {
      const ch = st.choices[unitKey(u)];
      if (ch.kind === 'file' && !ch.mine) components[ch.slot] = ch.path;
    }
    const body = { family: st.family, name: st.name.trim() || null, components };
    if (st.source?.kind === 'path') body.path = st.source.path;
    else if (st.source?.kind === 'ref') body.ref = st.source.ref;
    let card;
    try {
      card = await api.createModel(body);
    } catch (err) {
      toast(err.message, { error: true });
      st.busy = false;
      render();
      return;
    }
    // Downloads go straight into the new card's slots; one that can't be matched opens the Hub.
    const failed = [];
    let started = 0;
    for (const u of st.units) {
      const ch = st.choices[unitKey(u)];
      if (ch.kind !== 'hub') continue;
      try {
        const known = hubInfo(ch);
        const pick = known?.variant && known.repo ? known : await hubPick(ch.ref, { model: card.id, slot: ch.slot });
        if (!pick) throw new Error('nothing in it fits');
        if (pick.local) await api.updateModel(card.id, { components: { [ch.slot]: pick.local } });
        else await api.hubDownload({ repo: pick.repo, revision: pick.sha ?? undefined, variant: pick.variant, model: card.id, key: ch.slot });
        started += pick.local ? 0 : 1;
      } catch {
        failed.push(ch);
      }
    }
    loadModels();
    box.close();
    navigate(`#/models/${card.id}`);
    if (failed.length) {
      hubDialog(modelTarget(card, failed[0].slot), { ref: failed[0].ref, onChange: () => loadModels() });
    } else {
      toast(started ? `Made ${card.name} · downloading ${plural(started, 'file')}` : `Made ${card.name}`);
    }
  }
}

// --- one card (2a) -----------------------------------------------------------------------------

export function modelScreen({ id, navigate }) {
  let d = null;
  let fam = null; // its family in full: slots, groups, defaults
  let lib = null;
  let error = null;
  let gone = false;
  let downloads = [];
  let poll = null;
  const opened = new Set(); // folded sections that are open
  let editing = null; // the preset cell being typed in: {name, key, draft}
  const hubs = new Map(); // `${slot}|${ref}` -> what the ref downloads into this card (null: asking)

  const status = statusLine();
  const meta = h('span.topbar-meta');
  const header = h('header.topbar.sub',
    iconButton('back', 'Models', () => navigate('#/models')),
    h('div.topbar-heading', h('span.topbar-title', 'Models'), meta),
    h('span.spacer'),
    h('div.only-wide', status.el));
  const main = h('main.links-main.lora-main.mc-main', { onscroll: () => track() });
  const strip = connectionStrip();
  const el = h('div.screen.lora.model', header, strip.el, main);

  async function load() {
    try {
      [d, lib] = await Promise.all([api.model(id), loadModels()]);
      await loadFamily();
      error = null;
    } catch (err) {
      error = err.message;
    }
    await followDownloads();
    if (!gone) render();
  }

  async function loadFamily() {
    if (!d?.family) {
      fam = null;
      return;
    }
    if (fam?.id === d.family) return;
    try {
      fam = await api.modelFamily(d.family);
    } catch {
      fam = null;
    }
  }

  const active = (j) => ['queued', 'running'].includes(j.state);

  /** Downloads into this card's slots: shown in their rows, and the card reloads when done. */
  async function followDownloads() {
    try {
      const mine = (await api.hubDownloads()).filter((j) => j.assign.some((a) => a.startsWith(`model:${id}.`)));
      const finished = downloads.some((j) => active(j) && !mine.some((m) => m.id === j.id && active(m)));
      const appeared = mine.some((m) => active(m) && !downloads.some((j) => j.id === m.id));
      downloads = mine;
      clearTimeout(poll);
      if (mine.some(active) && !gone) {
        poll = setTimeout(async () => {
          const changed = await followDownloads();
          if (gone) return;
          if (changed) render();
          else paintJobs();
        }, 2000);
      }
      if (finished) {
        d = await api.model(id);
        lib = await loadModels();
      }
      return finished || appeared;
    } catch {
      return false; // Hugging Face is off here: nothing to follow.
    }
  }

  const jobFor = (slots) => downloads.find((j) => active(j) && slots.some((s) => j.assign.includes(`model:${id}.${s}`)));

  /** Progress only, so a field being typed in isn't rebuilt every two seconds. */
  function paintJobs() {
    for (const node of main.querySelectorAll('[data-job]')) {
      const job = jobFor(node.dataset.job.split(','));
      if (job) node.textContent = node.classList.contains('long') ? `Downloading · ${jobPct(job).replace('↓ ', '')}` : jobPct(job);
    }
  }

  async function save(call, word = null) {
    try {
      d = await call();
      if (word) toast(word);
      await loadFamily();
      lib = await loadModels();
      loadBackends();
    } catch (err) {
      toast(err.message, { error: true });
    }
    if (!gone) render();
  }
  const patch = (body, word = null) => save(() => api.updateModel(id, body), word);
  const setDefaults = (values, word = null) => patch({ defaults: values }, word);

  function render() {
    keepFocus(main, () => {
      clear(main);
      if (!d) {
        main.append(h('div.notice', error ?? 'Loading…'),
          error && h('div', h('a.btn-line', { href: '#/models' }, icon('back', 15), 'The library')));
        return;
      }
      meta.textContent = d.main_file ?? 'no model file yet';
      append(main, [hero(), h('div.mc-body', rail(), h('div.mc-sections',
        filesSection(), samplingSection(), presetsSection(), moreSection()))]);
    });
    track();
  }

  // --- readiness -------------------------------------------------------------------------

  function units() {
    // A slot the card is missing though its recipe calls it optional: an edit model's vision
    // projector, which a GGUF text encoder needs.
    const comps = d.components.map((c) => (d.missing.includes(c.slot) ? { ...c, required: true } : c));
    return needUnits(comps, fam?.required_any ?? [], d.components.find((c) => MAIN_SLOTS.includes(c.slot))?.slot);
  }
  const filled = (u) => u.some((c) => c.path && !c.problem);
  const reuseFor = (u) => onDisk(lib, u.map((c) => c.slot), { family: d.family });

  /** What a Hub source downloads into a slot here, asked once. */
  function hubFor(slot, ref) {
    const key = `${slot}|${ref}`;
    if (!hubs.has(key)) {
      hubs.set(key, null);
      hubPick(ref, { model: id, slot }).then((v) => hubs.set(key, v ?? false), () => hubs.set(key, false))
        .finally(() => { if (!gone) render(); });
    }
    return hubs.get(key);
  }

  async function download(slot, ref) {
    try {
      const pick = hubFor(slot, ref) || await hubPick(ref, { model: id, slot });
      if (!pick) throw new Error('nothing in it fits');
      if (pick.local) {
        patch({ components: { [slot]: pick.local } }, 'Set from the Hugging Face cache');
        return;
      }
      await api.hubDownload({ repo: pick.repo, revision: pick.sha ?? undefined, variant: pick.variant, model: id, key: slot });
      toast(`Downloading ${pick.name}`);
      await followDownloads();
      if (!gone) render();
    } catch {
      hub(slot, ref);
    }
  }

  function hero() {
    const famPick = h('select.mc-family', {
      'aria-label': 'Family', title: 'Its family: the sd-server recipe it runs with', dataset: { key: 'family' },
      onchange: () => patch({ family: famPick.value || null }, 'Family changed'),
    }, h('option', { value: '' }, 'No family yet'), familyOptions(lib?.families ?? [], d.family));
    const caps = [d.group && `${d.group} family`, ...d.capabilities.map((c) => CAPS[c] ?? c)].filter(Boolean).join(' · ');
    const best = d.detected[0];
    const other = best && best.family !== d.family && best.score >= 0.5;
    const name = h('input.mc-name', {
      value: d.name, 'aria-label': 'Name', spellcheck: false, dataset: { key: 'name' },
      onchange: () => name.value.trim() && patch({ name: name.value }), onkeydown: enter,
    });
    const engines = d.engines.map((u) => [h('b', u.engine), u.role === 'edit' ? ' for edits' : '',
      ` (${u.loaded ? 'loaded' : u.running ? 'running' : 'stopped'})`,
      u.needs_restart ? h('span.accent', ' · restart to load the changes') : null]);
    const facts = [
      d.size_bytes ? `${bytes(d.size_bytes)} on disk` : 'nothing on disk yet',
      engines.length ? ['set on ', engines.flatMap((e, i) => (i ? [', ', e] : [e]))] : null,
      d.uses ? `${plural(d.uses, 'image')}${d.last_used_at ? `, the last ${ago(d.last_used_at)}` : ''}` : 'no images yet',
    ].filter(Boolean);
    return h('div.mc-hero',
      h('div.mc-title',
        h('div.mc-kicker',
          h('label.tag.tag-accent.mc-family-tag', famPick, icon('chevronDown', 13)),
          caps && h('span.mc-muted', caps)),
        h('div.mc-name-row', name,
          h('button.icon-btn.mc-pin', {
            type: 'button', class: d.pinned ? 'on' : '', title: d.pinned ? 'Pinned: unpin' : 'Pin', 'aria-pressed': String(d.pinned),
            onclick: () => patch({ pinned: !d.pinned }),
          }, icon('pin', 18))),
        h('span.mc-facts', facts.flatMap((f, i) => (i ? [' · ', f] : [f]))),
        d.warnings?.length ? h('span.mc-facts.accent', d.warnings.join(' · ')) : null,
        other && h('span.mc-facts', 'Its file looks like ', h('button.link-btn', {
          type: 'button', title: best.reasons.join('\n'), onclick: () => patch({ family: best.family }, 'Family changed'),
        }, `${best.name} (${pct(best.score)})`), '.')),
      plate());
  }

  /** The setup plate: how many files it has, and the one next action. */
  function plate() {
    const us = units();
    const have = us.filter(filled).length;
    const gaps = us.filter((u) => !filled(u));
    const useBtn = (primary) => {
      const b = h(primary ? 'button.btn.btn-primary.blueprint.mc-act-btn' : 'button.btn.btn-secondary.mc-act-btn', {
        type: 'button', disabled: !d.ready, onclick: () => useMenu(b),
      }, primary && corners(), primary && icon('play', 14), 'Use on an engine');
      return b;
    };
    const bars = h('div.mc-bars', { style: { gridTemplateColumns: `repeat(${Math.max(1, us.length)}, 1fr)` } },
      us.map((u) => h('i', { class: filled(u) ? 'on' : jobFor(u.map((c) => c.slot)) ? 'busy' : '' })));
    let kicker = 'Setup';
    let big = `${have} of ${us.length} ${us.length === 1 ? 'file' : 'files'}`;
    let text;
    let actions;
    if (!d.family) {
      big = 'No family';
      text = 'Pick its family above: it says which files the card needs and how to drive it.';
      actions = [useBtn(false)];
    } else if (d.problems.length) {
      kicker = 'Can\'t load';
      text = d.problems[0];
      actions = [h('button.btn.btn-primary.blueprint.mc-act-btn', { type: 'button', onclick: () => jump('files') }, corners(), 'Show the files'), useBtn(false)];
    } else if (gaps.length) {
      const names = gaps.map((u) => u.map((c) => c.what).join(' or '));
      const list = names.length > 1 ? `${names.slice(0, -1).join(', ')} and ${article(names.at(-1))} ${names.at(-1)}` : names[0];
      const local = gaps.map((u) => reuseFor(u)[0]);
      const job = jobFor(gaps.flatMap((u) => u.map((c) => c.slot)));
      const first = gaps[0];
      const ref = first.find((c) => c.sources.length)?.sources[0];
      const info = ref && !local.every(Boolean) ? hubFor(first.find((c) => c.sources.length).slot, ref) : null;
      let after;
      let primary;
      if (job) {
        after = ' It\'s downloading now.';
        primary = h('button.btn.btn-primary.blueprint.mc-act-btn', { type: 'button', disabled: true }, corners(),
          h('span.long', { dataset: { job: gaps.flatMap((u) => u.map((c) => c.slot)).join(',') }, class: 'long' }, `Downloading · ${jobPct(job).replace('↓ ', '')}`));
      } else if (local.every(Boolean)) {
        after = gaps.length === 1 ? ' One is already on disk — no download.' : ' Each is already on disk — no download.';
        primary = h('button.btn.btn-primary.blueprint.mc-act-btn', {
          type: 'button',
          onclick: () => patch({ components: Object.fromEntries(local.map((f) => [f.slot, f.path])) }, gaps.length === 1 ? 'Set' : 'All set'),
        }, corners(), gaps.length === 1 ? 'Use the one on disk' : 'Use the ones on disk');
      } else if (ref) {
        after = info?.local ? ' It\'s in the Hugging Face cache.' : info?.size_bytes ? ` ${bytes(info.size_bytes)} to download.` : ' Get it from Hugging Face.';
        const slot = first.find((c) => c.sources.length).slot;
        primary = h('button.btn.btn-primary.blueprint.mc-act-btn', { type: 'button', onclick: () => download(slot, ref) }, corners(),
          info?.local ? 'Use it' : 'Download it');
      } else {
        after = '';
        primary = h('button.btn.btn-primary.blueprint.mc-act-btn', { type: 'button', onclick: () => hub(first[0].slot, null) }, corners(), 'Find it');
      }
      text = [`Needs ${article(list)} `, h('b', list), `.${after}`];
      actions = [primary, useBtn(false)];
    } else {
      kicker = 'Ready';
      text = d.engines.length
        ? `Runs on ${d.engines.map((u) => (u.role === 'edit' ? `${u.engine} (for edits)` : u.engine)).join(', ')}.`
        : 'Every file it needs is here.';
      actions = [useBtn(true)];
    }
    return h('div.mc-plate.blueprint', { class: d.ready ? 'ready' : '' }, corners(),
      h('div.mc-plate-head', h('span.mc-plate-kicker', kicker), h('span.mc-plate-big', big)),
      bars,
      h('span.mc-plate-text', text),
      h('div.mc-plate-actions', actions));
  }

  function useMenu(anchor) {
    const engines = lib?.engines ?? [];
    if (!engines.length) {
      toast('No sd-server engine is configured', { error: true });
      return;
    }
    const edits = d.capabilities.includes('edit');
    menu(anchor, engines.flatMap((e) => [{
      label: `${e.name}${e.model_card === id ? ' · runs it' : ''}${e.launchable ? (e.running ? ' · restarts' : ' · loads on start') : ' · started elsewhere'}`,
      checked: e.model_card === id,
      onclick: async () => {
        if (e.running && e.model_card !== id && !(await dialog({ title: `Restart ${e.name}?`, body: `It stops, then starts with ${d.name}.`, confirm: 'Restart' }))) return;
        // A running engine restarts with it; a stopped one loads it when it is next started.
        save(() => api.useModel(id, { engine: e.name, start: false }), `${e.name} runs ${d.name}${e.running ? ' · restarting' : ''}`);
      },
    },
    // An edit model next to the engine's own: the engine switches to it for edits.
    edits && e.launchable && e.model_card !== id && {
      label: `${e.name} · for edits${e.edit_card === id ? ' · uses it' : ''}`,
      checked: e.edit_card === id,
      onclick: () => save(() => api.useModel(id, { engine: e.name, role: 'edit' }), `${e.name} edits with ${d.name}`),
    }].filter(Boolean)));
  }

  // --- the rail --------------------------------------------------------------------------

  const SECTIONS = [['files', 'Files'], ['sampling', 'Sampling'], ['presets', 'Presets'], ['prompting', 'Prompting'],
    ['launch', 'Launch'], ['notes', 'Notes & cover']];
  let current = 'files';

  function rail() {
    const us = units();
    const have = us.filter(filled).length;
    const count = { files: us.length ? h('span.mc-rail-badge', { class: d.ready ? '' : 'todo' }, `${have}/${us.length}`) : null,
      presets: Object.keys(d.presets).length ? h('span.mc-rail-n', String(Object.keys(d.presets).length)) : null };
    return h('nav.mc-rail', { 'aria-label': 'Sections' }, SECTIONS.map(([key, word], i) => h('a.mc-rail-item', {
      href: `#/models/${id}`, class: current === key ? 'on' : '', 'aria-current': current === key ? 'true' : null, dataset: { section: key },
      onclick: (e) => { e.preventDefault(); jump(key); },
    }, h('span.mc-rail-num', `0${i + 1}`), word, h('span.spacer'), count[key])));
  }

  function jump(key) {
    const section = main.querySelector(`#mc-${key}`);
    if (!section) return;
    if (section.tagName === 'DETAILS') {
      section.open = true;
      opened.add(key);
    }
    section.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
    current = key;
    paintRail();
  }

  /** The rail follows the section at the top of the scroll. */
  function track() {
    const top = main.getBoundingClientRect().top + 90;
    let now = SECTIONS[0][0];
    for (const [key] of SECTIONS) {
      const s = main.querySelector(`#mc-${key}`);
      if (s && s.getBoundingClientRect().top <= top) now = key;
    }
    if (main.scrollTop + main.clientHeight >= main.scrollHeight - 4) {
      const last = [...main.querySelectorAll('.mc-fold')].filter((f) => f.getBoundingClientRect().top < main.getBoundingClientRect().bottom).pop();
      if (last) now = last.id.replace('mc-', '');
    }
    if (now !== current) {
      current = now;
      paintRail();
    }
  }

  function paintRail() {
    for (const a of main.querySelectorAll('.mc-rail-item')) {
      const on = a.dataset.section === current;
      a.classList.toggle('on', on);
      if (on) a.setAttribute('aria-current', 'true');
      else a.removeAttribute('aria-current');
    }
  }

  const sectionHead = (title, text, ...extra) => h('div.mc-section-head', h('h3', title), text && h('span.mc-muted', text), ...extra);

  // --- 01 files --------------------------------------------------------------------------

  function filesSection() {
    if (!d.components.length) {
      return h('section.mc-section', { id: 'mc-files' }, sectionHead('Files', 'Pick its family to see which files it needs.'));
    }
    const us = units();
    const shown = new Set();
    const rows = [];
    for (const u of us) {
      if (filled(u) || u.some((c) => c.path)) {
        for (const c of u.filter((x) => x.path)) {
          rows.push(fileRow(c));
          shown.add(c.slot);
        }
      } else {
        rows.push(fixRow(u, us.filter((x) => !filled(x)).length === 1));
        u.forEach((c) => shown.add(c.slot));
      }
    }
    const rest = d.components.filter((c) => !shown.has(c.slot));
    for (const c of rest.filter((x) => x.path)) rows.push(fileRow(c));
    const empty = rest.filter((c) => !c.path);
    const more = empty.length > 0 && h('details.mc-optional', {
      open: opened.has('optional'), ontoggle: (e) => { if (e.target.open) opened.add('optional'); else opened.delete('optional'); },
    },
    h('summary', h('span.mc-summary-mark', '▸'), `${plural(empty.length, 'more file')} it can load`,
      h('span.mc-muted', empty.map((c) => slotName(c).short).join(', '))),
    empty.map((c) => fileRow(c)));
    const famWord = d.family_name ?? 'its family';
    return h('section.mc-section', { id: 'mc-files' },
      sectionHead('Files', `What sd-server loads for ${famWord}. Encoders and the VAE can be shared between cards.`),
      h('div.mc-table', rows, more));
  }

  function fileRow(c) {
    const bad = !!(c.path && c.problem);
    const nm = slotName(c, d.family_name);
    const job = jobFor([c.slot]);
    const note = bad ? `${c.problem}.`
      : [c.label !== c.what && !MAIN_SLOTS.includes(c.slot) && c.label, c.source && `From ${c.source.repo}`,
        c.shared_with.length && `Shared with ${c.shared_with.join(', ')}`].filter(Boolean).join(' · ');
    return h('div.mc-file', { class: bad ? 'bad' : c.path ? '' : 'empty' },
      h('span.mc-mark', { class: bad ? 'bad' : c.path ? 'ok' : '' }, c.path ? icon(bad ? 'alert' : 'check', 14) : null),
      h('span.mc-slot', h('b', nm.heading), h('span.mc-flag', c.flag)),
      h('span.mc-file-name',
        h('span.mc-path', { title: c.path ?? '' }, c.path ? fileName(c.path) : 'Not set'),
        note && h('span.mc-muted', { class: bad ? 'accent' : '' }, note)),
      job ? h('span.mc-size.accent', { dataset: { job: c.slot }, title: job.label }, jobPct(job)) : h('span.mc-size', bytes(c.size_bytes)),
      h('button.mc-link', { type: 'button', onclick: (e) => changeMenu(e.currentTarget, c) }, c.path ? 'Change' : 'Set'));
  }

  function changeMenu(anchor, c) {
    const others = reuseFor([c]).filter((s) => s.path !== c.path).slice(0, 6);
    menu(anchor, [
      { label: 'A path on this machine…', icon: 'folder', onclick: () => typePath(c) },
      ...others.map((s) => ({ label: `Reuse ${fileName(s.path)} · ${bytes(s.size_bytes)} · ${s.names.join(', ')}`, icon: 'layers',
        onclick: () => patch({ components: { [c.slot]: s.path } }, 'Reused') })),
      { label: 'Hugging Face…', icon: 'search', onclick: () => hub(c.slot, c.sources[0] ?? null) },
      c.path && { label: 'Clear it', icon: 'x', danger: true, onclick: () => patch({ components: { [c.slot]: null } }, 'Cleared') },
    ].filter(Boolean));
  }

  async function typePath(c) {
    const value = (await dialog({
      title: `${slotName(c).heading} file`, body: `${c.flag} · a path on this machine, or a Hugging Face URL or repo:QUANT.`,
      input: { value: c.path ?? '', placeholder: 'D:\\models\\…' }, confirm: 'Use it',
    }))?.trim();
    if (value == null || value === (c.path ?? '')) return;
    if (looksLikeRef(value)) hub(c.slot, value);
    else patch({ components: { [c.slot]: value || null } });
  }

  /** A missing file opens into the ways to get one: on disk, a download, or anything else. */
  function fixRow(u, last) {
    const c = u[0];
    const nm = slotName(c, d.family_name);
    const found = reuseFor(u);
    const withRef = u.find((x) => x.sources.length);
    const ref = withRef?.sources[0];
    const info = ref ? hubFor(withRef.slot, ref) : null;
    const slots = u.map((x) => x.slot);
    const job = jobFor(slots);
    const opts = [];
    if (found[0]) {
      const f = found[0];
      const more = found.slice(1, 7);
      opts.push(h('div.mc-opt.best.blueprint', corners(),
        h('span.mc-opt-kicker', 'Recommended · on disk'),
        h('span.mc-path', { title: f.path }, fileName(f.path)),
        h('span.mc-muted', [bytes(f.size_bytes), `used by ${f.names.join(', ')}`].filter(Boolean).join(' · ')),
        h('div.mc-opt-actions',
          h('button.btn.btn-primary.mc-opt-btn', { type: 'button', onclick: () => patch({ components: { [f.slot]: f.path } }, 'Set') }, 'Use this file'),
          more.length > 0 && h('button.mc-link', {
            type: 'button',
            onclick: (e) => menu(e.currentTarget, more.map((s) => ({
              label: `${fileName(s.path)} · ${bytes(s.size_bytes)} · ${s.names.join(', ')}`, onclick: () => patch({ components: { [s.slot]: s.path } }, 'Set'),
            }))),
          }, `${more.length} more`))));
    }
    if (ref) {
      const { quant } = refParts(ref);
      opts.push(h('div.mc-opt',
        h('span.mc-opt-kicker', info?.local ? 'In the Hugging Face cache' : `Download${quant ? ` · ${quant.toUpperCase()}` : ''}`),
        h('span.mc-path', { title: ref }, info?.name ?? sourceLabel(ref)),
        h('span.mc-muted', [info?.size_bytes && bytes(info.size_bytes), refParts(ref).repo.split('/')[0]].filter(Boolean).join(' · ')),
        h('div.mc-opt-actions', job
          ? h('span.mc-size.accent', { dataset: { job: slots.join(',') } }, jobPct(job))
          : h('button.btn.btn-secondary.mc-opt-btn', { type: 'button', onclick: () => download(withRef.slot, ref) }, info?.local ? 'Use it' : 'Download'),
        withRef.sources.length > 1 && !job && h('button.mc-link', {
          type: 'button',
          onclick: (e) => menu(e.currentTarget, withRef.sources.slice(1).map((r) => ({ label: sourceLabel(r), icon: 'download', onclick: () => download(withRef.slot, r) }))),
        }, `${withRef.sources.length - 1} more`))));
    }
    opts.push(h('div.mc-opt',
      h('span.mc-opt-kicker', 'Something else'),
      h('span.mc-muted', 'A path on this machine, or another file from Hugging Face.'),
      h('div.mc-opt-actions',
        h('button.btn.btn-secondary.mc-opt-btn', { type: 'button', onclick: () => typePath(c) }, 'Path…'),
        h('button.btn.btn-ghost.mc-opt-btn', { type: 'button', onclick: () => hub(withRef?.slot ?? c.slot, ref ?? null) }, 'Hugging Face')),
      u.length > 1 && h('span.mc-muted', 'Or ', u.slice(1).map((x, i) => [i ? ', ' : '',
        h('button.link-btn', { type: 'button', onclick: () => typePath(x) }, `${article(x.what)} ${x.what}`)]), ' instead.')));
    return h('div.mc-fix',
      h('div.mc-fix-head',
        h('span.mc-mark.dashed'),
        h('span.mc-slot', h('b', u.length > 1 ? u.map((x) => slotName(x).heading).join(' or ') : nm.heading),
          h('span.mc-flag', `${u.map((x) => x.flag).join(' / ')} · required`)),
        h('span.mc-fix-text', last ? 'Not set. Pick one of these — the card is ready right after.' : 'Not set. Pick one of these.')),
      h('div.mc-opts', { style: { gridTemplateColumns: opts.map((_, i) => (i === 0 && found[0] ? '1.25fr' : '1fr')).join(' ') } }, opts));
  }

  function hub(slot, ref) {
    hubDialog(modelTarget(d, slot), { ref, onChange: async () => { await followDownloads(); d = await api.model(id); lib = await loadModels(); if (!gone) render(); } });
  }

  // --- 02 sampling -----------------------------------------------------------------------

  // A family whose recipe names no size gets 1024 (library.DEFAULT_SIZE).
  const famDefault = (key) => fam?.defaults?.[key] ?? (key === 'width' || key === 'height' ? 1024 : null);
  const hasGuidance = () => famDefault('guidance') != null || d.defaults.guidance != null;
  // Image guidance (--img-cfg-scale): only families whose recipe has image CFG use it.
  const hasImgCfg = () => (fam?.features ?? []).includes('img_cfg') || d.defaults.img_cfg_scale != null;
  const takesRefs = () => (fam?.features ?? []).includes('ref_images');

  function numInput(key, value, { step = 1, integer = true, min = null, label, className = 'mc-num', onset = null } = {}) {
    const input = h(`input.${className}`, {
      type: 'number', step, min, value: value ?? '', placeholder: 'engine', 'aria-label': label, dataset: { key: `d-${key}` },
      onchange: () => {
        const raw = input.value.trim();
        const v = integer ? Number.parseInt(raw, 10) : Number(raw);
        const next = raw === '' ? null : Number.isFinite(v) ? v : value ?? null;
        (onset ?? ((x) => setDefaults({ [key]: x })))(next);
      },
      onkeydown: enter,
    });
    return input;
  }

  /** "Family default", or what the family has, clickable to go back to it. */
  function versus(keys) {
    if (!fam) return h('span.mc-versus');
    const edited = keys.filter((k) => !same(d.defaults[k], famDefault(k)));
    if (!edited.length) return h('span.mc-versus', 'Family default');
    const was = edited.map((k) => (famDefault(k) == null ? 'engine' : String(famDefault(k)))).join(' × ');
    return h('button.mc-versus.edited', {
      type: 'button', title: 'Back to the family\'s',
      onclick: () => setDefaults(Object.fromEntries(edited.map((k) => [k, famDefault(k)])), 'Back to the family\'s'),
    }, h('i'), `Edited · family: ${was}`);
  }

  function samplingSection() {
    const df = d.defaults;
    const w = df.width ?? famDefault('width') ?? 1024;
    const ht = df.height ?? famDefault('height') ?? 1024;
    const m = Math.max(64, fam?.size_multiple ?? 64);
    const area = w * ht;
    const aspects = ASPECTS.map(([a, b]) => {
      const ww = Math.max(m, Math.round(Math.sqrt((area * a) / b) / m) * m);
      const hh = Math.max(m, Math.round(area / ww / m) * m);
      const on = Math.abs(w / ht - a / b) < 0.03;
      const big = Math.max(a, b);
      return h('button.mc-aspect', {
        type: 'button', class: on ? 'on' : '', title: `${a}:${b} · ${ww} × ${hh}`, 'aria-label': `${a}:${b}`, 'aria-pressed': String(on),
        onclick: () => !on && setDefaults({ width: ww, height: hh }),
      }, h('i', { style: { width: `${Math.round((20 * a) / big)}px`, height: `${Math.round((20 * b) / big)}px` } }));
    });
    const steps = df.steps ?? famDefault('steps');
    const stepBy = (n) => setDefaults({ steps: Math.max(1, (steps ?? 20) + n) });
    const cfg = df.cfg_scale ?? famDefault('cfg_scale');
    const noCfg = cfg != null && cfg <= 1;
    const guidance = df.guidance ?? famDefault('guidance') ?? 3.5;
    const range = h('input.mc-range', {
      type: 'range', min: 0, max: 10, step: 0.1, value: guidance, 'aria-label': 'Guidance', dataset: { key: 'd-guidance-range' },
      oninput: () => { range.style.setProperty('--at', `${(range.value / 10) * 100}%`); gnum.value = range.value; },
      onchange: () => setDefaults({ guidance: Number(range.value) }),
    });
    range.style.setProperty('--at', `${(guidance / 10) * 100}%`);
    const gnum = numInput('guidance', df.guidance ?? famDefault('guidance'), { step: 0.1, integer: false, label: 'Guidance' });
    const select = (key, options, label) => {
      const value = df[key] ?? '';
      const list = value && !options.includes(value) ? [value, ...options] : options;
      const s = h('select.mc-select', {
        'aria-label': label, dataset: { key: `d-${key}` }, onchange: () => setDefaults({ [key]: s.value || null }),
      }, h('option', { value: '' }, 'engine default'), list.map((o) => h('option', { value: o, selected: o === value }, o)));
      return s;
    };
    const row = (label, control, keys) => h('div.mc-srow', h('span.mc-slabel', label), h('div.mc-scontrol', control), versus(keys));
    const advanced = ['flow_shift', 'clip_skip', 'sigmas', 'negative_prompt', ...(takesRefs() ? ['ref_image_args'] : [])];
    const editedAdv = fam ? advanced.filter((k) => !same(k === 'negative_prompt' ? df[k] || null : df[k], k === 'negative_prompt' ? famDefault(k) || null : famDefault(k))) : [];
    const sigmas = h('input.seed-input.text', {
      value: Array.isArray(df.sigmas) ? df.sigmas.join(', ') : df.sigmas ?? '', placeholder: 'none · a list, or a preset name', spellcheck: false,
      'aria-label': 'Sigmas', dataset: { key: 'd-sigmas' },
      onchange: () => {
        const text = sigmas.value.trim();
        const list = text.split(/[\s,]+/).filter(Boolean).map(Number);
        setDefaults({ sigmas: !text ? null : list.every(Number.isFinite) ? list : text });
      },
      onkeydown: enter,
    });
    const negative = h('textarea.edit-text.model-negative', {
      'aria-label': 'Negative prompt', placeholder: noCfg ? 'Has no effect while CFG is off' : 'What to keep out of every image',
      dataset: { key: 'd-negative' }, onchange: () => setDefaults({ negative_prompt: negative.value }),
    });
    negative.value = df.negative_prompt ?? '';
    const refArgs = takesRefs() && h('input.seed-input.text', {
      value: df.ref_image_args ?? '', placeholder: 'none · e.g. resize_before_vae=false', spellcheck: false,
      'aria-label': 'Reference image args', dataset: { key: 'd-ref-args' },
      onchange: () => setDefaults({ ref_image_args: refArgs.value.trim() || null }),
      onkeydown: enter,
    });
    return h('section.mc-section', { id: 'mc-sampling' },
      sectionHead('Sampling', 'Every image starts here. A message\'s own settings still win.', h('span.spacer'),
        h('button.btn.btn-ghost.mc-small-btn', { type: 'button', onclick: () => patch({ reset_defaults: true }, 'Back to the family\'s defaults') }, 'Reset to family')),
      h('div.mc-table',
        row('Size', [h('div.mc-aspects', aspects),
          // A size can't be empty: clearing it keeps the one it had.
          h('span.mc-size-pair', numInput('width', w, { step: 8, label: 'Width', className: 'mc-big', onset: (x) => (x ? setDefaults({ width: x }) : render()) }), '×',
            numInput('height', ht, { step: 8, label: 'Height', className: 'mc-big', onset: (x) => (x ? setDefaults({ height: x }) : render()) }))], ['width', 'height']),
        row('Steps', h('div.mc-stepper',
          h('button', { type: 'button', 'aria-label': 'One step fewer', onclick: () => stepBy(-1) }, '−'),
          numInput('steps', steps, { min: 1, label: 'Steps', className: 'mc-big' }),
          h('button', { type: 'button', 'aria-label': 'One step more', onclick: () => stepBy(1) }, '+')), ['steps']),
        hasGuidance() && row('Guidance', [range, gnum, h('span.mc-muted', 'Distilled guidance — how closely it follows the prompt.')], ['guidance']),
        row('CFG', [numInput('cfg_scale', cfg, { step: 0.1, integer: false, label: 'CFG', className: 'mc-big' }),
          noCfg && h('span.tag.tag-neutral.mc-off', 'OFF'),
          h('span.mc-muted', noCfg ? 'Distilled model — so no negative prompt either.' : 'How strongly it follows the prompt and avoids the negative one.')], ['cfg_scale']),
        hasImgCfg() && row('Image guidance', [numInput('img_cfg_scale', df.img_cfg_scale ?? famDefault('img_cfg_scale'), { step: 0.1, integer: false, min: 0, label: 'Image guidance', className: 'mc-big' }),
          h('span.mc-muted', 'Edits only: how closely the result keeps to the images it is given. Empty is 1.')], ['img_cfg_scale']),
        row('Sampler', [select('sampler', SAMPLERS, 'Sampler'), h('span.mc-muted', 'with'), select('scheduler', SCHEDULERS, 'Scheduler'),
          h('span.mc-muted', 'schedule')], ['sampler', 'scheduler']),
        h('details.mc-adv', {
          open: opened.has('advanced'), ontoggle: (e) => { if (e.target.open) opened.add('advanced'); else opened.delete('advanced'); },
        },
        h('summary', h('span.mc-summary-mark', '▸'), 'Advanced',
          h('span.mc-muted', `flow shift, CLIP skip, sigmas, negative prompt${takesRefs() ? ', reference images' : ''} · ${editedAdv.length ? `${plural(editedAdv.length, 'edit')} over the family's` : 'all from the family'}`)),
        h('div.mc-adv-body',
          h('div.model-fields',
            h('label.model-field', h('span.eyebrow', 'Flow shift'), h('span.seed-field', numInput('flow_shift', df.flow_shift, { step: 0.1, integer: false, label: 'Flow shift', className: 'seed-input' }))),
            h('label.model-field', h('span.eyebrow', 'CLIP skip'), h('span.seed-field', numInput('clip_skip', df.clip_skip, { min: 1, label: 'CLIP skip', className: 'seed-input' })))),
          h('label.model-field.wide', h('span.eyebrow', 'Sigmas'), h('span.seed-field', sigmas)),
          h('label.model-field.wide', h('span.eyebrow', 'Negative prompt'), negative),
          refArgs && h('label.model-field.wide', { title: "sd-server's ref_image_args for edits (stable-diffusion.cpp docs/edit.md)" },
            h('span.eyebrow', 'Reference image args'), h('span.seed-field', refArgs))))));
  }

  // --- 03 presets ------------------------------------------------------------------------

  function presetsSection() {
    const df = d.defaults;
    const names = Object.keys(d.presets);
    const rows = [['steps', 'Steps'], hasGuidance() && ['guidance', 'Guidance'], ['cfg_scale', 'CFG'],
      hasImgCfg() && ['img_cfg_scale', 'Image guidance'],
      (df.flow_shift != null || names.some((n) => d.presets[n].flow_shift != null)) && ['flow_shift', 'Flow shift'],
      ['sampler', 'Sampler'], ['size', 'Size'], ['loras', 'LoRAs']].filter(Boolean);
    const lorasText = (list) => (list == null ? null : list.length ? list.map((l) => `${fileName(l.path).replace(/\.[^.]+$/, '')} · ${l.weight}`).join(', ') : 'none');
    const show = (key, src) => {
      if (key === 'sampler') return src.sampler || src.scheduler ? [src.sampler ?? 'engine', src.scheduler].filter(Boolean).join(' · ') : null;
      if (key === 'size') return `${df.width} × ${df.height}`;
      if (key === 'loras') return lorasText(src.loras);
      if (key === 'cfg_scale' && src.cfg_scale != null && src === df && src.cfg_scale <= 1) return `${src.cfg_scale.toFixed(1)} · off`;
      const v = src[key];
      return v == null ? null : typeof v === 'number' && !Number.isInteger(v) ? String(v) : key === 'cfg_scale' || key === 'guidance' ? Number(v).toFixed(1) : String(v);
    };
    const setPreset = (name, values, word = null) => patch({ presets: { [name]: values } }, word);
    const cells = [];
    const head = (name) => {
      const p = d.presets[name];
      return h('button.mc-mhead.preset', {
        type: 'button', title: 'Describe, or delete',
        onclick: (e) => menu(e.currentTarget, [
          { label: 'Describe it…', icon: 'pencil', onclick: async () => {
            const text = await dialog({ title: `Describe ${name}`, body: 'Shown where it is picked.', input: { value: p.description ?? '', placeholder: 'e.g. 8-step draft' }, confirm: 'Save' });
            if (text != null) setPreset(name, { description: text.trim() });
          } },
          { label: `Delete ${name}`, icon: 'trash', danger: true, onclick: () => setPreset(name, null, `Deleted ${name}`) },
        ]),
      }, h('b', name), h('span', p.description || 'no description'));
    };
    cells.push(h('span.mc-mhead'), h('span.mc-mhead', h('b', 'Default'), h('span', 'every image')), names.map(head),
      h('button.mc-mhead.add', {
        type: 'button',
        onclick: async () => {
          const name = (await dialog({ title: 'New preset', body: 'Letters, digits, - and _.', input: { placeholder: 'e.g. lightning' }, confirm: 'Add' }))?.trim();
          if (name) setPreset(name, { description: '' }, `Added ${name}`);
        },
      }, '+ Preset'));
    for (const [key, label] of rows) {
      cells.push(h('span.mc-mlabel', label));
      cells.push(h('span.mc-mcell', { class: show(key, df) == null ? 'dim' : '' }, show(key, df) ?? 'engine'));
      for (const name of names) cells.push(presetCell(name, key, show));
      cells.push(h('span.mc-mcell.blank'));
    }
    const grid = h('div.mc-matrix', { style: { gridTemplateColumns: `minmax(110px, 180px) repeat(${names.length + 1}, minmax(120px, 1fr)) 120px` } }, cells);
    return h('section.mc-section', { id: 'mc-presets' },
      sectionHead('Presets', names.length ? 'Each inherits the sampling above; filled cells are what it changes. Click a cell to change it.'
        : 'None yet. A preset picks LoRAs with the steps and CFG they were made for.'),
      h('div.mc-matrix-scroll', grid),
      h('span.mc-muted.mc-foot-note', 'They belong to this model: an engine shows them while it runs it. Add a LoRA to one from the ',
        h('a', { href: '#/loras' }, 'LoRA library'), '.'));
  }

  function presetCell(name, key, show) {
    const p = d.presets[name];
    const df = d.defaults;
    const own = key === 'sampler' ? p.sampler != null || p.scheduler != null : key === 'size' ? false : p[key] != null;
    const text = own ? show(key, p) : show(key, df) ?? 'engine';
    if (key === 'size') return h('span.mc-mcell.dim', { title: 'Presets keep the model\'s size' }, text);
    const set = (values, word = null) => patch({ presets: { [name]: values } }, word);
    const onclick = (e) => {
      const anchor = e.currentTarget;
      if (key === 'loras') {
        menu(anchor, [
          { label: 'The model\'s LoRAs', checked: p.loras == null, onclick: () => set({ loras: null }) },
          { label: 'No LoRAs', checked: Array.isArray(p.loras) && !p.loras.length, onclick: () => set({ loras: [] }) },
          ...(p.loras ?? []).map((l) => ({ label: `Remove ${fileName(l.path)}`, icon: 'x', onclick: () => set({ loras: p.loras.filter((x) => x !== l) }) })),
          { label: 'Add one from the LoRA library', icon: 'plus', onclick: () => navigate('#/loras') },
        ]);
        return;
      }
      if (key === 'sampler') {
        menu(anchor, [
          { label: 'The model\'s sampler', checked: !own, onclick: () => set({ sampler: null, scheduler: null }) },
          ...SAMPLERS.map((s) => ({ label: s, checked: p.sampler === s, onclick: () => set({ sampler: s }) })),
        ]);
        return;
      }
      // Numbers: edited in place (kept across a re-render); empty goes back to the model's.
      editing = { name, key, draft: p[key] == null ? '' : String(p[key]) };
      render();
      const input = main.querySelector(`[data-key="${CSS.escape(`p-${name}-${key}`)}"]`);
      input?.focus();
      input?.select();
    };
    const label = `${name} · ${key === 'cfg_scale' ? 'CFG' : key === 'img_cfg_scale' ? 'image guidance' : key.replace('_', ' ')}`;
    if (editing?.name === name && editing.key === key) {
      const integer = key === 'steps';
      const commit = () => {
        const raw = editing.draft.trim();
        editing = null;
        const v = integer ? Number.parseInt(raw, 10) : Number(raw);
        const next = raw === '' ? null : Number.isFinite(v) ? v : p[key] ?? null;
        if (same(next, p[key])) render();
        else set({ [key]: next });
      };
      const input = h('input.mc-cell-input', {
        type: 'number', step: integer ? 1 : 0.1, value: editing.draft, placeholder: show(key, df) ?? 'engine', 'aria-label': label,
        dataset: { key: `p-${name}-${key}` },
        oninput: () => { if (editing) editing.draft = input.value; },
        onkeydown: (ev) => {
          if (ev.key === 'Enter') { ev.preventDefault(); commit(); }
          if (ev.key === 'Escape') { ev.preventDefault(); ev.stopPropagation(); editing = null; render(); }
        },
        // Only when the person leaves it, not when a re-render takes it away.
        onblur: () => { if (input.isConnected && editing?.name === name && editing.key === key) commit(); },
      });
      return h('span.mc-mcell.editing', input);
    }
    return h('button.mc-mcell.edit', {
      type: 'button', class: own ? 'own' : 'dim', 'aria-label': `${label}: ${text}${own ? '' : ' (the model\'s)'}`,
      title: own ? 'Click to change · empty: the model\'s' : 'Click to override', onclick,
    }, text);
  }

  // --- 04-06 folded: prompting, launch, notes & cover ---------------------------------------

  function fold(key, title, summaryText, body) {
    return h('details.mc-fold', {
      id: `mc-${key}`, open: opened.has(key),
      ontoggle: (e) => { if (e.target.open) opened.add(key); else opened.delete(key); },
    },
    h('summary', h('span.mc-fold-title', title), h('span.mc-muted', summaryText), icon('chevronRight', 16)),
    h('div.mc-fold-body', body));
  }

  function moreSection() {
    const args = Object.keys(d.launch.model_args).length;
    const launchText = [d.launch.vae_format ? `--vae-format ${d.launch.vae_format}` : 'Family\'s VAE format',
      args ? plural(args, 'model arg') : 'model args',
      d.launch.extra_args.length ? d.launch.extra_args.join(' ') : 'no extra arguments'].join(' · ').replace(/^(Family's VAE format) · model args/, '$1 and model args');
    const notesText = [d.notes ? 'Notes' : 'No notes', d.tags.length ? plural(d.tags.length, 'tag') : 'no tags',
      d.cover ? `cover from ${d.cover.source}` : 'cover comes from its first image'].join(' · ');
    const promptText = d.prompt_hint ? `Its own: ${d.prompt_hint.slice(0, 90)}${d.prompt_hint.length > 90 ? '…' : ''}`
      : d.family_prompt_hint ? `Family's note: ${d.family_prompt_hint.slice(0, 90)}${d.family_prompt_hint.length > 90 ? '…' : ''}` : 'No note yet';
    return h('section.mc-folds',
      fold('prompting', 'Prompting', promptText, [d.family_note && h('div.tray-cell', h('span.eyebrow', 'About this family'), h('span.tray-hint.wrap', d.family_note)), promptCell()]),
      fold('launch', 'Launch', launchText, launchCell()),
      fold('notes', 'Notes & cover', notesText, [h('div.cell-pair.mc-pair', notesCell(), tagsCell()), coverCell(), deleteCell()]));
  }

  function launchCell() {
    const fmt = h('select.seed-input', { 'aria-label': '--vae-format', dataset: { key: 'l-vae' }, onchange: () => patch({ launch: { vae_format: fmt.value } }) },
      VAE_FORMATS.map((f) => h('option', { value: f, selected: (d.launch.vae_format ?? '') === f }, f || 'its family\'s')));
    const args = h('input.seed-input.text', {
      value: Object.entries(d.launch.model_args).map(([k, v]) => `${k}=${v}`).join(','), placeholder: 'key=value,…', spellcheck: false,
      'aria-label': '--model-args', dataset: { key: 'l-args' },
      onchange: () => {
        const parsed = Object.fromEntries(args.value.split(',').map((x) => x.trim()).filter(Boolean)
          .map((x) => { const [k, ...v] = x.split('='); return [k.trim(), v.join('=').trim() || 'true']; }));
        patch({ launch: { model_args: parsed } });
      },
      onkeydown: enter,
    });
    const extra = h('input.seed-input.text', {
      value: formatArgv(d.launch.extra_args), placeholder: 'e.g. --offload-to-cpu --vae-tiling', spellcheck: false, 'aria-label': 'Extra arguments',
      dataset: { key: 'l-extra' },
      onchange: () => {
        try {
          const extra_args = parseArgv(extra.value);
          extra.setCustomValidity('');
          patch({ launch: { extra_args } });
        } catch (err) {
          extra.setCustomValidity(err.message);
          extra.reportValidity();
        }
      },
      onkeydown: enter,
    });
    return h('div.tray-cell',
      h('label.model-field', h('span.mono-meta', '--vae-format'), h('span.seed-field', fmt)),
      h('label.model-field.wide', h('span.mono-meta', '--model-args'), h('span.seed-field', args)),
      h('label.model-field.wide', h('span.mono-meta', 'more arguments'), h('span.seed-field', extra)),
      h('span.tray-hint.wrap', 'Added to the engine\'s own when it starts with this model; its family\'s recommended model args come first.'));
  }

  function promptCell() {
    const area = h('textarea.edit-text.lib-notes', {
      'aria-label': 'How to prompt it', placeholder: d.family_prompt_hint ?? 'Natural language: a detailed description of the image.',
      dataset: { key: 'prompt' }, onchange: () => patch({ prompt_hint: area.value }),
    });
    area.value = d.prompt_hint ?? '';
    return h('div.tray-cell', h('span.eyebrow', 'How to prompt it'), area,
      h('span.tray-hint.wrap', 'Told to the chat model when it writes prompts for this model. Empty: the family\'s note.'));
  }

  function notesCell() {
    const area = h('textarea.edit-text.lib-notes', {
      'aria-label': 'Notes', placeholder: 'What it is good at, settings that work…', dataset: { key: 'notes' }, onchange: () => patch({ notes: area.value }),
    });
    area.value = d.notes;
    return h('div.tray-cell', h('span.eyebrow', 'Notes'), area);
  }

  function tagsCell() {
    const toggle = (t) => patch({ tags: d.tags.includes(t) ? d.tags.filter((x) => x !== t) : [...d.tags, t] });
    const all = [...new Set([...d.tags, ...(lib?.tags ?? []), 'Photo', 'Art', 'Fast'])];
    return h('div.tray-cell', h('span.eyebrow', 'Tags'),
      h('div.tray-opts', all.map((t) => h('button.pick.lib-tag', {
        type: 'button', class: d.tags.includes(t) ? 'on' : '', 'aria-pressed': String(d.tags.includes(t)), onclick: () => toggle(t),
      }, t)),
      h('button.pick.lib-tag.new', {
        type: 'button',
        onclick: async () => {
          const t = (await dialog({ title: 'New tag', input: { placeholder: 'Tag' }, confirm: 'Add' }))?.trim();
          if (t) patch({ tags: [...d.tags, t] });
        },
      }, '+ New')));
  }

  function coverCell() {
    const frame = h('div.tile-img.lib-cover', { style: { aspectRatio: d.cover?.width ? `${d.cover.width} / ${d.cover.height}` : '4 / 5' } });
    if (d.cover) {
      const img = h('img.tile-pic', { alt: `${d.name} cover` });
      setImageUrl(img, d.cover.url);
      frame.append(img);
    } else {
      frame.append(h('span.tile-desc', 'no cover yet'));
    }
    const fromHub = d.components.some((c) => c.source);
    const reel = d.recent_images.slice(0, 10).map((a) => {
      const pic = h('img.reel-pic', { alt: '' });
      setImage(pic, a);
      return h('button.reel-item.lib-reel', {
        type: 'button', class: d.cover?.artifact_id === a.id ? 'on' : '', title: `Use ${a.label} as the cover`,
        onclick: () => save(() => api.setModelCover(id, { artifact_id: a.id }), 'Cover set'),
      }, h('span.reel-thumb', pic), a.label);
    });
    return h('div.tray-cell.mc-cover',
      h('div.blueprint.lib-cover-frame.mc-cover-frame', corners(), frame),
      h('div.mc-cover-side',
        h('div.log-head', h('span.eyebrow', 'Cover'), h('span.spacer'), d.cover && h('span.mono-meta', d.cover.source),
          d.cover && iconButton('x', 'Remove the cover', () => save(() => api.clearModelCover(id)), 14)),
        h('div.lib-cover-actions',
          h('button.mini', {
            type: 'button', disabled: !fromHub, title: fromHub ? 'The first image on its Hugging Face card' : 'None of its files came from the Hub',
            onclick: () => save(() => api.setModelCover(id, { hub: true }), 'Cover set'),
          }, 'Use Hub thumbnail')),
        h('span.tray-hint', d.recent_images.length ? `Images made with it, newest first (${plural(d.uses, 'image')}):` : 'Images made with it show here; the first becomes the cover.'),
        reel.length > 0 && h('div.lib-reel-grid', reel)));
  }

  function deleteCell() {
    const users = d.engines.map((u) => u.engine);
    return h('div.tray-cell.row-cell',
      h('div.cell-grow', h('span.eyebrow', 'Remove'),
        h('span.mono-meta.wrap', users.length ? `${users.join(', ')} runs it: give ${users.length > 1 ? 'them' : 'it'} another model first.`
          : 'Forgets the card. Its files stay where they are; past images keep their record.')),
      h('button.mini', {
        type: 'button', disabled: users.length > 0,
        onclick: async () => {
          if (!(await dialog({ title: `Remove ${d.name}?`, body: 'The card goes; its files stay where they are.', confirm: 'Remove', danger: true }))) return;
          try {
            await api.deleteModel(id);
            loadModels();
            toast(`Removed ${d.name}`);
            navigate('#/models', { replace: true });
          } catch (err) {
            toast(err.message, { error: true });
          }
        },
      }, icon('trash', 13), 'Remove card…'));
  }

  load();
  return {
    el,
    destroy() {
      gone = true;
      clearTimeout(poll);
      status.destroy();
      strip.destroy();
    },
  };
}
