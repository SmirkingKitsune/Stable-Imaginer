// Agents (design "Agent Mockups" 1a, 1b): a chat model plus a persona.
//
// #/agents: a tile per agent, its portrait as the cover, mirroring the model library: its
// model, quant and size, how much it has been used, whether it is ready, and the engine it
// runs on (● running, ○ not). Filters (pinned, vision, tools, running) and a table view.
//
// #/agents/<id>: the portrait, name and pin over a plate with what it runs on and the one
// next action (New conversation, Use on the engine, Start, Stop), then a section rail:
// its portrait (from the Hub uploader, a self-portrait it paints with set_portrait, or an
// upload), its persona (name, system prompt, what the viewer's thought bubble shows), the
// tools it may call, its files, and its sampling. Every edit saves as you leave its field.
//
// Adding one takes three steps, like importing a model: where its model is (a GGUF here, or
// Hugging Face, where only models that see images are listed), whether it can be an agent
// (it must see images: a vision projector; it should call tools: a chat template that takes
// them), and who it is (name, engine, system prompt).

import { api, setImageUrl } from './api.js';
import { connectionStrip, dialog, menu, modal, modelName, statusLine, toast } from './chrome.js';
import { append, corners, h, icon, iconButton, clear } from './dom.js';
import { ago, plural } from './format.js';
import { agentTarget, hubDialog, looksLikeRef } from './hub.js';
import { lastPlace, loadAgents, loadBackends, imageEngine } from './store.js';

const VIEW_KEY = 'imaginer.agents';
const FILTERS = [['all', 'All'], ['pinned', 'Pinned'], ['vision', 'Vision'], ['tools', 'Tools'], ['running', 'Running']];
const SAMPLING = [
  ['temperature', 'Temperature', 0.05], ['top_p', 'Top-p', 0.05], ['top_k', 'Top-k', 1],
  ['min_p', 'Min-p', 0.01], ['repeat_penalty', 'Repeat penalty', 0.05], ['max_tokens', 'Max tokens', 64],
];
const POLL_MS = 2000;
const bytes = (n) => (n == null || !n ? '' : n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB`
  : n >= 1e6 ? `${Math.round(n / 1e6)} MB` : `${Math.max(1, Math.round(n / 1e3))} KB`);
const fileName = (path) => (path ? path.split(/[\\/]/).pop() : '');
const enter = (e) => { if (e.key === 'Enter' && e.target.tagName === 'INPUT') e.target.blur(); };
const hasFiles = (e) => [...(e.dataTransfer?.types ?? [])].includes('Files');

function remembered() {
  try {
    return { view: 'tiles', filter: 'all', ...JSON.parse(localStorage.getItem(VIEW_KEY) || '{}') };
  } catch {
    return { view: 'tiles', filter: 'all' };
  }
}

function remember(prefs) {
  try {
    localStorage.setItem(VIEW_KEY, JSON.stringify(prefs));
  } catch {
    // Private mode: kept for this page only.
  }
}

// --- portraits -------------------------------------------------------------------------------

/**
 * An agent's portrait as a square of `size` px: its image, or its monogram on the accent
 * tint. framed: a blueprint frame with registration marks; origin: the "HUB · QWEN" chip.
 */
export function portrait(agent, { size = 26, framed = false, origin = false, cls = '' } = {}) {
  const p = agent?.portrait;
  const el = h(framed ? 'div.portrait.blueprint' : 'span.portrait', {
    class: cls,
    // size 0: it fills its box (a tile's cover), sized by CSS.
    style: size ? { width: `${size}px`, height: `${size}px`, fontSize: `${Math.round(size * 0.58)}px` } : null,
    'aria-hidden': 'true',
  }, framed && corners());
  if (p?.url) {
    const img = h('img.portrait-pic', { alt: '', decoding: 'async' });
    setImageUrl(img, p.url);
    el.append(img);
  } else {
    el.append(h('span.portrait-mono', p?.monogram ?? '?'));
  }
  if (origin && p?.origin) el.append(h('span.tile-origin.portrait-origin', p.origin));
  return el;
}

/** An image of the agent's that isn't necessarily the one in use (a portrait option). */
function portraitImage(image, size) {
  const el = h('span.portrait', { style: size ? { width: `${size}px`, height: `${size}px` } : null });
  const img = h('img.portrait-pic', { alt: '', loading: 'lazy' });
  setImageUrl(img, image.url);
  el.append(img);
  return el;
}

/** READY, RUNNING, NEEDS MMPROJ or CAN'T LOAD. */
export function agentState(a) {
  if (a.working) return h('span.state.run', 'RUNNING');
  if (a.problems.length) return h('span.state.fail', { title: a.problems.join('\n') }, 'CAN\'T LOAD');
  if (a.missing.length) return h('span.state.wait', `NEEDS ${a.missing.map((m) => m.replace('_', ' ')).join(' · ').toUpperCase()}`);
  return h('span.state.done', 'READY');
}

/** "● nas · llama": its engine, filled when it is up and running this agent. */
function engineWord(a) {
  if (!a.use) return a.engine ? `○ ${a.engine}` : 'no engine';
  return `${a.use.running ? '●' : '○'} ${a.use.link ? `${a.use.link} · ` : ''}${a.use.engine}`;
}

function paintingLine(p) {
  if (!p) return null;
  if (p.status === 'running' && p.total_steps) return `PAINTING · ${p.step}/${p.total_steps}`;
  return p.status === 'queued' ? 'PAINTING · QUEUED' : 'PAINTING · WRITING';
}

// --- the library (1a) ------------------------------------------------------------------------

export function agentsScreen({ navigate }) {
  let lib = null;
  let error = null;
  let query = '';
  let gone = false;
  let poll = null;
  const prefs = remembered();

  const status = statusLine();
  const meta = h('span.topbar-meta');
  const search = h('input.search-input', {
    type: 'search', placeholder: 'Search names, models, uploaders', 'aria-label': 'Search agents',
    oninput: () => { query = search.value.trim().toLowerCase(); render(); },
    onkeydown: (e) => { if (e.key === 'Escape') { search.value = ''; query = ''; render(); search.blur(); } },
  });
  const add = h('button.btn.btn-primary.blueprint.engine-go', { type: 'button', onclick: () => addAgent(lib, navigate) },
    corners(), icon('plus', 14), h('span.only-wide', 'Add an agent'));
  const header = h('header.topbar.sub',
    iconButton('back', 'Back', () => navigate(lastPlace())),
    h('div.topbar-heading', h('span.topbar-title', 'Agents'), meta),
    h('span.spacer'),
    h('label.search.lib-search', icon('search', 15), search, h('kbd', '/')),
    h('div.only-wide', status.el),
    add);
  const body = h('div.lib-body');
  const scroller = h('div.gallery-scroll.lib-scroll', body);
  const strip = connectionStrip();
  const el = h('div.screen.loras.models.agents', header, strip.el, scroller);

  async function refresh() {
    try {
      lib = await loadAgents();
      if (!lib) throw new Error('the agents could not be loaded');
      error = null;
    } catch (err) {
      error = err.message;
    }
    if (gone) return;
    render();
    clearTimeout(poll);
    if (lib?.items.some((a) => a.working || a.portrait.painting)) poll = setTimeout(refresh, POLL_MS);
  }

  const searched = (a) => !query || [a.name, a.model, a.family, a.engine, a.portrait.hub_owner, a.source?.repo, ...a.tags]
    .filter(Boolean).join(' ').toLowerCase().includes(query);
  const filtered = (a, key) => key === 'all' || (key === 'pinned' && a.pinned) || (key === 'vision' && a.vision)
    || (key === 'tools' && a.tools) || (key === 'running' && a.use?.running);

  function render() {
    clear(body);
    if (!lib) {
      body.append(h('div.notice', error ?? 'Reading the agents…'));
      return;
    }
    const running = lib.items.filter((a) => a.use?.running).length;
    const linked = new Map();
    for (const a of lib.items) if (a.use?.link) linked.set(a.use.link, (linked.get(a.use.link) ?? 0) + 1);
    meta.textContent = [plural(lib.items.length, 'agent'), `${running} running`,
      ...[...linked].map(([link, n]) => `${n} on ${link}`)].join(' · ');
    if (!lib.items.length) {
      body.append(h('div.notice.lib-empty',
        h('span.eyebrow.accent', 'No agents yet'),
        h('span', 'An agent is a chat model plus a persona: a name, a system prompt and a portrait. Add one from a GGUF on this machine or from Hugging Face.'),
        h('div.lib-empty-actions', h('button.btn-line', { type: 'button', onclick: () => addAgent(lib, navigate) }, icon('plus', 15), 'Add an agent'))));
      return;
    }
    const seg = (key, word) => {
      const n = lib.items.filter((a) => filtered(a, key)).length;
      return h('button.pick', {
        type: 'button', class: prefs.filter === key ? 'on' : '', 'aria-pressed': String(prefs.filter === key),
        onclick: () => { prefs.filter = key; remember(prefs); render(); },
      }, word, ' ', h('span.mono-meta', String(n)));
    };
    const view = (key, ico, word) => h('button.pick', {
      type: 'button', class: prefs.view === key ? 'on' : '', 'aria-pressed': String(prefs.view === key),
      onclick: () => { prefs.view = key; remember(prefs); render(); },
    }, icon(ico, 14), word);
    const list = lib.items.filter((a) => searched(a) && filtered(a, prefs.filter));
    append(body, [
      h('div.lib-bar',
        h('div.tray-seg.lib-filters', { role: 'group', 'aria-label': 'Show' }, FILTERS.map(([k, w]) => seg(k, w))),
        h('span.spacer'),
        h('div.tray-seg.small.lib-views', { role: 'group', 'aria-label': 'View' }, view('tiles', 'grid', 'Tiles'), view('table', 'list', 'Table'))),
      prefs.view === 'table' ? table(list) : h('div.model-tiles.agent-tiles', list.map(tile)),
      !list.length && h('div.notice', query ? `Nothing matches “${query}”.` : 'No agent fits this filter.'),
    ]);
  }

  function tile(a) {
    const painting = a.portrait.painting;
    const cover = painting
      ? h('div.tile-img.agent-cover.live', h('span.tile-desc', 'painting its portrait…'))
      : h('div.tile-img.agent-cover', portrait(a, { size: 0, cls: 'fill' }),
        a.portrait.origin && h('span.tile-origin', a.portrait.origin));
    const line = [a.quant, bytes(a.size_bytes), a.turns ? plural(a.turns, 'turn') : 'unused',
      a.use?.kind === 'linked' && a.use.link && `on ${a.use.link}`, a.last_used_at && ago(a.last_used_at)].filter(Boolean).join(' · ');
    return h('a.tile.result.lib-tile.model-tile.agent-tile', { href: `#/agents/${a.id}`, 'aria-label': a.name },
      cover,
      painting && h('div.tile-run',
        h('span.tile-chip', paintingLine(painting)),
        h('div.tile-progress', h('i', { style: { width: `${painting.total_steps ? Math.round((100 * painting.step) / painting.total_steps) : 0}%` } }))),
      h('div.tile-foot.lib-foot',
        h('div.tile-foot-text',
          h('span.tile-title', a.pinned ? [icon('pin', 12), ' ', a.name] : a.name, a.default && h('span.agent-default', 'default')),
          h('span.mono-meta', line),
          h('span.model-tile-state', agentState(a), h('span.mono-meta', engineWord(a))))));
  }

  function table(list) {
    const row = (a) => h('a.agent-row', { href: `#/agents/${a.id}` },
      portrait(a, { size: 32 }),
      h('span.agent-row-name', h('b', a.pinned ? [icon('pin', 12), ' ', a.name] : a.name),
        h('span.mono-meta', a.model ? fileName(a.model) : 'whatever its engine runs')),
      h('span.mono-meta', engineWord(a)),
      agentState(a),
      h('span.mono-meta.right', a.turns ? plural(a.turns, 'turn') : 'unused'),
      h('span.mono-meta.right', a.last_used_at ? ago(a.last_used_at) : '—'));
    return h('div.agent-table',
      h('div.agent-row.head', h('span'), h('span.eyebrow', 'Agent'), h('span.eyebrow', 'Engine'), h('span.eyebrow', 'State'),
        h('span.eyebrow.right', 'Turns'), h('span.eyebrow.right', 'Last used')),
      list.map(row));
  }

  refresh();
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

// --- adding an agent -------------------------------------------------------------------------

/** What a Hub reference downloads into an agent: its model (by quant) and vision projector. */
async function hubFiles(ref) {
  const r = await api.hubRepo(ref);
  const quant = (ref.split(':')[1] ?? '').toLowerCase();
  const models = r.variants.filter((v) => v.component === 'model' && v.format === 'gguf');
  const model = models.find((v) => quant && v.quant?.toLowerCase() === quant)
    ?? models.find((v) => /q4_k_m/i.test(v.quant ?? '')) ?? models[0] ?? null;
  const projectors = r.variants.filter((v) => v.component === 'mmproj');
  const mmproj = projectors.find((v) => /f16/i.test(v.quant ?? v.name)) ?? projectors[0] ?? null;
  return { repo: r.id, sha: r.sha ?? null, model, mmproj };
}

/** Download a Hub reference's model (and its vision projector) into an agent. */
async function downloadInto(agentId, ref, keys = ['model', 'mmproj']) {
  const found = await hubFiles(ref);
  let started = 0;
  for (const key of keys) {
    const v = found[key];
    if (!v) continue;
    if (v.local) {
      await api.updateAgent(agentId, { files: { [key]: v.local.path } });
      continue;
    }
    await api.hubDownload({ repo: found.repo, revision: found.sha ?? undefined, variant: v.id, agent: agentId, key });
    started += 1;
  }
  if (!found.model && keys.includes('model')) throw new Error(`${found.repo} has no GGUF model`);
  return started;
}

/** Make an agent in three steps, like importing a model: where its model is, whether it can
 * be an agent (it must see images; it should call tools), and who it is. */
function addAgent(lib, navigate) {
  const engines = lib?.engines ?? [];
  const def = engines.find((e) => e.default) ?? engines[0];
  const elsewhere = engines.filter((e) => !e.launchable);
  const st = {
    step: 1,
    text: '', // the path field
    source: null, // {kind: 'path'|'ref'|'engine', text, name}
    note: null, // {text, warn, title} under the drop zone
    found: [], // paths a dropped file's name matched
    busy: false,
    detected: null, // what /v1/agents/detect said
    model: null, // the model option picked (its id)
    projector: null, // the projector option picked (its id)
    template: '', // a chat template file replacing the model's own
    all: new Set(), // choice lists shown in full
    name: '',
    named: false,
    engine: def?.name ?? '',
    prompt: '',
  };
  const d = () => st.detected;
  const option = (list, id) => list?.find((o) => o.id === id) ?? null;
  const pickedModel = () => option(d()?.models, st.model);
  const pickedProjector = () => option(d()?.projectors, st.projector);
  const canMake = () => !!d()?.fit.allowed && (d().kind === 'engine' || (!!pickedModel() && !!pickedProjector()));
  const isRef = (value) => looksLikeRef(value)
    || (/^[\w.-]+\/[\w.-]+$/.test(value) && !/^[A-Za-z]:/.test(value) && !value.startsWith('/'));

  let box;
  modal((close) => {
    box = h('div.dialog.blueprint.mi.ag-wizard', { role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Add an agent' });
    box.close = close;
    render();
    return box;
  });

  function render() {
    if (!box) return;
    keepFocus(box, () => {
      clear(box).append(...corners());
      const body = st.step === 1 ? sourceStep() : st.step === 2 ? checkStep() : personaStep();
      append(box, [steps(), h('div.mi-main', body), foot()]);
    });
  }

  // --- the step bar ------------------------------------------------------------------------

  function fitWords() {
    const f = d()?.fit;
    if (!f) return '';
    const sees = f.vision === false ? 'Blind' : f.vision ? 'Sees images' : 'Vision unknown';
    const calls = f.tools ? 'calls tools' : f.tools === false ? 'no tools' : 'tools unknown';
    return `${sees} · ${calls}`;
  }

  function steps() {
    const cells = [
      { word: 'Source', ask: 'Where is the model?', done: st.source?.name ?? '', mono: st.source?.kind !== 'engine' },
      { word: 'Check', ask: 'Can it see and call tools?', done: fitWords() },
      { word: 'Persona', ask: 'Who is it?' },
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
      const other = elsewhere.find((e) => e.default) ?? elsewhere[0];
      return h('div.mi-foot',
        other && h('span.mi-foot-text', 'Or ', h('button.link-btn', { type: 'button', onclick: () => fromEngine(other.name) },
          `use what ${other.name} runs`), ', an engine started elsewhere.'),
        h('span.spacer'), cancel,
        primary(st.busy ? 'Checking…' : 'Next: check it', () => next(), !st.text.trim()));
    }
    if (st.step === 2) {
      return h('div.mi-foot', back(1),
        h('span.mi-foot-text.mi-status', { class: canMake() ? 'ok' : '' }, checkStatus()),
        cancel,
        primary(st.busy ? 'Checking…' : 'Next: persona', () => { st.step = 3; render(); }, !canMake()));
    }
    const s = summary();
    return h('div.mi-foot', back(2),
      h('span.mi-foot-text.mi-status', { class: s.ready ? 'ok' : '' }, s.text),
      primary(st.busy ? 'Making it…' : s.button, () => create(), !canMake()));
  }

  function checkStatus() {
    const f = d()?.fit;
    if (!f) return '';
    if (!f.allowed) return 'It can’t be an agent';
    if (d().kind !== 'engine' && !pickedProjector()) return 'Pick its vision projector';
    return f.tools === false ? 'It can be an agent, but makes no images itself' : 'It can be an agent';
  }

  // --- 01 source -----------------------------------------------------------------------------

  function sourceStep() {
    const picker = h('input', {
      type: 'file', accept: '.gguf', hidden: true,
      onchange: () => { if (picker.files[0]) dropped(picker.files[0]); },
    });
    const zone = h('div.mi-drop', {
      tabindex: '0', role: 'button', 'aria-label': 'Drop a GGUF model here, or choose one',
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
    h('span.mi-drop-title', 'Drop a chat model’s .gguf'),
    h('span.mi-drop-sub', 'Its vision projector (mmproj) is looked for beside it.'));
    const path = h('input.seed-input.path-input', {
      type: 'text', value: st.text, spellcheck: false, placeholder: 'or paste a path — D:\\models\\…\\model-Q4_K_M.gguf', 'aria-label': 'A model file on this machine',
      dataset: { key: 'ag-path' },
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
        h('span.mi-copy', 'Find one on Hugging Face. Only models that see images are listed, and those that call tools come first.'),
        h('button.btn.btn-secondary.mi-btn.mi-hub-btn', { type: 'button', onclick: searchHub }, icon('search', 14), 'Search Hugging Face'),
        h('span.spacer'),
        h('span.mi-small', 'Its model and projector download into the new agent.')));
  }

  function syncNext() {
    const go = box.querySelector('.mi-go');
    if (go) go.disabled = !st.text.trim() || st.busy;
  }

  async function dropped(file) {
    if (!/\.gguf$/i.test(file.name)) {
      st.note = { warn: true, text: `${file.name} isn't a GGUF: llama-server loads .gguf models.` };
      render();
      return;
    }
    st.busy = true;
    st.found = [];
    st.note = { text: `Looking for ${file.name} on this machine…` };
    render();
    let res = null;
    try {
      res = await api.locateAgentModel({ name: file.name, size_bytes: file.size });
    } catch (err) {
      st.note = { warn: true, text: err.message };
    }
    st.busy = false;
    if (res?.paths.length === 1) {
      st.text = res.paths[0];
      st.note = null;
      next();
      return;
    }
    if (res) {
      st.found = res.paths;
      if (res.paths.length) st.text = res.paths[0];
      st.note = res.paths.length
        ? { text: `${res.paths.length} files here are called ${file.name}. Pick one, then Next.` }
        : { warn: true, text: `${file.name} isn't in the model folders here. Paste its full path below.`, title: `Looked in:\n${res.searched.join('\n')}` };
    }
    render();
    if (!res?.paths.length) box.querySelector('[data-key="ag-path"]')?.focus();
  }

  function searchHub() {
    hubDialog(agentTarget(), {
      onPick: (p) => {
        st.text = p.ref;
        next();
      },
    });
  }

  /** Check what the source field names, and go on to the check. */
  async function next() {
    const value = st.text.trim();
    if (!value || st.busy) return;
    const ref = isRef(value);
    st.source = { kind: ref ? 'ref' : 'path', text: value, name: ref ? value.replace(/^hf:\/\//, '') : fileName(value) };
    st.template = '';
    await check(ref ? { ref: value } : { path: value }, { fresh: true });
  }

  async function fromEngine(engine) {
    st.source = { kind: 'engine', text: engine, name: `What ${engine} runs` };
    st.template = '';
    st.engine = engine;
    await check({ engine }, { fresh: true });
  }

  /** Ask the server; fresh: a new source (its picks and name start over). */
  async function check(body, { fresh = false } = {}) {
    st.busy = true;
    render();
    try {
      const found = await api.detectAgent(body);
      // A file here checked again keeps the models its folder offered (a dropped projector's).
      const was = st.detected;
      if (!fresh && was?.kind === 'path' && found.kind === 'path' && was.models.length > found.models.length) found.models = was.models;
      const keep = (list, id) => (!fresh && list.some((o) => o.id === id) ? id : list[0]?.id ?? null);
      st.model = keep(found.models, st.model);
      st.projector = keep(found.projectors, st.projector);
      st.detected = found;
      // A dropped projector: the source is named after the model picked beside it.
      if (fresh && found.kind === 'path' && found.models[0]) st.source.name = found.models[0].name;
      if (fresh) st.all.clear();
      if (!st.named) st.name = found.name;
      st.step = 2;
    } catch (err) {
      toast(err.message, { error: true });
    }
    st.busy = false;
    render();
  }

  /** Check again with the picks so far (a file here: its model, projector and template). */
  function recheck(changes = {}) {
    Object.assign(st, changes);
    const cur = d();
    if (cur.kind === 'path') {
      check({ path: pickedModel()?.id ?? st.source.text, mmproj: pickedProjector()?.id ?? null, chat_template: st.template || null });
    } else if (cur.kind === 'ref') {
      check({ ref: st.source.text, chat_template: st.template || null });
    } else render();
  }

  // --- 02 check ------------------------------------------------------------------------------

  function checkStep() {
    const cur = d();
    if (!cur) return h('div.ag-check', h('span.mi-copy.muted', 'Checking…'));
    const f = cur.fit;
    const model = pickedModel();
    const facts = [f.params, f.architecture, f.context_length && `${Math.round(f.context_length / 1024)}k context`,
      cur.repo && cur.kind === 'path' ? `from ${cur.repo}` : null].filter(Boolean).join(' · ');
    const title = cur.kind === 'ref' ? cur.repo : cur.kind === 'engine' ? (cur.models[0]?.name ?? st.source.name) : (model?.name ?? st.source.name);
    const other = f.problems.filter((p) => !/vision projector|see images/.test(p));
    return h('div.ag-check',
      h('div.ag-model', h('b', title), facts && h('span.mono-meta', facts)),
      h('div.ag-reqs', visionRow(f), toolsRow(f)),
      other.map((p) => h('span.mi-note.warn', p)),
      f.notes.map((n) => h('span.mi-note', n)),
      cur.kind !== 'engine' && cur.models.length > 1 && choices('model', 'Model file', cur.models, st.model, (id) => {
        if (cur.kind === 'path') recheck({ model: id });
        else { st.model = id; render(); }
      }),
      cur.kind !== 'engine' && (cur.projectors.length > 0 || cur.kind === 'path') && choices('projector', 'Vision projector', cur.projectors, st.projector,
        (id) => {
          if (cur.kind === 'path') recheck({ projector: id });
          else { st.projector = id; render(); }
        },
        cur.kind === 'path' ? () => typeProjector() : null));
  }

  function requirement({ state, title, flag, text, action = null }) {
    const mark = state === 'ok' ? icon('check', 14) : state === 'bad' ? icon('x', 14) : h('span', '?');
    return h('div.ag-req', { class: state },
      h('span.mc-mark', { class: state === 'ok' ? 'ok' : state === 'bad' ? 'bad' : 'dashed' }, mark),
      h('span.ag-req-text', h('b', title, h('span.mono-meta', flag)), h('span', text), action),
      h('span.tag.ag-req-tag', { class: state === 'ok' ? 'tag-accent' : 'tag-outline' }, flag === '--mmproj' ? 'MUST' : 'SHOULD'));
  }

  function visionRow(f) {
    const cur = d();
    const p = pickedProjector();
    let text;
    if (cur.kind === 'engine') {
      text = f.vision ? `${st.source.text} reports vision: it was started with a projector.`
        : f.vision === false ? f.problems.find((x) => /see images/.test(x)) : `${st.source.text} isn't answering, so whether it sees images is unknown.`;
    } else if (p) {
      text = `With ${p.name}: it reads the images it makes and the ones you give it.`;
    } else {
      text = f.problems.find((x) => /vision projector/.test(x)) ?? 'Pick its vision projector below.';
    }
    let state = p ? 'ok' : 'bad';
    if (cur.kind === 'engine') state = f.vision ? 'ok' : f.vision === false ? 'bad' : 'unknown';
    return requirement({ state, title: 'Sees images', flag: '--mmproj', text });
  }

  function toolsRow(f) {
    const cur = d();
    const fix = cur.kind !== 'engine' && h('div.ag-req-actions',
      h('button.mc-link', { type: 'button', onclick: () => typeTemplate() }, st.template ? 'Change the chat template file' : 'Use a chat template file that takes tools…'),
      st.template && h('button.mc-link', { type: 'button', onclick: () => recheck({ template: '' }) }, 'Use the model’s own'));
    const own = st.template ? `${fileName(st.template)} takes` : 'Its chat template takes';
    const text = f.tools ? `${own} tools: it can make and edit images itself (generate_image, edit_image).`
      : f.tools === false ? `${st.template ? `${fileName(st.template)} takes` : 'Its chat template takes'} no tools: it can talk about images but can't make them. A model that calls tools, or a chat template that takes them, can.`
        : cur.kind === 'engine' ? 'Unknown until the engine answers.' : 'There is no chat template to read here; llama-server says once it runs.';
    return requirement({ state: f.tools ? 'ok' : f.tools === false ? 'bad' : 'unknown', title: 'Calls tools', flag: 'chat template', text, action: (f.tools !== true || st.template) && fix });
  }

  function choices(key, word, list, current, pick, other = null) {
    const all = st.all.has(key);
    const shown = all ? list : list.slice(0, 6);
    const rows = shown.map((o, i) => {
      const on = o.id === current;
      const where = o.local ? (d().kind === 'ref' ? 'On disk · used as is' : o.local) : 'Downloads into the agent';
      return h('button.mi-choice', { type: 'button', role: 'radio', 'aria-checked': String(on), class: on ? 'on' : '', onclick: () => !on && pick(o.id) },
        h('span.mi-radio'),
        h('span.mi-choice-text', h('b', { title: o.id }, o.name, o.shards > 1 ? ` · ${o.shards} shards` : ''), h('span', where)),
        h('span.mi-choice-size', bytes(o.size_bytes)),
        i === 0 ? h('span.tag.tag-accent.mi-best', 'BEST') : h('span'));
    });
    return h('div.mi-choices',
      h('div.mi-choices-head', h('b', word), h('span.mono-meta', key === 'projector' ? '--mmproj' : '-m')),
      h('div.mi-choice-list', { role: 'radiogroup', 'aria-label': word }, rows,
        list.length > 6 && h('button.mi-choice.quiet', { type: 'button', onclick: () => { if (all) st.all.delete(key); else st.all.add(key); render(); } },
          h('span'), h('span.mi-choice-text', h('b', all ? 'Show fewer' : `Show all ${list.length}`)), h('span'), h('span')),
        other && h('button.mi-choice.quiet', { type: 'button', onclick: other },
          h('span.mi-radio.square'), h('span.mi-choice-text', h('b', 'Another file on this machine…'), h('span', 'A path')), h('span'), h('span'))));
  }

  async function typeProjector() {
    const value = (await dialog({
      title: 'Vision projector', body: 'An mmproj .gguf on this machine, made for this model.',
      input: { value: pickedProjector()?.id ?? '', placeholder: 'D:\\models\\…\\mmproj-F16.gguf' }, confirm: 'Use it',
    }))?.trim();
    if (!value) return;
    st.projector = value;
    check({ path: pickedModel()?.id ?? st.source.text, mmproj: value, chat_template: st.template || null });
  }

  async function typeTemplate() {
    const value = (await dialog({
      title: 'Chat template file',
      body: 'A Jinja template (.jinja) for --chat-template-file. It replaces the one in the model; one that takes tools lets it call them.',
      input: { value: st.template, placeholder: 'D:\\templates\\…\\tool-use.jinja' }, confirm: 'Use it',
    }))?.trim();
    if (value == null || value === st.template) return;
    recheck({ template: value });
  }

  // --- 03 persona ----------------------------------------------------------------------------

  function personaStep() {
    const cur = d();
    const name = h('input.seed-input.text', {
      value: st.name, placeholder: cur?.name ?? 'Its name', 'aria-label': 'Name', spellcheck: false, dataset: { key: 'ag-name' },
      oninput: () => { st.name = name.value; st.named = true; },
      onkeydown: (e) => { if (e.key === 'Enter') { e.preventDefault(); create(); } },
    });
    const engine = h('select.seed-input', {
      'aria-label': 'Runs on', dataset: { key: 'ag-engine' }, disabled: cur?.kind === 'engine',
      onchange: () => { st.engine = engine.value; render(); },
    }, engines.map((e) => h('option', { value: e.name, selected: e.name === st.engine },
      `${e.name}${e.link ? ` · on ${e.link}` : ''}${e.launchable ? '' : ' · started elsewhere'}`)));
    const prompt = h('textarea.seed-input.text.lib-notes.agent-prompt', {
      rows: 4, placeholder: 'Leave it empty for the built-in one: an image director that plans, makes and edits images.',
      'aria-label': 'System prompt', dataset: { key: 'ag-prompt' },
      oninput: () => { st.prompt = prompt.value; },
    });
    prompt.value = st.prompt;
    const e = engines.find((x) => x.name === st.engine);
    const runs = !e ? 'Configure a chat engine first, on the Engines screen.'
      : cur?.kind === 'engine' ? `It answers as whatever ${e.name} runs.`
        : e.launchable ? `${e.name} restarts with its model when you use it.`
          : `${e.name} is started elsewhere and runs its own model: these files are kept for an engine launched here.`;
    const portrait = cur?.owner ? `Its portrait: ${cur.owner}’s avatar from Hugging Face, until it paints its own.`
      : 'Its portrait: its monogram, until it paints its own or you upload one.';
    return h('div.ag-persona',
      h('label.ag-field', h('span.eyebrow', 'Name'), h('span.seed-field', name)),
      h('label.ag-field', h('span.eyebrow', 'Runs on'), h('span.seed-field', engine), h('span.mi-small', runs)),
      h('label.ag-field', h('span.eyebrow', 'System prompt'), prompt, h('span.mi-small', 'What it is told before every conversation. You can change it later.')),
      h('span.mi-small', portrait));
  }

  function summary() {
    const cur = d();
    if (!cur || cur.kind === 'engine') return { ready: true, text: 'Nothing to download', button: 'Make the agent' };
    const files = [pickedModel(), pickedProjector()].filter(Boolean);
    const gets = cur.kind === 'ref' ? files.filter((o) => !o.local) : [];
    const total = gets.reduce((n, o) => n + (o.size_bytes ?? 0), 0);
    return gets.length
      ? { ready: false, text: `${bytes(total) || plural(gets.length, 'file')} to download once it’s made`, button: `Make it · get ${bytes(total) || plural(gets.length, 'file')}` }
      : { ready: true, text: 'Ready to run · nothing to download', button: 'Make the agent' };
  }

  async function create() {
    if (st.busy || !canMake()) return;
    const cur = d();
    const model = pickedModel();
    const projector = pickedProjector();
    const body = { name: st.name.trim() || null, engine: st.engine || null, system_prompt: st.prompt.trim() || null };
    if (st.template) body.chat_template = st.template;
    if (cur.kind === 'path') Object.assign(body, { path: model.id, mmproj: projector.id });
    else if (cur.kind === 'ref') body.ref = model.quant ? `${cur.repo}:${model.quant}` : `hf://${cur.repo}/${model.id}`;
    st.busy = true;
    render();
    let made;
    try {
      made = await api.createAgent(body);
    } catch (err) {
      toast(err.message, { error: true });
      st.busy = false;
      render();
      return;
    }
    let started = 0;
    const failed = [];
    if (cur.kind === 'ref') {
      for (const [key, o] of [['model', model], ['mmproj', projector]]) {
        try {
          if (o.local) await api.updateAgent(made.id, { files: { [key]: o.local } });
          else {
            await api.hubDownload({ repo: cur.repo, revision: cur.sha ?? undefined, variant: o.id, agent: made.id, key });
            started += 1;
          }
        } catch (err) {
          failed.push(`${o.name}: ${err.message}`);
        }
      }
    }
    box.close();
    await loadAgents();
    navigate(`#/agents/${made.id}`);
    if (failed.length) toast(`Made ${made.name}, but: ${failed.join('; ')}`, { error: true });
    else toast(started ? `Made ${made.name} · downloading ${plural(started, 'file')}` : `Made ${made.name}`);
  }
}

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

// --- one agent (1b) --------------------------------------------------------------------------

export function agentScreen({ id, navigate }) {
  let d = null;
  let error = null;
  let gone = false;
  let poll = null;
  let downloads = [];
  const opened = new Set(['prompt']); // folded parts that are open

  const status = statusLine();
  const meta = h('span.topbar-meta');
  const header = h('header.topbar.sub',
    iconButton('back', 'Agents', () => navigate('#/agents')),
    h('div.topbar-heading', h('span.topbar-title', 'Agents'), meta),
    h('span.spacer'),
    h('div.only-wide', status.el));
  const main = h('main.links-main.lora-main.mc-main.agent-main', { onscroll: () => track() });
  const strip = connectionStrip();
  const el = h('div.screen.lora.model.agent', header, strip.el, main);

  async function load() {
    try {
      d = await api.agent(id);
      error = null;
    } catch (err) {
      error = err.message;
    }
    await followDownloads();
    if (!gone) render();
    schedule();
  }

  /** While it paints, works or downloads, the page follows along. */
  function schedule() {
    clearTimeout(poll);
    if (gone || !d) return;
    if (d.portrait.painting || d.working || downloads.some(active)) {
      poll = setTimeout(async () => {
        const was = d;
        try {
          d = await api.agent(id);
        } catch {
          // Gone meanwhile: keep the last view.
        }
        await followDownloads();
        if (gone) return;
        if (JSON.stringify(was) !== JSON.stringify(d)) {
          render();
          if (was.portrait.key !== d.portrait.key) loadAgents();
        }
        schedule();
      }, POLL_MS);
    }
  }

  const active = (j) => ['queued', 'running'].includes(j.state);

  async function followDownloads() {
    try {
      const mine = (await api.hubDownloads()).filter((j) => j.assign.some((a) => a.startsWith(`agent:${id}.`)));
      const finished = downloads.some((j) => active(j) && !mine.some((m) => m.id === j.id && active(m)));
      downloads = mine;
      if (finished) d = await api.agent(id);
    } catch {
      downloads = []; // Hugging Face is off here.
    }
  }

  const jobFor = (key) => downloads.find((j) => active(j) && j.assign.includes(`agent:${id}.${key}`));

  async function save(call, word = null) {
    try {
      d = await call();
      if (word) toast(word);
      loadAgents();
      loadBackends();
    } catch (err) {
      toast(err.message, { error: true });
    }
    if (!gone) render();
    schedule();
  }
  const patch = (body, word = null) => save(() => api.updateAgent(id, body), word);

  function render() {
    const active = document.activeElement;
    const key = main.contains(active) ? active?.dataset?.key : null;
    const top = main.scrollTop;
    clear(main);
    if (!d) {
      main.append(h('div.notice', error ?? 'Loading…'),
        error && h('div', h('a.btn-line', { href: '#/agents' }, icon('back', 15), 'The agents')));
      return;
    }
    meta.textContent = `${d.name} · ${d.working ? 'working' : d.use?.running ? 'running' : d.ready ? 'ready' : 'not ready'}`;
    document.title = `${d.name} · Agents · Stable-Imaginer`;
    append(main, [hero(), h('div.mc-body', rail(), h('div.mc-sections',
      portraitSection(), personaSection(), toolsSection(), filesSection(), samplingSection(), moreSection()))]);
    main.scrollTop = top;
    if (key) main.querySelector(`[data-key="${CSS.escape(key)}"]`)?.focus({ preventScroll: true });
    track();
  }

  // --- the hero and its plate --------------------------------------------------------------

  function hero() {
    const name = h('input.mc-name', {
      value: d.name, 'aria-label': 'Name', spellcheck: false, dataset: { key: 'name' },
      onchange: () => name.value.trim() && patch({ name: name.value }), onkeydown: enter,
    });
    const tags = [d.family && h('span.tag.tag-accent.agent-tag', d.family), d.vision && h('span.tag.tag-outline.agent-tag', 'vision'),
      d.tools && h('span.tag.tag-outline.agent-tag', 'tool calling'), d.default && h('span.tag.tag-outline.agent-tag', 'default')];
    const files = d.files.filter((f) => f.path);
    const mmproj = d.files.find((f) => f.key === 'mmproj' && f.path);
    const facts = [
      d.source?.repo ? h('b', d.source.repo) : d.model ? h('b', fileName(d.model)) : null,
      d.quant, files.length ? bytes(d.size_bytes) : null,
      mmproj && `mmproj ${(fileName(mmproj.path).match(/(f16|bf16|f32|q8_0)/i)?.[1] ?? '').toUpperCase() || 'set'}`,
      d.context && `${Math.round(d.context / 1024)}k context`,
      h('span.accent', d.turns ? `${plural(d.turns, 'turn')}, ${plural(d.images, 'image')}` : 'no turns yet'),
    ].filter(Boolean);
    return h('div.mc-hero.agent-hero',
      portrait(d, { size: 168, framed: true, origin: true, cls: 'agent-hero-portrait' }),
      h('div.mc-title',
        h('div.mc-kicker', tags),
        h('div.mc-name-row', name,
          h('button.icon-btn.mc-pin', {
            type: 'button', class: d.pinned ? 'on' : '', title: d.pinned ? 'Pinned: unpin' : 'Pin', 'aria-pressed': String(d.pinned),
            onclick: () => patch({ pinned: !d.pinned }),
          }, icon('pin', 18))),
        h('span.mc-facts', facts.flatMap((f, i) => (i ? [' · ', f] : [f])))),
      plate());
  }

  function plate() {
    const use = d.use;
    const bars = [];
    if (d.own_files) {
      for (const f of d.files) if (f.path || f.required) bars.push(f.path && !f.problem ? 'on' : jobFor(f.key) ? 'busy' : '');
    } else bars.push(use?.status === 'ok' ? 'on' : '');
    let kicker = 'Ready';
    let big = use ? `${use.running ? 'Running' : 'Stopped'} · ${use.engine}` : 'No engine';
    let text;
    const actions = [];
    const primary = (label, ico, onclick, disabled = false) => h('button.btn.btn-primary.blueprint.mc-act-btn', { type: 'button', disabled, onclick },
      corners(), ico && icon(ico, 15), label);
    const secondary = (label, onclick) => h('button.btn.btn-secondary.mc-act-btn', { type: 'button', onclick }, label);
    if (!use) {
      kicker = 'No engine';
      big = 'Nothing to run on';
      text = 'Configure a chat engine on the Engines screen, then pick it under Files.';
    } else if (d.problems.length) {
      kicker = 'Can\'t load';
      text = d.problems[0];
      actions.push(primary('Show the files', null, () => jump('files')));
    } else if (d.missing.length) {
      kicker = 'Setup';
      big = `Needs ${d.missing.map((m) => m.replace('_', ' ')).join(' and ')}`;
      text = d.missing.includes('model') ? `${use.engine} needs a model file to run it.` : 'Its model sees images with a vision projector (mmproj). Add one under Files.';
      actions.push(primary('Show the files', null, () => jump('files')));
    } else if (!use.runs_it) {
      kicker = 'Ready';
      big = `${use.engine} runs ${use.runs_other ?? 'another model'}`;
      text = [`Using it restarts `, h('b', use.engine), ` with its model.`];
      actions.push(primary(`Use on ${use.engine}`, 'play', () => useIt()));
    } else {
      const parts = d.own_files
        ? `${listWords(d.files.filter((f) => f.path).map((f) => f.key === 'model' ? 'Model' : f.key === 'mmproj' ? 'mmproj' : 'chat template'))} ${d.files.filter((f) => f.path).length > 1 ? 'are' : 'is'} here.`
        : `It uses whatever ${use.engine} runs${d.model ? ` (${modelName(d.model)})` : ''}.`;
      const where = use.default ? ' It is the default chat engine, so new prompts go to it.' : '';
      text = [parts, where, use.needs_restart ? h('span.accent', ' Restart to load the changes.') : null];
      if (!use.running && use.launchable) {
        big = `Stopped · ${use.engine}`;
        actions.push(primary('Start', 'play', () => useIt()));
      } else if (!use.running) {
        kicker = 'Unavailable';
        text = `${use.engine} isn't answering (${use.status}). It is started elsewhere.`;
      } else {
        actions.push(primary('New conversation', 'chat', () => newConversation()));
      }
      if (use.running && use.launchable) actions.push(secondary(use.needs_restart ? 'Restart' : 'Stop', () => (use.needs_restart ? restart() : stopIt())));
      if (use.default && !d.default) {
        // Several agents on one engine: the engine answers as one of them by default.
        actions.push(secondary('Make it the default', () => save(() => api.useAgent(id, { start: false }), `New prompts go to ${d.name}`)));
      }
    }
    if (d.working && !d.portrait.painting) kicker = 'Working';
    return h('div.mc-plate.blueprint', { class: d.ready ? 'ready' : '' }, corners(),
      h('div.mc-plate-head', h('span.mc-plate-kicker', kicker), h('span.mc-plate-big', big)),
      h('div.mc-bars', { style: { gridTemplateColumns: `repeat(${Math.max(1, bars.length)}, 1fr)` } }, bars.map((c) => h('i', { class: c }))),
      h('span.mc-plate-text', text),
      actions.length > 0 && h('div.mc-plate-actions', actions));
  }

  async function useIt() {
    const use = d.use;
    if (use?.running && !use.runs_it && !(await dialog({ title: `Restart ${use.engine}?`, body: `It stops, then starts with ${d.name}.`, confirm: 'Restart' }))) return;
    save(() => api.useAgent(id, { start: true }), `${use.engine} runs ${d.name}`);
  }

  async function restart() {
    try {
      await api.stopAgent(id);
      await save(() => api.useAgent(id, { start: true }), `Restarting ${d.use.engine}`);
    } catch (err) {
      toast(err.message, { error: true });
    }
  }

  const stopIt = () => save(() => api.stopAgent(id), `${d.use.engine} stopped`);

  async function newConversation() {
    try {
      const c = await api.createConversationWith(null, id);
      navigate(`#/c/${c.id}/chat`, { focusBar: true });
    } catch (err) {
      toast(err.message, { error: true });
    }
  }

  // --- the rail --------------------------------------------------------------------------

  const SECTIONS = [['portrait', 'Portrait'], ['persona', 'Persona'], ['tools', 'Tools'], ['files', 'Files'], ['sampling', 'Sampling']];
  let current = 'portrait';

  function rail() {
    const on = d.tool_list.filter((t) => t.enabled).length;
    const need = d.files.filter((f) => f.path || f.required);
    const have = need.filter((f) => f.path && !f.problem).length;
    const count = {
      tools: h('span.mc-rail-n', String(on)),
      files: d.own_files ? h('span.mc-rail-badge', { class: d.ready ? '' : 'todo' }, `${have}/${need.length}`) : null,
    };
    return h('nav.mc-rail', { 'aria-label': 'Sections' }, SECTIONS.map(([key, word], i) => h('a.mc-rail-item', {
      href: `#/agents/${id}`, class: current === key ? 'on' : '', 'aria-current': current === key ? 'true' : null, dataset: { section: key },
      onclick: (e) => { e.preventDefault(); jump(key); },
    }, h('span.mc-rail-num', `0${i + 1}`), word, h('span.spacer'), count[key])));
  }

  function jump(key) {
    main.querySelector(`#ag-${key}`)?.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
    current = key;
    paintRail();
  }

  function track() {
    const top = main.getBoundingClientRect().top + 90;
    let now = SECTIONS[0][0];
    for (const [key] of SECTIONS) {
      const s = main.querySelector(`#ag-${key}`);
      if (s && s.getBoundingClientRect().top <= top) now = key;
    }
    if (main.scrollTop + main.clientHeight >= main.scrollHeight - 4) now = SECTIONS.at(-1)[0];
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

  const sectionHead = (title, text) => h('div.mc-section-head', h('h3', title), text && h('span.mc-muted', text));

  // --- 01 portrait -----------------------------------------------------------------------

  function portraitSection() {
    const p = d.portrait;
    const hub = p.images.find((i) => i.key === 'hub');
    const upload = p.images.find((i) => i.key === 'upload');
    const selves = p.images.filter((i) => i.source === 'self');
    const inUse = (key) => p.key === key;
    const useLink = (key) => (inUse(key) ? h('span.state.done', 'IN USE')
      : h('button.mc-link', { type: 'button', onclick: () => save(() => api.setAgentPortrait(id, { key }), 'Portrait changed') }, 'Use it'));

    const hubOpt = h('div.mc-opt', { class: hub && inUse('hub') ? 'best' : '' },
      h('div.agent-opt-head', h('span.mc-opt-kicker', 'From Hugging Face'), hub && useLink('hub')),
      h('div.agent-opt-who',
        hub ? portraitImage(hub, 64) : h('span.portrait', { style: { width: '64px', height: '64px', fontSize: '38px' } }, h('span.portrait-mono', (p.hub_owner ?? p.monogram)[0].toUpperCase())),
        h('div.agent-opt-text',
          h('b', p.hub_owner ?? 'Not from the Hub'),
          h('span.mc-muted', p.hub_owner ? 'The uploader of the GGUF repo. Fetched from hf.co with the model.' : 'Its model didn’t come from Hugging Face, so there is no uploader to show.'))),
      p.hub_owner && h('div.mc-opt-actions',
        h('button.mc-link', {
          type: 'button', onclick: () => save(() => api.setAgentPortrait(id, { refresh: true }), 'Fetched from the Hub'),
        }, hub ? 'Refresh from the Hub' : 'Fetch from the Hub'),
        p.hub_fetched_at && h('span.mono-meta', `fetched ${ago(p.hub_fetched_at)}`)));

    const cells = selves.slice(-4).map((img) => h('button.agent-self', {
      type: 'button', class: inUse(img.key) ? 'on' : '', title: `Self-portrait v${img.version}${inUse(img.key) ? ' (in use)' : ': use it'}`,
      onclick: () => !inUse(img.key) && save(() => api.setAgentPortrait(id, { key: img.key }), `Self-portrait v${img.version}`),
    }, portraitImage(img, 0)));
    if (p.painting) {
      const step = p.painting.total_steps ? `${p.painting.step}/${p.painting.total_steps}` : p.painting.status === 'queued' ? 'queued' : '…';
      cells.push(h('a.agent-self.live', { href: `#/c/${p.painting.conversation_id}`, title: 'Painting: open its conversation' }, h('span', step)));
    }
    while (cells.length < 4) cells.push(h('span.agent-self.empty'));
    const engine = imageEngine();
    const selfOpt = h('div.mc-opt', { class: p.source === 'self' ? 'best' : '' },
      h('div.agent-opt-head', h('span.mc-opt-kicker', 'Self-portrait'), p.source === 'self' && h('span.state.done', 'IN USE')),
      h('div.agent-selves', cells.slice(-4)),
      h('span.mc-muted', 'The agent writes its own prompt and paints it on the image engine with ', h('code', 'set_portrait'), '.'),
      h('div.mc-opt-actions',
        h('button.btn.btn-secondary.mc-opt-btn.agent-paint', {
          type: 'button', disabled: !!p.painting || !d.use?.running,
          title: d.use?.running ? 'It writes a prompt and paints it' : 'Its engine isn\'t running',
          onclick: () => paint(),
        }, icon('sparkles', 14), p.painting ? 'Painting…' : selves.length ? 'Paint another' : 'Paint one'),
        h('span.mono-meta', engine ? `${modelName(engine.model) || engine.name} · 1:1` : 'no image engine')));

    const pick = h('input', { type: 'file', accept: 'image/*', hidden: true, onchange: () => pick.files[0] && uploadFile(pick.files[0]) });
    const drop = h('button.mi-drop.agent-drop', {
      type: 'button', onclick: () => pick.click(),
      ondragover: (e) => { if (hasFiles(e)) { e.preventDefault(); e.stopPropagation(); drop.classList.add('over'); } },
      ondragleave: () => drop.classList.remove('over'),
      ondrop: (e) => {
        if (!hasFiles(e)) return;
        e.preventDefault();
        e.stopPropagation();
        drop.classList.remove('over');
        if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
      },
    }, upload ? portraitImage(upload, 56) : icon('upload', 20), h('span.mi-drop-sub', upload ? 'Drop or pick another' : 'Drop a square image, or pick one'));
    const uploadOpt = h('div.mc-opt', { class: upload && inUse('upload') ? 'best' : '' },
      h('div.agent-opt-head', h('span.mc-opt-kicker', 'Upload'), upload && useLink('upload')),
      drop, pick,
      h('span.mc-muted', upload?.label ? `From ${upload.label} in its conversation; cropped to a square.` : 'Cropped to a square; an image from a conversation stays there.'));

    const last = selves.at(-1);
    return h('section.mc-section', { id: 'ag-portrait' },
      sectionHead('Portrait', 'Shown in the viewer while it thinks, and beside every reply in the conversation.'),
      h('div.mc-opts.agent-opts', hubOpt, selfOpt, uploadOpt),
      last?.prompt && h('details.mc-optional', {
        open: opened.has('prompt'), ontoggle: (e) => { if (e.target.open) opened.add('prompt'); else opened.delete('prompt'); },
      },
      h('summary', h('span.mc-summary-mark', icon('chevronRight', 14)), 'Last self-portrait prompt ',
        h('span.mc-muted', `written by the agent · v${last.version}`)),
      h('div.reason-box', last.prompt),
      last.conversation_id && h('a.mc-link.agent-open', { href: `#/c/${last.conversation_id}${last.label ? `/${last.label}` : ''}` }, `Open ${last.label ?? 'it'} in its conversation`)),
      p.key && h('div', h('button.mc-link.agent-clear', {
        type: 'button', onclick: () => save(() => api.setAgentPortrait(id, { key: 'none' }), 'Showing its monogram'),
      }, 'Show its monogram instead')));
  }

  async function paint() {
    try {
      const res = await api.paintAgent(id);
      d = res.agent;
      toast(`${d.name} is painting a self-portrait`, { action: { label: 'Watch', onclick: () => navigate(`#/c/${res.conversation_id}`) } });
      loadAgents();
    } catch (err) {
      toast(err.message, { error: true });
    }
    if (!gone) render();
    schedule();
  }

  async function uploadFile(file) {
    if (!file.type.startsWith('image/')) {
      toast('That isn’t an image', { error: true });
      return;
    }
    save(() => api.uploadAgentPortrait(id, file), 'Portrait uploaded');
  }

  // --- 02 persona ------------------------------------------------------------------------

  function personaSection() {
    const name = h('input.seed-input.text', {
      value: d.name, dataset: { key: 'persona-name' }, 'aria-label': 'Name',
      onchange: () => name.value.trim() && patch({ name: name.value }), onkeydown: enter,
    });
    const prompt = h('textarea.seed-input.text.lib-notes.agent-prompt', {
      dataset: { key: 'system-prompt' }, 'aria-label': 'System prompt', rows: 4,
      onchange: () => {
        const text = prompt.value.trim();
        if (text === (d.system_prompt_custom ? d.system_prompt : d.default_system_prompt).trim()) return;
        patch({ system_prompt: text }, text ? 'System prompt saved' : 'Back to the built-in prompt');
      },
    });
    prompt.value = d.system_prompt;
    const viewerPick = (key, word) => h('button.pick', {
      type: 'button', class: d.viewer === key ? 'on' : '', 'aria-pressed': String(d.viewer === key),
      onclick: () => d.viewer !== key && patch({ viewer: key }),
    }, word);
    return h('section.mc-section', { id: 'ag-persona' },
      sectionHead('Persona', 'What the agent is told before every conversation.'),
      h('div.mc-table',
        h('div.mc-srow', h('span.mc-slabel', 'Name'),
          h('div.mc-scontrol', h('span.seed-field.agent-name', name)),
          h('span.mc-versus', 'defaults to the model’s name')),
        h('div.mc-srow.top', h('span.mc-slabel', 'System prompt'),
          h('div.mc-scontrol', prompt),
          d.system_prompt_custom
            ? h('button.mc-versus.edited', { type: 'button', title: 'Back to the built-in prompt', onclick: () => patch({ system_prompt: '' }, 'Back to the built-in prompt') }, h('i'), 'edited · reset')
            : h('span.mc-versus', 'the built-in one')),
        h('div.mc-srow', h('span.mc-slabel', 'In the viewer'),
          h('div.mc-scontrol', h('div.tray-seg.agent-viewer-seg', { role: 'group', 'aria-label': 'The thought bubble shows' },
            viewerPick('status', 'Status only'), viewerPick('reasoning', 'Status + reasoning'))),
          h('span.mc-versus', 'the thought bubble'))));
  }

  // --- 03 tools --------------------------------------------------------------------------

  function toolsSection() {
    const rows = d.tool_list.map((t) => h('div.mc-file.agent-tool', { class: t.enabled ? '' : 'empty' },
      h('span.mc-mark', { class: t.enabled ? 'ok' : '' }, t.enabled ? icon('check', 14) : null),
      h('div.mc-slot', h('b', t.name, t.name === 'set_portrait' && h('span.tag.tag-accent.agent-new', 'NEW')), h('span.mc-flag', t.flags)),
      h('span.mc-muted', t.description, !t.available && h('span.accent', ' No image engine can serve it now.')),
      h('button.mc-link', {
        type: 'button', onclick: () => patch({ tools: { [t.name]: !t.enabled } }, `${t.name} ${t.enabled ? 'off' : 'on'}`),
      }, t.enabled ? 'Turn off' : 'Turn on')));
    return h('section.mc-section', { id: 'ag-tools' },
      sectionHead('Tools', d.use && !d.tools && d.use.running ? 'Calls it may make. Its chat template doesn’t take tools, so it makes none.' : 'Calls it may make.'),
      h('div.mc-table', rows));
  }

  // --- 04 files --------------------------------------------------------------------------

  function filesSection() {
    const engines = (agentsLib()?.engines ?? []);
    const enginePick = h('select.input.agent-engine', {
      'aria-label': 'Engine', dataset: { key: 'engine' },
      onchange: () => patch({ engine: enginePick.value }, 'Engine changed'),
    }, h('option', { value: '', selected: !d.engine_set }, `The default chat engine${engines.find((e) => e.default) ? ` (${engines.find((e) => e.default).name})` : ''}`),
    engines.map((e) => h('option', { value: e.name, selected: d.engine_set === e.name }, `${e.name}${e.link ? ` · on ${e.link}` : ''}${e.launchable ? '' : ' · started elsewhere'}`)));
    const launchable = d.use?.launchable;
    const rows = d.files.map((f) => fileRow(f, launchable));
    return h('section.mc-section', { id: 'ag-files' },
      sectionHead('Files', launchable ? `What ${d.use.engine} loads to run it.` : `${d.use?.engine ?? 'Its engine'} is started elsewhere: it runs its own model, so files here aren't used.`),
      h('div.mc-table',
        h('div.mc-srow', h('span.mc-slabel', 'Runs on'), h('div.mc-scontrol', enginePick),
          h('span.mc-versus', d.use ? `${d.use.kind === 'linked' ? 'shared by a linked instance' : d.use.kind === 'local' ? 'started here' : 'started elsewhere'}` : '')),
        rows));
  }

  function fileRow(f, launchable) {
    const bad = !!(f.path && f.problem);
    const job = jobFor(f.key);
    // An engine started elsewhere loads its own files: say what it runs instead.
    const elsewhere = launchable === false && !f.path;
    const shown = f.path ? fileName(f.path)
      : elsewhere ? (f.key === 'model' ? (d.model ? fileName(d.model) : `What ${d.use?.engine ?? 'its engine'} runs`) : 'As its engine has it')
        : f.key === 'chat_template' ? 'The one in the model' : 'Not set';
    const note = bad ? `${f.problem}.` : [
      f.source?.repo && `From ${f.source.repo}`,
      elsewhere && f.key === 'model' && `Reported by ${d.use?.engine ?? 'its engine'}`,
      f.key === 'mmproj' && f.required && !f.path && 'Its repo has one: with it, it sees images.',
    ].filter(Boolean).join(' · ');
    return h('div.mc-file', { class: bad ? 'bad' : f.path || (elsewhere && f.key === 'model' && d.model) ? '' : 'empty' },
      h('span.mc-mark', { class: bad ? 'bad' : f.path ? 'ok' : '' }, f.path ? icon(bad ? 'alert' : 'check', 14) : null),
      h('span.mc-slot', h('b', f.what[0].toUpperCase() + f.what.slice(1)), h('span.mc-flag', f.flag)),
      h('span.mc-file-name',
        h('span.mc-path', { title: f.path ?? '' }, shown),
        note && h('span.mc-muted', { class: bad ? 'accent' : '' }, note)),
      job ? h('span.mc-size.accent', `↓ ${Math.floor((100 * job.done_bytes) / (job.total_bytes || 1))}%`) : h('span.mc-size', bytes(f.size_bytes)),
      launchable !== false || f.path
        ? h('button.mc-link', { type: 'button', onclick: (e) => changeMenu(e.currentTarget, f) }, f.path ? 'Change' : 'Set')
        : h('span'));
  }

  function changeMenu(anchor, f) {
    menu(anchor, [
      { label: 'A path on this machine…', icon: 'folder', onclick: () => typePath(f) },
      f.key !== 'chat_template' && { label: 'Hugging Face…', icon: 'search', onclick: () => fromHub(f) },
      f.path && { label: 'Clear it', icon: 'x', danger: true, onclick: () => patch({ files: { [f.key]: null } }, 'Cleared') },
    ].filter(Boolean));
  }

  async function typePath(f) {
    const value = (await dialog({
      title: `${f.what[0].toUpperCase()}${f.what.slice(1)} file`,
      body: f.key === 'chat_template' ? 'A Jinja template (.jinja) on this machine, for --chat-template-file.' : `${f.flag} · a GGUF on this machine.`,
      input: { value: f.path ?? '', placeholder: 'D:\\models\\…' }, confirm: 'Use it',
    }))?.trim();
    if (value == null || value === (f.path ?? '')) return;
    patch({ files: { [f.key]: value || null } });
  }

  async function fromHub(f) {
    const ref = (await dialog({
      title: f.key === 'mmproj' ? 'Vision projector from Hugging Face' : 'Model from Hugging Face',
      body: 'A repo, with its quant: Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M. A copy already on this machine is used as is.',
      input: { value: d.source?.repo ? `${d.source.repo}${d.quant ? `:${d.quant}` : ''}` : '', placeholder: 'owner/repo:QUANT' },
      confirm: 'Get it',
    }))?.trim();
    if (!ref) return;
    try {
      const n = await downloadInto(id, ref, [f.key]);
      toast(n ? `Downloading into ${d.name}` : 'Set from this machine');
      await followDownloads();
      d = await api.agent(id);
    } catch (err) {
      toast(err.message, { error: true });
    }
    if (!gone) render();
    schedule();
  }

  // --- 05 sampling -----------------------------------------------------------------------

  function samplingSection() {
    const rows = SAMPLING.map(([key, word, step]) => {
      const own = d.sampling[key];
      const engine = d.engine_sampling[key];
      const input = h('input.seed-input', {
        type: 'number', step: String(step), value: own ?? '', placeholder: engine != null ? String(engine) : 'engine’s',
        dataset: { key: `s-${key}` }, 'aria-label': word,
        onchange: () => patch({ sampling: { [key]: input.value === '' ? null : Number(input.value) } }),
        onkeydown: enter,
      });
      return h('div.mc-srow', h('span.mc-slabel', word),
        h('div.mc-scontrol', h('span.seed-field.agent-num', input)),
        own != null
          ? h('button.mc-versus.edited', { type: 'button', onclick: () => patch({ sampling: { [key]: null } }) }, h('i'), `${engine != null ? `engine ${engine}` : 'the engine’s'} · reset`)
          : h('span.mc-versus', engine != null ? `the engine’s, ${engine}` : 'the engine’s default'));
    });
    const ctx = h('input.seed-input', {
      type: 'number', step: '1024', min: '0', value: d.context ?? '', placeholder: 'the model’s', dataset: { key: 'context' }, 'aria-label': 'Context size',
      onchange: () => patch({ context: ctx.value === '' ? 0 : Number(ctx.value) }, 'Context size saved'), onkeydown: enter,
    });
    return h('section.mc-section', { id: 'ag-sampling' },
      sectionHead('Sampling', 'Sent with each of its requests, over the engine’s own.'),
      h('div.mc-table', rows,
        h('div.mc-srow', h('span.mc-slabel', 'Context'),
          h('div.mc-scontrol', h('span.seed-field.agent-num', ctx), h('span.mc-muted', 'tokens · -c')),
          h('span.mc-versus', d.use?.launchable ? 'applies when the engine restarts' : 'set where its engine runs'))));
  }

  function moreSection() {
    const notes = h('textarea.seed-input.text.lib-notes.agent-prompt', {
      dataset: { key: 'notes' }, rows: 3, placeholder: 'Anything to remember about it', 'aria-label': 'Notes',
      onchange: () => patch({ notes: notes.value }),
    });
    notes.value = d.notes;
    return h('section.mc-section', { id: 'ag-more' },
      sectionHead('Notes', null),
      notes,
      h('div', h('button.btn-line.agent-delete', { type: 'button', onclick: () => remove() }, icon('trash', 15), 'Forget this agent')));
  }

  async function remove() {
    const ok = await dialog({
      title: `Forget ${d.name}?`,
      body: 'Its portraits go; its model files and the replies it wrote stay.',
      confirm: 'Forget it', danger: true,
    });
    if (!ok) return;
    try {
      await api.deleteAgent(id);
      await loadAgents();
      navigate('#/agents');
    } catch (err) {
      toast(err.message, { error: true });
    }
  }

  let lib = null;
  const agentsLib = () => lib;
  loadAgents().then((l) => { lib = l; if (d && !gone) render(); });
  load();
  return {
    el,
    destroy() {
      gone = true;
      clearTimeout(poll);
      status.destroy();
      strip.destroy();
      document.title = 'Stable-Imaginer';
    },
  };
}

function listWords(words) {
  if (words.length < 2) return words[0] ?? '';
  return `${words.slice(0, -1).join(', ')} and ${words.at(-1)}`;
}
