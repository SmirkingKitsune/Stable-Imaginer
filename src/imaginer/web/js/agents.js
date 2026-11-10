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

import { api, setImageUrl } from './api.js';
import { connectionStrip, dialog, menu, modal, modelName, statusLine, toast } from './chrome.js';
import { append, corners, h, icon, iconButton, clear } from './dom.js';
import { ago, plural } from './format.js';
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

function addAgent(lib, navigate) {
  const engines = lib?.engines ?? [];
  const def = engines.find((e) => e.default) ?? engines[0];
  modal((close) => {
    const name = h('input.input', { placeholder: 'Defaults to the model’s name', 'aria-label': 'Name' });
    const engine = h('select.input', { 'aria-label': 'Engine' }, engines.map((e) => h('option', {
      value: e.name, selected: e === def,
    }, `${e.name}${e.link ? ` · on ${e.link}` : ''}${e.launchable ? '' : ' · started elsewhere'}`)));
    const model = h('input.input', { placeholder: 'D:\\models\\…\\model-Q4_K_M.gguf, or Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M', 'aria-label': 'Model' });
    const mmproj = h('input.input', { placeholder: 'mmproj-F16.gguf (optional: for a vision model)', 'aria-label': 'Vision projector' });
    const hint = h('span.mc-muted');
    const sync = () => {
      const e = engines.find((x) => x.name === engine.value);
      const ref = /^[\w.-]+\/[\w.-]+(:[\w.-]+)?$/.test(model.value.trim()) || /huggingface\.co|hf\.co/.test(model.value);
      mmproj.disabled = ref;
      hint.textContent = !e ? 'Configure a chat engine first, on the Engines screen.'
        : !e.launchable ? `${e.name} is started elsewhere: leave the model empty to use whatever it runs (${e.model ? modelName(e.model) : 'its model'}).`
          : ref ? 'From Hugging Face: its GGUF and vision projector download into the agent.'
            : `${e.name} restarts with this model when you use the agent.`;
    };
    engine.addEventListener('change', sync);
    model.addEventListener('input', sync);
    sync();
    const go = h('button.btn.btn-primary', {
      type: 'button',
      onclick: async () => {
        const text = model.value.trim();
        const isRef = text && (/^[\w.-]+\/[\w.-]+(:[\w.-]+)?$/.test(text) || /huggingface\.co|hf\.co/.test(text));
        go.disabled = true;
        try {
          const body = { name: name.value.trim() || null, engine: engine.value || null };
          if (isRef) body.ref = text;
          else if (text) {
            body.path = text;
            if (mmproj.value.trim()) body.mmproj = mmproj.value.trim();
          }
          const made = await api.createAgent(body);
          if (isRef) {
            try {
              const n = await downloadInto(made.id, text);
              if (n) toast(`Downloading ${plural(n, 'file')} into ${made.name}`);
            } catch (err) {
              toast(`Made ${made.name}, but: ${err.message}`, { error: true });
            }
          }
          close();
          await loadAgents();
          navigate(`#/agents/${made.id}`);
        } catch (err) {
          go.disabled = false;
          toast(err.message, { error: true });
        }
      },
    }, 'Add the agent');
    return h('div.dialog.blueprint.agent-add', { role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Add an agent' },
      corners(),
      h('div.dialog-title', 'Add an agent'),
      h('div.dialog-body', 'A chat model plus a persona. Its portrait comes from whoever uploaded the model to Hugging Face, until it paints its own.'),
      h('label.field', h('span.eyebrow', 'Model'), model),
      h('label.field', h('span.eyebrow', 'Vision projector'), mmproj),
      h('label.field', h('span.eyebrow', 'Runs on'), engine),
      h('label.field', h('span.eyebrow', 'Name'), name),
      hint,
      h('div.dialog-actions', h('button.btn.btn-secondary', { type: 'button', onclick: () => close() }, 'Cancel'), go));
  });
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
