// Hugging Face (from the Engines screen): find a model repo, see which of its files can fill
// each of an engine's model settings, and use a copy already on this machine or download one.
//
// The server reads each repo's layout (GGUF quants and shards, mmproj projectors, Diffusers
// pipelines, ComfyUI repackages) and matches it to the engine, so this dialog only shows the
// result. Downloads run on the server and keep going after the dialog closes; once done, they
// set the engine's setting, as an edit on the Engines screen would.

import { api, readEvents } from './api.js';
import { announce, modal, toast } from './chrome.js';
import { append, corners, h, icon, clear } from './dom.js';

const FORMATS = [['gguf', 'GGUF'], ['diffusers', 'Diffusers'], ['safetensors', 'Safetensors'], ['any', 'Any']];
const DEFAULT_FORMAT = { llamacpp: 'gguf', sdcpp: 'any', diffusers: 'diffusers' };
// What the trending list (no search yet) shows for each engine.
const TRENDING_TASK = { sdcpp: 'text-to-image', diffusers: 'text-to-image' };
const KIND = {
  gguf: 'GGUF quants', diffusers: 'Diffusers pipeline', comfy: 'ComfyUI files',
  transformers: 'transformers checkpoint', safetensors: 'safetensors files', unknown: 'no model files',
};
const COMPONENT = {
  model: 'model', mmproj: 'vision projector', diffusion_model: 'diffusion model', vae: 'VAE',
  text_encoder: 'text encoder', lora: 'LoRA (not loaded here)', other: 'not loaded here',
};
const SHOWN = 6; // candidates per setting before "show all"

const size = (bytes) => (bytes == null ? '' : bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB`
  : bytes >= 1e6 ? `${Math.round(bytes / 1e6)} MB` : `${Math.max(1, Math.round(bytes / 1e3))} KB`);
const count = (n) => (n == null ? '' : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(n));
const day = (iso) => (iso ? iso.slice(0, 10) : '');
const norm = (path) => path.replace(/\\/g, '/');

/** A value typed into a path field that is meant as a Hub reference, not a local path:
 * a URL, an hf:// path, or llama.cpp's owner/name:QUANT. Mirrors hub/refs.py. */
export function looksLikeRef(value) {
  const v = value.trim();
  if (/^(https?:\/\/|hf:\/\/)/.test(v) || /^(www\.)?(huggingface\.co|hf\.co)\//.test(v)) return true;
  return /^[A-Za-z0-9][\w.-]*\/[A-Za-z0-9][\w.-]*:[\w.-]+$/.test(v);
}

/** A repo id on its own (owner/name): opened directly rather than searched for. */
const isRepoId = (v) => /^[A-Za-z0-9][\w.-]*\/[A-Za-z0-9][\w.-]*$/.test(v.trim());

/** Open the dialog for engine e. ref opens that repo straight away. */
export function hubDialog(e, { ref = null, onChange = () => {} } = {}) {
  const state = {
    view: ref ? 'repo' : 'search',
    query: '',
    format: DEFAULT_FORMAT[e.type] ?? 'any',
    base: null, // {id} when listing a repo's quantized versions
    results: null,
    repo: null,
    ref,
    loading: false,
    expanded: new Set(),
    status: null,
    focusSearch: !ref, // once, when the search view is first shown
    editingFolder: false,
  };
  const jobs = new Map(); // download id -> latest job
  const meters = new Map(); // variant id ('' for the pipeline) -> {bar, label}
  let events = null;
  let box;
  const layer = modal((close) => {
    box = h('div.dialog.blueprint.hub-dialog', { role: 'dialog', 'aria-modal': 'true', 'aria-label': `Hugging Face models for ${e.name}` });
    box.close = () => { events?.abort(); close(); onChange(); };
    return box;
  }, { onDismiss: () => { events?.abort(); onChange(); } });

  // --- rendering -------------------------------------------------------------------------

  function render() {
    meters.clear();
    clear(box).append(...corners());
    append(box, [head(), state.view === 'repo' ? repoView() : searchView(), foot()]);
  }

  function head() {
    return h('div.dialog-head.row',
      h('div',
        h('span.eyebrow.accent', `${e.name} · Hugging Face`),
        h('div.dialog-title', state.view === 'repo' && state.repo ? state.repo.id : 'Find a model')),
      h('button.icon-btn', { type: 'button', 'aria-label': 'Close', title: 'Close', onclick: () => box.close() }, icon('x', 16)));
  }

  function foot() {
    const s = state.status;
    if (!s) return null;
    const token = s.token ? `Hub token from ${s.token_source}` : 'no Hub token: gated repos need HF_TOKEN or `hf auth login`';
    if (state.editingFolder) {
      const input = h('input.seed-input.path-input', {
        type: 'text', value: s.models_dir, spellcheck: false, 'aria-label': 'Models folder',
        onkeydown: (ev) => { if (ev.key === 'Enter') { ev.preventDefault(); saveFolder(input.value); } },
      });
      setTimeout(() => input.select());
      return h('div.hub-foot.hub-folder',
        h('label.seed-field', icon('folder', 13), input),
        h('button.mini.primary', { type: 'button', onclick: () => saveFolder(input.value) }, 'Save'),
        h('button.mini', { type: 'button', onclick: () => { state.editingFolder = false; render(); } }, 'Cancel'));
    }
    return h('div.hub-foot',
      h('span.tray-hint.wrap', `${token} · downloads go to ${s.models_dir}${s.hf_cache ? ' · copies in the Hugging Face cache are used in place' : ''}`),
      h('button.mini', { type: 'button', onclick: () => { state.editingFolder = true; render(); } }, 'Change folder'));
  }

  async function saveFolder(value) {
    if (!value.trim()) return;
    try {
      state.status = await api.updateHub({ models_dir: value.trim() });
      state.editingFolder = false;
      toast(`Downloads now go to ${state.status.models_dir}`);
      if (state.repo) await reloadRepo();
      render();
    } catch (err) {
      toast(err.message, { error: true });
    }
  }

  function searchView() {
    const input = h('input.seed-input', {
      type: 'search', value: state.query, spellcheck: false, 'aria-label': 'Search the Hub, or paste a repo',
      placeholder: 'Search, or paste a repo, URL or repo:QUANT',
      onkeydown: (ev) => { if (ev.key === 'Enter') { ev.preventDefault(); go(input.value); } },
    });
    if (state.focusSearch) {
      state.focusSearch = false;
      setTimeout(() => input.focus());
    }
    const seg = h('div.tray-seg.hub-formats', FORMATS.map(([key, text]) => h('button.pick', {
      type: 'button', class: state.format === key ? 'on' : '', 'aria-pressed': String(state.format === key),
      onclick: () => { state.format = key; state.base = null; search(); },
    }, text)));
    const rows = state.results == null
      ? [h('span.tray-hint', state.loading ? 'Searching…' : '')]
      : state.results.length
        ? state.results.map(resultRow)
        : [h('span.tray-hint', 'Nothing found.')];
    return h('div.hub-body',
      h('div.hub-search',
        h('label.seed-field', icon('search', 14), input),
        h('button.btn.btn-primary', { type: 'button', onclick: () => go(input.value) }, 'Search')),
      seg,
      state.base && h('div.hub-banner', h('span.mono-meta.wrap', `quantized versions of ${state.base}`),
        h('button.mini', { type: 'button', onclick: () => { state.base = null; search(); } }, 'Clear')),
      h('span.eyebrow', state.query || state.base ? 'Results'
        : ['Trending', TRENDING_TASK[e.type], FORMATS.find(([k]) => k === state.format)[1]].filter(Boolean).join(' · ')),
      h('div.hub-results', rows));
  }

  function resultRow(r) {
    const meta = [r.task, r.downloads != null && `↓ ${count(r.downloads)}`, r.likes != null && `♥ ${count(r.likes)}`,
      day(r.updated_at), ...r.formats.map((f) => f.toUpperCase())].filter(Boolean).join(' · ');
    return h('button.hub-result', { type: 'button', onclick: () => openRepo(r.id) },
      h('span.hub-result-id', r.id, r.gated && h('span.state.wait', 'GATED')),
      h('span.mono-meta', meta));
  }

  function repoView() {
    const r = state.repo;
    const back = h('button.mini', { type: 'button', onclick: () => { state.view = 'search'; render(); if (state.results == null) search(); } },
      icon('chevronLeft', 13), 'Search');
    if (!r) {
      return h('div.hub-body', h('div', back), h('span.tray-hint', state.loading ? `Reading ${state.ref}…` : ''));
    }
    const meta = [KIND[r.kind], r.task, r.downloads != null && `↓ ${count(r.downloads)}`, r.likes != null && `♥ ${count(r.likes)}`,
      r.sha && `commit ${r.sha.slice(0, 7)}`].filter(Boolean).join(' · ');
    const body = h('div.hub-body',
      h('div.hub-repo-head', back, h('span.mono-meta.wrap', meta), h('span.spacer'),
        h('a.mini', { href: r.url, target: '_blank', rel: 'noopener noreferrer' }, 'Open on the Hub', icon('external', 12))),
      r.gated && h('div.tool-error', h('span', state.status?.token
        ? 'Gated: accept its terms on the Hub with the account whose token this server uses.'
        : 'Gated: accept its terms on the Hub, then set HF_TOKEN (or run `hf auth login`) on this server.')),
      r.notes.map((n) => h('span.tray-hint.wrap', n)));
    if (e.type === 'diffusers') body.append(pipelineCell(r));
    else body.append(...r.roles.map((role) => roleCell(r, role)));
    const quantizable = ['transformers', 'diffusers', 'safetensors'].includes(r.kind) || !r.roles.some((x) => x.candidates.length);
    if (quantizable) {
      body.append(h('div', h('button.mini', { type: 'button', onclick: () => quantized(r.id) }, icon('layers', 13), 'Quantized versions of this model')));
    }
    if (r.base_models.length) {
      body.append(h('div.hub-bases', h('span.mono-meta', 'based on'),
        r.base_models.map((b) => h('button.mini', { type: 'button', onclick: () => openRepo(b) }, b))));
    }
    const fits = new Set(r.roles.flatMap((x) => x.candidates));
    const rest = r.variants.filter((v) => !fits.has(v.id));
    if (rest.length && e.type !== 'diffusers') {
      body.append(h('details.hub-rest', h('summary.mono-meta', `${rest.length} other file${rest.length === 1 ? '' : 's'} this engine doesn't load`),
        rest.map((v) => h('div.hub-variant.dim', h('span.hub-name', v.name), h('span.mono-meta', COMPONENT[v.component]),
          h('span.mono-meta', size(v.size_bytes)), h('span'), h('span')))));
    }
    return body;
  }

  function roleCell(r, role) {
    const byId = new Map(r.variants.map((v) => [v.id, v]));
    const ordered = role.candidates.map((id) => byId.get(id)).filter(Boolean);
    if (role.suggested) ordered.sort((a, b) => (b.id === role.suggested) - (a.id === role.suggested));
    const all = state.expanded.has(role.key);
    const shown = all ? ordered : ordered.slice(0, SHOWN);
    const current = role.current ? role.current.split(/[\\/]/).pop() : 'not set';
    return h('div.tray-cell.hub-role',
      h('div.cell-head', h('span.eyebrow', `${role.flag} · ${role.what}`), h('span.spacer'),
        h('span.mono-meta', { title: role.current ?? '' }, `now: ${current}`)),
      ordered.length
        ? shown.map((v) => variantRow(r, role, v))
        : h('span.tray-hint', 'Nothing in this repo fits here.'),
      ordered.length > SHOWN && h('div', h('button.mini', {
        type: 'button', onclick: () => { if (all) state.expanded.delete(role.key); else state.expanded.add(role.key); render(); },
      }, all ? 'Show fewer' : `Show all ${ordered.length}`)));
  }

  function variantRow(r, role, v) {
    const target = `${e.name}.${role.key}`;
    const local = v.local;
    const where = local ? (local.source === 'hf_cache' ? 'in the HF cache' : 'downloaded') : v.partial_bytes ? `${Math.floor((100 * v.partial_bytes) / v.size_bytes)}% downloaded` : '';
    return h('div.hub-variant', { class: v.id === role.suggested ? 'suggested' : '' },
      h('span.hub-name', { title: v.files.map((f) => f.path).join('\n') }, v.name,
        v.id === role.suggested && h('span.hub-tag', 'suggested'),
        v.shards > 1 && h('span.hub-tag', `${v.shards} shards`)),
      h('span.mono-meta', v.quant ?? v.format),
      h('span.mono-meta', size(v.size_bytes)),
      h('span.mono-meta', where),
      action(r, v.id, local, target, () => ({ variant: v.id, key: role.key }), v.partial_bytes ? 'Resume' : 'Download'));
  }

  function pipelineCell(r) {
    const p = r.pipeline;
    const role = r.roles[0];
    const cell = h('div.tray-cell.hub-role', h('div.cell-head', h('span.eyebrow', 'model_id · Diffusers pipeline'), h('span.spacer'),
      h('span.mono-meta', { title: role?.current ?? '' }, `now: ${role?.current ?? 'not set'}`)));
    if (!p) {
      cell.append(h('span.tray-hint.wrap', 'Not a Diffusers pipeline: the diffusers backend loads a repo with a model_index.json.'));
      return cell;
    }
    const target = `${e.name}.model_id`;
    const usingId = role?.current === r.id;
    // In the Hugging Face cache, Diffusers finds it by repo id (and follows later commits).
    const cached = p.local?.source === 'hf_cache';
    const button = !p.compatible ? h('span')
      : cached ? h('button.mini', { type: 'button', disabled: usingId, onclick: () => assign('model_id', r.id) }, usingId ? [icon('check', 12), 'In use'] : 'Use')
        : action(r, '', p.local, target, () => ({ key: 'model_id' }), p.partial_bytes ? 'Resume' : 'Download');
    append(cell, [
      h('div.hub-variant',
        h('span.hub-name', p.pipeline_class ?? 'Diffusers pipeline'),
        h('span.mono-meta', `${p.components.length} parts`),
        h('span.mono-meta', size(p.size_bytes)),
        h('span.mono-meta', p.local ? (cached ? 'in the HF cache' : 'downloaded') : p.partial_bytes ? `${Math.floor((100 * p.partial_bytes) / p.size_bytes)}% downloaded` : ''),
        button),
      h('span.tray-hint.wrap', p.compatible ? p.components.join(' · ') : `Can't be used: ${p.problem}.`),
      p.compatible && !p.local && h('div.hub-alt',
        h('button.mini', { type: 'button', disabled: usingId, onclick: () => assign('model_id', r.id) }, usingId ? 'Using the repo id' : 'Use the repo id instead'),
        h('span.tray-hint.wrap', 'Diffusers then downloads it into the Hugging Face cache when the model first loads, without progress here.')),
    ]);
    return cell;
  }

  /** Use / In use / progress / Download for one variant ('' is the pipeline). */
  function action(r, id, local, target, body, word) {
    const job = activeJob(r, id);
    if (job) return meter(id, job);
    if (local) {
      const using = local.in_use.includes(target);
      return h('button.mini', { type: 'button', disabled: using, title: local.path, onclick: () => assign(target.split('.').pop(), local.path) },
        using ? [icon('check', 12), 'In use'] : 'Use');
    }
    return h('button.mini.primary', { type: 'button', onclick: () => download(r, body()) }, icon('download', 12), word);
  }

  function meter(id, job) {
    const bar = h('i');
    const label = h('span.mono-meta');
    const el = h('div.hub-meter', h('div.step-bar', bar), label,
      h('button.icon-btn.hub-cancel', { type: 'button', title: 'Cancel download', 'aria-label': 'Cancel download', onclick: () => cancel(job.id) }, icon('x', 12)));
    meters.set(id, { bar, label });
    paint(id, job);
    return el;
  }

  function paint(id, job) {
    const m = meters.get(id);
    if (!m) return;
    const pct = job.total_bytes ? Math.min(100, Math.floor((100 * job.done_bytes) / job.total_bytes)) : 0;
    m.bar.style.width = `${pct}%`;
    const speed = job.speed_bps ? ` · ${size(job.speed_bps)}/s` : '';
    m.label.textContent = job.state === 'queued' ? 'queued' : `${pct}%${speed}`;
    m.label.title = [job.current_file, job.warning].filter(Boolean).join('\n');
  }

  function activeJob(r, id) {
    const end = norm(id ? `/${r.id}/${id}` : `/${r.id}`);
    for (const job of jobs.values()) {
      if ((job.state === 'running' || job.state === 'queued') && norm(job.target).endsWith(end)) return job;
    }
    return null;
  }

  // --- actions -------------------------------------------------------------------------------

  async function go(text) {
    const value = text.trim();
    state.query = value;
    state.base = null;
    if (value && (looksLikeRef(value) || isRepoId(value))) {
      if (await openRepo(value, { quiet: isRepoId(value) && !looksLikeRef(value) })) return;
    }
    search();
  }

  async function search() {
    state.view = 'search';
    state.loading = true;
    state.results = null;
    render();
    const trending = !state.query && !state.base;
    const params = { q: state.query, format: state.format, sort: trending ? 'trending' : 'downloads', limit: 30 };
    if (trending && TRENDING_TASK[e.type]) params.task = TRENDING_TASK[e.type];
    if (state.base) Object.assign(params, { base_model: state.base, relation: 'quantized' });
    try {
      state.results = await api.hubSearch(params);
    } catch (err) {
      state.results = [];
      toast(err.message, { error: true });
    }
    state.loading = false;
    if (state.view === 'search') render();
  }

  function quantized(id) {
    state.base = id;
    state.query = '';
    state.format = e.type === 'llamacpp' ? 'gguf' : 'any';
    search();
  }

  /** Show a repo; resolves false if it couldn't be read (quiet: say nothing, e.g. a search
   * term that only looked like a repo id). */
  async function openRepo(refText, { quiet = false } = {}) {
    state.view = 'repo';
    state.ref = refText;
    state.repo = null;
    state.loading = true;
    state.expanded.clear();
    render();
    try {
      state.repo = await api.hubRepo(refText, e.name);
      state.loading = false;
      render();
      return true;
    } catch (err) {
      state.loading = false;
      if (!quiet) toast(err.message, { error: true });
      state.view = 'search';
      render();
      return false;
    }
  }

  async function reloadRepo() {
    if (!state.repo) return;
    try {
      state.repo = await api.hubRepo(state.ref, e.name);
      if (state.view === 'repo') render();
    } catch {
      // Keep what is shown.
    }
  }

  async function assign(key, value) {
    const patch = key === 'model_id' ? { model_id: value } : { launch: { files: { [key]: value } } };
    try {
      await api.updateEngine(e.name, patch);
      toast(`${key} set to ${value.split(/[\\/]/).pop()}`);
      onChange();
      reloadRepo();
    } catch (err) {
      toast(err.message, { error: true });
    }
  }

  async function download(r, body) {
    try {
      const job = await api.hubDownload({ repo: r.id, revision: r.sha ?? undefined, engine: e.name, ...body });
      jobs.set(job.id, job);
      announce(`Downloading ${job.label}`);
      render();
    } catch (err) {
      toast(err.message, { error: true });
    }
  }

  async function cancel(id) {
    try {
      jobs.set(id, await api.cancelHubDownload(id));
    } catch (err) {
      toast(err.message, { error: true });
    }
  }

  /** Download progress while the dialog is open. */
  async function follow() {
    events = new AbortController();
    const mine = events;
    try {
      for (const job of await api.hubDownloads()) jobs.set(job.id, job);
      if (state.view === 'repo') render();
      const resp = await api.hubEvents(mine.signal);
      await readEvents(resp, (type, _id, job) => {
        if (type !== 'download' || mine.signal.aborted) return;
        const before = jobs.get(job.id);
        jobs.set(job.id, job);
        const r = state.repo;
        if (job.state === 'running' && before?.state === 'running') {
          if (r) for (const id of meters.keys()) if (activeJob(r, id)?.id === job.id) paint(id, job);
          return;
        }
        if (job.state === 'succeeded') {
          announce(`Downloaded ${job.label}`);
          toast(job.warning ? `${job.label}: ${job.warning}` : `Downloaded ${job.label}${job.assign.length ? ` · set ${job.assign.join(', ')}` : ''}`, { error: !!job.warning });
          onChange();
          reloadRepo();
        } else if (job.state === 'failed') {
          toast(`${job.label}: ${job.error}`, { error: true });
          reloadRepo();
        } else if (job.state === 'cancelled') {
          reloadRepo();
        } else if (state.view === 'repo') render();
      });
    } catch {
      // Closed, or the connection dropped.
    }
  }

  api.hub().then((s) => { state.status = s; render(); }, () => {});
  if (ref) openRepo(ref);
  else search();
  follow();
  return layer;
}
