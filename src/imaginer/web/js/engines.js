// Engines (designs 1a–1d): one backend at a time, with its runtime, model files, endpoint,
// capabilities and startup log; engines that have launch settings start and stop here.
// Runtime installs run in a dialog (1b plan, 1c progress) and keep going in the background.
// The status line in any header opens a popover of the default engines (1d).
//
// Changes save as you leave a field. The server applies them at once and writes them to
// config/<profile>.local.yaml, over the profile; API keys stay in the environment.

import { api, readEvents } from './api.js';
import { announce, connectionStrip, modal, statusLine, toast } from './chrome.js';
import { corners, h, icon, iconButton, clear } from './dom.js';
import { clock, plural } from './format.js';
import { loadBackends } from './store.js';

const POLL_MS = 5000;
const TITLES = {
  sdcpp: 'stable-diffusion.cpp sd-server',
  llamacpp: 'llama.cpp llama-server',
  diffusers: 'Diffusers, in process',
  linked: 'Shared by a linked instance',
};
const EXE = { sdcpp: 'sd-server', llamacpp: 'llama-server' };
const CAPS = {
  image: ['t2i', 'edit', 'multi_ref', 'rgba'],
  llm: ['chat', 'stream', 'tools', 'vision_in'],
};
const STATUS_WORD = { ok: 'ready', degraded: 'loading', down: 'down', unavailable: 'off', unknown: '…' };

const roleOf = (e) => (e.kind === 'llm' ? 'chat' : 'image');
const size = (bytes) => (bytes == null ? '' : bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB`
  : bytes >= 1e6 ? `${Math.round(bytes / 1e6)} MB` : `${Math.max(1, Math.round(bytes / 1e3))} KB`);
const endpointOf = (e) => (e.type === 'linked' ? 'linked instance' : e.host ? `${e.host}:${e.port}` : 'in-process');

/** "EXITED · CODE 1", "RUNNING · PID 4120", or the backend's health when we didn't start it. */
function stateChip(e) {
  const p = e.process;
  if (e.install?.state === 'running') return h('span.state.run', `INSTALLING · ${e.install.step || 1}/${e.install.total || '?'}`);
  if (p?.state === 'exited') return h('span.state.fail', `EXITED · CODE ${p.exit_code}`);
  if (p?.state === 'running' || p?.state === 'starting') {
    return h('span.state.run', e.status === 'ok' ? `RUNNING · PID ${p.pid}` : `LOADING · PID ${p.pid}`);
  }
  if (p?.state === 'stopping') return h('span.state.wait', 'STOPPING');
  if (e.status === 'ok') return h('span.state.done', 'READY');
  if (!e.enabled) return h('span.state.wait', 'DISABLED');
  return h('span.state.wait', e.launchable ? 'STOPPED' : (STATUS_WORD[e.status] ?? e.status).toUpperCase());
}

/** One line for the header and the popover: what is going on with this engine. */
function stateLine(e) {
  const p = e.process;
  if (p?.state === 'exited') {
    return [p.exited_at && clock(p.exited_at), p.summary ?? `exited with code ${p.exit_code}`,
      e.host && e.status !== 'ok' && `port ${e.port} not listening`].filter(Boolean).join(' · ');
  }
  if (p?.state === 'running' || p?.state === 'starting') {
    return [`since ${clock(p.started_at)}`, e.status === 'ok' ? `listening on ${endpointOf(e)}` : 'loading models'].join(' · ');
  }
  return [e.model, STATUS_WORD[e.status] ?? e.status, e.detail].filter(Boolean).join(' · ');
}

// --- the screen (design 1a) -------------------------------------------------------------

export function enginesScreen({ name: wanted = null, navigate }) {
  let engines = [];
  let current = wanted;
  let logLines = [];
  let logFilter = 'all';
  let stream = null; // AbortController of the selected engine's event stream
  let poll = null;
  let gone = false;

  const status = statusLine();
  const meta = h('span.topbar-meta');
  const header = h('header.topbar.sub',
    iconButton('back', 'Back', () => (history.length > 1 ? history.back() : navigate('#/'))),
    h('div.topbar-heading', h('span.topbar-title', 'Engines'), meta),
    h('span.spacer'),
    h('div.only-wide', status.el),
    h('button.btn-line', { type: 'button', onclick: () => refresh({ probe: true }) }, icon('retry', 15), 'Check all'));

  const railList = h('div.rail-list');
  const routing = h('div.routing');
  const railHint = h('p.rail-hint');
  const rail = h('aside.rail.engines-rail', { 'aria-label': 'Backends' },
    h('span.eyebrow', 'Backends'), railList,
    h('span.eyebrow.routing-head', 'Routing'), routing, railHint);
  const main = h('main.engines-main');
  const strip = connectionStrip();
  const el = h('div.screen.engines', header, strip.el, rail, main);

  let shownSig = null;
  let deferred = false;

  async function refresh({ probe = false } = {}) {
    let next;
    try {
      next = await api.engines();
    } catch (err) {
      if (!gone && !engines.length) clear(main).append(h('div.notice', err.message));
      return;
    }
    if (gone) return;
    engines = next;
    if (probe) loadBackends();
    if (!current || !engines.some((e) => e.name === current)) {
      current = (engines.find((e) => e.launchable) ?? engines[0])?.name ?? null;
    }
    // A poll that changes nothing leaves the view alone; one that does waits while a
    // field is being edited, so typing is never lost.
    const sig = JSON.stringify([engines, current, logFilter]);
    if (sig === shownSig && !probe) return;
    if (main.contains(document.activeElement) && document.activeElement.matches('input')) {
      deferred = true;
      return;
    }
    render();
  }
  main.addEventListener('focusout', () => setTimeout(() => {
    if (deferred && !main.contains(document.activeElement)) { deferred = false; render(); }
  }));

  function select(name) {
    if (name === current) return;
    current = name;
    history.replaceState(null, '', `#/engines/${name}`);
    logLines = [];
    render();
    follow();
  }

  function render() {
    shownSig = JSON.stringify([engines, current, logFilter]);
    const profile = engines[0]?.settings_file;
    meta.textContent = [profile && `saves to ${profile.split(/[\\/]/).slice(-2).join('/')}`, plural(engines.length, 'backend')]
      .filter(Boolean).join(' · ');
    renderRail();
    const e = engines.find((x) => x.name === current);
    clear(main);
    // Narrow screens hide the rail: the backends become tabs.
    main.append(h('div.engine-tabs', { role: 'group', 'aria-label': 'Backends' }, engines.map((x) => h('button.pick', {
      type: 'button', class: x.name === current ? 'on' : '', 'aria-pressed': String(x.name === current),
      onclick: () => select(x.name),
    }, x.name))));
    if (!e) {
      main.append(h('div.notice', engines.length ? 'Pick a backend.' : 'No backends are configured.'));
      return;
    }
    main.append(...engineView(e));
  }

  function renderRail() {
    clear(railList).append(...engines.map((e) => {
      const caps = e.capabilities.length ? `caps ${e.capabilities.join(',')}` : null;
      const role = e.default_for.length ? e.default_for.join(', ') : roleOf(e);
      const where = e.launchable ? `local · ${endpointOf(e)}` : endpointOf(e);
      const word = e.process?.state === 'exited' ? 'stopped' : e.process?.state === 'running' && e.status !== 'ok'
        ? 'loading' : !e.enabled ? 'disabled' : STATUS_WORD[e.status] ?? e.status;
      return h('button.rail-item', {
        type: 'button', class: e.name === current ? 'on' : '', 'aria-current': e.name === current ? 'true' : null,
        style: { opacity: e.enabled ? '' : '.6' }, onclick: () => select(e.name),
      },
      h('span.rail-icon', icon(e.type === 'linked' ? 'link' : e.type === 'diffusers' ? 'layers' : e.launchable ? 'cpu' : 'server', 16)),
      h('span.rail-text',
        h('span.rail-name', e.name),
        h('span.mono-meta', { class: e.status === 'ok' ? '' : 'accent' }, `${role} · ${word}`),
        h('span.mono-meta', e.enabled ? where : e.detail ?? 'disabled'),
        e.enabled && caps && h('span.mono-meta', caps)));
    }));
    clear(routing).append(...['chat', 't2i', 'edit'].map((task) => h('div.routing-row',
      h('span.muted', task), h('span', engines.find((e) => e.default_for.includes(task))?.name ?? '—'))));
    const file = engines[0]?.settings_file;
    railHint.textContent = file
      ? `Changes are saved to ${file.split(/[\\/]/).slice(-2).join('/')}, over the profile. API keys stay in the environment and are only named here.`
      : 'This server was started without a profile, so changes last until it restarts. API keys stay in the environment.';
  }

  // --- one engine ---------------------------------------------------------------------

  function engineView(e) {
    const p = e.process;
    const alive = p && ['starting', 'running', 'stopping'].includes(p.state);
    const installing = e.install?.state === 'running';
    const actions = [];
    if (p?.command) {
      actions.push(h('button.btn-line', {
        type: 'button', onclick: () => copy(p.command.map(quote).join(' ')),
      }, icon('terminal', 15), 'Copy command'));
    }
    if (e.launchable) {
      actions.push(alive
        ? h('button.btn.btn-secondary.engine-go', { type: 'button', disabled: p.state === 'stopping', onclick: () => act(() => api.stopEngine(e.name), 'Stopped') },
          h('span.stop-square.hollow'), `Stop ${EXE[e.type] ?? e.name}`)
        : h('button.btn.btn-primary.blueprint.engine-go', {
          type: 'button', disabled: installing || !e.enabled || e.status === 'ok',
          title: e.status === 'ok' ? `Already answering on ${endpointOf(e)}, started outside this server` : null,
          onclick: () => act(() => api.startEngine(e.name), 'Starting'),
        }, corners(), icon('play', 14), `Start ${EXE[e.type] ?? e.name}`));
    }
    const head = h('div.engine-head',
      h('div.engine-title',
        h('span.eyebrow.accent', `${roleOf(e) === 'chat' ? 'Chat' : 'Image'} backend · ${e.name}`),
        h('h2.hero-title', TITLES[e.type] ?? e.type),
        h('div.engine-state', stateChip(e), h('span.mono-meta', stateLine(e)))),
      ...actions);

    const left = h('div.engine-cells');
    if (e.runtime) left.append(runtimeCell(e));
    if (e.launchable) left.append(filesCell(e));
    else if (e.files.length) left.append(remoteCell(e));
    left.append(h('div.cell-pair', endpointCell(e), capsCell(e)));
    const right = e.launchable ? logPanel(e) : remoteNote(e);
    return [head, h('div.engine-grid', left, right)];
  }

  function runtimeCell(e) {
    const r = e.runtime;
    const job = e.install;
    const busy = job?.state === 'running';
    const line = r.installed
      ? [r.version, r.accelerator, r.method].filter(Boolean).join(' · ')
      : 'not installed';
    return h('div.tray-cell.row-cell',
      h('div.cell-grow',
        h('span.eyebrow', 'Runtime · this machine'),
        h('span.cell-big', line),
        h('span.mono-meta.wrap', busy ? `installing · ${job.label || 'starting'}` : r.executable ?? r.directory),
        job && !busy && job.state !== 'succeeded' && h('span.mono-meta.accent.wrap', `last install ${job.state}${job.error ? `: ${job.error}` : ''}`)),
      busy
        ? h('button.mini', { type: 'button', onclick: () => installDialog(e, refresh) }, 'Show install')
        : [
          r.installed && h('button.mini', { type: 'button', onclick: () => checkUpdate(e) }, 'Check update'),
          h('button.mini', { type: 'button', onclick: () => installDialog(e, refresh, { force: r.installed }) },
            r.installed ? 'Reinstall…' : 'Install…'),
        ]);
  }

  function filesCell(e) {
    const grid = h('div.files-grid');
    for (const f of e.files) {
      const input = h('input.seed-input.path-input', {
        type: 'text', value: f.path ?? '', placeholder: f.what, spellcheck: false, 'aria-label': `${f.flag} (${f.what})`,
        onchange: () => save(e, { launch: { files: { [f.key]: input.value.trim() || null } } }),
        onkeydown: (ev) => { if (ev.key === 'Enter') input.blur(); },
      });
      const bad = !!(f.path && f.problem);
      grid.append(
        h('span.mono-meta', { class: bad ? 'strong' : '' }, f.flag),
        h('label.seed-field', { class: bad ? 'bad' : '' }, icon('folder', 13), input),
        h('span.mono-meta', size(f.size_bytes)),
        h('span.file-mark', { class: bad ? 'bad' : f.path ? 'ok' : '', title: f.problem ?? (f.path ? f.format : 'not set') },
          f.path ? icon(bad ? 'alert' : 'check', 15) : null));
      if (bad) grid.append(h('span'), h('span.file-problem', `${f.problem}.`));
    }
    const needsVision = e.type === 'sdcpp' && !e.files.find((f) => f.key === 'llm_vision')?.path;
    return h('div.tray-cell',
      h('div.cell-head', h('span.eyebrow', 'Model files'), h('span.spacer'),
        h('span.tray-hint', 'paths on this machine · checked on save')),
      grid,
      needsVision && h('span.tray-hint', 'Without --llm_vision, sd-server can only do t2i: drop edit and multi_ref below.'));
  }

  function remoteCell(e) {
    return h('div.tray-cell',
      h('span.eyebrow', 'Launch'),
      h('span.tray-hint.wrap', `${e.name} is not started from here${e.loopback ? '' : `: it runs on ${e.host}`}. Add launch settings to start a local ${EXE[e.type] ?? 'engine'} with this server.`),
      e.loopback && h('div', h('button.mini', {
        type: 'button', onclick: () => save(e, { launch: { files: {} } }),
      }, icon('plus', 13), 'Set up launch')));
  }

  function endpointCell(e) {
    if (e.type === 'linked') {
      return h('div.tray-cell', h('span.eyebrow', 'Endpoint'), h('span.cell-big', 'linked instance'),
        h('span.tray-hint', 'Reached through the link, with its token.'));
    }
    if (!e.host) {
      return h('div.tray-cell', h('span.eyebrow', 'Endpoint'), h('span.cell-big', 'in process'),
        h('span.tray-hint', 'Diffusers runs inside this server; no endpoint to manage.'));
    }
    const host = h('input.seed-input', {
      type: 'text', value: e.host, 'aria-label': 'Host', spellcheck: false,
      onchange: () => host.value.trim() && save(e, { host: host.value.trim() }),
    });
    const port = h('input.seed-input', {
      type: 'number', min: 1, max: 65535, value: String(e.port), 'aria-label': 'Port',
      onchange: () => { const v = Number(port.value); if (v >= 1 && v <= 65535) save(e, { port: v }); },
    });
    const hint = e.api_key_env
      ? `API key from $${e.api_key_env}`
      : e.type === 'sdcpp' ? 'sd-server has no auth. Keep it on loopback or a trusted LAN.' : 'no API key';
    return h('div.tray-cell', h('span.eyebrow', 'Endpoint'),
      h('div.endpoint-row', h('label.seed-field', host), h('label.seed-field', port)),
      h('span.tray-hint', hint));
  }

  function capsCell(e) {
    const all = CAPS[e.kind === 'llm' ? 'llm' : 'image'];
    const declared = e.declared_capabilities;
    const live = new Set(e.capabilities);
    const picked = new Set(declared ?? all);
    const toggle = (cap) => {
      const next = new Set(picked);
      if (next.has(cap)) next.delete(cap); else next.add(cap);
      save(e, { declared_capabilities: [...next] });
    };
    const hint = e.type === 'sdcpp'
      ? (e.files.find((f) => f.key === 'llm_vision')?.path ? 'edit and multi_ref need --llm_vision ✓' : 'edit and multi_ref need --llm_vision')
      : e.status === 'ok' ? `the engine reports ${e.capabilities.join(', ') || 'none'}` : 'checked against the engine when it is up';
    return h('div.tray-cell', h('span.eyebrow', 'Capabilities'),
      h('div.tray-opts', all.map((cap) => h('button.pick.cap', {
        type: 'button', class: picked.has(cap) ? 'on' : '', 'aria-pressed': String(picked.has(cap)),
        title: picked.has(cap) && e.status === 'ok' && !live.has(cap) ? `${cap}: declared, but the engine doesn't report it` : cap,
        onclick: () => toggle(cap),
      }, cap))),
      h('span.tray-hint', hint));
  }

  function remoteNote(e) {
    if (e.type === 'linked') {
      return h('div.engine-note',
        h('span.eyebrow', 'Health'),
        h('div.tool-row.wrap', h('span.tool-name', 'LINK'), h('span.tool-args', `${STATUS_WORD[e.status] ?? e.status}${e.detail ? ` · ${e.detail}` : ''}`)),
        h('p.rail-hint', 'The whole model runs on the linked instance that shares it. Pair, unlink and choose what to use on the Links screen.'),
        h('div', h('a.btn-line', { href: '#/links' }, icon('link', 15), 'Linked instances')));
    }
    return h('div.engine-note',
      h('span.eyebrow', 'Health'),
      h('div.tool-row.wrap', h('span.tool-name', 'GET'), h('span.tool-args', `${e.base_url ?? 'in process'} · ${STATUS_WORD[e.status] ?? e.status}${e.detail ? ` · ${e.detail}` : ''}`)),
      h('p.rail-hint', e.loopback
        ? 'Start it yourself, or set up launch to start it from here.'
        : `This engine runs on ${e.host}. Start and update it there; this screen manages only how this server reaches it.`));
  }

  // --- startup log -------------------------------------------------------------------------

  const logBox = h('div.engine-log', { role: 'log', 'aria-live': 'off', tabindex: '0', 'aria-label': 'Startup log' });
  const logFrame = h('div.engine-log-frame.blueprint', corners(), logBox);

  function logPanel(e) {
    const p = e.process;
    const seg = (key, text) => h('button.pick', {
      type: 'button', class: logFilter === key ? 'on' : '', 'aria-pressed': String(logFilter === key),
      onclick: () => { logFilter = key; render(); },
    }, text);
    const span = p?.started_at ? `last run · ${clock(p.started_at)}${p.exited_at ? ` → ${clock(p.exited_at)}` : ' → now'}` : 'not started yet';
    paintLog(e);
    return h('div.engine-logcol',
      h('div.log-head',
        h('span.eyebrow', 'Startup log'), h('span.mono-meta', span), h('span.spacer'),
        h('div.tray-seg.small', seg('all', 'All'), seg('errors', 'Errors')),
        p?.log_path && iconButton('fileText', 'Open log file', () => openLogFile(e), 15)),
      logFrame);
  }

  function paintLog(e) {
    clear(logBox);
    const lines = logFilter === 'errors' ? logLines.filter((l) => l.error) : logLines;
    if (!lines.length) {
      logBox.append(h('div.muted', logFilter === 'errors' ? 'No errors in this run.' : 'Start the engine to see its output here.'));
    }
    for (const line of lines.slice(-400)) {
      const cmd = line.text.startsWith('$ ');
      logBox.append(h('div.log-line', { class: line.error ? 'error' : cmd ? 'cmd' : '' },
        !cmd && h('span.log-time', `${clock(line.at)}:${new Date(line.at).getSeconds().toString().padStart(2, '0')} `), line.text));
    }
    logBox.scrollTop = logBox.scrollHeight;
  }

  async function loadLog() {
    if (!current) return;
    try {
      logLines = await api.engineLog(current);
    } catch {
      logLines = [];
    }
    const e = engines.find((x) => x.name === current);
    if (e?.launchable) paintLog(e);
  }

  /** The selected engine's live output and state, while this screen is open. */
  async function follow() {
    stream?.abort();
    if (!current) return;
    const ctrl = new AbortController();
    stream = ctrl;
    const name = current;
    await loadLog();
    try {
      const resp = await api.engineEvents(name, ctrl.signal);
      await readEvents(resp, (type, _id, data) => {
        if (ctrl.signal.aborted || name !== current) return;
        if (type === 'line') {
          logLines.push(data);
          if (logLines.length > 2000) logLines.shift();
          const e = engines.find((x) => x.name === name);
          if (e) paintLog(e);
        } else if (type === 'state' || type === 'install') {
          refresh();
          if (type === 'state') {
            if (data === 'exited') announce(`${name} exited`);
            if (data === 'running') announce(`${name} started`);
          }
        }
      });
    } catch {
      // Aborted, or the connection dropped; the poll keeps the view current.
    }
  }

  // --- actions -------------------------------------------------------------------------------

  async function save(e, patch) {
    try {
      const updated = await api.updateEngine(e.name, patch);
      engines = engines.map((x) => (x.name === e.name ? updated : x));
      render();
      loadBackends();
      toast(updated.settings_file ? `Saved to ${updated.settings_file.split(/[\\/]/).slice(-2).join('/')}` : 'Applied until the server restarts');
    } catch (err) {
      toast(err.message, { error: true });
      render();
    }
  }

  async function act(call, word) {
    try {
      const updated = await call();
      engines = engines.map((x) => (x.name === updated.name ? updated : x));
      render();
      announce(`${word} ${updated.name}`);
      setTimeout(() => { refresh(); loadBackends(); }, 1500);
    } catch (err) {
      toast(err.message, { error: true });
    }
  }

  async function checkUpdate(e) {
    try {
      const u = await api.checkUpdate(e.name);
      if (u.error) toast(`Update check failed: ${u.error}`, { error: true });
      else if (u.update_available) {
        toast(`${u.latest_version} is available (installed ${u.current_version})`, {
          error: true, action: { label: 'Update…', onclick: () => installDialog(e, refresh, { force: true }) },
        });
      } else toast(`Up to date · ${u.current_version || e.runtime.version}`);
    } catch (err) {
      toast(err.message, { error: true });
    }
  }

  async function openLogFile(e) {
    try {
      const text = await api.engineLogFile(e.name);
      const url = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
      window.open(url, '_blank', 'noopener');
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
    } catch (err) {
      toast(err.message, { error: true });
    }
  }

  function copy(text) {
    navigator.clipboard?.writeText(text).then(() => toast('Command copied'), () => toast(text));
  }

  refresh().then(follow);
  poll = setInterval(() => { if (!document.hidden) refresh(); }, POLL_MS);

  return {
    el,
    show(name) {
      if (!engines.length) current = name; // still loading: open on it once the list is in
      else if (engines.some((e) => e.name === name)) select(name);
    },
    destroy() {
      gone = true;
      clearInterval(poll);
      stream?.abort();
      status.destroy();
      strip.destroy();
    },
  };
}

const quote = (arg) => (/[\s"]/.test(arg) ? `"${arg.replace(/"/g, '\\"')}"` : arg);

// --- install dialog (designs 1b, 1c) ------------------------------------------------------------

/** Plan, then follow an install of e's runtime. onChange refreshes whoever opened it. */
export function installDialog(e, onChange = () => {}, { force = false } = {}) {
  const exe = EXE[e.type] ?? e.type;
  const req = { method: 'auto', accelerator: '', version: 'latest', force };
  let plan = null;
  let job = e.install?.state === 'running' ? e.install : null;
  let ctrl = null;
  let box;
  const layer = modal((close) => {
    box = h('div.dialog.blueprint.install-dialog', { role: 'dialog', 'aria-modal': 'true', 'aria-label': `Install ${exe}` });
    box.close = () => { ctrl?.abort(); close(); onChange(); };
    return box;
  }, { onDismiss: () => { ctrl?.abort(); onChange(); } });

  function render() {
    clear(box).append(...corners());
    if (job) renderJob();
    else renderPlan();
  }

  function seg(options, value, pickFn) {
    return h('div.tray-seg', options.map(([key, text]) => h('button.pick', {
      type: 'button', class: value === key ? 'on' : '', 'aria-pressed': String(value === key), onclick: () => pickFn(key),
    }, text)));
  }

  function renderPlan() {
    const hw = plan?.hardware;
    const accel = req.accelerator || hw?.accelerator || '';
    const accels = [...new Set([hw?.accelerator, 'cuda', 'vulkan', 'cpu'].filter((a) => a && (hw?.accelerators ?? []).includes(a)))];
    const detected = hw
      ? [hw.gpu ?? (hw.accelerator ?? 'cpu').toUpperCase(), hw.cuda_arch && `sm_${hw.cuda_arch}`].filter(Boolean).join(' · ')
      : 'detecting…';
    const steps = [];
    (plan?.attempts ?? []).forEach((a, i) => {
      if (i > 0) steps.push(h('span.tray-hint', `if that fails: ${a.attempt} (${a.method}, ${a.accelerator})`));
      else a.steps.forEach((label, n) => steps.push(h('div.lin-row', h('span.lin-thumb.arrow.pending.step-n', String(n + 1)), h('span', label))));
    });
    const tag = h('input.seed-input', {
      type: 'text', value: req.version, 'aria-label': 'Release tag or git ref', spellcheck: false,
      onchange: () => { req.version = tag.value.trim() || 'latest'; load(); },
    });
    box.append(
      h('div.dialog-head', h('span.eyebrow.accent', `${e.name} · ${e.runtime?.installed ? `installed ${e.runtime.version}` : 'not installed'}`),
        h('div.dialog-title', `${e.runtime?.installed ? 'Reinstall' : 'Install'} ${exe}`)),
      h('div.cell-grid',
        h('div.tray-cell', h('span.eyebrow', 'Method'),
          seg([['auto', 'Auto'], ['release', 'Release'], ['source', 'Source']], req.method, (m) => { req.method = m; load(); })),
        h('div.tray-cell', h('span.eyebrow', 'Accelerator'),
          accels.length ? seg(accels.map((a) => [a, a === 'cuda' ? 'CUDA' : a === 'cpu' ? 'CPU' : a[0].toUpperCase() + a.slice(1)]), accel, (a) => { req.accelerator = a; load(); })
            : h('span.tray-hint', 'detecting…')),
        h('div.tray-cell', h('span.eyebrow', 'This machine'), h('span.cell-big', detected),
          h('span.tray-hint', !hw ? '' : hw.driver_cuda ? `driver CUDA ${hw.driver_cuda}`
            : hw.gpu ? 'driver CUDA version not reported' : 'no NVIDIA GPU found')),
        h('div.tray-cell', h('span.eyebrow', 'Tag'), h('label.seed-field', tag),
          h('span.tray-hint', req.version === 'latest' ? 'the newest release, resolved when installing' : 'a release tag or a git ref'))),
      h('div.install-steps',
        h('span.eyebrow', 'Plan · nothing is downloaded yet'),
        plan?.directory && h('span.tray-hint.wrap', `Installs on this machine, into ${plan.directory}.`),
        plan ? steps : h('span.tray-hint', 'Working out the plan…'),
        h('span.tray-hint', 'Auto falls back to a source build, then CPU. Any fallback is reported before activating.')),
      h('div.dialog-actions',
        h('button.btn.btn-secondary', { type: 'button', onclick: () => box.close() }, 'Cancel'),
        h('button.btn.btn-primary', { type: 'button', disabled: !plan?.attempts?.length, onclick: start }, e.runtime?.installed ? 'Reinstall' : 'Install')));
  }

  function renderJob() {
    const done = job.state !== 'running';
    const pct = job.fraction >= 0 ? Math.round(job.fraction * 100) : null;
    const rows = [];
    const total = Math.max(job.total, 1);
    for (let n = 1; n <= total; n += 1) {
      const past = n < job.step || (done && job.state === 'succeeded');
      const now = n === job.step && !done;
      rows.push(h('div.lin-row', { class: !past && !now ? 'later' : '' },
        h('span.lin-thumb.arrow.step-n', { class: now ? 'now' : past ? '' : 'pending' }, past ? icon('check', 13) : String(n)),
        h('span', { class: now ? 'lin-label' : '' }, n === job.step ? job.label : past ? `step ${n}` : `step ${n}`)));
      if (now) {
        rows.push(h('div.job-progress.step-progress',
          h('div.job-step', h('span.job-step-n', job.label || 'Working'), h('span.mono-meta.accent', pct == null ? '' : `${pct}%`)),
          h('div.step-bar', h('i', { style: { width: `${pct ?? 0}%` } }))));
      }
    }
    const chip = job.state === 'running' ? h('span.state.run', `STEP ${job.step || 1} / ${job.total || '?'}`)
      : job.state === 'succeeded' ? h('span.state.done', 'INSTALLED') : h('span.state.fail', job.state.toUpperCase());
    box.append(
      h('div.dialog-head.row', h('div', h('span.eyebrow.accent', `${e.name} · ${done ? job.state : 'installing'}`),
        h('div.dialog-title', `Install ${exe}`)), chip),
      h('div.install-steps', rows),
      job.error && h('div.msg-error', job.error),
      job.warning && h('div.tool-error', h('span', job.warning)),
      h('div.reason-box.install-output', (job.output ?? []).slice(-6).join('\n') || 'starting…'),
      h('div.dialog-actions', done
        ? h('button.btn.btn-primary', { type: 'button', onclick: () => box.close() }, 'Close')
        : [
          h('button.btn.btn-secondary', { type: 'button', onclick: cancel }, 'Cancel install'),
          h('button.btn.btn-secondary', { type: 'button', onclick: () => box.close() }, 'Run in background'),
        ]));
  }

  async function load() {
    render();
    try {
      plan = await api.installPlan(e.name, req);
      if (!req.accelerator && plan.hardware?.accelerator) req.accelerator = plan.hardware.accelerator;
    } catch (err) {
      plan = { attempts: [], hardware: plan?.hardware };
      toast(err.message, { error: true });
    }
    if (!job) render();
  }

  async function start() {
    try {
      const updated = await api.install(e.name, req);
      job = updated.install;
      render();
      followJob();
    } catch (err) {
      toast(err.message, { error: true });
    }
  }

  async function cancel() {
    try {
      job = (await api.cancelInstall(e.name)).install ?? job;
      render();
    } catch (err) {
      toast(err.message, { error: true });
    }
  }

  async function followJob() {
    ctrl?.abort();
    ctrl = new AbortController();
    const mine = ctrl;
    try {
      const resp = await api.engineEvents(e.name, mine.signal);
      await readEvents(resp, (type, _id, data) => {
        if (type !== 'install' || mine.signal.aborted) return;
        job = data;
        if (box.isConnected) render();
        if (data.state !== 'running') {
          announce(`Install ${data.state}`);
          onChange();
          mine.abort();
        }
      });
    } catch {
      // Closed or aborted.
    }
  }

  if (job) {
    render();
    followJob();
  } else load();
  return layer;
}

// --- the status line's popover (design 1d) ---------------------------------------------------

/** Under the status line: the default engines, with Start and Log for local ones. */
export async function enginePopover(anchor) {
  let list;
  try {
    list = await api.engines();
  } catch (err) {
    toast(err.message, { error: true });
    return;
  }
  const shown = list.filter((e) => e.default_for.length);
  const rect = anchor.getBoundingClientRect();
  anchor.setAttribute('aria-expanded', 'true');
  anchor.classList.add('open');
  const close = ({ restore = true } = {}) => {
    sheet.remove();
    anchor.setAttribute('aria-expanded', 'false');
    anchor.classList.remove('open');
    document.removeEventListener('pointerdown', outside, true);
    document.removeEventListener('keydown', onKey, true);
    window.removeEventListener('hashchange', onRoute);
    if (restore && anchor.isConnected) anchor.focus();
  };
  const onRoute = () => close({ restore: false });
  const outside = (ev) => { if (!sheet.contains(ev.target) && !anchor.contains(ev.target)) close({ restore: false }); };
  const onKey = (ev) => { if (ev.key === 'Escape') { ev.preventDefault(); close(); } };
  const go = (hash) => { close({ restore: false }); location.hash = hash; };
  const rows = shown.map((e) => {
    // Something already answers on its port (started elsewhere): nothing to start.
    const stopped = e.launchable && e.status !== 'ok' && !['starting', 'running', 'stopping'].includes(e.process?.state);
    const roles = e.default_for.map((t) => (t === 't2i' ? 'image' : t)).join(' · ');
    return h('div.pop-engine',
      h('div.pop-engine-row',
        h('span.ring', { class: e.status === 'ok' ? 'ok' : '' }),
        h('div.engine-text',
          h('span.engine-name', e.name, h('span.engine-default', roles)),
          h('span.engine-meta', e.process?.state === 'exited'
            ? [`exited with code ${e.process.exit_code}`, e.process.exited_at && clock(e.process.exited_at), e.process.summary].filter(Boolean).join(' · ')
            : [e.model, STATUS_WORD[e.status] ?? e.status, e.status === 'ok' && e.detail == null ? null : e.detail].filter(Boolean).join(' · ')))),
      (stopped || e.process) && h('div.pop-actions',
        stopped && e.enabled && h('button.mini.primary', {
          type: 'button',
          onclick: async () => {
            try {
              await api.startEngine(e.name);
              toast(`Starting ${e.name}`);
              setTimeout(loadBackends, 1500);
              close();
            } catch (err) {
              toast(err.message, { error: true, action: { label: 'Engines', onclick: () => go(`#/engines/${e.name}`) } });
            }
          },
        }, icon('play', 12), 'Start'),
        e.process && h('button.mini', { type: 'button', onclick: () => go(`#/engines/${e.name}`) }, icon('fileText', 13), 'Log')));
  });
  const sheet = h('div.menu.blueprint.engine-pop', {
    role: 'dialog', 'aria-label': 'Engines',
    style: { top: `${rect.bottom + 8}px`, right: `${Math.max(8, window.innerWidth - rect.right)}px` },
  },
  corners(),
  rows.length ? rows : h('div.pop-engine', h('span.tray-hint', 'No default engines are configured.')),
  h('button.menu-item.pop-settings', { type: 'button', onclick: () => go('#/engines') },
    icon('sliders', 15), 'Engine settings', h('span.spacer'), icon('arrowRight', 15)),
  h('button.menu-item.pop-settings', { type: 'button', onclick: () => go('#/links') },
    icon('link', 15), 'Linked instances', h('span.spacer'), icon('arrowRight', 15)));
  document.body.append(sheet);
  setTimeout(() => document.addEventListener('pointerdown', outside, true));
  document.addEventListener('keydown', onKey, true);
  window.addEventListener('hashchange', onRoute);
  sheet.querySelector('button')?.focus();
}
