// Client state: the gallery, loaded conversations with their live turns, drafts and the
// quick options. Views subscribe to topics and re-render from this state:
//   'backends'  engine list and health
//   'gallery'   the conversation list, from this server and every linked instance
//   'turns'     any turn's running state or job progress (gallery tiles)
//   `conv:<id>` anything in one conversation
//
// A conversation's messages form a tree (parent_id): edits and retried tool calls are
// siblings. Each conversation shows one path through it, see pathOf().

import { api, readEvents } from './api.js';
import { announce } from './chrome.js';
import { titleFrom } from './format.js';
import { place } from './origin.js';

// --- topics ---------------------------------------------------------------------

const listeners = new Map();
const pending = new Set();
let scheduled = false;

export function on(topic, fn) {
  if (!listeners.has(topic)) listeners.set(topic, new Set());
  listeners.get(topic).add(fn);
  return () => listeners.get(topic)?.delete(fn);
}

export function emit(...topics) {
  topics.forEach((t) => pending.add(t));
  if (scheduled) return;
  scheduled = true;
  // Normally the next frame renders. Frames are paused in background tabs and some
  // embedded views, so a timer makes sure the page is current when it's shown again.
  requestAnimationFrame(flush);
  setTimeout(flush, 100);
}

function flush() {
  if (!scheduled) return;
  scheduled = false;
  const now = [...pending];
  pending.clear();
  for (const topic of now) for (const fn of [...(listeners.get(topic) ?? [])]) fn();
}

// --- engines --------------------------------------------------------------------

export const state = {
  backends: [],
  gallery: { items: [], sources: [], cursor: null, done: false, loading: null, loaded: false, archived: false },
  convs: new Map(),
  conn: { status: 'online' }, // see connection(), below
};

// The backends shown are those of the instance the screen's conversation lives on: a turn
// in a linked instance's conversation runs there, with its engines.
let backendsBase = '';

/** Show the backends behind `base` ('' for this server's) from now on. */
export function backendsFrom(base) {
  if (base === backendsBase) return;
  backendsBase = base;
  state.backends = [];
  loadBackends();
}

export async function loadBackends() {
  try {
    state.backends = await api.backends(backendsBase);
    // The regular poll can find the server again after the retries ran out.
    if (state.conn.status === 'offline' && !looping) restored();
  } catch (err) {
    // Unreachable: keep the last list, the status line reads "offline" meanwhile.
    if (err.status === 0) connectionLost();
    else state.backends = [];
  }
  emit('backends');
}

export const defaultBackend = (task) => state.backends.find((b) => b.default_for.includes(task));
export const imageBackends = () => state.backends.filter((b) => b.kind === 'image' && b.enabled);

// --- connection (design 4a/4b) ----------------------------------------------------
// One recovery loop for the whole page. A dropped turn stream or a failed engine poll
// starts it; it probes the server with backoff and, once back, re-attaches every turn
// that lost its stream from its last event id. Turns keep running on the server.
//
//   status  'online' | 'reconnecting' (attempt n of max, next try at nextAt)
//           | 'offline' (retries exhausted) | 'restored' (back, shown for 3 s)

const BACKOFF_S = [1, 2, 4, 8, 16];
const RESTORED_MS = 3000;
const resumeQueue = new Set(); // conversations whose turn stream must be re-attached
let wake = null; // resolves the current backoff wait early ("Retry now")
let looping = false;

export const connection = () => state.conn;
export const offline = () => state.conn.status === 'reconnecting' || state.conn.status === 'offline';

function setConn(patch) {
  state.conn = { ...state.conn, ...patch };
  emit('conn', 'backends', 'turns');
  for (const id of resumeQueue) emit(`conv:${id}`);
}

/** Something could not reach the server. c: a conversation whose turn stream dropped. */
export function connectionLost(c = null) {
  if (c) resumeQueue.add(c.id);
  if (looping) {
    if (c) emit(`conv:${c.id}`);
    return;
  }
  const lastEvent = c?.turn ? { lastEventId: c.turn.lastEventId, lastEventAt: c.turn.lastEventAt ?? Date.now() } : {};
  state.conn = { status: 'reconnecting', attempt: 0, max: BACKOFF_S.length, startedAt: Date.now(), ...lastEvent };
  reconnectLoop();
}

/** Skip the rest of the current wait, or start over once retries ran out. */
export function reconnectNow() {
  if (wake) wake();
  else if (!looping) {
    state.conn = { ...state.conn, status: 'reconnecting', attempt: 0, startedAt: Date.now() };
    reconnectLoop();
  }
}

async function reconnectLoop() {
  looping = true;
  try {
    for (let i = 0; i < BACKOFF_S.length; i += 1) {
      setConn({ status: 'reconnecting', attempt: i + 1, nextAt: Date.now() + BACKOFF_S[i] * 1000 });
      await new Promise((resolve) => {
        const timer = setTimeout(resolve, BACKOFF_S[i] * 1000);
        wake = () => { clearTimeout(timer); resolve(); };
      });
      wake = null;
      setConn({ nextAt: null });
      try {
        state.backends = await api.backends();
      } catch (err) {
        if (err.status === 0) continue;
      }
      await restored();
      return;
    }
    setConn({ status: 'offline', nextAt: null, gaveUpAt: Date.now() });
  } finally {
    looping = false;
    wake = null;
  }
}

async function restored() {
  const from = state.conn.lastEventId ?? null;
  setConn({ status: 'restored', resumedFrom: from, caughtUp: 0, at: Date.now() });
  const queued = [...resumeQueue];
  resumeQueue.clear();
  for (const id of queued) {
    const c = state.convs.get(id);
    if (!c?.turn?.running) continue;
    const turn = await api.turn(id).catch(() => null);
    if (turn?.running) attach(c, c.turn.lastEventId, { counting: true });
    else {
      finish(c, c.turn.ctrl);
      refresh(c);
    }
  }
  loadGallery({ fresh: true });
  const at = state.conn.at;
  setTimeout(() => { if (state.conn.at === at) setConn({ status: 'online' }); }, RESTORED_MS);
}

/** Throw away what is loaded for a conversation and read it again from the server. */
export async function reloadConversation(id) {
  const c = state.convs.get(id);
  if (c) {
    c.turn?.ctrl.abort();
    c.turn = null;
    c.loaded = false;
    c.messages.clear();
    c.artifacts.clear();
    c.jobs.clear();
    c.rev += 1;
  }
  resumeQueue.delete(id);
  try {
    await ensureLoaded(id);
    if (state.conn.status === 'offline' && !resumeQueue.size) setConn({ status: 'online' });
  } catch (err) {
    if (err.status === 0) connectionLost();
    throw err;
  }
}

// --- gallery --------------------------------------------------------------------

export function setArchived(archived) {
  if (state.gallery.archived === archived) return;
  state.gallery = { items: [], sources: [], cursor: null, done: false, loading: null, loaded: false, archived };
  emit('gallery');
}

/** Load the next page, or (fresh) reload the first page and keep what follows. */
export function loadGallery({ fresh = false } = {}) {
  const g = state.gallery;
  if (g.loading) return g.loading;
  if (!fresh && g.done) return Promise.resolve();
  const view = g;
  g.loading = (async () => {
    try {
      const page = await api.gallery({ cursor: fresh ? null : g.cursor, archived: g.archived });
      if (state.gallery !== view) return;
      for (const c of page.items) place(c.id, c.origin?.link_id);
      if (fresh || !g.loaded) g.sources = page.sources;
      const items = g.archived ? page.items.filter((c) => c.archived_at) : page.items;
      if (fresh) {
        const seen = new Set(items.map((c) => c.id));
        g.items = [...items, ...g.items.filter((c) => !seen.has(c.id))];
        if (!g.loaded) [g.cursor, g.done] = [page.next_cursor, !page.next_cursor];
      } else {
        const seen = new Set(g.items.map((c) => c.id));
        g.items.push(...items.filter((c) => !seen.has(c.id)));
        [g.cursor, g.done] = [page.next_cursor, !page.next_cursor];
      }
      g.loaded = true;
      for (const c of items) if (c.turn_running) watchTurn(c.id);
    } catch (err) {
      if (err.status === 0) connectionLost();
      else throw err;
    } finally {
      g.loading = null;
      emit('gallery');
    }
  })();
  return g.loading;
}

function patchGalleryItem(id, fn) {
  const item = state.gallery.items.find((c) => c.id === id);
  if (item) {
    fn(item);
    emit('gallery');
  }
}

function forgetGalleryItem(id) {
  state.gallery.items = state.gallery.items.filter((c) => c.id !== id);
  emit('gallery');
}

// --- conversations --------------------------------------------------------------

function conv(id) {
  let c = state.convs.get(id);
  if (!c) {
    c = {
      id, info: null, messages: new Map(), artifacts: new Map(), jobs: new Map(),
      lineage: new Map(), loading: null, loaded: false, missing: false,
      turn: null, // {running, lastEventId, ctrl, follow}
      error: null, // the last turn's error event, if it failed
      branches: [], // from the server: names, counts, forks
      sel: null, // the message whose newest branch is shown; null = the latest branch
      rev: 0, // bumped when messages are added or replaced, for the tree cache
      tree: null,
      branchesLoading: null, branchesStale: false,
    };
    state.convs.set(id, c);
  }
  return c;
}

export const getConv = (id) => state.convs.get(id) ?? null;

/** Load a conversation fully, then attach to its running turn, if any. */
export function ensureLoaded(id) {
  const c = conv(id);
  if (c.loaded) return Promise.resolve(c);
  if (c.loading) return c.loading;
  c.loading = (async () => {
    try {
      const [info, messages, artifacts, jobs, turn, branches] = await Promise.all([
        api.conversation(id), api.messages(id), api.artifacts(id), api.jobs(id), api.turn(id),
        api.branches(id),
      ]);
      c.info = info;
      c.branches = branches;
      for (const m of messages) if (!c.messages.has(m.id) || !c.turn?.running) c.messages.set(m.id, m);
      c.rev += 1;
      for (const a of artifacts) c.artifacts.set(a.id, a);
      for (const j of jobs) if (!c.jobs.has(j.id) || !c.turn?.running) c.jobs.set(j.id, j);
      c.loaded = true;
      // Replay the running turn from its start on top of this snapshot.
      if (turn?.running) attach(c, 0);
      return c;
    } catch (err) {
      if (err.status === 404) c.missing = true;
      throw err;
    } finally {
      c.loading = null;
      emit(`conv:${id}`);
    }
  })();
  return c.loading;
}

export const images = (c) =>
  [...c.artifacts.values()].filter((a) => a.kind === 'image').sort((a, b) => (a.id < b.id ? -1 : 1));

export const byLabel = (c, label) => [...c.artifacts.values()].find((a) => a.label === label) ?? null;

export const orderedMessages = (c) => [...c.messages.values()].sort((a, b) => a.seq - b.seq);

// --- branches -------------------------------------------------------------------

function treeOf(c) {
  if (c.tree?.rev === c.rev) return c.tree;
  const ordered = orderedMessages(c);
  const children = new Map(); // parent id ('' for first messages) -> children, oldest first
  const attachedBy = new Map(); // artifact id -> the first message attaching it
  for (const m of ordered) {
    const key = m.parent_id ?? '';
    if (!children.has(key)) children.set(key, []);
    children.get(key).push(m);
    for (const aid of m.attachments) if (!attachedBy.has(aid)) attachedBy.set(aid, m);
  }
  // The newest leaf under every message: walking up from the newest messages first, the
  // first leaf to reach a message is the newest one below it.
  const newest = new Map();
  for (let i = ordered.length - 1; i >= 0; i -= 1) {
    const leaf = ordered[i];
    for (let m = leaf; m && !newest.has(m.id); m = c.messages.get(m.parent_id)) newest.set(m.id, leaf);
  }
  c.tree = { rev: c.rev, ordered, children, attachedBy, newest };
  return c.tree;
}

/** The messages shown, oldest first: the path to the newest leaf under the selection. */
export function pathOf(c) {
  const t = treeOf(c);
  const anchor = c.sel ? t.newest.get(c.sel) : null;
  const path = [];
  for (let m = anchor ?? t.ordered[t.ordered.length - 1]; m; m = c.messages.get(m.parent_id)) path.push(m);
  return path.reverse();
}

export const leafOf = (c) => pathOf(c).at(-1) ?? null;

/** m and its alternatives at the fork, oldest first: edits of a user message, or attempts
 * at one tool call. */
export function siblingsOf(c, m) {
  return (treeOf(c).children.get(m.parent_id ?? '') ?? [])
    .filter((s) => s.role === m.role && (m.role !== 'tool' || s.tool_call_id === m.tool_call_id));
}

/** Show the newest branch through this message. */
export function select(c, messageId) {
  c.sel = messageId;
  emit(`conv:${c.id}`);
}

/** The server's branch whose leaf is the newest one under this message. */
export function branchAt(c, messageId) {
  const leaf = treeOf(c).newest.get(messageId);
  return c.branches.find((b) => b.leaf_id === leaf?.id) ?? null;
}

/** The branch shown. While a turn extends it, the list may still name its previous leaf. */
export function currentBranch(c) {
  const path = pathOf(c);
  const leaf = path.at(-1);
  if (!leaf) return null;
  const exact = branchAt(c, leaf.id);
  if (exact) return exact;
  const ids = new Set(path.map((m) => m.id));
  return c.branches.find((b) => ids.has(b.leaf_id)) ?? null;
}

/** The message that brought an artifact into the conversation (upload or tool result). */
export const attacherOf = (c, artifactId) => treeOf(c).attachedBy.get(artifactId) ?? null;

/** Artifact ids attached somewhere on the shown path. */
export const onPath = (c) => new Set(pathOf(c).flatMap((m) => m.attachments));

function loadBranches(c) {
  if (c.branchesLoading) {
    c.branchesStale = true;
    return;
  }
  c.branchesLoading = api.branches(c.id).then((list) => { c.branches = list; }, () => {}).finally(() => {
    c.branchesLoading = null;
    emit(`conv:${c.id}`);
    if (c.branchesStale) {
      c.branchesStale = false;
      loadBranches(c);
    }
  });
}

export async function lineageOf(c, artifactId) {
  if (!c.lineage.has(artifactId)) {
    c.lineage.set(artifactId, api.lineage(artifactId, c.id).catch(() => null));
  }
  return c.lineage.get(artifactId);
}

export async function rename(id, title) {
  const info = await api.updateConversation(id, { title: title || null });
  conv(id).info = info;
  patchGalleryItem(id, (item) => { item.title = info.title; });
  emit(`conv:${id}`);
}

export async function setArchivedFlag(id, archived) {
  const info = await api.updateConversation(id, { archived });
  conv(id).info = info;
  if (archived !== state.gallery.archived) forgetGalleryItem(id);
  emit(`conv:${id}`);
}

export async function remove(id) {
  await api.deleteConversation(id);
  state.convs.get(id)?.turn?.ctrl.abort();
  state.convs.delete(id);
  forgetGalleryItem(id);
}

// --- turns ----------------------------------------------------------------------

export const running = (id) => !!state.convs.get(id)?.turn?.running;

/** Progress of the turn's newest unfinished job as 0..1, or null if none. */
export function progressOf(id) {
  const c = state.convs.get(id);
  if (!c?.turn?.running) return null;
  const live = [...c.jobs.values()].filter((j) => j.status === 'running' || j.status === 'queued');
  const job = live[live.length - 1];
  const p = job?.progress;
  if (!p?.total_steps) return job ? 0 : null;
  return Math.min(1, ((p.image ?? 1) - 1 + p.step / p.total_steps) / (p.images ?? 1));
}

/**
 * Send a message: create the conversation first if there is none (cid = null), upload
 * new files, then start the turn. Resolves with the conversation id once the server has
 * accepted the turn; the turn then streams in the background. A brand-new conversation
 * whose first message is refused is deleted again, so a failed send leaves nothing behind.
 * The message continues the branch shown, or with editOf replaces that message on a new one.
 */
export async function send(cid, { text, attachments, options, editOf = null }) {
  const fresh = !cid;
  const c = fresh ? adopt(await api.createConversation(titleFrom(text))) : await ensureLoaded(cid);
  try {
    const ids = [];
    for (const att of attachments) {
      const artifact = att.artifact ?? (await api.upload(c.id, att.file));
      c.artifacts.set(artifact.id, artifact);
      ids.push(artifact.id);
    }
    const body = { content: text, attachments: ids };
    if (options && Object.keys(options).length) body.image_options = options;
    if (editOf) body.edit_of = editOf;
    else if (leafOf(c)) body.parent_id = leafOf(c).id;
    await start(c, (signal) => api.postMessage(c.id, body, signal));
  } catch (err) {
    if (fresh) {
      state.convs.delete(c.id);
      forgetGalleryItem(c.id);
      api.deleteConversation(c.id).catch(() => {});
    }
    throw err;
  }
  return c.id;
}

function adopt(info) {
  const c = conv(info.id);
  c.info = info;
  c.loaded = true;
  if (!state.gallery.archived) {
    state.gallery.items.unshift({ ...info, image_count: 0, cover: null, turn_running: false });
    emit('gallery');
  }
  return c;
}

/** Run a tool call again (toolMessageId is its result); the new attempt is shown. */
export async function retry(cid, toolMessageId) {
  const c = await ensureLoaded(cid);
  await start(c, (signal) => api.retry(c.id, toolMessageId, signal));
}

async function start(c, open) {
  const ctrl = new AbortController();
  const resp = await open(ctrl.signal);
  c.error = null;
  // follow: show the branch of the turn's first message (the new user message or attempt).
  c.turn = { running: true, lastEventId: 0, ctrl, follow: true };
  emit(`conv:${c.id}`, 'turns');
  follow(c, resp);
}

/** Watch a turn started elsewhere (another tab, before a reload) for gallery progress. */
export function watchTurn(id) {
  const c = conv(id);
  if (!c.turn?.running) attach(c, 0);
}

/** Follow a running turn from after its event `after`. counting: a resume after an outage,
 * whose replayed events are counted for the "caught up" note. */
async function attach(c, after, { counting = false } = {}) {
  c.turn?.ctrl.abort();
  const ctrl = new AbortController();
  c.turn = { running: true, lastEventId: after, lastEventAt: Date.now(), ctrl, counting };
  emit(`conv:${c.id}`, 'turns');
  try {
    follow(c, await api.attachTurn(c.id, after, ctrl.signal));
  } catch (err) {
    if (err.name === 'AbortError') return;
    if (err.status === 0) connectionLost(c);
    else finish(c, ctrl);
  }
}

async function follow(c, resp) {
  const { ctrl } = c.turn;
  try {
    await readEvents(resp, (type, id, data) => {
      if (c.turn?.ctrl !== ctrl) return;
      if (id != null) {
        c.turn.lastEventId = id;
        c.turn.lastEventAt = Date.now();
      }
      if (c.turn.counting && state.conn.status === 'restored') {
        state.conn = { ...state.conn, caughtUp: (state.conn.caughtUp ?? 0) + 1 };
        emit('conn');
      }
      apply(c, type, data);
    });
  } catch (err) {
    if (err.name === 'AbortError') return;
  }
  if (c.turn?.ctrl !== ctrl || !c.turn.running) return;
  // The stream dropped before the turn ended: pick it up where it stopped, if it still runs.
  let turn;
  try {
    turn = await api.turn(c.id);
  } catch (err) {
    if (err.status === 0) {
      connectionLost(c);
      return;
    }
    turn = null;
  }
  if (turn?.running) attach(c, c.turn.lastEventId);
  else {
    finish(c, ctrl);
    refresh(c);
  }
}

/** The turn's stream is down and waiting for the connection to come back. */
export const stale = (id) => resumeQueue.has(id) && offline();

function finish(c, ctrl) {
  if (c.turn?.ctrl !== ctrl) return;
  c.turn.running = false;
  emit(`conv:${c.id}`, 'turns');
}

async function refresh(c) {
  try {
    const [messages, artifacts, jobs] = await Promise.all([
      api.messages(c.id), api.artifacts(c.id), api.jobs(c.id),
    ]);
    for (const m of messages) c.messages.set(m.id, m);
    c.rev += 1;
    loadBranches(c);
    for (const a of artifacts) c.artifacts.set(a.id, a);
    for (const j of jobs) c.jobs.set(j.id, j);
  } finally {
    emit(`conv:${c.id}`);
  }
}

function apply(c, type, data) {
  switch (type) {
    case 'message': {
      const m = { ...data.message };
      if (m.status === 'streaming' && m.role === 'assistant') {
        m.content_text ??= '';
        m.reasoning_text ??= '';
      }
      const isNew = !c.messages.has(m.id);
      c.messages.set(m.id, m);
      c.rev += 1;
      if (isNew && c.turn?.follow) {
        c.turn.follow = false;
        c.sel = m.id;
      }
      loadBranches(c);
      break;
    }
    case 'delta': {
      const m = c.messages.get(data.message_id);
      if (!m) break;
      if (data.content) m.content_text = (m.content_text ?? '') + data.content;
      if (data.reasoning) m.reasoning_text = (m.reasoning_text ?? '') + data.reasoning;
      break;
    }
    case 'job_update': {
      const before = c.jobs.get(data.job.id)?.status;
      c.jobs.set(data.job.id, data.job);
      if (data.job.status === 'succeeded' && before && before !== 'succeeded') {
        announce(`${data.job.outputs.length === 1 ? 'Image' : `${data.job.outputs.length} images`} ready`);
      }
      emit('turns');
      break;
    }
    case 'artifact_created': {
      const a = data.artifact;
      const isNew = !c.artifacts.has(a.id);
      c.artifacts.set(a.id, a);
      if (isNew && a.kind === 'image') {
        patchGalleryItem(c.id, (item) => {
          item.image_count += 1;
          item.cover = a;
        });
      }
      loadBranches(c);
      break;
    }
    case 'done':
    case 'error':
      if (type === 'error') c.error = data;
      c.turn.running = false;
      emit('turns');
      loadBranches(c);
      api.conversation(c.id).then((info) => {
        c.info = info;
        patchGalleryItem(c.id, (item) => Object.assign(item, info, { turn_running: false }));
        emit(`conv:${c.id}`);
      }, () => {});
      break;
    default:
      break;
  }
  emit(`conv:${c.id}`);
}

export async function stop(id) {
  await api.cancelTurn(id);
}

export async function cancelJob(c, jobId) {
  const job = await api.cancelJob(jobId, c.id);
  c.jobs.set(job.id, job);
  emit(`conv:${c.id}`);
}

// --- drafts and quick options ---------------------------------------------------

const drafts = new Map(); // conversation id, or 'new' for the gallery bar

export function draft(key) {
  if (!drafts.has(key)) drafts.set(key, { text: '', attachments: [] });
  return drafts.get(key);
}

/** Empty a draft in place: an open bar keeps editing the same object. */
export function clearDraft(key) {
  const d = draft(key);
  for (const att of d.attachments) if (att.preview) URL.revokeObjectURL(att.preview);
  d.text = '';
  d.attachments.length = 0;
}

const OPTIONS_KEY = 'imaginer.imageOptions';

/** {aspect_ratio, n, seedFixed, seed, backend, sizeExact, width, height, steps, cfg_scale,
 * negative, transparent}; null fields mean "let the model (or the backend) decide". */
export const options = (() => {
  const base = {
    aspect_ratio: null, n: null, seedFixed: false, seed: 41207, backend: null,
    sizeExact: false, width: 1184, height: 888, steps: null, cfg_scale: null, negative: '', transparent: false,
  };
  try {
    return { ...base, ...JSON.parse(localStorage.getItem(OPTIONS_KEY) || '{}') };
  } catch {
    return base;
  }
})();

/** quiet: store without re-rendering, e.g. while a text field is being typed in. */
export function setOption(patch, { quiet = false } = {}) {
  Object.assign(options, patch);
  try {
    localStorage.setItem(OPTIONS_KEY, JSON.stringify(options));
  } catch {
    // Not persisted in private mode; still applies to this page.
  }
  if (!quiet) emit('options');
}

/** The image_options body for a message: only what the user actually pinned. */
export function requestOptions() {
  const out = {};
  if (options.sizeExact && options.width && options.height) {
    out.width = options.width;
    out.height = options.height;
  } else if (options.aspect_ratio) out.aspect_ratio = options.aspect_ratio;
  if (options.n) out.n = options.n;
  if (options.seedFixed && Number.isInteger(options.seed)) out.seed = options.seed;
  if (options.steps) out.steps = options.steps;
  if (options.cfg_scale != null) out.cfg_scale = options.cfg_scale;
  if (options.negative.trim()) out.negative_prompt = options.negative.trim();
  if (options.transparent && imageEngine()?.capabilities.includes('rgba')) out.transparent = true;
  const backend = options.backend && imageBackends().find((b) => b.name === options.backend);
  if (backend && !backend.default_for.includes('t2i')) out.backend = backend.name;
  return out;
}

/** The image backend the next message will use. */
export const imageEngine = () =>
  (options.backend && imageBackends().find((b) => b.name === options.backend)) || defaultBackend('t2i') || null;
