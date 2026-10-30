// The /v1 client: JSON calls, the turn's server-sent events over fetch, authed images.
//
// Paths are relative, so the UI also works when the server sits behind a path prefix.
// When the server has a bearer token, the user enters it once; it stays in this browser.
// A conversation on a linked instance is reached through this server's proxy: calls about
// it take the prefix origin.js knows for it, so callers never have to.

import { baseOf } from './origin.js';

const TOKEN_KEY = 'imaginer.token';

let token = readToken();
let onUnauthorized = () => {};

function readToken() {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(value) {
  token = value || null;
  blobs.clear();
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    // Private mode: the token lasts for this page only.
  }
}

export const whenUnauthorized = (fn) => { onUnauthorized = fn; };

export class ApiError extends Error {
  constructor(status, problem) {
    super(problem?.detail || problem?.title || `request failed (${status})`);
    this.status = status;
    this.code = problem?.code || 'http_error';
    this.problem = problem;
  }
}

const rel = (path) => path.replace(/^\//, '');

async function request(method, path, { json, body, headers = {}, signal } = {}) {
  const sent = { ...headers };
  if (token) sent.Authorization = `Bearer ${token}`;
  if (json !== undefined) {
    sent['Content-Type'] = 'application/json';
    body = JSON.stringify(json);
  }
  let resp;
  try {
    resp = await fetch(rel(path), { method, headers: sent, body, signal });
  } catch (err) {
    if (err.name === 'AbortError') throw err;
    throw new ApiError(0, { code: 'network_error', detail: 'the server is unreachable' });
  }
  if (resp.ok) return resp;
  let problem = null;
  try {
    problem = await resp.json();
  } catch {
    // Not a problem+json body.
  }
  const error = new ApiError(resp.status, problem);
  if (resp.status === 401) onUnauthorized(error);
  throw error;
}

const getJson = async (path, opts) => (await request('GET', path, opts)).json();
const sendJson = async (method, path, json) => {
  const resp = await request(method, path, { json });
  return resp.status === 204 ? null : resp.json();
};

/** Every page of a cursor-paged collection. */
async function allPages(path, params = {}) {
  const items = [];
  let cursor = null;
  do {
    const q = new URLSearchParams({ ...params, limit: 500, ...(cursor ? { cursor } : {}) });
    const page = await getJson(`${path}?${q}`);
    items.push(...page.items);
    cursor = page.next_cursor;
  } while (cursor);
  return items;
}

const at = (id, path) => `${baseOf(id)}${path}`;

export const api = {
  /** This server's backends, or (base) those of the linked instance behind that prefix. */
  backends: (base = '') => getJson(`${base}v1/backends`),

  conversations({ cursor = null, limit = 48, archived = false } = {}) {
    const q = new URLSearchParams({ limit });
    if (cursor) q.set('cursor', cursor);
    if (archived) q.set('include_archived', 'true');
    return getJson(`v1/conversations?${q}`);
  },
  /** This server's conversations and every linked instance's: items carry their origin. */
  gallery({ cursor = null, limit = 48, archived = false } = {}) {
    const q = new URLSearchParams({ limit });
    if (cursor) q.set('cursor', cursor);
    if (archived) q.set('include_archived', 'true');
    return getJson(`v1/gallery?${q}`);
  },
  /** One page of a server-side search: {items (with .match), scanned, total, next_cursor}. */
  search({ q, fields = null, archived = false, cursor = null, limit = 60 }) {
    const p = new URLSearchParams({ q, limit });
    if (fields) p.set('fields', fields);
    if (archived) p.set('include_archived', 'true');
    if (cursor) p.set('cursor', cursor);
    return getJson(`v1/conversations/search?${p}`);
  },
  conversation: (id) => getJson(at(id, `v1/conversations/${id}`)),
  createConversation: (title) => sendJson('POST', 'v1/conversations', { title }),
  updateConversation: (id, patch) => sendJson('PATCH', at(id, `v1/conversations/${id}`), patch),
  deleteConversation: (id) => sendJson('DELETE', at(id, `v1/conversations/${id}`)),

  /** Every message of every branch; parent_id links them into a tree. */
  async messages(id) {
    const items = [];
    let after = 0;
    for (;;) {
      const page = await getJson(at(id, `v1/conversations/${id}/messages?all_branches=true&after_seq=${after}&limit=500`));
      items.push(...page.items);
      if (!page.has_more || !page.items.length) return items;
      after = page.items[page.items.length - 1].seq;
    }
  },
  branches: (id) => getJson(at(id, `v1/conversations/${id}/branches`)),
  artifacts: (id) => allPages(at(id, `v1/conversations/${id}/artifacts`)),
  jobs: (id) => allPages(at(id, `v1/conversations/${id}/jobs`)),
  lineage: (artifactId, conversationId) => getJson(at(conversationId, `v1/artifacts/${artifactId}/lineage`)),

  /** kind 'mask' uploads a painted mask (labelled mask-N) for the image labelled maskFor. */
  async upload(conversationId, file, { kind = null, maskFor = null } = {}) {
    const form = new FormData();
    form.append('file', file, file.name || 'image');
    const q = new URLSearchParams();
    if (kind) q.set('kind', kind);
    if (maskFor) q.set('mask_for', maskFor);
    const query = q.size ? `?${q}` : '';
    const resp = await request('POST', at(conversationId, `v1/conversations/${conversationId}/artifacts${query}`), { body: form });
    return resp.json();
  },

  async turn(conversationId) {
    try {
      return await getJson(at(conversationId, `v1/conversations/${conversationId}/turn`));
    } catch (err) {
      if (err.status === 404) return null;
      throw err;
    }
  },
  cancelTurn: (conversationId) => sendJson('POST', at(conversationId, `v1/conversations/${conversationId}/turn/cancel`)),
  cancelJob: (jobId, conversationId) => sendJson('POST', at(conversationId, `v1/jobs/${jobId}/cancel`)),

  /** Send a message. Resolves once the turn is accepted; read it with readEvents. */
  postMessage: (conversationId, body, signal) =>
    request('POST', at(conversationId, `v1/conversations/${conversationId}/messages`), {
      json: body, signal, headers: { Accept: 'text/event-stream' },
    }),

  /** Run a tool call again as a new attempt (a new branch); streams like postMessage. */
  retry: (conversationId, toolMessageId, signal) =>
    request('POST', at(conversationId, `v1/conversations/${conversationId}/messages/${toolMessageId}/retry`), {
      signal, headers: { Accept: 'text/event-stream' },
    }),

  // --- engines (designs 1a–1d) ---
  engines: () => getJson('v1/engines'),
  engine: (name) => getJson(`v1/engines/${name}`),
  updateEngine: (name, patch) => sendJson('PATCH', `v1/engines/${name}`, patch),
  startEngine: (name) => sendJson('POST', `v1/engines/${name}/start`),
  stopEngine: (name) => sendJson('POST', `v1/engines/${name}/stop`),
  engineLog: (name, errors = false) => getJson(`v1/engines/${name}/log${errors ? '?errors=true' : ''}`),
  installPlan: (name, body) => sendJson('POST', `v1/engines/${name}/install/plan`, body),
  install: (name, body) => sendJson('POST', `v1/engines/${name}/install`, body),
  cancelInstall: (name) => sendJson('POST', `v1/engines/${name}/install/cancel`),
  checkUpdate: (name) => getJson(`v1/engines/${name}/update`),
  /** Live `line`, `state` and `install` events; read them with readEvents. */
  engineEvents: (name, signal) =>
    request('GET', `v1/engines/${name}/events`, { signal, headers: { Accept: 'text/event-stream' } }),
  async engineLogFile(name) {
    const resp = await request('GET', `v1/engines/${name}/log/file`);
    return resp.text();
  },

  // --- linked instances ---
  links: () => getJson('v1/links'),
  discovered: () => getJson('v1/links/discovered'),
  pairingCode: () => sendJson('POST', 'v1/links/pairing-code'),
  link: (body) => sendJson('POST', 'v1/links', body),
  unlink: (id) => sendJson('DELETE', `v1/links/${id}`),
  refreshLink: (id) => sendJson('POST', `v1/links/${id}/refresh`),
  useLinked: (id, body) => sendJson('POST', `v1/links/${id}/use`, body),
  linkSettings: (patch) => sendJson('PATCH', 'v1/links/settings', patch),

  /** Re-attach to a running turn; its events after lastEventId are replayed first. */
  attachTurn: (conversationId, lastEventId, signal) =>
    request('GET', at(conversationId, `v1/conversations/${conversationId}/turn/events`), {
      signal, headers: { Accept: 'text/event-stream', 'Last-Event-ID': String(lastEventId) },
    }),
};

/** Feed a text/event-stream response to onEvent(type, id, data) until it ends. */
export async function readEvents(resp, onEvent) {
  const reader = resp.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let end;
    while ((end = buffer.search(/\r?\n\r?\n/)) !== -1) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end).replace(/^\r?\n\r?\n/, '');
      dispatch(block, onEvent);
    }
  }
  if (buffer.trim()) dispatch(buffer, onEvent);
}

function dispatch(block, onEvent) {
  let type = 'message';
  let id = null;
  const data = [];
  for (const line of block.split(/\r?\n/)) {
    if (!line || line.startsWith(':')) continue;
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? '' : line.slice(colon + 1).replace(/^ /, '');
    if (field === 'event') type = value;
    else if (field === 'id') id = Number(value);
    else if (field === 'data') data.push(value);
  }
  if (!data.length) return;
  onEvent(type, id, JSON.parse(data.join('\n')));
}

// --- image bytes ----------------------------------------------------------------
// Without a token, <img> loads content URLs directly (and the browser caches them, the
// server marks them immutable). With one, bytes go through fetch so the header is sent.

const blobs = new Map();

const contentOf = (artifact) => at(artifact.conversation_id, rel(artifact.content_url));

export function setImage(img, artifact) {
  const src = contentOf(artifact);
  if (!token) {
    img.src = src;
    return;
  }
  let pending = blobs.get(src);
  if (!pending) {
    pending = request('GET', src).then((r) => r.blob()).then((b) => URL.createObjectURL(b));
    pending.catch(() => blobs.delete(src));
    blobs.set(src, pending);
  }
  pending.then((u) => { img.src = u; }, () => {});
}

export async function download(artifact) {
  const resp = await request('GET', contentOf(artifact));
  const href = URL.createObjectURL(await resp.blob());
  const ext = artifact.mime.split('/').pop().replace('jpeg', 'jpg');
  const a = Object.assign(document.createElement('a'), { href, download: `${artifact.label}.${ext}` });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(href), 30_000);
}
