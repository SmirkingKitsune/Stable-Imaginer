// Entry point: hash routes, screen lifecycle, drag-and-drop onto the bar, and the token
// prompt for servers that require one.
//
//   #/                 gallery            #/archived      archived conversations
//   #/c/<id>           viewer (latest)    #/c/<id>/img-3  viewer on img-3
//   #/c/<id>/chat      conversation       #/c/<id>/mask/img-3  mask editor for img-3
//   #/engines          engine settings    #/engines/sdcpp      on one backend
//   #/links            linked instances
//   #/l/<link>/c/...   the same conversation routes, for one that lives on a linked instance

import { setToken, whenUnauthorized } from './api.js';
import { dialog } from './chrome.js';
import { h } from './dom.js';
import { conversationScreen } from './conversation.js';
import { enginesScreen } from './engines.js';
import { galleryScreen } from './gallery.js';
import { linksScreen } from './links.js';
import { maskScreen } from './mask.js';
import { baseOf, place } from './origin.js';
import { backendsFrom, loadBackends } from './store.js';
import { viewerScreen } from './viewer.js';

const BACKENDS_REFRESH_MS = 30_000;

const root = document.getElementById('app');
let screen = null;
let pendingFocus = false;

function navigate(hash, { replace = false, focusBar = false } = {}) {
  pendingFocus = focusBar;
  if (location.hash === hash) return;
  if (replace) {
    history.replaceState(null, '', hash);
    route();
  } else {
    location.hash = hash;
  }
}

function parse(hash) {
  let parts = hash.replace(/^#\/?/, '').split('/').filter(Boolean).map(decodeURIComponent);
  let link = null;
  if (parts[0] === 'l' && parts[1] && parts[2] === 'c') [link, parts] = [parts[1], parts.slice(2)];
  if (parts[0] === 'archived') return { name: 'gallery', archived: true };
  if (parts[0] === 'engines') return { name: 'engines', engine: parts[1] ?? null };
  if (parts[0] === 'links') return { name: 'links' };
  if (parts[0] === 'c' && parts[1]) {
    if (parts[2] === 'chat') return { name: 'conversation', id: parts[1], link };
    if (parts[2] === 'mask' && parts[3]) return { name: 'mask', id: parts[1], label: parts[3], link };
    return { name: 'viewer', id: parts[1], label: parts[2] ?? null, link };
  }
  return { name: 'gallery', archived: false };
}

let mounted = null; // the route the current screen was built for

function route() {
  const r = parse(location.hash);
  // The viewer and the Engines screen rewrite their own hash; that is not a new screen.
  if (mounted && r.name === 'viewer' && mounted.name === 'viewer' && mounted.id === r.id) return;
  if (mounted && r.name === 'engines' && mounted.name === 'engines') {
    if (r.engine) screen.show(r.engine); // a link, or back/forward, to another engine
    mounted = r;
    return;
  }
  const focusBar = pendingFocus;
  pendingFocus = false;
  screen?.destroy();
  root.replaceChildren();
  if (r.id) place(r.id, r.link);
  backendsFrom(r.id ? baseOf(r.id) : '');
  if (r.name === 'conversation') screen = conversationScreen({ id: r.id, navigate, focusBar });
  else if (r.name === 'mask') screen = maskScreen({ id: r.id, label: r.label, navigate });
  else if (r.name === 'engines') screen = enginesScreen({ name: r.engine, navigate });
  else if (r.name === 'links') screen = linksScreen({ navigate });
  else if (r.name === 'viewer') screen = viewerScreen({ id: r.id, label: r.label, navigate });
  else screen = galleryScreen({ archived: r.archived, navigate });
  mounted = r;
  root.append(screen.el);
}

// --- drop images anywhere onto the current bar -------------------------------------

const dropHint = h('div.drop-hint', { hidden: true }, h('div.drop-box.blueprint', 'Drop images to attach'));
document.body.append(dropHint);
let dragDepth = 0;
const hasFiles = (e) => [...(e.dataTransfer?.types ?? [])].includes('Files');

window.addEventListener('dragenter', (e) => {
  if (!hasFiles(e)) return;
  dragDepth += 1;
  dropHint.hidden = false;
});
window.addEventListener('dragleave', (e) => {
  if (!hasFiles(e)) return;
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) dropHint.hidden = true;
});
window.addEventListener('dragover', (e) => { if (hasFiles(e)) e.preventDefault(); });
window.addEventListener('drop', (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  dropHint.hidden = true;
  screen?.bar?.addFiles(e.dataTransfer.files);
});

// --- access token ---------------------------------------------------------------------

let asking = false;
whenUnauthorized(async () => {
  if (asking) return;
  asking = true;
  const value = await dialog({
    title: 'Access token',
    body: 'This server requires a token (IMAGINER_SERVER__AUTH_TOKEN). It is kept in this browser.',
    input: { type: 'password', placeholder: 'Token' },
    confirm: 'Connect',
  });
  asking = false;
  if (value) {
    setToken(value.trim());
    location.reload();
  }
});

window.addEventListener('hashchange', route);
loadBackends();
setInterval(() => { if (!document.hidden) loadBackends(); }, BACKENDS_REFRESH_MS);
route();
