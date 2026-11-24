// The artifact library: every image uploaded or made in any conversation, on this server and
// on linked instances, newest first and each distinct image once. Pick images to use in the
// next prompt: they are copied into that conversation when it is sent, where they get a
// label of their own (and can be @-mentioned like any other).
//
// #/artifacts: a masonry with filters (images or masks, where they came from, pinned, which
// instance) and a server-side search over prompts, file names, labels, titles and pins.
// Selecting tiles shows a bar to use, pin or unpin them together; a tile opens its details:
// how it was made, where it is, and the same actions.
//
// Pinning keeps a copy of an image in the library, so it outlives its conversation (and the
// linked instance it came from). An image whose conversation is gone is listed by its pin.
//
// libraryPicker() is the same library as a dialog: "From the artifact library" in the
// prompt bar's paperclip menu.

import { api, downloadLibraryItem, setLibraryImage } from './api.js';
import { announce, connectionStrip, dialog, modal, modelName, statusLine, toast } from './chrome.js';
import { corners, h, icon, iconButton, clear } from './dom.js';
import { ago, loraName, plural, truncate } from './format.js';
import { convHash, place } from './origin.js';
import { addLibraryToDraft, draft, lastPlace, placeDraft } from './store.js';

const PREFS_KEY = 'imaginer.artifacts';
const PAGE = 60;
const MAX_ATTACHMENTS = 16; // MessageCreate.attachments
const SEARCH_DEBOUNCE_MS = 250;
const SOURCES = [['all', 'All'], ['generated', 'Generated'], ['edited', 'Edited'], ['upload', 'Uploaded']];
const KINDS = [['image', 'Images'], ['mask', 'Masks'], ['all', 'Both']];
const SOURCE_WORD = { upload: 'uploaded', generated: 'generated', edited: 'edited', derived: 'derived' };

const keyOf = (item) => `${item.origin.link_id ?? 'local'}:${item.id}`;
const titleOf = (item) => item.pin?.name || item.conversation_title || item.original_filename || item.label || 'Image';
const bytes = (n) => (!n ? '' : n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1e3))} KB`);

function remembered() {
  const base = { kind: 'image', source: 'all', pinned: false, origin: null };
  try {
    return { ...base, ...JSON.parse(localStorage.getItem(PREFS_KEY) || '{}') };
  } catch {
    return base;
  }
}

function remember(prefs) {
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
  } catch {
    // Private mode: kept for this page only.
  }
}

/**
 * Pages of the library for one set of filters. A page can come back short (even empty)
 * while there is more: an image this server already has is left out of a linked
 * instance's page. more() keeps asking until it got something or the end.
 */
function pager(params) {
  const state = { items: [], sources: [], cursor: null, done: false, loading: null, error: null };
  const seen = new Set();
  async function more() {
    if (state.done) return state;
    if (state.loading) return state.loading;
    state.loading = (async () => {
      try {
        for (let round = 0; round < 5 && !state.done; round += 1) {
          const page = await api.library({ ...params(), cursor: state.cursor, limit: PAGE });
          state.sources = page.sources;
          state.cursor = page.next_cursor;
          state.done = !page.next_cursor;
          const fresh = page.items.filter((i) => !seen.has(keyOf(i)));
          fresh.forEach((i) => seen.add(keyOf(i)));
          state.items.push(...fresh);
          if (fresh.length) break;
        }
        state.error = null;
      } catch (err) {
        state.error = err.message;
        state.done = true;
      } finally {
        state.loading = null;
      }
      return state;
    })();
    return state.loading;
  }
  return { state, more };
}

/** "nas offline · old-box needs an update" for the sources that didn't list. */
function sourcesNote(sources) {
  const off = sources.filter((s) => s.status !== 'ok');
  return {
    text: off.map((s) => `${s.name} ${s.status === 'unreachable' ? 'offline'
      : s.status === 'unsupported' ? 'needs an update' : 'unavailable'}`).join(' · '),
    title: off.map((s) => `${s.name}: ${s.detail ?? s.status}`).join('\n'),
  };
}

// --- using images ------------------------------------------------------------------------

/** Put images into the prompt the user was working on, and go back there. */
function useInNextPrompt(items, navigate, { edit = false } = {}) {
  const key = placeDraft();
  const d = draft(key);
  const room = MAX_ATTACHMENTS - d.attachments.length;
  if (room <= 0) {
    toast(`The next prompt has ${MAX_ATTACHMENTS} images already`, { error: true });
    return;
  }
  const added = addLibraryToDraft(key, items, { max: MAX_ATTACHMENTS });
  if (edit && !d.text.trim()) {
    const att = d.attachments.find((a) => (a.library?.sha256 ?? a.artifact?.sha256) === items[0].sha256);
    const ref = att?.artifact?.label ?? att?.ref;
    if (ref) d.text = `Edit @${ref}: `;
  }
  if (items.length > room) toast(`Only ${plural(room, 'image')} fit in the next prompt`);
  else if (!added) toast(items.length === 1 ? 'It is in the next prompt already' : 'They are in the next prompt already');
  navigate(key === 'new' ? lastPlace() : convHash(key, 'chat'), { focusBar: true });
}

async function pin(item) {
  const kept = await api.pinImage({ ref: item.ref });
  announce('Pinned to the artifact library');
  return kept;
}

/** Unpin; one whose conversation is gone goes for good, so that asks first. */
async function unpin(item) {
  if (!item.pin) return item;
  if (!item.live) {
    const ok = await dialog({
      title: 'Unpin this image?',
      body: 'Its conversation is gone: the pin is the only copy left, and it is deleted with it.',
      confirm: 'Unpin and delete', danger: true,
    });
    if (!ok) return item;
  }
  await api.unpin(item.pin.id);
  announce('Unpinned');
  return item.live ? { ...item, pin: null } : null;
}

function openConversation(item, navigate) {
  if (!item.live || !item.conversation_id) return;
  place(item.conversation_id, item.origin.link_id);
  navigate(convHash(item.conversation_id, item.label));
}

// --- the screen ----------------------------------------------------------------------------

export function artifactsScreen({ navigate }) {
  const prefs = remembered();
  let query = '';
  let debounce = null;
  let gone = false;
  let run = 0;
  let list = pager(() => params());
  const checked = new Map(); // key -> item
  const tiles = new Map(); // key -> {el, sig, ratio}

  const params = () => ({
    kind: prefs.kind,
    source: prefs.source === 'all' ? [] : [prefs.source],
    pinned: prefs.pinned,
    origin: prefs.origin,
    q: query,
  });

  const status = statusLine();
  const meta = h('span.topbar-meta');
  const offNote = h('span.topbar-tag.sources-off', { hidden: true });
  const search = h('input.search-input', {
    type: 'search', placeholder: 'Search prompts, files, titles, pins', 'aria-label': 'Search the artifact library',
    oninput: () => {
      clearTimeout(debounce);
      debounce = setTimeout(() => { query = search.value.trim(); reload(); }, SEARCH_DEBOUNCE_MS);
    },
    onkeydown: (e) => {
      if (e.key === 'Escape') { search.value = ''; query = ''; reload(); search.blur(); }
    },
  });
  const header = h('header.topbar.sub',
    iconButton('back', 'Back', () => navigate(lastPlace())),
    h('div.topbar-heading', h('span.topbar-title', 'Artifact library'), meta),
    offNote,
    h('span.spacer'),
    h('label.search.lib-search', icon('search', 15), search, h('kbd', '/')),
    h('div.only-wide', status.el));
  const bar = h('div.lib-bar');
  const columns = h('div.masonry.art-masonry');
  const notice = h('div');
  const sentinel = h('div.sentinel');
  const end = h('div.lib-end');
  const scroller = h('div.gallery-scroll.lib-scroll', bar, notice, columns, sentinel, end);
  const bulk = h('div.pbar.lib-bulk', { hidden: true, role: 'toolbar', 'aria-label': 'Selected images' });
  const strip = connectionStrip();
  const el = h('div.screen.loras.artifacts', header, strip.el, scroller, bulk);

  function reload() {
    run += 1;
    list = pager(() => params());
    tiles.clear();
    checked.clear();
    render();
    loadMore();
  }

  async function loadMore() {
    const mine = run;
    const before = list.state.items.length;
    render();
    await list.more();
    if (gone || mine !== run) return;
    render();
    // Still room under the tiles: keep filling.
    if (!list.state.done && list.state.items.length > before && sentinelVisible()) loadMore();
  }

  const sentinelVisible = () => sentinel.getBoundingClientRect().top < scroller.getBoundingClientRect().bottom + 900;

  // --- filters ---------------------------------------------------------------------------

  function renderBar() {
    const seg = (options, key, label) => h('div.tray-seg.lib-filters', { role: 'group', 'aria-label': label },
      options.map(([value, word]) => h('button.pick', {
        type: 'button', class: prefs[key] === value ? 'on' : '', 'aria-pressed': String(prefs[key] === value),
        onclick: () => { prefs[key] = value; remember(prefs); reload(); },
      }, word)));
    const pinnedBtn = h('button.pick', {
      type: 'button', class: prefs.pinned ? 'on' : '', 'aria-pressed': String(prefs.pinned),
      title: 'Only images kept in the library',
      onclick: () => { prefs.pinned = !prefs.pinned; remember(prefs); reload(); },
    }, icon('pin', 13), 'Pinned');
    const sources = list.state.sources;
    const origins = sources.length > 1 || prefs.origin
      ? h('div.tray-seg.lib-filters', { role: 'group', 'aria-label': 'Instance' },
        [['', 'Everywhere'], ...sources.map((s) => [s.id, s.name])].map(([value, word]) => {
          const on = (prefs.origin ?? '') === value;
          return h('button.pick', {
            type: 'button', class: on ? 'on' : '', 'aria-pressed': String(on),
            onclick: () => { prefs.origin = value || null; remember(prefs); reload(); },
          }, value && value !== 'local' && icon('link', 12), word);
        }))
      : null;
    clear(bar).append(...[
      seg(SOURCES, 'source', 'Source'),
      h('div.tray-seg.lib-filters', pinnedBtn),
      origins,
      h('span.spacer'),
      seg(KINDS, 'kind', 'Kind'),
    ].filter(Boolean));
  }

  // --- tiles -------------------------------------------------------------------------------

  const sigOf = (item) => JSON.stringify([item.pin?.id, item.pin?.name, item.live, checked.has(keyOf(item))]);

  function tile(item) {
    const key = keyOf(item);
    const sig = sigOf(item);
    const cached = tiles.get(key);
    if (cached?.sig === sig) return cached;
    const img = h('img.tile-pic', { alt: titleOf(item), loading: 'lazy', decoding: 'async' });
    setLibraryImage(img, item);
    const remote = item.origin.link_id ? item.origin.name : null;
    const check = h('input', {
      type: 'checkbox', checked: checked.has(key), 'aria-label': `Select ${titleOf(item)}`,
      onchange: () => {
        if (check.checked) checked.set(key, item);
        else checked.delete(key);
        node.classList.toggle('on', check.checked); // in place: no new tile, so focus stays
        entry.sig = sigOf(item);
        renderBulk();
      },
    });
    const node = h('div.tile.art-tile', { class: checked.has(key) ? 'on' : '', dataset: { key } },
      h('button.art-open', { type: 'button', 'aria-label': `${titleOf(item)}, details`, onclick: () => openDetail(item) },
        h('div.tile-img', { style: { aspectRatio: item.width ? `${item.width} / ${item.height}` : '4 / 3' } }, img),
        h('div.tile-hover.blueprint', corners(),
          h('span.label-chip', item.label ?? 'pinned'),
          h('div.tile-foot', h('div.tile-foot-text',
            h('span.tile-title', titleOf(item)),
            h('span.mono-meta', [SOURCE_WORD[item.source], item.kind === 'mask' && 'mask', ago(item.created_at)].filter(Boolean).join(' · ')))))),
      h('label.art-check', { title: 'Select' }, check),
      h('div.art-badges',
        item.pin && h('span.tile-origin.art-badge', { title: item.live ? 'Pinned' : 'Pinned · its conversation is gone' }, icon('pin', 11), item.live ? 'pinned' : 'kept'),
        remote && h('span.tile-origin.art-badge', { title: `On ${remote}` }, icon('link', 11), remote)));
    const entry = { el: node, sig, ratio: item.width ? item.height / item.width : 0.75 };
    tiles.set(key, entry);
    return entry;
  }

  function columnCount() {
    const w = scroller.clientWidth || window.innerWidth;
    return w >= 1600 ? 6 : w >= 1200 ? 5 : w >= 900 ? 4 : w >= 600 ? 3 : 2;
  }

  function render() {
    if (gone) return;
    renderBar();
    const s = list.state;
    const note = sourcesNote(s.sources);
    offNote.hidden = !note.text;
    offNote.textContent = note.text;
    offNote.title = note.title;
    meta.textContent = s.items.length ? `${plural(s.items.length, 'image')}${s.done ? '' : '+'}` : '';
    clear(notice);
    if (s.error) notice.append(h('div.notice', s.error));
    else if (!s.items.length && s.done) {
      notice.append(h('div.notice.lib-empty',
        h('span.eyebrow.accent', query || prefs.pinned || prefs.source !== 'all' || prefs.origin ? 'Nothing here' : 'No images yet'),
        h('span', query ? `Nothing matches “${query}”.`
          : prefs.pinned ? 'Pin an image to keep it here, even after its conversation is deleted.'
            : 'Images you upload or make in any conversation show up here, ready for the next prompt.')));
    } else if (!s.items.length) {
      notice.append(h('div.notice', 'Reading the library…'));
    }
    const n = columnCount();
    const cols = Array.from({ length: n }, () => h('div.masonry-col'));
    const heights = new Array(n).fill(0);
    for (const item of s.items) {
      const entry = tile(item);
      const i = heights.indexOf(Math.min(...heights));
      cols[i].append(entry.el);
      heights[i] += entry.ratio + 0.05;
    }
    clear(columns).append(...cols);
    clear(end);
    if (s.done && s.items.length) {
      end.append(h('div.rule-label', h('span.rule'), h('span', `END OF LIBRARY · ${plural(s.items.length, 'IMAGE')}`), h('span.rule')));
    }
    renderBulk();
  }

  function replace(item, next) {
    const s = list.state;
    const at = s.items.indexOf(item);
    if (at === -1) return;
    tiles.delete(keyOf(item));
    checked.delete(keyOf(item));
    if (next) {
      // A pin of a linked instance's image is a copy here: it stands where that image stood.
      s.items.splice(at, 1, next);
    } else {
      s.items.splice(at, 1);
    }
    render();
  }

  // --- the bulk bar ------------------------------------------------------------------------

  function renderBulk() {
    const items = [...checked.values()];
    el.classList.toggle('selecting', items.length > 0);
    bulk.hidden = !items.length;
    if (bulk.hidden) return;
    const pinnable = items.filter((i) => !i.pin);
    const pinned = items.filter((i) => i.pin);
    clear(bulk).append(
      h('span.lib-bulk-count', `${items.length} selected`),
      h('button.mini.primary', { type: 'button', onclick: () => useInNextPrompt(items, navigate) }, icon('arrowRight', 13), 'Use in next prompt'),
      h('span.lib-bulk-sep'),
      h('button.mini', { type: 'button', disabled: !pinnable.length, onclick: () => bulkPin(pinnable) }, icon('pin', 13), 'Pin'),
      h('button.mini', { type: 'button', disabled: !pinned.length, onclick: () => bulkUnpin(pinned) }, 'Unpin'),
      h('button.att-x', { type: 'button', title: 'Clear selection', 'aria-label': 'Clear selection', onclick: () => { checked.clear(); tiles.clear(); render(); } }, icon('x', 13)));
  }

  async function bulkPin(items) {
    for (const item of items) {
      try {
        replace(item, await pin(item));
      } catch (err) {
        toast(err.message, { error: true });
        return;
      }
    }
  }

  async function bulkUnpin(items) {
    const orphans = items.filter((i) => !i.live).length;
    if (orphans) {
      const ok = await dialog({
        title: `Unpin ${plural(items.length, 'image')}?`,
        body: `${plural(orphans, 'of them has', 'of them have')} no conversation left: unpinning deletes ${orphans === 1 ? 'it' : 'them'} for good.`,
        confirm: 'Unpin', danger: true,
      });
      if (!ok) return;
    }
    for (const item of items) {
      try {
        await api.unpin(item.pin.id);
        replace(item, item.live ? { ...item, pin: null } : null);
      } catch (err) {
        toast(err.message, { error: true });
        return;
      }
    }
    announce('Unpinned');
  }

  // --- one image -----------------------------------------------------------------------------

  function openDetail(item) {
    modal((close) => detailDialog(item, {
      close,
      onUse: (edit) => { close(); useInNextPrompt([item], navigate, { edit }); },
      onOpen: () => { close(); openConversation(item, navigate); },
      onChange: (next) => { replace(item, next); close(); if (next) openDetail(next); },
    }));
  }

  // --- wiring ------------------------------------------------------------------------------

  const observer = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) loadMore();
  }, { root: scroller, rootMargin: '900px 0px' });
  observer.observe(sentinel);
  let lastCols = 0;
  const resize = new ResizeObserver(() => {
    const n = columnCount();
    if (n !== lastCols) { lastCols = n; render(); }
  });
  resize.observe(scroller);
  const onSlash = (e) => {
    if (e.key !== '/' || e.target.closest?.('input, textarea, [contenteditable]')) return;
    e.preventDefault();
    search.focus();
  };
  document.addEventListener('keydown', onSlash);

  render();
  loadMore();
  return {
    el,
    destroy() {
      gone = true;
      clearTimeout(debounce);
      observer.disconnect();
      resize.disconnect();
      document.removeEventListener('keydown', onSlash);
      status.destroy();
      strip.destroy();
    },
  };
}

/** One image: big, with how it was made and where it is, and what can be done with it. */
function detailDialog(item, { close, onUse, onOpen, onChange }) {
  const img = h('img.art-detail-pic', { alt: titleOf(item) });
  setLibraryImage(img, item);
  const g = item.generation;
  const r = g?.resolved ?? {};
  const rows = [];
  const row = (key, value) => { if (value != null && value !== '' && value !== false) rows.push(h('dt', key), h('dd', value)); };
  const copyable = (text) => h('span.art-copy', h('span.call-text', text), h('button.mini', {
    type: 'button', onclick: () => copy(text),
  }, 'Copy'));
  row('prompt', item.prompt && copyable(item.prompt));
  row('negative', r.negative_prompt && copyable(r.negative_prompt));
  row('model', g && [g.backend, g.model && modelName(g.model), g.model_card && `card ${g.model_card}`].filter(Boolean).join(' · '));
  row('seed', g?.seed != null ? String(g.seed) : null);
  row('settings', g && [
    r.steps != null && plural(r.steps, 'step'), r.cfg_scale != null && `cfg ${r.cfg_scale}`,
    r.sampler, r.scheduler, g.alpha && 'transparent',
  ].filter(Boolean).join(' · '));
  row('loras', r.loras?.length ? r.loras.map((l) => `${loraName(l.path)} ×${l.weight ?? 1}`).join(', ') : null);
  row('image', [SOURCE_WORD[item.source], item.kind === 'mask' && 'mask', item.width && `${item.width}×${item.height}`,
    bytes(item.size_bytes), ago(item.created_at)].filter(Boolean).join(' · '));
  row('file', item.original_filename);
  row('where', item.live
    ? [item.label, item.conversation_title || 'Untitled', item.origin.link_id && `on ${item.origin.name}`,
      item.copies > 1 && `${item.copies} copies`].filter(Boolean).join(' · ')
    : ['its conversation is gone', item.pin?.source_instance_name && `from ${item.pin.source_instance_name}`].filter(Boolean).join(' · '));

  const pinFields = item.pin && !item.origin.link_id ? pinEditor(item) : null;
  const pinBtn = h('button.btn.btn-secondary', {
    type: 'button',
    onclick: async () => {
      try {
        onChange(item.pin ? await unpin(item) : await pin(item));
      } catch (err) {
        toast(err.message, { error: true });
      }
    },
  }, icon('pin', 14), item.pin ? 'Unpin' : 'Pin');
  return h('div.dialog.blueprint.art-detail', { role: 'dialog', 'aria-modal': 'true', 'aria-label': titleOf(item) },
    corners(),
    h('div.art-detail-head',
      h('div.dialog-title', truncate(titleOf(item), 80)),
      h('span.spacer'),
      iconButton('x', 'Close', () => close(), 16)),
    h('div.art-detail-body',
      h('div.art-detail-frame', { style: { aspectRatio: item.width ? `${item.width} / ${item.height}` : '4 / 3' } }, img),
      h('div.art-detail-side',
        h('dl.call-detail.art-facts', rows),
        pinFields)),
    h('div.dialog-actions.art-detail-actions',
      h('button.btn.btn-secondary', { type: 'button', onclick: () => downloadLibraryItem(item).catch((err) => toast(err.message, { error: true })) },
        icon('download', 14), 'Download'),
      item.live && item.conversation_id && h('button.btn.btn-secondary', { type: 'button', onclick: onOpen }, icon('external', 14), 'Open conversation'),
      pinBtn,
      h('span.spacer'),
      item.kind === 'image' && h('button.btn.btn-secondary', { type: 'button', onclick: () => onUse(true) }, icon('pencil', 14), 'Edit in next prompt'),
      h('button.btn.btn-primary', { type: 'button', onclick: () => onUse(false) }, icon('arrowRight', 14), 'Use in next prompt')));
}

/** A pin's name and notes, saved as you leave each field. */
function pinEditor(item) {
  const save = async (patch) => {
    try {
      const next = await api.updatePin(item.pin.id, patch);
      Object.assign(item, { pin: next.pin });
      announce('Saved');
    } catch (err) {
      toast(err.message, { error: true });
    }
  };
  const name = h('input.input', {
    value: item.pin.name ?? '', placeholder: 'A name for it', maxlength: 200, 'aria-label': 'Pin name',
    onchange: () => save({ name: name.value }),
    onkeydown: (e) => { if (e.key === 'Enter') name.blur(); },
  });
  const notes = h('textarea.input.lib-notes', {
    placeholder: 'Notes', maxlength: 4000, 'aria-label': 'Pin notes',
    onchange: () => save({ notes: notes.value }),
  });
  notes.value = item.pin.notes ?? '';
  return h('div.art-pin-fields', h('span.eyebrow', 'Pinned'), name, notes,
    h('span.tray-hint', `kept ${ago(item.pin.pinned_at)}${item.live ? '' : ' · the only copy left'}`));
}

function copy(text) {
  if (!navigator.clipboard) {
    toast('Select the text to copy it: this page can\'t use the clipboard');
    return;
  }
  navigator.clipboard.writeText(text).then(() => announce('Copied'), () => toast('Couldn\'t copy', { error: true }));
}

// --- the picker (the prompt bar's "From the library") -------------------------------------

/**
 * The library as a dialog: pick up to `room` images, in order; onPick(items) gets them.
 * Masks are offered behind a toggle, since a mask goes with an image it was painted for.
 */
export function libraryPicker({ room = MAX_ATTACHMENTS, onPick }) {
  let query = '';
  let kind = 'image';
  let debounce = null;
  let run = 0;
  let list = pager(() => ({ kind, q: query }));
  const chosen = [];
  const keys = () => chosen.map(keyOf);

  const grid = h('div.lib-pick-grid.art-pick-grid');
  const status = h('div.art-pick-status');
  const moreBtn = h('button.btn-line', { type: 'button', hidden: true, onclick: () => load() }, 'Load more');
  const addBtn = h('button.btn.btn-primary', { type: 'button', disabled: true }, 'Add');
  const search = h('input.search-input', {
    type: 'search', placeholder: 'Search prompts, files, titles', 'aria-label': 'Search the artifact library',
    oninput: () => {
      clearTimeout(debounce);
      debounce = setTimeout(() => { query = search.value.trim(); restart(); }, SEARCH_DEBOUNCE_MS);
    },
  });
  const kindBtn = (value, word) => h('button.pick', {
    type: 'button', dataset: { kind: value },
    onclick: () => { kind = value; restart(); },
  }, word);
  const kinds = h('div.tray-seg.small', { role: 'group', 'aria-label': 'Kind' }, kindBtn('image', 'Images'), kindBtn('mask', 'Masks'));

  function restart() {
    run += 1;
    list = pager(() => ({ kind, q: query }));
    render();
    load();
  }

  async function load() {
    const mine = run;
    render();
    await list.more();
    if (mine === run) render();
  }

  function toggle(item) {
    const at = keys().indexOf(keyOf(item));
    if (at !== -1) chosen.splice(at, 1);
    else if (chosen.length >= room) {
      toast(`Room for ${plural(room, 'more image')} in this prompt`, { error: true });
      return;
    } else chosen.push(item);
    render();
  }

  function render() {
    const s = list.state;
    for (const b of kinds.querySelectorAll('[data-kind]')) {
      const on = b.dataset.kind === kind;
      b.classList.toggle('on', on);
      b.setAttribute('aria-pressed', String(on));
    }
    const picked = keys();
    clear(grid).append(...s.items.map((item) => {
      const n = picked.indexOf(keyOf(item));
      const thumb = h('img', { alt: titleOf(item), loading: 'lazy' });
      setLibraryImage(thumb, item);
      return h('button.lib-pick.art-pick', {
        type: 'button', class: n !== -1 ? 'on' : '', 'aria-pressed': String(n !== -1),
        title: [item.label, titleOf(item), item.origin.link_id && `on ${item.origin.name}`].filter(Boolean).join(' · '),
        onclick: () => toggle(item),
      },
      h('span.art-pick-frame', thumb,
        n !== -1 && h('span.art-pick-n', String(n + 1)),
        item.origin.link_id && h('span.art-pick-origin', icon('link', 10), item.origin.name),
        item.pin && h('span.art-pick-pin', icon('pin', 10))),
      h('span.mono-meta', truncate(item.conversation_title || item.label || titleOf(item), 22)));
    }));
    const note = sourcesNote(s.sources);
    clear(status).append(...[
      s.error && h('span.tray-hint', s.error),
      !s.error && !s.items.length && h('span.tray-hint', s.done ? (query ? `Nothing matches “${query}”.` : 'No images yet.') : 'Reading the library…'),
      note.text && h('span.tray-hint', { title: note.title }, note.text),
    ].filter(Boolean));
    moreBtn.hidden = s.done;
    moreBtn.disabled = !!s.loading;
    addBtn.disabled = !chosen.length;
    addBtn.textContent = chosen.length ? `Add ${plural(chosen.length, 'image')}` : 'Add';
  }

  modal((close) => {
    addBtn.onclick = () => { close(); onPick(chosen.slice()); };
    return h('div.dialog.blueprint.lib-pick-dialog.art-picker', { role: 'dialog', 'aria-modal': 'true', 'aria-label': 'From the artifact library' },
      corners(),
      h('div.art-picker-head',
        h('div.dialog-title', 'From the artifact library'),
        h('span.spacer'),
        h('label.search', icon('search', 15), search),
        kinds),
      h('span.tray-hint', `Pick up to ${plural(room, 'image')}. Each is copied into this conversation when you send.`),
      grid,
      status,
      h('div.art-picker-more', moreBtn),
      h('div.dialog-actions', h('button.btn.btn-secondary', { type: 'button', onclick: () => close() }, 'Cancel'), addBtn));
  });
  search.focus();
  load();
}
