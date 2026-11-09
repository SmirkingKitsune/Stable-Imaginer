// The LoRA library (design 2a) and one LoRA's page (design 2b).
//
// #/loras: every LoRA the image engines can apply, one per file however many folders it is
// in. Tiles (covers lead) is the normal view, Table (grouped by folder, with a bulk bar for
// the ones checked) the advanced one; the toggle, filter and sort are remembered in the
// browser. Tags filter both; pins come first.
//
// #/loras/<id>: what the user adds to it, every edit saved as you leave its field, like
// Engines. Compatibility has a row per model the engines run: what was detected from the
// file is a first guess, the user's call wins. Usage comes from the LoRAs each job records.

import { api, setImage, setImageUrl } from './api.js';
import { announce, connectionStrip, dialog, menu, modal, statusLine, toast } from './chrome.js';
import { append, corners, h, icon, iconButton, clear } from './dom.js';
import { ago, plural } from './format.js';
import { hubDialog } from './hub.js';
import { convHash } from './origin.js';
import { imageEngine, lastPlace, loadLoras, queueLoras } from './store.js';

const VIEW_KEY = 'imaginer.loras';
const SUGGESTED_TAGS = ['Style', 'Character', 'Photo', 'Lighting', 'Speed', 'Utility'];
const WEIGHTS = [0.6, 0.8, 1];
const SORTS = {
  last: ['Last used', (a, b) => time(b.last_used_at) - time(a.last_used_at) || b.uses - a.uses],
  uses: ['Most used', (a, b) => b.uses - a.uses || time(b.last_used_at) - time(a.last_used_at)],
  name: ['Name', (a, b) => a.name.localeCompare(b.name)],
  size: ['Size', (a, b) => (b.size_bytes ?? 0) - (a.size_bytes ?? 0)],
  newest: ['Newest file', (a, b) => time(b.modified_at) - time(a.modified_at)],
};
const COMPAT_CHIP = { ok: ['done', 'OK'], maybe: ['wait', '?'], unknown: ['wait', '?'], no: ['fail', 'NO'] };

const time = (iso) => (iso ? new Date(iso).getTime() : 0);
const bytes = (n) => (n == null ? '' : n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB`
  : n >= 1e6 ? `${Math.round(n / 1e6)} MB` : `${Math.max(1, Math.round(n / 1e3))} KB`);
const weightText = (w) => `×${Number(w).toFixed(2).replace(/\.?0+$/, '')}`;
const folderOf = (item) => (item.local ? (item.folder ? `${item.folder}/` : 'top folder') : `on ${item.origin}`);

function remembered() {
  try {
    return { view: 'tiles', tag: 'All', sort: 'last', ...JSON.parse(localStorage.getItem(VIEW_KEY) || '{}') };
  } catch {
    return { view: 'tiles', tag: 'All', sort: 'last' };
  }
}

function remember(prefs) {
  try {
    localStorage.setItem(VIEW_KEY, JSON.stringify(prefs));
  } catch {
    // Private mode: it lasts for this page.
  }
}

/** The chip for a LoRA's standing with its engines' models. */
function compatChip(compat) {
  const [kind, word] = COMPAT_CHIP[compat.state];
  return h('span.state', { class: kind, title: compat.bad ?? compat.text }, compat.state === 'no' && compat.bad ? 'WRONG BASE' : word);
}

/** Put a LoRA's mention in the next prompt bar that opens, for the image engine in use. */
function useInNextPrompt(items, navigate) {
  const engine = imageEngine();
  const picks = [];
  for (const item of items) {
    const use = item.engines.find((e) => e.engine === engine?.name);
    if (use) picks.push({ path: use.path, weight: item.default_weight });
  }
  if (!picks.length) {
    toast(engine ? `${engine.name} doesn't list ${items.length === 1 ? 'this LoRA' : 'these LoRAs'}` : 'No image engine is set', { error: true });
    return;
  }
  queueLoras(picks);
  // Back to the conversation (or gallery) the library was opened from.
  navigate(lastPlace(), { focusBar: true });
  const missing = items.length - picks.length;
  if (missing) toast(`${engine.name} doesn't list ${missing === 1 ? 'one of them' : `${missing} of them`}`);
}

// --- the library (design 2a) -------------------------------------------------------------

export function lorasScreen({ navigate }) {
  const prefs = remembered();
  let lib = null;
  let error = null;
  let query = '';
  const checked = new Set();
  let gone = false;

  const status = statusLine();
  const meta = h('span.topbar-meta');
  const search = h('input.search-input', {
    type: 'search', placeholder: 'Search names, triggers, tags', 'aria-label': 'Search LoRAs',
    oninput: () => { query = search.value.trim().toLowerCase(); render(); },
    onkeydown: (e) => { if (e.key === 'Escape') { search.value = ''; query = ''; render(); search.blur(); } },
  });
  const install = h('button.btn.btn-primary.blueprint.engine-go', { type: 'button', onclick: () => installLora() },
    corners(), icon('download', 14), h('span.only-wide', 'Install from Hugging Face'));
  const header = h('header.topbar.sub',
    iconButton('back', 'Back', () => (history.length > 1 ? history.back() : navigate('#/'))),
    h('div.topbar-heading', h('span.topbar-title', 'LoRAs'), meta),
    h('span.spacer'),
    h('label.search.lib-search', icon('search', 15), search, h('kbd', '/')),
    h('div.only-wide', status.el),
    install);
  const bar = h('div.lib-bar');
  const body = h('div.lib-body');
  const scroller = h('div.gallery-scroll.lib-scroll', bar, body);
  const bulk = h('div.pbar.lib-bulk', { hidden: true, role: 'toolbar', 'aria-label': 'Selected LoRAs' });
  const strip = connectionStrip();
  const el = h('div.screen.loras', header, strip.el, scroller, bulk);

  async function refresh() {
    try {
      lib = await loadLoras();
      if (!lib) throw new Error('the library could not be loaded');
      error = null;
    } catch (err) {
      error = err.message;
    }
    if (gone) return;
    for (const id of [...checked]) if (!lib?.items.some((i) => i.id === id)) checked.delete(id);
    render();
  }

  const save = (patch) => { Object.assign(prefs, patch); remember(prefs); render(); };

  function matches(item, tag) {
    if (tag === 'All') return true;
    if (tag === 'Pinned') return item.pinned;
    if (tag === 'Incompatible') return item.compat.state === 'no';
    return item.tags.includes(tag);
  }

  function searched(item) {
    if (!query) return true;
    const text = [item.name, item.file_name, item.path, item.notes, ...item.tags, ...item.triggers.map((t) => t.text)]
      .join(' ').toLowerCase();
    return query.split(/\s+/).every((word) => text.includes(word));
  }

  function shown() {
    const [, compare] = SORTS[prefs.sort] ?? SORTS.last;
    return lib.items.filter((i) => matches(i, prefs.tag) && searched(i))
      .sort((a, b) => (b.pinned - a.pinned) || compare(a, b));
  }

  function render() {
    renderMeta();
    clear(bar);
    clear(body);
    if (error && !lib) {
      body.append(h('div.notice', error));
      bulk.hidden = true;
      return;
    }
    if (!lib) {
      body.append(h('div.notice', 'Looking through the LoRA folders…'));
      return;
    }
    if (!lib.items.length) {
      body.append(h('div.notice.lib-empty',
        h('span.eyebrow.accent', 'No LoRAs yet'),
        h('span', 'Set an image engine\'s LoRA folder on the Engines screen, or install one from Hugging Face.'),
        h('div.lib-empty-actions', h('a.btn-line', { href: '#/engines' }, icon('sliders', 15), 'Engines'))));
      bulk.hidden = true;
      return;
    }
    if (!['All', 'Pinned', 'Incompatible', ...lib.tags].includes(prefs.tag)) prefs.tag = 'All';
    const list = shown();
    renderBar(list);
    if (prefs.view === 'table') renderTable(list);
    else renderTiles(list);
    renderBulk();
  }

  function renderMeta() {
    if (!lib) { meta.textContent = ''; return; }
    const remote = Object.entries(lib.origins).map(([name, n]) => `${n} on ${name}`);
    meta.textContent = [plural(lib.items.length, 'LoRA'), lib.total_bytes && bytes(lib.total_bytes),
      [lib.local_count && `${lib.local_count} on this machine`, ...remote].filter(Boolean).join(', ')]
      .filter(Boolean).join(' · ');
    meta.title = lib.roots.length ? `LoRA folders: ${lib.roots.join(', ')}` : '';
  }

  function renderBar(list) {
    const tags = ['All', 'Pinned', ...lib.tags, 'Incompatible'];
    const count = (t) => lib.items.filter((i) => matches(i, t)).length;
    const filters = h('div.tray-seg.lib-filters', { role: 'group', 'aria-label': 'Filter' },
      tags.filter((t) => t === 'All' || count(t)).map((t) => h('button.pick', {
        type: 'button', class: prefs.tag === t ? 'on' : '', 'aria-pressed': String(prefs.tag === t),
        onclick: () => save({ tag: t }),
      }, t, h('span.mono-meta', String(count(t))))));
    const sortBtn = h('button.mini', {
      type: 'button',
      onclick: () => menu(sortBtn, Object.entries(SORTS).map(([key, [label]]) => ({
        label, checked: prefs.sort === key, onclick: () => save({ sort: key }),
      }))),
    }, icon('clock', 13), (SORTS[prefs.sort] ?? SORTS.last)[0], icon('chevronDown', 12));
    const view = (key, label, iconName) => h('button.pick', {
      type: 'button', class: prefs.view === key ? 'on' : '', 'aria-pressed': String(prefs.view === key), title: label,
      onclick: () => save({ view: key }),
    }, icon(iconName, 13), h('span', label));
    append(bar, [
      filters,
      h('span.spacer'),
      h('span.mono-meta', `${plural(list.length, 'LoRA')} · pinned first`),
      sortBtn,
      h('div.tray-seg.small.lib-views', { role: 'group', 'aria-label': 'View' }, view('tiles', 'Tiles', 'grid'), view('table', 'Table', 'list')),
    ]);
  }

  // --- tiles --------------------------------------------------------------------------

  function columnCount() {
    const w = scroller.clientWidth || window.innerWidth;
    return w >= 1500 ? 5 : w >= 1060 ? 4 : w >= 760 ? 3 : 2;
  }

  function renderTiles(list) {
    const n = columnCount();
    const cols = Array.from({ length: n }, () => h('div.masonry-col'));
    const heights = new Array(n).fill(0);
    for (const item of list) {
      const ratio = item.cover?.width ? item.cover.height / item.cover.width : 1;
      const i = heights.indexOf(Math.min(...heights));
      cols[i].append(tile(item));
      heights[i] += ratio + 0.3;
    }
    body.append(h('div.masonry', cols),
      h('div.gallery-end.lib-end', h('div.rule-label', h('span.rule'),
        h('span', list.length ? `END OF LIBRARY · ${plural(list.length, 'LORA')}` : `NOTHING MATCHES${query ? ` “${query.toUpperCase()}”` : ''}`),
        h('span.rule'))));
  }

  function tile(item) {
    const bad = item.compat.state === 'no';
    const ratio = item.cover?.width ? `${item.cover.width} / ${item.cover.height}` : '1 / 1';
    let img;
    if (item.cover) {
      img = h('img.tile-pic', { alt: item.name, loading: 'lazy', decoding: 'async' });
      setImageUrl(img, item.cover.url);
    } else {
      img = h('span.tile-desc', 'no cover · set one from a conversation');
    }
    const words = item.triggers.filter((t) => t.on).map((t) => t.text);
    const metaLine = [item.tags[0]?.toLowerCase(), weightText(item.default_weight),
      item.uses ? plural(item.uses, 'use') : 'unused', item.last_used_at && ago(item.last_used_at)].filter(Boolean).join(' · ');
    return h('a.tile.result.lib-tile', {
      href: `#/loras/${item.id}`, 'aria-label': `${item.name}${bad ? ', incompatible' : ''}`,
    },
    h('div.tile-img', { class: bad ? 'dim' : '', style: { aspectRatio: ratio } }, img),
    item.origin && h('span.tile-origin', { title: `Lives on ${item.origin}` }, icon('link', 11), item.origin),
    bad && h('span.state.fail.lib-bad', { title: item.compat.bad }, (item.compat.bad ?? 'incompatible').toUpperCase()),
    h('div.tile-foot.lib-foot',
      h('div.tile-foot-text',
        h('span.tile-title', item.name),
        h('span.mono-meta', metaLine),
        words.length > 0 && h('span.lib-trigger', { title: words.join(', ') }, words.join(', '))),
      pinButton(item, 15)));
  }

  function pinButton(item, size) {
    return h('button.icon-btn.lib-pin', {
      type: 'button', class: item.pinned ? 'on' : '', title: item.pinned ? 'Unpin' : 'Pin',
      'aria-label': item.pinned ? `Unpin ${item.name}` : `Pin ${item.name}`, 'aria-pressed': String(item.pinned),
      onclick: (e) => { e.preventDefault(); e.stopPropagation(); change([item], { pinned: !item.pinned }); },
    }, icon('pin', size));
  }

  // --- table --------------------------------------------------------------------------

  function renderTable(list) {
    const head = h('div.lib-row.lib-head',
      h('label.check', h('input', {
        type: 'checkbox', 'aria-label': 'Select all shown', checked: list.length > 0 && list.every((i) => checked.has(i.id)),
        onchange: (e) => { for (const i of list) e.target.checked ? checked.add(i.id) : checked.delete(i.id); render(); },
      })),
      h('span'), h('span'),
      h('span.eyebrow', 'Name · file'), h('span.eyebrow', 'Trigger'), h('span.eyebrow', 'Tags'),
      h('span.eyebrow', 'Compatibility'), h('span.eyebrow.right', 'Size'), h('span.eyebrow.right', 'Uses'),
      h('span.eyebrow.right', 'Last'), h('span'));
    const groups = new Map();
    for (const item of list) {
      const key = folderOf(item);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(item);
    }
    const table = h('div.lib-table', { role: 'table', 'aria-label': 'LoRAs' }, head);
    for (const [name, items] of [...groups].sort(([a], [b]) => a.localeCompare(b))) {
      table.append(h('div.lib-group', icon(name.startsWith('on ') ? 'link' : 'folder', 14),
        h('span.lib-group-name', name), h('span.mono-meta', plural(items.length, 'LoRA'))));
      table.append(...items.map(row));
    }
    if (!list.length) table.append(h('div.notice', `Nothing matches${query ? ` “${query}”` : ''}.`));
    body.append(table, h('div.lib-table-end'));
  }

  function row(item) {
    const thumb = h('span.lin-thumb.lib-thumb', { class: item.compat.state === 'no' ? 'dim' : '' });
    if (item.cover) {
      const img = h('img', { alt: '', loading: 'lazy' });
      setImageUrl(img, item.cover.url);
      thumb.append(img);
    }
    const words = item.triggers.filter((t) => t.on).map((t) => t.text).join(', ');
    const more = h('button.icon-btn.lib-more', {
      type: 'button', title: 'More', 'aria-label': `More for ${item.name}`,
      onclick: (e) => {
        e.stopPropagation();
        menu(more, [
          { label: 'Open', icon: 'arrowRight', onclick: () => navigate(`#/loras/${item.id}`) },
          { label: 'Use in next prompt', icon: 'puzzle', onclick: () => useInNextPrompt([item], navigate) },
          { label: item.pinned ? 'Unpin' : 'Pin', icon: 'pin', onclick: () => change([item], { pinned: !item.pinned }) },
          ...(item.local ? [{ label: 'Delete file…', icon: 'trash', danger: true, onclick: () => remove([item]) }] : []),
        ]);
      },
    }, icon('more', 16));
    const on = checked.has(item.id);
    return h('div.lib-row', {
      role: 'row', class: on ? 'on' : '', tabindex: '0',
      onclick: (e) => { if (!e.target.closest('button, input, label, a')) navigate(`#/loras/${item.id}`); },
      onkeydown: (e) => { if (e.key === 'Enter' && e.target === e.currentTarget) navigate(`#/loras/${item.id}`); },
    },
    h('label.check', h('input', {
      type: 'checkbox', checked: on, 'aria-label': `Select ${item.name}`,
      onchange: () => { if (checked.has(item.id)) checked.delete(item.id); else checked.add(item.id); render(); },
    })),
    pinButton(item, 14),
    thumb,
    h('span.lib-name',
      h('span.lib-name-line', h('span.tile-title', item.name),
        item.origin && h('span.hub-tag.lib-origin', icon('link', 10), item.origin)),
      h('span.mono-meta', { title: item.path }, item.file_name)),
    h('span.lib-word', { title: words }, words || '—'),
    h('span.lib-tags', item.tags.map((t) => h('span.hub-tag', t))),
    h('span.lib-compat', compatChip(item.compat), h('span.mono-meta', { title: item.compat.text }, item.compat.text)),
    h('span.mono-meta.right', bytes(item.size_bytes)),
    h('span.mono-meta.right.strong', String(item.uses)),
    h('span.mono-meta.right', item.last_used_at ? ago(item.last_used_at) : 'never'),
    more);
  }

  // --- the bulk bar (table view) ----------------------------------------------------------

  function renderBulk() {
    const items = lib.items.filter((i) => checked.has(i.id));
    bulk.hidden = prefs.view !== 'table' || !items.length;
    if (bulk.hidden) return;
    const local = items.filter((i) => i.local);
    const tagBtn = h('button.mini', { type: 'button', onclick: () => tagMenu(tagBtn, items) }, icon('tag', 13), 'Tag');
    const presetBtn = h('button.mini', { type: 'button', onclick: () => presetMenu(presetBtn, items) }, icon('layers', 13), 'Add to preset');
    clear(bulk).append(
      h('span.lib-bulk-count', `${items.length} selected`),
      h('span.mono-meta', bytes(local.reduce((n, i) => n + (i.size_bytes ?? 0), 0))),
      tagBtn,
      h('button.mini', { type: 'button', disabled: !local.length, onclick: () => moveTo(local) }, icon('folder', 13), 'Move to folder'),
      presetBtn,
      h('button.mini.primary', { type: 'button', onclick: () => useInNextPrompt(items, navigate) }, icon('puzzle', 13), 'Use in next prompt'),
      h('span.lib-bulk-sep'),
      h('button.mini', { type: 'button', disabled: !local.length, title: 'Delete files', onclick: () => remove(local) }, icon('trash', 13), 'Delete'),
      h('button.att-x', { type: 'button', title: 'Clear selection', 'aria-label': 'Clear selection', onclick: () => { checked.clear(); render(); } }, icon('x', 13)));
  }

  function tagMenu(anchor, items) {
    const tags = [...new Set([...lib.tags, ...SUGGESTED_TAGS])];
    menu(anchor, [
      ...tags.map((t) => ({
        label: t, checked: items.every((i) => i.tags.includes(t)),
        onclick: () => {
          const all = items.every((i) => i.tags.includes(t));
          Promise.all(items.map((i) => api.updateLora(i.id, { tags: all ? i.tags.filter((x) => x !== t) : [...i.tags, t] })))
            .then(() => { announce(`${all ? 'Untagged' : 'Tagged'} ${t}`); refresh(); }, (err) => toast(err.message, { error: true }));
        },
      })),
      {
        label: 'New tag…', icon: 'plus',
        onclick: async () => {
          const t = (await dialog({ title: 'New tag', input: { placeholder: 'Tag' }, confirm: 'Tag' }))?.trim();
          if (!t) return;
          await Promise.all(items.map((i) => api.updateLora(i.id, { tags: [...i.tags, t] }))).catch((err) => toast(err.message, { error: true }));
          refresh();
        },
      },
    ]);
  }

  /** The presets of the first one's engines; each LoRA goes in at its default weight. */
  async function presetMenu(anchor, items) {
    let presets;
    try {
      presets = (await api.lora(items[0].id)).presets;
    } catch (err) {
      toast(err.message, { error: true });
      return;
    }
    if (!presets.length) {
      toast('Its engines have no presets: make one from a LoRA\'s page', { error: true });
      return;
    }
    menu(anchor, presets.map((p) => ({
      label: `${p.name} · ${p.engine}`, icon: 'layers',
      onclick: () => Promise.all(items.filter((i) => i.engines.some((e) => e.engine === p.engine))
        .map((i) => api.loraPreset(i.id, { engine: p.engine, preset: p.name, weight: i.default_weight })))
        .then(() => { toast(`Added to ${p.name}`); refresh(); }, (err) => toast(err.message, { error: true })),
    })));
  }

  async function moveTo(items) {
    const folder = await dialog({
      title: `Move ${items.length === 1 ? items[0].name : plural(items.length, 'LoRA')}`,
      body: 'To a folder in its LoRA folder. Presets that name a file by its path are updated.',
      input: { placeholder: 'styles/', value: items[0].folder ? `${items[0].folder}/` : '' },
      confirm: 'Move',
    });
    if (folder == null) return;
    for (const i of items) {
      try {
        await api.moveLora(i.id, folder.trim());
      } catch (err) {
        toast(`${i.name}: ${err.message}`, { error: true });
      }
    }
    refresh();
  }

  async function remove(items) {
    const one = items.length === 1;
    const ok = await dialog({
      title: one ? `Delete ${items[0].name}?` : `Delete ${plural(items.length, 'LoRA')}?`,
      body: `${one ? 'Its file is' : 'Their files are'} deleted from this machine and taken out of the presets that name ${one ? 'it' : 'them'}. Past images keep their record.`,
      confirm: 'Delete', danger: true,
    });
    if (!ok) return;
    for (const i of items) {
      try {
        await api.deleteLora(i.id);
        checked.delete(i.id);
      } catch (err) {
        toast(`${i.name}: ${err.message}`, { error: true });
      }
    }
    refresh();
  }

  async function change(items, patch) {
    try {
      await Promise.all(items.map((i) => api.updateLora(i.id, patch)));
    } catch (err) {
      toast(err.message, { error: true });
    }
    refresh();
  }

  async function installLora() {
    let engines;
    try {
      engines = await api.engines();
    } catch (err) {
      toast(err.message, { error: true });
      return;
    }
    const able = engines.filter((e) => e.loras && e.enabled);
    const engine = able.find((e) => e.default_for.includes('t2i')) ?? able[0];
    if (!engine) {
      toast('No image engine here takes LoRAs: an sd-server launched here, or Diffusers', { error: true });
      return;
    }
    hubDialog(engine, { onChange: refresh });
  }

  // --- lifecycle ------------------------------------------------------------------------

  let lastCols = 0;
  const resize = new ResizeObserver(() => {
    const n = columnCount();
    if (n !== lastCols && lib && prefs.view !== 'table') { lastCols = n; render(); }
  });
  resize.observe(scroller);
  const onSlash = (e) => {
    if (e.key !== '/' || e.target.closest?.('input, textarea, [contenteditable]')) return;
    e.preventDefault();
    search.focus();
  };
  document.addEventListener('keydown', onSlash);
  render();
  refresh();

  return {
    el,
    destroy() {
      gone = true;
      resize.disconnect();
      document.removeEventListener('keydown', onSlash);
      status.destroy();
      strip.destroy();
    },
  };
}

// --- one LoRA (design 2b) ----------------------------------------------------------------

export function loraScreen({ id, navigate }) {
  let d = null;
  let error = null;
  let gone = false;

  const status = statusLine();
  const meta = h('span.topbar-meta');
  const header = h('header.topbar.sub',
    iconButton('back', 'LoRAs', () => navigate('#/loras')),
    h('div.topbar-heading', h('span.topbar-title', 'LoRAs'), meta),
    h('span.spacer'),
    h('div.only-wide', status.el));
  const main = h('main.links-main.lora-main');
  const strip = connectionStrip();
  const el = h('div.screen.lora', header, strip.el, main);

  async function load() {
    try {
      d = await api.lora(id);
      error = null;
    } catch (err) {
      error = err.message;
    }
    if (!gone) render();
  }

  /** Save, then show what the server made of it; the pickers' copy reloads too. */
  async function save(call, word = null) {
    try {
      d = await call();
      if (word) toast(word);
      loadLoras();
    } catch (err) {
      toast(err.message, { error: true });
    }
    if (!gone) render();
  }
  const patch = (body, word = null) => save(() => api.updateLora(id, body), word);

  function render() {
    clear(main);
    if (!d) {
      main.append(h('div.notice', error ?? 'Loading…'),
        error && h('div', h('a.btn-line', { href: '#/loras' }, icon('back', 15), 'The library')));
      return;
    }
    meta.textContent = d.root ? d.path : `${d.path} · ${d.origin ? `on ${d.origin}` : 'listed by its engine'}`;
    meta.title = d.files.map((f) => f.path).join('\n');
    const left = h('div.engine-cells',
      h('div.cell-pair', nameCell(), weightCell()),
      wordsCell(),
      notesCell(),
      h('div.cell-pair', tagsCell(), d.local ? folderCell() : remoteCell()),
      d.presets.length || d.local ? presetsCell() : null,
      compatCell(),
      d.local && fileCell(),
      d.local && deleteCell());
    append(main, [head(), h('div.engine-grid', left, coverCol()), usageSection()]);
  }

  function head() {
    const works = d.compat_models.filter((m) => m.state === 'ok').length;
    const untested = d.compat_models.filter((m) => m.state === 'maybe' || m.state === 'unknown').length;
    const chip = works
      ? h('span.state.done', `WORKS WITH ${plural(works, 'MODEL', 'MODELS')}${untested ? ` · ${untested} UNTESTED` : ''}`)
      : untested ? h('span.state.wait', `${plural(untested, 'MODEL', 'MODELS')} UNTESTED`) : h('span.state.fail', 'WORKS WITH NO MODEL HERE');
    const where = d.local ? (d.source ? 'from Hugging Face' : 'on this machine') : `on ${d.origin}`;
    const facts = [d.base, d.format, d.rank && `rank ${d.rank}`, d.size_bytes && bytes(d.size_bytes), plural(d.uses, 'image')];
    const presetBtn = h('button.btn-line', { type: 'button', onclick: () => presetMenu(presetBtn) }, icon('layers', 15), 'Add to preset');
    return h('div.engine-head',
      h('div.engine-title',
        h('span.eyebrow.accent', ['LoRA', d.folder ? `${d.folder}/` : null, where].filter(Boolean).join(' · ')),
        h('h2.hero-title', d.name),
        h('div.engine-state', chip, h('span.mono-meta', facts.filter(Boolean).join(' · ')))),
      h('button.icon-btn.lib-pin-big', {
        type: 'button', class: d.pinned ? 'on' : '', title: d.pinned ? 'Pinned: unpin' : 'Pin', 'aria-pressed': String(d.pinned),
        onclick: () => patch({ pinned: !d.pinned }),
      }, icon('pin', 18)),
      d.presets.length > 0 && presetBtn,
      h('button.btn.btn-primary.blueprint.engine-go', { type: 'button', onclick: () => useInNextPrompt([d], navigate) },
        corners(), icon('plus', 14), 'Use in next prompt'));
  }

  function presetMenu(anchor) {
    menu(anchor, d.presets.map((p) => ({
      label: `${p.name} · ${p.engine}`, checked: p.included,
      onclick: () => save(() => api.loraPreset(id, { engine: p.engine, preset: p.name, weight: p.included ? null : d.default_weight }),
        p.included ? `Taken out of ${p.name}` : `Added to ${p.name}`),
    })));
  }

  // --- cells ---------------------------------------------------------------------------

  const enter = (e) => { if (e.key === 'Enter' && e.target.tagName === 'INPUT') e.target.blur(); };

  function nameCell() {
    const input = h('input.seed-input.text', {
      value: d.custom_name ? d.name : '', placeholder: d.file_name.replace(/\.[^.]+$/, ''), 'aria-label': 'Display name',
      onchange: () => patch({ name: input.value }), onkeydown: enter,
    });
    return h('div.tray-cell', h('span.eyebrow', 'Display name'), h('label.seed-field', input),
      h('span.tray-hint', 'The file keeps its name; prompts and presets still use the path.'));
  }

  function weightCell() {
    const input = h('input.seed-input', {
      type: 'number', step: 0.05, min: -10, max: 10, value: String(d.default_weight), 'aria-label': 'Default weight',
      onchange: () => {
        const v = Number(input.value);
        if (Number.isFinite(v) && v >= -10 && v <= 10) patch({ default_weight: v });
        else input.value = String(d.default_weight);
      },
      onkeydown: enter,
    });
    return h('div.tray-cell', h('span.eyebrow', 'Default weight'),
      h('div.size-row', h('label.seed-field', h('span.mono-meta', '×'), input),
        h('div.tray-seg.small.lib-weights', WEIGHTS.map((w) => h('button.pick', {
          type: 'button', class: d.default_weight === w ? 'on' : '', 'aria-pressed': String(d.default_weight === w),
          onclick: () => patch({ default_weight: w }),
        }, w.toFixed(1))))),
      h('span.tray-hint', 'What the prompt bar\'s LoRAs and the @ picker start it at.'));
  }

  function wordsCell() {
    const words = d.triggers;
    const put = (next) => patch({ triggers: next });
    const add = h('input.seed-input.text', {
      placeholder: 'Add a word, Enter', 'aria-label': 'Add a trigger word',
      onkeydown: (e) => {
        if (e.key !== 'Enter' || e.isComposing) return;
        e.preventDefault();
        const text = add.value.trim();
        if (text) put([...words, { text, on: true }]);
      },
    });
    const chips = words.map((t, i) => h('span.mention.lib-word-chip', { class: t.on ? '' : 'off' },
      h('button.lib-word-toggle', {
        type: 'button', title: t.on ? 'On: added to the prompt. Click to turn off' : 'Off. Click to turn on', 'aria-pressed': String(t.on),
        onclick: () => put(words.map((w, j) => (j === i ? { ...w, on: !w.on } : w))),
      }, t.text, !t.on && h('span.muted', '\u00a0(off)')),
      h('button.lib-word-x', { type: 'button', title: `Remove ${t.text}`, 'aria-label': `Remove ${t.text}`, onclick: () => put(words.filter((_, j) => j !== i)) }, icon('x', 11))));
    const from = d.triggers_from
      ? `From ${d.triggers_from}: ${d.file_triggers.map((w) => `“${w}”`).join(', ')}.`
      : d.file_triggers.length ? 'Your words.' : 'Neither the file nor its Hub card names any.';
    return h('div.tray-cell',
      h('div.cell-head', h('span.eyebrow', 'Trigger words'), h('span.spacer'),
        h('span.tray-hint.only-wide', 'added to the prompt when this LoRA is picked · click one to turn it off')),
      h('div.lib-words', chips, h('label.seed-field.lib-word-add', add)),
      h('span.tray-hint.wrap', from,
        !d.triggers_from && d.file_triggers.length > 0 && [' ', h('button.link-btn', { type: 'button', onclick: () => patch({ reset_triggers: true }) }, `Back to ${d.file_triggers.map((w) => `“${w}”`).join(', ')}`)]));
  }

  function notesCell() {
    const area = h('textarea.edit-text.lib-notes', {
      'aria-label': 'Notes', placeholder: 'What it is good at, the weight that works…',
      onchange: () => patch({ notes: area.value }),
    });
    area.value = d.notes;
    return h('div.tray-cell', h('span.eyebrow', 'Notes'), area);
  }

  function tagsCell() {
    const all = [...new Set([...d.tags, ...SUGGESTED_TAGS])];
    const toggle = (t) => patch({ tags: d.tags.includes(t) ? d.tags.filter((x) => x !== t) : [...d.tags, t] });
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

  function folderCell() {
    const input = h('input.seed-input.path-input', {
      value: d.folder ? `${d.folder}/` : '', placeholder: 'the top folder', spellcheck: false, 'aria-label': 'Folder',
      onchange: () => {
        const folder = input.value.trim().replace(/\\/g, '/').replace(/^\/+|\/+$/g, '');
        if (folder === d.folder) return;
        save(() => api.moveLora(id, folder), `Moved to ${folder ? `${folder}/` : 'the top folder'}`);
      },
      onkeydown: enter,
    });
    return h('div.tray-cell', h('span.eyebrow', 'Folder'), h('label.seed-field', icon('folder', 13), input),
      h('span.tray-hint', 'Moving it changes its path: presets that name it are updated.'));
  }

  function remoteCell() {
    return h('div.tray-cell', h('span.eyebrow', 'Where'), h('span.cell-big', d.origin ? `on ${d.origin}` : d.engines.map((e) => e.engine).join(', ')),
      h('span.tray-hint.wrap', d.origin ? `${d.origin} keeps its file and applies it: move or delete it there.` : 'Its engine lists it: its file is on that engine\'s machine.'));
  }

  function presetsCell() {
    const rows = d.presets.map((p) => {
      const weight = p.included && h('label.seed-field.lora-weight', h('span.mono-meta', '×'), h('input.seed-input', {
        type: 'number', step: 0.05, min: -10, max: 10, value: String(p.weight ?? 1), 'aria-label': `${p.name} weight`,
        onchange: (e) => {
          const v = Number(e.target.value);
          if (Number.isFinite(v)) save(() => api.loraPreset(id, { engine: p.engine, preset: p.name, weight: v }));
        },
        onkeydown: enter,
      }));
      return h('div.lora-row.lib-preset',
        h('input', {
          type: 'checkbox', checked: p.included, 'aria-label': `In ${p.name}`,
          onchange: () => save(() => api.loraPreset(id, { engine: p.engine, preset: p.name, weight: p.included ? null : d.default_weight })),
        }),
        h('span.lib-preset-name', h('span.lib-preset-title', p.name), h('span.mono-meta', [p.engine, p.description].filter(Boolean).join(' · '))),
        weight || h('span'),
        h('span.mono-meta', { title: p.meta }, p.meta));
    });
    return h('div.tray-cell',
      h('div.cell-head', h('span.eyebrow', 'Presets'), h('span.spacer'), h('span.tray-hint.only-wide', 'named settings, saved with the engines\'')),
      rows.length ? h('div.lib-presets', rows) : h('span.tray-hint', 'Its engines have no presets yet.'),
      d.engines.some((e) => d.presets.some((p) => p.engine === e.engine) || d.local) && h('button.mini', {
        type: 'button', style: { alignSelf: 'flex-start' }, onclick: () => newPreset(),
      }, '+ New preset with this LoRA'));
  }

  function newPreset() {
    const engines = [...new Set(d.engines.map((e) => e.engine))];
    let engine = engines[0];
    const nameInput = h('input.input', { placeholder: 'clay', 'aria-label': 'Preset name', spellcheck: false });
    const descInput = h('input.input', { placeholder: 'What it is for (optional)', 'aria-label': 'Description' });
    const layer = modal((close) => {
      const done = async () => {
        const name = nameInput.value.trim().toLowerCase();
        if (!/^[a-z0-9][a-z0-9_-]{0,31}$/.test(name)) {
          toast('A preset name is lowercase letters, digits, - and _', { error: true });
          return;
        }
        close();
        save(() => api.loraPreset(id, { engine, preset: name, weight: d.default_weight, create: true, description: descInput.value.trim() }), `Made ${name}`);
      };
      nameInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') done(); });
      const seg = engines.length > 1 && h('div.tray-seg', engines.map((n) => h('button.pick', {
        type: 'button', class: n === engine ? 'on' : '',
        onclick: (e) => { engine = n; for (const b of e.target.parentElement.children) b.classList.toggle('on', b === e.target); },
      }, n)));
      return h('div.dialog.blueprint', { role: 'dialog', 'aria-modal': 'true', 'aria-label': 'New preset' },
        corners(),
        h('div.dialog-title', 'New preset with this LoRA'),
        h('div.dialog-body', `It applies ${d.name} at ${weightText(d.default_weight)}, on top of the engine's default LoRAs. Steps and CFG can be set in the config.`),
        seg, nameInput, descInput,
        h('div.dialog-actions',
          h('button.btn.btn-secondary', { type: 'button', onclick: () => close() }, 'Cancel'),
          h('button.btn.btn-primary', { type: 'button', onclick: done }, 'Make preset')));
    });
    nameInput.focus();
    return layer;
  }

  function compatCell() {
    const grid = h('div.lib-compat-grid', { role: 'table', 'aria-label': 'Compatibility per model' },
      h('span.eyebrow', 'Model'), h('span.eyebrow', 'Detected'), h('span.eyebrow', 'Your call'));
    for (const m of d.compat_models) {
      const shownAs = m.state === 'ok' ? 'shown' : m.state === 'no' ? 'hidden' : 'shown with ?';
      grid.append(
        h('span.lib-compat-model',
          h('span.lib-compat-name', h('span.lib-model', { title: m.name }, m.name),
            h('span.mono-meta', { class: m.state === 'no' ? '' : 'accent' }, `${shownAs}${m.call === 'auto' ? '' : ' · your call'}`)),
          h('span.mono-meta', { title: m.where }, m.where)),
        h('span.lib-compat-detected', h('span.state', { class: COMPAT_CHIP[m.detected][0] }, COMPAT_CHIP[m.detected][1]),
          h('span.mono-meta.wrap', m.detected_text)),
        h('span.lib-compat-call', h('span.tray-seg.small', [['auto', 'Auto'], ['works', 'Works'], ['no', 'Doesn\'t']].map(([key, label]) => h('button.pick', {
          type: 'button', class: m.call === key ? 'on' : '', 'aria-pressed': String(m.call === key),
          'aria-label': `${m.name}: ${label}`,
          onclick: () => m.call !== key && patch({ compat: { [m.key]: key } }),
        }, label)))));
    }
    const lineage = h('input.seed-input.text', {
      value: d.lineage, placeholder: d.lineage_default || 'e.g. Qwen-Image 2.x', 'aria-label': 'Lineage', spellcheck: false,
      onchange: () => patch({ lineage: lineage.value }), onkeydown: enter,
    });
    return h('div.tray-cell',
      h('div.cell-head', h('span.eyebrow', 'Compatibility · per model'), h('span.spacer'),
        h('span.tray-hint.only-wide', 'detected from the file\'s keys · your call wins')),
      d.compat_models.length ? grid : h('span.tray-hint', 'No image engine here runs a model it can be checked against.'),
      h('div.size-row.lib-lineage', h('span.eyebrow', 'Lineage'), h('label.seed-field', lineage),
        h('span.tray-hint', 'A model in this lineage whose weights can\'t be read here starts as untested, not unknown.')),
      h('span.tray-hint.wrap', 'Pickers show this LoRA for a model when it works, detected or by your call; untested ones carry a “?”. Your calls follow the file\'s SHA-256, so they survive renames and moves.'));
  }

  function fileCell() {
    const grid = h('div.files-grid.lib-files');
    for (const f of d.files) {
      grid.append(h('span.mono-meta', 'path'), h('span.mono-meta.strong.wrap', f.path));
      if (d.files.length > 1) grid.append(h('span.mono-meta', 'used by'), h('span.mono-meta', f.engines.join(', ')));
    }
    grid.append(h('span.mono-meta', 'sha256'), h('span.mono-meta', { title: d.sha256 },
      `${d.sha256.slice(0, 10)}…${d.sha256.slice(-4)}${d.files[0]?.hashed_at ? ` · hashed ${ago(d.files[0].hashed_at)}` : ''}`));
    if (d.source) {
      grid.append(h('span.mono-meta', 'source'), h('a.mono-meta.lib-source', { href: d.source.url, target: '_blank', rel: 'noopener' }, `${d.source.repo} ↗`));
    }
    return h('div.tray-cell', h('span.eyebrow', 'File'), grid);
  }

  function deleteCell() {
    const named = d.presets.filter((p) => p.included).map((p) => p.name);
    const roots = [...new Set(d.files.map((f) => f.root))].join(', ');
    return h('div.tray-cell.row-cell',
      h('div.cell-grow', h('span.eyebrow', 'Delete'),
        h('span.mono-meta.wrap', `Removes the file from ${roots}${named.length ? ` and from the ${named.join(', ')} preset${named.length === 1 ? '' : 's'}` : ''}. Past images keep their record.`)),
      h('button.mini', {
        type: 'button',
        onclick: async () => {
          const ok = await dialog({
            title: `Delete ${d.name}?`, body: `${d.files.map((f) => f.path).join('\n')}\n\nThis can't be undone. Past images keep their record.`,
            confirm: 'Delete', danger: true,
          });
          if (!ok) return;
          try {
            await api.deleteLora(id);
            loadLoras();
            toast(`Deleted ${d.name}`);
            navigate('#/loras', { replace: true });
          } catch (err) {
            toast(err.message, { error: true });
          }
        },
      }, icon('trash', 13), 'Delete file…'));
  }

  // --- the cover ------------------------------------------------------------------------

  function coverCol() {
    const frame = h('div.tile-img.lib-cover', { style: { aspectRatio: d.cover?.width ? `${d.cover.width} / ${d.cover.height}` : '4 / 5' } });
    if (d.cover) {
      const img = h('img.tile-pic', { alt: `${d.name} cover` });
      setImageUrl(img, d.cover.url);
      frame.append(img);
    } else {
      frame.append(h('span.tile-desc', 'no cover yet · pick one below'));
    }
    const reel = d.recent_images.slice(0, 5).map((a) => {
      const pic = h('img.reel-pic', { alt: '' });
      setImage(pic, a);
      return h('button.reel-item.lib-reel', {
        type: 'button', class: d.cover?.artifact_id === a.id ? 'on' : '', title: `Use ${a.label} as the cover`,
        onclick: () => save(() => api.setLoraCover(id, { artifact_id: a.id }), 'Cover set'),
      }, h('span.reel-thumb', pic), a.label);
    });
    return h('div.engine-logcol',
      h('div.log-head', h('span.eyebrow', 'Cover'), h('span.spacer'), d.cover && h('span.mono-meta', d.cover.source),
        d.cover && iconButton('x', 'Remove the cover', () => save(() => api.clearLoraCover(id)), 14)),
      h('div.blueprint.lib-cover-frame', corners(), frame),
      h('div.lib-cover-actions',
        h('button.mini', { type: 'button', onclick: () => pickCover() }, icon('image', 13), 'Pick from a conversation'),
        h('button.mini', {
          type: 'button', disabled: !d.source, title: d.source ? `The first image on ${d.source.repo}'s card` : 'It didn\'t come from the Hub',
          onclick: () => save(() => api.setLoraCover(id, { hub: true }), 'Cover set'),
        }, 'Use Hub thumbnail')),
      h('span.tray-hint', d.recent_images.length ? 'Images made with this LoRA, newest first:' : 'Picks from the images made with this LoRA first.'),
      reel.length > 0 && h('div.lib-reel-grid', reel));
  }

  async function pickCover() {
    let others = [];
    try {
      others = (await api.gallery({ limit: 30 })).items.filter((c) => c.cover && !c.origin?.link_id).map((c) => ({ ...c.cover, title: c.title }));
    } catch {
      // Only the images made with it, then.
    }
    const mine = new Set(d.recent_images.map((a) => a.id));
    const pick = (a, close) => h('button.lib-pick', {
      type: 'button', title: [a.label, a.title].filter(Boolean).join(' · '),
      onclick: () => { close(); save(() => api.setLoraCover(id, { artifact_id: a.id }), 'Cover set'); },
    }, (() => { const img = h('img', { alt: a.label, loading: 'lazy' }); setImage(img, a); return img; })(), h('span.mono-meta', a.label));
    modal((close) => h('div.dialog.blueprint.lib-pick-dialog', { role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Pick a cover' },
      corners(),
      h('div.dialog-title', 'Pick a cover'),
      d.recent_images.length > 0 && [h('span.eyebrow', 'Made with this LoRA'), h('div.lib-pick-grid', d.recent_images.map((a) => pick(a, close)))],
      h('span.eyebrow', 'Latest image of other conversations'),
      h('div.lib-pick-grid', others.filter((a) => !mine.has(a.id)).map((a) => pick(a, close))),
      h('div.dialog-actions', h('button.btn.btn-secondary', { type: 'button', onclick: () => close() }, 'Cancel'))));
  }

  // --- usage ------------------------------------------------------------------------------

  function usageSection() {
    const tiles = d.used_in.map((u) => {
      const a = u.artifact;
      const img = a ? h('img.tile-pic', { alt: u.title ?? '', loading: 'lazy' }) : h('span.tile-desc', 'no image');
      if (a) setImage(img, a);
      return h('a.tile.result', { href: a ? convHash(u.conversation_id, a.label) : convHash(u.conversation_id, 'chat') },
        h('div.tile-img', { style: { aspectRatio: '1 / 1' } }, img),
        h('div.tile-foot', h('div.tile-foot-text',
          h('span.tile-title', u.title || 'Untitled'),
          h('span.mono-meta', [u.weight != null && weightText(u.weight), plural(u.images, 'image'), ago(u.last_used_at)].filter(Boolean).join(' · ')))));
    });
    return h('section.lib-usage',
      h('div.lib-usage-head', h('span.search-count', d.conversations ? `Used in ${plural(d.conversations, 'conversation')}` : 'Not used yet'),
        h('span.mono-meta', `${plural(d.uses, 'image')} · from the LoRAs each job records · newest first`)),
      tiles.length > 0 && h('div.lib-usage-grid', tiles));
  }

  load();
  return {
    el,
    destroy() {
      gone = true;
      status.destroy();
      strip.destroy();
    },
  };
}
