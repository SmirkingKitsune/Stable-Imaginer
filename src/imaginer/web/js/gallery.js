// The gallery (designs 1d–1g, 1k): every conversation as a tile showing its latest image,
// newest first, in a masonry that grows as you scroll. The prompt bar is docked at the
// bottom; past the last tile it rises into the open space under a "New conversation" title.
// With no conversations at all, the bar sits centred (1e).
//
// Search (design 6a) asks the server, which looks through titles, prompts, labels and
// replies of every conversation, archived ones too if asked, a page at a time: results
// arrive while a bar under the header tracks how much has been searched. Result tiles
// always show their footer, with the matching text highlighted.
//
// Keyboard (design 7a): the tiles are one tab stop; arrow keys move between them.
//
// Linked instances' conversations appear here too, newest activity first with this
// server's own, and carry their instance's name. They stay where they are: opening one
// goes through this server to that instance. One that doesn't answer is noted in the header.

import { api, setImage } from './api.js';
import { announce, brand, connectionStrip, statusLine } from './chrome.js';
import { corners, h, icon, iconButton, clear } from './dom.js';
import { ago, plural, ratioOf, truncate } from './format.js';
import { convHash } from './origin.js';
import { promptBar } from './promptbar.js';
import { connectionLost, loadGallery, on, progressOf, running, send, setArchived, state, watchTurn } from './store.js';

const END_SPACER = 560; // px of open space after the last tile (design 1d)
const END_RULE = 72; // px from the top of the end block to below its rule, plus a margin
const END_EASE = 160; // px of spare room over which the bar eases from docked to centred
const SEARCH_DEBOUNCE_MS = 250;
const SEARCH_PAGE = 60;
const FILTERS = { all: 'title,prompt,label,reply', titles: 'title', prompts: 'prompt' };

export function galleryScreen({ archived = false, navigate }) {
  setArchived(archived);
  let endP = 0;
  // Search: q is what was typed; the rest is the server's answer so far.
  const found = {
    q: '', filter: 'all', archived, items: [], scanned: 0, total: 0, done: true, error: null, run: 0,
  };
  let debounce = null;
  let roving = null; // id of the tile that holds the tab stop

  const clearBtn = h('button.att-x.search-clear', {
    type: 'button', title: 'Clear search', 'aria-label': 'Clear search', hidden: true,
    onclick: () => { resetSearch(); search.focus(); },
  }, icon('x', 12));
  const search = h('input.search-input', {
    type: 'search', placeholder: archived ? 'Search archived' : 'Search conversations',
    'aria-label': 'Search conversations',
    oninput: () => {
      clearBtn.hidden = !search.value;
      clearTimeout(debounce);
      debounce = setTimeout(() => startSearch(search.value.trim()), SEARCH_DEBOUNCE_MS);
    },
    onkeydown: (e) => {
      if (e.key === 'Escape') { resetSearch(); search.blur(); }
      if (e.key === 'Enter') {
        // Enter opens the first result.
        e.preventDefault();
        clearTimeout(debounce);
        if (found.q !== search.value.trim()) startSearch(search.value.trim());
        else if (found.items[0]) columns.querySelector('.tile')?.click();
      }
    },
    onblur: () => { if (!search.value) header.classList.remove('searching'); },
  });
  const status = statusLine();
  const statusShort = statusLine({ short: true });
  const emptyStatus = statusLine();
  const searchBox = h('label.search', icon('search', 15), search, clearBtn, h('kbd', '/'));
  const searchToggle = iconButton('search', 'Search', () => {
    header.classList.toggle('searching');
    search.focus();
  });
  searchToggle.classList.add('only-narrow');

  const offlineNote = h('span.topbar-tag.sources-off', { hidden: true });
  const header = h('header.topbar',
    brand(),
    archived && h('span.topbar-tag', 'Archived'),
    offlineNote,
    h('span.spacer'),
    searchBox,
    h('div.only-wide', status.el),
    h('div.only-narrow', statusShort.el),
    searchToggle,
    h('a.icon-btn', {
      href: archived ? '#/' : '#/archived', title: archived ? 'Back to gallery' : 'Archived',
      'aria-label': archived ? 'Back to gallery' : 'Archived',
    }, icon(archived ? 'back' : 'archive')),
  );

  const progress = h('div.search-progress', { hidden: true, 'aria-hidden': 'true' }, h('i'));
  const resultCount = h('span.search-count');
  const resultMeta = h('span.mono-meta');
  const filterBtn = (key, text) => h('button.pick', { type: 'button', dataset: { key }, onclick: () => setFilter(key) }, text);
  const archivedBtn = h('button.pick', { type: 'button', onclick: () => toggleArchived() }, '+ Archived');
  const filters = h('div.tray-seg.search-filters', { role: 'group', 'aria-label': 'Search in' },
    filterBtn('all', 'All'), filterBtn('titles', 'Titles'), filterBtn('prompts', 'Prompts'),
    !archived && archivedBtn);
  const searchHead = h('div.search-head', { hidden: true }, resultCount, resultMeta, h('span.spacer'), filters);

  const columns = h('div.masonry', { onkeydown: (e) => roveKey(e), onfocusin: (e) => noteFocus(e) });
  const end = h('div.gallery-end');
  const sentinel = h('div.sentinel');
  const scroller = h('div.gallery-scroll', { onscroll: () => updateEnd() }, searchHead, columns, sentinel, end);
  const scrim = h('div.scrim');
  const hero = h('div.dock-hero',
    h('span.eyebrow.accent', 'New conversation'),
    h('h1.hero-title', 'Describe an image, or drop one in.'));
  const bar = promptBar({
    draftKey: 'new',
    placeholder: 'Describe an image, or drop one in…',
    onSubmit: async (msg) => {
      const id = await send(null, msg);
      navigate(`#/c/${id}/chat`);
    },
    onStop: () => {},
  });
  const dock = h('div.dock', hero, bar.el);
  const empty = h('div.empty');
  const strip = connectionStrip();
  const el = h('div.screen.gallery', header, strip.el, progress, scroller, scrim, dock, empty);

  // --- tiles ------------------------------------------------------------------

  const tiles = new Map(); // key -> {el, sig, ratio}

  function tile(c, match = null) {
    const key = match ? `s:${c.id}` : c.id;
    const sig = JSON.stringify([c.title, c.image_count, c.cover?.id, c.updated_at, c.origin?.link_id, match, match && found.q]);
    const cached = tiles.get(key);
    if (cached?.sig === sig) return cached;
    const title = c.title || 'Untitled';
    const img = c.cover
      ? h('img.tile-pic', { alt: title, loading: 'lazy', decoding: 'async' })
      : h('span.tile-desc', truncate(title, 48));
    if (c.cover) setImage(img, c.cover);
    const run = h('div.tile-run', h('div.tile-progress', h('i')), h('span.tile-chip'));
    const href = c.image_count ? convHash(c.id) : convHash(c.id, 'chat');
    const remote = c.origin?.link_id ? c.origin.name : null;
    const badge = remote && h('span.tile-origin', { title: `Lives on ${remote}` }, icon('link', 11), remote);
    const node = match
      ? h('a.tile.result', {
        href, dataset: { id: c.id }, tabindex: '-1',
        'aria-label': `${title}, matched in ${match.field === 'prompt' && match.label ? `${match.label} prompt` : match.field}`,
      },
      h('div.tile-img', { style: { aspectRatio: ratioOf(c.cover) } }, img),
      badge,
      run,
      h('div.tile-foot',
        h('div.tile-foot-text',
          h('span.tile-title', match.field === 'title' ? highlight(title) : title),
          h('span.mono-meta.match', matchLine(c, match)))))
      : h('a.tile', {
        href, dataset: { id: c.id }, tabindex: '-1',
        'aria-label': `${title}, ${plural(c.image_count, 'image')}${remote ? `, on ${remote}` : ''}`,
      },
      h('div.tile-img', { style: { aspectRatio: ratioOf(c.cover) } }, img),
      badge,
      run,
      h('div.tile-hover.blueprint',
        corners(),
        c.cover && h('span.label-chip', c.cover.label),
        h('div.tile-foot',
          h('div.tile-foot-text',
            h('span.tile-title', title),
            h('span.mono-meta', `${plural(c.image_count, 'image')} · ${ago(c.updated_at)}`)),
          icon('arrowRight', 16))));
    const entry = { el: node, sig, run, id: c.id, ratio: (c.cover?.width ? c.cover.height / c.cover.width : 0.75) + (match ? 0.18 : 0) };
    tiles.set(key, entry);
    paintRun(c.id, entry);
    return entry;
  }

  /** "prompt · “…a tiny lighthouse glyph…”", "img-7 prompt · …", "5 images · 2h · title". */
  function matchLine(c, match) {
    if (match.field === 'title') return `${plural(c.image_count, 'image')} · ${ago(c.updated_at)} · title`;
    if (match.field === 'label') return [`${match.label} · label`, ago(c.updated_at)].join(' · ');
    const where = match.field === 'prompt' ? (match.label ? `${match.label} prompt` : 'prompt') : 'reply';
    return [`${where} · “`, ...highlight(match.snippet ?? ''), '”', match.field === 'reply' ? ` · ${ago(c.updated_at)}` : ''];
  }

  /** The text with each occurrence of the query in a <mark>. */
  function highlight(text) {
    const q = found.q.toLowerCase();
    if (!q) return [text];
    const parts = [];
    let at = 0;
    const lower = text.toLowerCase();
    for (let i = lower.indexOf(q); i !== -1; i = lower.indexOf(q, i + q.length)) {
      parts.push(text.slice(at, i), h('mark', text.slice(i, i + q.length)));
      at = i + q.length;
    }
    parts.push(text.slice(at));
    return parts;
  }

  function paintRun(id, entry) {
    const live = running(id);
    entry.run.hidden = !live;
    if (!live) return;
    const p = progressOf(id);
    const pct = p == null ? null : `${Math.round(p * 100)}%`;
    entry.run.querySelector('i').style.width = pct ?? '0%';
    entry.run.querySelector('.tile-chip').textContent = pct == null ? 'WORKING' : `GENERATING ${pct}`;
  }

  function columnCount() {
    const w = scroller.clientWidth || window.innerWidth;
    return w >= 1600 ? 5 : w >= 1060 ? 4 : w >= 760 ? 3 : 2;
  }

  /** Shortest-column masonry: appending a page never moves the tiles above it. */
  function layout() {
    const g = state.gallery;
    const off = g.sources.filter((src) => src.status !== 'ok');
    offlineNote.hidden = !off.length;
    offlineNote.textContent = off.map((src) => `${src.name} ${src.status === 'unreachable' ? 'offline' : 'unavailable'}`).join(' · ');
    offlineNote.title = off.map((src) => `${src.name}: ${src.detail ?? src.status}`).join('\n');
    const searching = !!found.q;
    const entries = searching
      ? found.items.map((hit) => tile(hit, hit.match))
      : g.items.map((c) => tile(c));
    const n = columnCount();
    const cols = Array.from({ length: n }, () => h('div.masonry-col'));
    const heights = new Array(n).fill(0);
    const place = (node, ratio) => {
      const i = heights.indexOf(Math.min(...heights));
      cols[i].append(node);
      heights[i] += ratio + 0.05;
    };
    for (const entry of entries) place(entry.el, entry.ratio);
    if (searching && !found.done) {
      place(h('div.tile.searching', { 'aria-hidden': 'true' },
        h('div.tile-img', { style: { aspectRatio: '4 / 3' } }, h('span.tile-desc', 'searching older…'))), 0.75);
    }
    clear(columns).append(...cols);
    setRoving();
    renderSearchHead();

    const isEmpty = g.loaded && !g.items.length && !searching;
    el.classList.toggle('is-empty', isEmpty && !archived);
    clear(empty);
    if (isEmpty && !archived) {
      empty.append(
        h('div.empty-hero',
          h('span.eyebrow.accent', 'New conversation'),
          h('h1.hero-title.big', 'Describe an image, or drop one in.')),
        bar.el,
        emptyStatus.el);
    } else if (!dock.contains(bar.el)) {
      dock.append(bar.el);
    }

    clear(end);
    const finished = searching ? found.done : g.done;
    if (finished) {
      end.style.height = `${END_SPACER}px`;
      const label = searching
        ? found.error ? `SEARCH FAILED · ${found.error.toUpperCase()}`
          : found.items.length ? `END OF RESULTS · ${plural(found.items.length, 'CONVERSATION')}`
            : `NOTHING MATCHES “${found.q.toUpperCase()}”`
        : archived && !g.items.length
          ? 'NO ARCHIVED CONVERSATIONS'
          : `END OF ${archived ? 'ARCHIVE' : 'GALLERY'} · ${plural(g.items.length, 'CONVERSATION')}`;
      end.append(h('div.rule-label', h('span.rule'), h('span', label), h('span.rule')));
    } else {
      end.style.height = '0px';
    }
    updateEnd();
  }

  // --- search (design 6a) ---------------------------------------------------------------

  function renderSearchHead() {
    const searching = !!found.q;
    searchHead.hidden = !searching;
    progress.hidden = !searching || found.done;
    if (!searching) return;
    resultCount.textContent = plural(found.items.length, 'conversation');
    const where = { all: 'titles, prompts, labels, replies', titles: 'titles', prompts: 'prompts' }[found.filter];
    resultMeta.textContent = found.done
      ? `searched ${found.total} · ${where}${found.archived ? ' · archived too' : ''}`
      : `searched ${found.scanned} of ${found.total} · ${where}`;
    progress.firstChild.style.width = `${found.total ? Math.round((found.scanned / found.total) * 100) : 0}%`;
    for (const b of filters.querySelectorAll('[data-key]')) {
      const on = b.dataset.key === found.filter;
      b.classList.toggle('on', on);
      b.setAttribute('aria-pressed', String(on));
    }
    archivedBtn.classList.toggle('on', found.archived);
    archivedBtn.setAttribute('aria-pressed', String(found.archived));
  }

  /** Ask the server page by page; a newer search (or clearing it) stops this one. */
  async function startSearch(q) {
    const run = (found.run += 1);
    Object.assign(found, { q, items: [], scanned: 0, total: 0, done: !q, error: null });
    layout();
    if (!q) return;
    let cursor = null;
    do {
      let page;
      try {
        page = await api.search({
          q, fields: FILTERS[found.filter], archived: found.archived, cursor, limit: SEARCH_PAGE,
        });
      } catch (err) {
        if (run !== found.run) return;
        if (err.status === 0) connectionLost();
        Object.assign(found, { done: true, error: err.message });
        layout();
        return;
      }
      if (run !== found.run) return;
      found.items.push(...page.items);
      found.scanned += page.scanned;
      found.total = page.total;
      cursor = page.next_cursor;
      for (const hit of page.items) if (hit.turn_running) watchTurn(hit.id);
      found.done = !cursor;
      layout();
    } while (cursor);
    announce(`${plural(found.items.length, 'conversation')} found`);
  }

  function resetSearch() {
    clearTimeout(debounce);
    search.value = '';
    clearBtn.hidden = true;
    startSearch('');
  }

  function setFilter(key) {
    found.filter = key;
    startSearch(found.q);
  }

  function toggleArchived() {
    found.archived = !found.archived;
    startSearch(found.q);
  }

  // --- keyboard: one tab stop, arrows between tiles (design 7a) ---------------------------

  const allTiles = () => [...columns.querySelectorAll('a.tile')];

  function setRoving() {
    const list = allTiles();
    if (!list.length) return;
    const keep = list.find((t) => t.dataset.id === roving) ?? list[0];
    for (const t of list) t.tabIndex = t === keep ? 0 : -1;
  }

  function noteFocus(e) {
    const t = e.target.closest?.('a.tile');
    if (!t) return;
    roving = t.dataset.id;
    for (const other of allTiles()) other.tabIndex = other === t ? 0 : -1;
  }

  function roveKey(e) {
    const from = e.target.closest?.('a.tile');
    if (!from || !['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
    e.preventDefault();
    const list = allTiles();
    let next = null;
    if (e.key === 'Home') next = list[0];
    else if (e.key === 'End') next = list[list.length - 1];
    else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
      const col = [...from.parentElement.querySelectorAll('a.tile')];
      next = col[col.indexOf(from) + (e.key === 'ArrowDown' ? 1 : -1)] ?? null;
    } else {
      // The tile in the next column whose middle is nearest this one's.
      const cols = [...columns.children];
      const target = cols[cols.indexOf(from.parentElement) + (e.key === 'ArrowRight' ? 1 : -1)];
      if (target) {
        const r = from.getBoundingClientRect();
        const mid = r.top + r.height / 2;
        let best = Infinity;
        for (const t of target.querySelectorAll('a.tile')) {
          const tr = t.getBoundingClientRect();
          const d = Math.abs(tr.top + tr.height / 2 - mid);
          if (d < best) { best = d; next = t; }
        }
      }
    }
    if (next) {
      next.focus();
      next.scrollIntoView({ block: 'nearest' });
    }
  }

  /** Docked at the bottom until the open space under the end rule can hold the bar and
   * its title; then it eases into the middle of that space (designs 1d, 1g). */
  function updateEnd() {
    const g = state.gallery;
    const base = window.innerWidth < 720 ? 14 : 24;
    let p = 0;
    let bottom = base;
    const finished = found.q ? found.done : g.done && g.loaded;
    if (finished && end.firstChild) {
      const free = el.getBoundingClientRect().bottom - end.getBoundingClientRect().top - END_RULE;
      const room = free - dock.offsetHeight;
      p = Math.max(0, Math.min(1, room / END_EASE));
      bottom = base + p * Math.max(0, room / 2 - base);
    }
    if (Math.abs(p - endP) < 0.004 && el.dataset.ready) return;
    endP = p;
    el.dataset.ready = '1';
    dock.style.bottom = `${Math.round(bottom)}px`;
    hero.style.opacity = String(p);
    hero.style.visibility = p > 0.02 ? 'visible' : 'hidden';
    scrim.style.opacity = String(1 - p);
  }

  const observer = new IntersectionObserver((entries) => {
    if (!found.q && entries.some((e) => e.isIntersecting)) loadGallery();
  }, { root: scroller, rootMargin: '900px 0px' });
  observer.observe(sentinel);

  let lastCols = 0;
  const resize = new ResizeObserver(() => {
    const n = columnCount();
    if (n !== lastCols) { lastCols = n; layout(); }
    updateEnd();
  });
  resize.observe(scroller);

  const onSlash = (e) => {
    if (e.key !== '/' || e.target.closest?.('input, textarea, [contenteditable]')) return;
    e.preventDefault();
    header.classList.add('searching');
    search.focus();
  };
  document.addEventListener('keydown', onSlash);

  const unsubscribe = [
    on('gallery', layout),
    on('turns', () => { for (const entry of tiles.values()) paintRun(entry.id, entry); }),
  ];
  layout();
  loadGallery({ fresh: true });

  return {
    el,
    bar,
    destroy() {
      found.run += 1;
      clearTimeout(debounce);
      observer.disconnect();
      resize.disconnect();
      document.removeEventListener('keydown', onSlash);
      unsubscribe.forEach((u) => u());
      [status, statusShort, emptyStatus, bar, strip].forEach((part) => part.destroy());
    },
  };
}
