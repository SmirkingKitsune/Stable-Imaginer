// The viewer (designs 2a, 2e): a tile opens here, on the conversation's latest image, with a
// reel of every other image beside it (above the bar on narrow screens). Focusing the bar,
// or pressing Conversation, opens the full thread with the draft kept, on the branch of the
// image shown. Images from other branches than the one shown carry a branch mark in the
// reel; Retry runs an image's tool call again on a new branch.
//
// Inspection (designs 5a, 5b): a zoom bar in the stage (fit, 1:1, steps up to 400%), drag
// to pan once the image outgrows the stage, with a minimap. Compare pairs an edit with its
// edit_source from lineage, as a slider or side by side; zoom applies to both sides.

import { download, setImage } from './api.js';
import { connectionStrip, toast } from './chrome.js';
import { corners, h, icon, iconButton, clear } from './dom.js';
import { imagesSummary, specOf, truncate } from './format.js';
import { convHash } from './origin.js';
import { promptBar } from './promptbar.js';
import {
  attacherOf, branchAt, byLabel, currentBranch, ensureLoaded, images, lineageOf, on, onPath, retry,
  running, select, send, stop,
} from './store.js';

const SOURCE_WORD = { upload: 'upload', generated: 'generated', edited: 'edited', derived: 'derived' };
const STEPS = [0.5, 1, 2, 4]; // zoom levels besides fit
const MAX_ZOOM = 8;
const PAN_STEP = 60; // px per arrow key while zoomed
const SIDE_GAP = 12;

export function viewerScreen({ id, label, navigate }) {
  let c = null;
  let current = null; // artifact on stage
  const chat = convHash(id, 'chat');

  // zoom: null = fit, else image px → screen px. pan: offset of the image centre, screen px.
  const view = { zoom: null, panX: 0, panY: 0, w: 0, h: 0 };
  // compare: null, or {before: artifact | null (picking one), mode: 'slider' | 'side', split}
  let compare = null;
  let drag = null;
  let spaceHeld = false;

  const title = h('span.topbar-title', '…');
  const meta = h('span.topbar-meta');
  const header = h('header.topbar.sub',
    iconButton('back', 'Back to gallery', () => navigate('#/')),
    h('div.topbar-heading', title, meta),
    h('span.spacer'),
    h('button.btn-line.only-wide', { type: 'button', onclick: () => current && save(current) }, icon('download', 15), 'Download'),
    h('button.btn-line.only-wide', { type: 'button', onclick: () => openChat() }, icon('chat', 15), 'Conversation'),
    h('span.only-narrow', iconButton('chat', 'Conversation', () => openChat())),
  );

  // --- stage: one pane, or two side by side -------------------------------------

  function pane() {
    const pic = h('img.zoom-pic', { alt: '', draggable: 'false' });
    // The slider's "before" image: on top of pic, clipped to the left of the divider.
    const under = h('img.zoom-pic.under', { alt: '', draggable: 'false', hidden: true });
    const labelChip = h('span.label-chip');
    const kindChip = h('span.kind-chip');
    const el = h('div.stage.blueprint', { tabindex: '-1' },
      corners(),
      h('div.stage-clip', pic, under),
      h('div.stage-chips', labelChip, kindChip));
    return { el, pic, under, labelChip, kindChip };
  }
  const main = pane(); // the image shown; "after" when comparing
  const side = pane(); // "before", side by side only
  side.el.hidden = true;
  const divider = h('div.cmp-line', { hidden: true });
  const handle = h('div.cmp-handle.blueprint', { hidden: true }, corners(), icon('move', 15));
  main.el.append(divider, handle);

  const miniView = h('div.mini-view');
  const miniPic = h('img', { alt: '' });
  const minimap = h('div.minimap.blueprint', { hidden: true, 'aria-hidden': 'true' },
    h('div.mini-clip', miniPic, miniView));
  const zoomPct = h('span.zoom-pct', { 'aria-live': 'polite' });
  const fitBtn = h('button.icon-btn.zoom-text', { type: 'button', title: 'Fit (0)', onclick: () => setZoom(null) }, icon('fit', 15), 'Fit');
  const oneBtn = h('button.icon-btn.zoom-text.mono', { type: 'button', title: 'Actual size (1)', onclick: () => setZoom(1) }, '1:1');
  const zoomBar = h('div.zoom-bar.blueprint', { onpointerdown: (e) => e.stopPropagation() },
    corners(),
    iconButton('zoomOut', 'Zoom out (−)', () => zoomStep(-1), 16),
    zoomPct,
    iconButton('zoomIn', 'Zoom in (+)', () => zoomStep(1), 16),
    h('span.zoom-sep'),
    fitBtn, oneBtn);
  main.el.append(minimap, zoomBar);
  const panes = h('div.stage-row', main.el, side.el);

  const prompt = h('span.stage-prompt');
  const source = h('span.mono-meta.accent');
  const spec = h('span.mono-meta');
  const branchMeta = h('span.mono-meta');
  const pixelNote = h('span.mono-meta');
  const retryBtn = iconButton('retry', 'Retry → new branch', () => current && retryCurrent());
  const compareBtn = iconButton('compare', 'Compare (C)', () => toggleCompare());
  const seg = (mode, text) => h('button.pick', { type: 'button', onclick: () => setMode(mode) }, text);
  const modeSeg = h('div.tray-seg.cmp-modes', { hidden: true, role: 'group', 'aria-label': 'Compare view' },
    seg('slider', 'Slider'), seg('side', 'Side by side'));
  const actions = h('span.stage-actions',
    retryBtn,
    iconButton('pencil', 'Edit this image', () => current && editCurrent()),
    iconButton('brush', 'Mask: paint the area to change', () => current && navigate(convHash(id, 'mask', current.label))),
    iconButton('download', 'Download', () => current && save(current)));
  const info = h('div.stage-info',
    h('div.stage-text', prompt, h('div.stage-meta', source, spec, branchMeta, pixelNote)),
    modeSeg, compareBtn, actions);
  const stageCol = h('div.stage-col', panes, info);
  const reel = h('div.reel', {
    role: 'listbox', tabindex: '0', 'aria-label': 'Images in this conversation', 'aria-orientation': 'vertical',
  });
  const body = h('div.viewer-body', stageCol, reel);

  const bar = promptBar({
    draftKey: id,
    placeholder: 'Reply to open the conversation…',
    isBusy: () => running(id),
    onFocus: () => openChat({ focusBar: true }),
    onSubmit: async (msg) => {
      showBranchOf(current);
      await send(id, msg);
      navigate(chat);
    },
    onStop: () => stop(id),
  });
  const strip = connectionStrip({ turnRunning: () => running(id) });
  const el = h('div.screen.viewer', header, strip.el, body, h('div.scrim.short'), h('div.dock', bar.el));

  // --- layout -------------------------------------------------------------------

  function fit() {
    if (!current) return;
    const narrow = window.innerWidth < 720;
    const maxW = narrow ? window.innerWidth - 20 : Math.min(1100, window.innerWidth - 200);
    const maxH = narrow
      ? window.innerHeight * 0.5
      : window.innerHeight - 56 - 22 - 64 - 190;
    const w = current.width || 4;
    const hgt = current.height || 3;
    const sideBySide = compare?.mode === 'side' && compare.before;
    const room = sideBySide ? (maxW - SIDE_GAP) / 2 : maxW;
    const scale = Math.min(room / w, Math.max(160, maxH) / hgt);
    view.w = Math.round(w * scale);
    view.h = Math.round(hgt * scale);
    for (const p of [main, side]) {
      p.el.style.width = `${view.w}px`;
      p.el.style.height = `${view.h}px`;
    }
    stageCol.style.width = `${sideBySide ? view.w * 2 + SIDE_GAP : view.w}px`;
    reel.style.maxHeight = narrow ? '' : `${view.h + 60}px`;
    clampPan();
    paint();
  }

  const fitZoom = () => (current?.width ? view.w / current.width : 1);
  const zoomOf = () => view.zoom ?? fitZoom();
  const isFit = () => view.zoom == null || Math.abs(view.zoom - fitZoom()) < 0.001;

  function clampPan() {
    const z = zoomOf();
    const mx = Math.max(0, ((current?.width ?? 0) * z - view.w) / 2);
    const my = Math.max(0, ((current?.height ?? 0) * z - view.h) / 2);
    view.panX = Math.max(-mx, Math.min(mx, view.panX));
    view.panY = Math.max(-my, Math.min(my, view.panY));
  }

  /** Zoom to z (null = fit), keeping the image point under (ax, ay) in place; the anchor is
   * relative to the stage centre and defaults to it. */
  function setZoom(z, ax = 0, ay = 0) {
    const before = zoomOf();
    const fz = fitZoom();
    let next = z == null ? null : Math.max(Math.min(fz, 0.1), Math.min(MAX_ZOOM, z));
    if (next != null && Math.abs(next - fz) < 0.001) next = null;
    const after = next ?? fz;
    view.panX = ax - ((ax - view.panX) * after) / before;
    view.panY = ay - ((ay - view.panY) * after) / before;
    view.zoom = next;
    if (next == null) { view.panX = 0; view.panY = 0; }
    clampPan();
    paint();
  }

  function zoomStep(dir) {
    const z = zoomOf();
    const levels = [...new Set([fitZoom(), ...STEPS])].sort((a, b) => a - b);
    const next = dir > 0 ? levels.find((l) => l > z + 0.001) : levels.findLast((l) => l < z - 0.001);
    if (next != null) setZoom(next);
  }

  /** Position the images for the current zoom and pan; refresh the bar, minimap and note. */
  function paint() {
    if (!current) return;
    const z = zoomOf();
    const w = current.width * z;
    const hgt = current.height * z;
    const left = (view.w - w) / 2 + view.panX;
    const top = (view.h - hgt) / 2 + view.panY;
    const pixelated = z >= 2;
    for (const img of [main.pic, main.under, side.pic]) {
      Object.assign(img.style, { left: `${left}px`, top: `${top}px`, width: `${w}px`, height: `${hgt}px` });
      img.classList.toggle('pixelated', pixelated);
    }
    const overflows = w > view.w + 1 || hgt > view.h + 1;
    const sliding = compare?.before && compare.mode === 'slider';
    main.el.classList.toggle('pannable', overflows && (!sliding || spaceHeld));
    main.el.classList.toggle('sliding', !!sliding && !spaceHeld);
    side.el.classList.toggle('pannable', overflows);
    main.el.classList.toggle('panning', !!drag && drag.kind === 'pan');

    zoomPct.textContent = `${Math.round(z * 100)}%`;
    fitBtn.classList.toggle('on', isFit());
    oneBtn.classList.toggle('on', Math.abs(z - 1) < 0.001);
    pixelNote.textContent = compare?.before
      ? (compare.mode === 'slider' ? 'drag or ← → to move the divider' : `${Math.round(z * 100)}% · both sides`)
      : isFit() ? `fit · ${Math.round(z * 100)}%`
        : Math.abs(z - 1) < 0.001 ? 'actual size · 1 image px = 1 screen px'
          : `${Math.round(z * 100)}% · drag to pan`;

    minimap.hidden = !overflows || compare?.mode === 'side';
    if (!minimap.hidden) {
      // At most 132 px on its longer side, in the image's own shape.
      const k = 132 / Math.max(current.width, current.height);
      minimap.style.width = `${Math.round(current.width * k) + 8}px`;
      minimap.style.height = `${Math.round(current.height * k) + 8}px`;
      Object.assign(miniView.style, {
        left: `${Math.max(0, (-left / w) * 100)}%`,
        top: `${Math.max(0, (-top / hgt) * 100)}%`,
        width: `${Math.min(100, (view.w / w) * 100)}%`,
        height: `${Math.min(100, (view.h / hgt) * 100)}%`,
      });
    }
    paintCompare();
  }

  // --- compare (design 5b) ----------------------------------------------------------

  function paintCompare() {
    const on = !!compare;
    const ready = on && compare.before;
    compareBtn.classList.toggle('on', on);
    compareBtn.setAttribute('aria-pressed', String(on));
    modeSeg.hidden = !ready;
    actions.hidden = on;
    info.classList.toggle('comparing', on);
    for (const b of modeSeg.children) {
      const selected = b.textContent === (compare?.mode === 'side' ? 'Side by side' : 'Slider');
      b.classList.toggle('on', selected);
      b.setAttribute('aria-pressed', String(selected));
    }
    const slider = ready && compare.mode === 'slider';
    const sideBySide = ready && compare.mode === 'side';
    main.under.hidden = !slider;
    divider.hidden = !slider;
    handle.hidden = !slider;
    side.el.hidden = !sideBySide;
    panes.classList.toggle('side', !!sideBySide);
    if (slider) {
      // The "before" image covers the left of the divider, over the "after" image.
      main.under.style.clipPath = `inset(0 ${100 - compare.split}% 0 0)`;
      divider.style.left = `${compare.split}%`;
      handle.style.left = `${compare.split}%`;
    }
    main.labelChip.textContent = ready ? `${current.label} · after` : current.label;
    main.labelChip.classList.toggle('after', !!ready);
    main.kindChip.hidden = !!ready;
    if (slider) {
      main.kindChip.hidden = false;
      main.kindChip.textContent = `${compare.before.label} · before`;
      main.kindChip.classList.add('before');
      // Before on the left: swap the chips' order.
      main.labelChip.parentElement.classList.add('swap');
    } else {
      main.kindChip.classList.remove('before');
      main.labelChip.parentElement.classList.remove('swap');
      if (!ready) main.kindChip.textContent = SOURCE_WORD[current.source] ?? current.source;
    }
    if (sideBySide) {
      side.labelChip.textContent = `${compare.before.label} · before`;
      side.kindChip.hidden = true;
      // Before goes first.
      if (panes.firstChild !== side.el) panes.prepend(side.el);
    } else if (panes.firstChild !== main.el) {
      panes.prepend(main.el);
    }
    if (on && !ready) {
      pixelNote.textContent = 'pick an image in the reel to compare with';
    }
  }

  async function toggleCompare(force = !compare) {
    if (!force) {
      compare = null;
      fit();
      renderReel();
      return;
    }
    compare = { before: null, mode: compare?.mode ?? 'slider', split: 50 };
    const target = current;
    const before = await editSource(target);
    if (current !== target || !compare) return;
    if (before) setBefore(before);
    else {
      paint();
      renderReel();
    }
  }

  function setBefore(artifact) {
    compare.before = artifact;
    setImage(main.under, artifact);
    setImage(side.pic, artifact);
    fit();
    renderReel();
  }

  function setMode(mode) {
    if (!compare) return;
    compare.mode = mode;
    fit();
  }

  async function editSource(artifact) {
    if (artifact.source !== 'edited') return null;
    const lineage = await lineageOf(c, artifact.id);
    const src = lineage?.parents.find((p) => p.relation === 'edit_source');
    return src ? c.artifacts.get(src.artifact_id) ?? byLabel(c, src.label) : null;
  }

  function moveSplit(clientX) {
    const r = main.el.getBoundingClientRect();
    compare.split = Math.max(2, Math.min(98, ((clientX - r.left) / r.width) * 100));
    paintCompare();
  }

  // --- pointer: pan, divider, ctrl+wheel zoom -------------------------------------------

  function onDown(e, p) {
    if (e.button !== 0) return;
    const sliding = compare?.before && compare.mode === 'slider' && p === main && !spaceHeld;
    if (sliding) {
      drag = { kind: 'split' };
      moveSplit(e.clientX);
    } else if (p.el.classList.contains('pannable')) {
      drag = { kind: 'pan', x: e.clientX, y: e.clientY, px: view.panX, py: view.panY };
    } else return;
    p.el.setPointerCapture?.(e.pointerId);
    e.preventDefault();
    paint();
  }
  function onMove(e) {
    if (!drag) return;
    if (drag.kind === 'split') moveSplit(e.clientX);
    else {
      view.panX = drag.px + (e.clientX - drag.x);
      view.panY = drag.py + (e.clientY - drag.y);
      clampPan();
      paint();
    }
  }
  function onUp() {
    if (!drag) return;
    drag = null;
    paint();
  }
  function onWheel(e, p) {
    if (!e.ctrlKey) return; // ctrl+wheel, and trackpad pinches, zoom; plain wheel does nothing
    e.preventDefault();
    const r = p.el.getBoundingClientRect();
    const factor = Math.exp(-e.deltaY * 0.01);
    setZoom(zoomOf() * factor, e.clientX - r.left - r.width / 2, e.clientY - r.top - r.height / 2);
  }
  for (const p of [main, side]) {
    p.el.addEventListener('pointerdown', (e) => onDown(e, p));
    p.el.addEventListener('pointermove', onMove);
    p.el.addEventListener('pointerup', onUp);
    p.el.addEventListener('pointercancel', onUp);
    p.el.addEventListener('wheel', (e) => onWheel(e, p), { passive: false });
  }

  // --- the image shown ---------------------------------------------------------------

  async function show(artifact) {
    const changed = current?.id !== artifact.id;
    current = artifact;
    setImage(main.pic, artifact);
    main.pic.alt = artifact.metadata?.prompt || artifact.label;
    setImage(miniPic, artifact);
    prompt.textContent = artifact.source === 'upload'
      ? 'Your upload.'
      : truncate(artifact.metadata?.prompt || '', 400);
    prompt.title = artifact.metadata?.prompt || '';
    spec.textContent = specOf(artifact);
    source.textContent = artifact.source === 'upload' ? 'your upload'
      : artifact.source === 'edited' ? 'edit_image' : 'generate_image';
    if (changed) {
      view.zoom = null;
      view.panX = 0;
      view.panY = 0;
      if (compare) {
        // Keep comparing when the next image is an edit too; otherwise stop.
        compare = artifact.source === 'edited' ? { ...compare, before: null } : null;
        if (compare) toggleCompare(true);
      }
    }
    fit();
    renderBranch();
    renderReel();
    if (location.hash !== convHash(id, artifact.label)) {
      history.replaceState(null, '', convHash(id, artifact.label));
    }
    if (artifact.source === 'edited') {
      const lineage = await lineageOf(c, artifact.id);
      if (current !== artifact || !lineage) return;
      const src = lineage.parents.find((p) => p.relation === 'edit_source');
      const refs = lineage.parents.filter((p) => p.relation === 'reference').map((p) => p.label);
      const mask = lineage.parents.find((p) => p.relation === 'mask');
      source.textContent = [src && `edit of ${src.label}`, refs.length && `ref ${refs.join(', ')}`, mask && `mask ${mask.label}`]
        .filter(Boolean).join(' + ') || 'edit_image';
    }
  }

  /** "branch · retry of img-3", and Retry, for the image on stage. */
  function renderBranch() {
    const by = attacherOf(c, current.id);
    const branch = by && (onPath(c).has(current.id) ? currentBranch(c) : branchAt(c, by.id));
    branchMeta.textContent = c.branches.length > 1 && branch ? `branch · ${branch.name}` : '';
    retryBtn.hidden = by?.role !== 'tool';
    retryBtn.disabled = running(id) || by?.status !== 'complete';
  }

  /** The reel is a listbox (design 7a): one tab stop, the shown image is its active option. */
  function renderReel() {
    clear(reel);
    const shown = onPath(c);
    for (const a of images(c)) {
      const selected = a.id === current?.id;
      const before = compare?.before?.id === a.id;
      const elsewhere = !shown.has(a.id) && attacherOf(c, a.id) !== null;
      const thumb = h('img.reel-pic', { alt: '', loading: 'lazy' });
      setImage(thumb, a);
      reel.append(h('div.reel-item', {
        id: `reel-${a.id}`, role: 'option', 'aria-selected': String(selected),
        title: elsewhere ? `${a.label} (on another branch)` : before ? `${a.label} (before)` : a.label,
        class: [selected && 'on', before && 'before'].filter(Boolean).join(' '),
        onclick: () => pick(a),
      }, h('span.reel-thumb', thumb, elsewhere && h('span.reel-mark', icon('branch', 11))), h('span', a.label)));
    }
    if (current) reel.setAttribute('aria-activedescendant', `reel-${current.id}`);
    reel.querySelector('.on')?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }

  /** A reel click: show it, or, while Compare waits for one, use it as "before". */
  function pick(artifact) {
    if (compare && !compare.before && artifact.id !== current?.id) setBefore(artifact);
    else show(artifact);
  }

  function step(delta) {
    const all = images(c);
    const i = all.findIndex((a) => a.id === current?.id);
    const next = all[i + delta];
    if (next) show(next);
  }

  /** Before leaving for the thread: show the branch the image on stage belongs to. */
  function showBranchOf(artifact) {
    const by = artifact && attacherOf(c, artifact.id);
    if (by && !onPath(c).has(artifact.id)) select(c, by.id);
  }

  function openChat(opts) {
    showBranchOf(current);
    navigate(chat, opts);
  }

  function editCurrent() {
    bar.addArtifact(current, { first: true, prefill: `Edit ${current.label}: ` });
    openChat({ focusBar: true });
  }

  async function retryCurrent() {
    const by = attacherOf(c, current.id);
    if (by?.role !== 'tool') return;
    try {
      await retry(id, by.id);
      navigate(chat);
    } catch (err) {
      if (err.status === 409) {
        toast('A turn is already running here.', { error: true, action: { label: 'Show', onclick: () => openChat() } });
      } else toast(err.message, { error: true });
    }
  }

  async function save(artifact) {
    try {
      await download(artifact);
      toast(`Download started · ${artifact.label}.${artifact.mime.split('/').pop().replace('jpeg', 'jpg')}`);
    } catch (err) {
      toast(`Download failed: ${err.message}`, { error: true });
    }
  }

  function render() {
    if (!c) return;
    const info_ = c.info;
    title.textContent = info_?.title || 'Untitled';
    const all = images(c);
    meta.textContent = imagesSummary(all, c.branches.length);
    bar.refresh();
    if (!all.length) {
      navigate(chat, { replace: true });
      return;
    }
    const wanted = (label && byLabel(c, label)) || null;
    if (!current || !all.some((a) => a.id === current.id)) show(wanted ?? all[all.length - 1]);
    else {
      renderBranch();
      renderReel();
    }
  }

  // --- keys: + − 0 1 zoom, arrows pan (or switch images at fit), C compare -------------

  const onKey = (e) => {
    if (e.target.closest?.('input, textarea, [contenteditable], .dialog')) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const zoomed = !isFit();
    const sliding = compare?.before && compare.mode === 'slider';
    const arrows = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    if (e.key === ' ' && (e.target === document.body || e.target.closest?.('.stage, .reel'))) {
      e.preventDefault();
      if (!spaceHeld) { spaceHeld = true; paint(); }
    } else if (e.key === '+' || e.key === '=') { e.preventDefault(); zoomStep(1); }
    else if (e.key === '-' || e.key === '_') { e.preventDefault(); zoomStep(-1); }
    else if (e.key === '0') { e.preventDefault(); setZoom(null); }
    else if (e.key === '1') { e.preventDefault(); setZoom(1); }
    else if (e.key === 'c' || e.key === 'C') { e.preventDefault(); toggleCompare(); }
    else if (e.key === 'Escape' && compare) { e.preventDefault(); toggleCompare(false); }
    else if (e.key in arrows) {
      e.preventDefault();
      const [dx, dy] = arrows[e.key];
      if (sliding && dx && !zoomed) {
        compare.split = Math.max(2, Math.min(98, compare.split + dx * 2));
        paintCompare();
      } else if (zoomed) {
        view.panX -= dx * PAN_STEP;
        view.panY -= dy * PAN_STEP;
        clampPan();
        paint();
      } else step(dx + dy > 0 ? 1 : -1);
    }
  };
  const onKeyUp = (e) => {
    if (e.key === ' ' && spaceHeld) { spaceHeld = false; paint(); }
  };
  document.addEventListener('keydown', onKey);
  document.addEventListener('keyup', onKeyUp);
  window.addEventListener('resize', fit);
  const unsubscribe = [on(`conv:${id}`, render), on('turns', () => {
    bar.refresh();
    if (current) renderBranch();
  })];

  ensureLoaded(id).then((loaded) => {
    c = loaded;
    render();
  }, (err) => {
    clear(body).append(h('div.notice', err.status === 404 ? 'This conversation no longer exists.' : err.message));
  });

  return {
    el,
    bar,
    destroy() {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('resize', fit);
      unsubscribe.forEach((u) => u());
      bar.destroy();
      strip.destroy();
    },
  };
}
