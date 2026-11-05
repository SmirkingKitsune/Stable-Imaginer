// The viewer (designs 2a, 2e): a tile opens here, on the conversation's latest image. The
// stage takes all the room the screen has; beside it (below it on narrow screens) a column
// holds the image's actions, prompt and details over a reel of every image, and collapses
// to the reel alone. Under the stage, a one-line reply field: focusing it, or pressing
// Conversation, opens the full thread with the draft kept, on the branch of the image shown.
// Images from other branches than the one shown carry a branch mark in the reel; Retry runs
// an image's tool call again on a new branch.
//
// Inspection (designs 5a, 5b): scroll or pinch to zoom about the pointer, drag to pan once
// the image outgrows the stage, double-click (or double-tap) between fit and 1:1, and a zoom
// bar (fit, 1:1, steps up to 400%) with a minimap. At fit, a sideways swipe on a touch
// screen steps through the images. Compare pairs an edit with its edit_source from lineage,
// as a slider or side by side; zoom applies to both sides.

import { download, setImage } from './api.js';
import { connectionStrip, toast } from './chrome.js';
import { corners, h, icon, iconButton, clear } from './dom.js';
import { imagesSummary, specOf } from './format.js';
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
const WHEEL_RATE = 0.002; // zoom per wheel px: a ~100 px notch is ×1.22
const PINCH_RATE = 0.01; // a trackpad pinch arrives as ctrl+wheel in small steps
const LINE_GRAB = 20; // px either side of the compare divider that move it while zoomed
const SWIPE = 60; // px of sideways travel that steps to the next image at fit
const DOUBLE_TAP_MS = 300;
const SIDE_KEY = 'imaginer.viewerSide';
const ZOOM_HINT = matchMedia('(hover: none)').matches ? 'pinch to zoom' : 'scroll to zoom';

/** The column starts open on wide screens, until the user says otherwise. */
function sideOpenAtStart() {
  try {
    const saved = localStorage.getItem(SIDE_KEY);
    if (saved) return saved === 'open';
  } catch {
    // No storage (private mode): the default below.
  }
  return window.innerWidth >= 720;
}

export function viewerScreen({ id, label, navigate }) {
  let c = null;
  let current = null; // artifact on stage
  const chat = convHash(id, 'chat');

  // zoom: null = fit, else image px → screen px. pan: offset of the image centre, screen px.
  const view = { zoom: null, panX: 0, panY: 0, w: 0, h: 0 };
  // compare: null, or {before: artifact | null (picking one), mode: 'slider' | 'side', split}
  let compare = null;
  let drag = null; // {kind: 'split'} | {kind: 'pan', x, y, px, py} | {kind: 'swipe'}
  let pinch = null; // two fingers down: {p, d0, z0, mx, my, px, py} as the pinch began
  const pointers = new Map(); // pointerId → {x, y, x0, y0} while down on a stage
  let tap = null; // the last tap, for a double tap: {x, y, t}
  let spaceHeld = false;
  let sideOpen = sideOpenAtStart();

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

  // --- side column: actions, prompt and details, the reel ---------------------------

  const sideLabel = h('span.side-label');
  const sideKind = h('span.eyebrow');
  const sideBtn = h('button.icon-btn.side-toggle', {
    type: 'button', 'aria-label': 'Details', onclick: () => setSide(!sideOpen),
  }, icon('info', 17));
  const action = (name, text, hint, onclick) =>
    h('button.side-action', { type: 'button', title: hint, onclick }, icon(name, 15), h('span', text));
  const retryBtn = action('retry', 'Retry', 'Retry → new branch', () => current && retryCurrent());
  const compareBtn = action('compare', 'Compare', 'Compare (C)', () => toggleCompare());
  const seg = (mode, text) => h('button.pick', { type: 'button', onclick: () => setMode(mode) }, text);
  const modeSeg = h('div.tray-seg.cmp-modes', { hidden: true, role: 'group', 'aria-label': 'Compare view' },
    seg('slider', 'Slider'), seg('side', 'Side by side'));
  const actions = h('div.side-actions',
    action('pencil', 'Edit', 'Edit this image', () => current && editCurrent()),
    action('brush', 'Mask', 'Mask: paint the area to change', () => current && navigate(convHash(id, 'mask', current.label))),
    retryBtn,
    action('download', 'Download', 'Download', () => current && save(current)));
  const pixelNote = h('span.mono-meta.side-note');
  const prompt = h('p.side-prompt');
  const source = h('span.mono-meta.accent');
  const spec = h('span.mono-meta');
  const branchMeta = h('span.mono-meta');
  const reelCount = h('span.eyebrow');
  const reel = h('div.reel', {
    role: 'listbox', tabindex: '0', 'aria-label': 'Images in this conversation', 'aria-orientation': 'vertical',
  });
  const sideEl = h('aside.viewer-side',
    h('div.side-head', h('div.side-title', sideLabel, sideKind), sideBtn),
    h('div.side-tools', actions, compareBtn, modeSeg, pixelNote),
    h('div.side-info', prompt, h('div.side-meta', source, spec, branchMeta)),
    h('div.side-reel', h('div.side-reel-head', reelCount), reel));

  const bar = promptBar({
    draftKey: id,
    compact: true,
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
  const body = h('div.viewer-body', h('div.stage-col', panes, h('div.viewer-dock', bar.el)), sideEl);
  const el = h('div.screen.viewer', header, strip.el, body);

  function setSide(open, { remember = true } = {}) {
    sideOpen = open;
    sideEl.classList.toggle('collapsed', !open);
    sideBtn.classList.toggle('on', open);
    sideBtn.title = open ? 'Hide details (I)' : 'Show details (I)';
    sideBtn.setAttribute('aria-expanded', String(open));
    if (!remember) return;
    try {
      localStorage.setItem(SIDE_KEY, open ? 'open' : 'closed');
    } catch {
      // Not persisted in private mode; still applies to this page.
    }
  }
  setSide(sideOpen, { remember: false });

  // --- layout -------------------------------------------------------------------

  /** One pane, or before and after side by side ("before" first). */
  function arrangePanes() {
    const sideBySide = !!compare?.before && compare.mode === 'side';
    side.el.hidden = !sideBySide;
    panes.classList.toggle('side', sideBySide);
    const first = sideBySide ? side.el : main.el;
    if (panes.firstChild !== first) panes.prepend(first);
  }

  /** The layout sizes the stage; measure it, and keep the image where it was. */
  function fit() {
    arrangePanes();
    view.w = main.el.clientWidth;
    view.h = main.el.clientHeight;
    if (!current) return;
    clampPan();
    paint();
  }

  const fitZoom = () => (current?.width && view.w ? Math.min(view.w / current.width, view.h / current.height) : 1);
  const zoomOf = () => view.zoom ?? fitZoom();
  const isFit = () => view.zoom == null || Math.abs(view.zoom - fitZoom()) < 0.001;
  const isSlider = () => !!compare?.before && compare.mode === 'slider';

  function overflows() {
    const z = zoomOf();
    return current.width * z > view.w + 1 || current.height * z > view.h + 1;
  }

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
    if (!current || !view.w) return;
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
    const big = overflows();
    const slider = isSlider();
    // Zoomed in, a drag pans; on the slider it moves the divider instead, near it or at fit.
    main.el.classList.toggle('pannable', big);
    main.el.classList.toggle('sliding', slider && !big && !spaceHeld);
    if (!slider || !big || spaceHeld) main.el.classList.remove('near-line');
    side.el.classList.toggle('pannable', big);
    main.el.classList.toggle('panning', drag?.kind === 'pan' || !!pinch);

    const pct = `${Math.round(z * 100)}%`;
    zoomPct.textContent = pct;
    fitBtn.classList.toggle('on', isFit());
    oneBtn.classList.toggle('on', Math.abs(z - 1) < 0.001);
    pixelNote.textContent = compare?.before
      ? (slider ? (big ? 'drag the divider, or the image to pan' : 'drag or ← → to move the divider') : `${pct} · both sides`)
      : isFit() ? `fit · ${pct} · ${ZOOM_HINT}`
        : Math.abs(z - 1) < 0.001 ? 'actual size · 1 image px = 1 screen px'
          : `${pct} · drag to pan`;

    minimap.hidden = !big || compare?.mode === 'side';
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
    }
    if (on && !ready) {
      pixelNote.textContent = 'pick an image below to compare with';
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
      // The reel is where "before" gets picked.
      if (!sideOpen) setSide(true, { remember: false });
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

  function nearLine(clientX) {
    const r = main.el.getBoundingClientRect();
    return Math.abs(clientX - (r.left + (r.width * compare.split) / 100)) <= LINE_GRAB;
  }

  // --- pointer: drag to pan, pinch and wheel to zoom, the divider, double tap -----------

  /** A point on screen, relative to the centre of pane p. */
  function anchor(p, x, y) {
    const r = p.el.getBoundingClientRect();
    return [x - r.left - r.width / 2, y - r.top - r.height / 2];
  }

  function onDown(e, p) {
    if (e.button !== 0 || !current) return;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY });
    p.el.setPointerCapture?.(e.pointerId);
    if (pointers.size === 2) {
      // A second finger: pinch, whatever the first one was doing.
      const [a, b] = pointers.values();
      const [mx, my] = anchor(p, (a.x + b.x) / 2, (a.y + b.y) / 2);
      pinch = { p, d0: Math.hypot(a.x - b.x, a.y - b.y) || 1, z0: zoomOf(), mx, my, px: view.panX, py: view.panY };
      drag = null;
      tap = null;
    } else if (pointers.size === 1) {
      if (isSlider() && p === main && !spaceHeld && (!overflows() || nearLine(e.clientX))) {
        drag = { kind: 'split' };
        moveSplit(e.clientX);
      } else if (overflows()) {
        drag = { kind: 'pan', x: e.clientX, y: e.clientY, px: view.panX, py: view.panY };
      } else if (e.pointerType !== 'mouse') {
        drag = { kind: 'swipe' };
      }
    }
    if (!drag && !pinch) return; // a click at fit: only a double one does something
    e.preventDefault();
    paint();
  }

  function onMove(e, p) {
    const pt = pointers.get(e.pointerId);
    if (!pt) {
      // Hovering a zoomed slider: show where a drag moves the divider rather than the image.
      if (p === main && current) {
        main.el.classList.toggle('near-line', isSlider() && !spaceHeld && overflows() && nearLine(e.clientX));
      }
      return;
    }
    pt.x = e.clientX;
    pt.y = e.clientY;
    if (pinch) movePinch();
    else if (drag?.kind === 'split') moveSplit(e.clientX);
    else if (drag?.kind === 'pan') {
      view.panX = drag.px + (e.clientX - drag.x);
      view.panY = drag.py + (e.clientY - drag.y);
      clampPan();
      paint();
    }
  }

  function movePinch() {
    const [a, b] = pointers.values();
    const [mx, my] = anchor(pinch.p, (a.x + b.x) / 2, (a.y + b.y) / 2);
    const fz = fitZoom();
    const scaled = (pinch.z0 * Math.hypot(a.x - b.x, a.y - b.y)) / pinch.d0;
    const z = Math.max(Math.min(fz, pinch.z0), Math.min(MAX_ZOOM, scaled));
    view.zoom = Math.abs(z - fz) < 0.001 ? null : z;
    // The image point that was under the fingers' midpoint stays under it.
    view.panX = mx - ((pinch.mx - pinch.px) * z) / pinch.z0;
    view.panY = my - ((pinch.my - pinch.py) * z) / pinch.z0;
    clampPan();
    paint();
  }

  function onUp(e, p) {
    const pt = pointers.get(e.pointerId);
    if (!pt) return;
    pointers.delete(e.pointerId);
    if (pinch) {
      if (pointers.size < 2) {
        pinch = null;
        // The finger left down carries on as a pan.
        const [rest] = pointers.values();
        drag = rest && overflows() ? { kind: 'pan', x: rest.x, y: rest.y, px: view.panX, py: view.panY } : null;
        paint();
      }
      return;
    }
    const kind = drag?.kind;
    drag = null;
    const dx = e.clientX - pt.x0;
    const dy = e.clientY - pt.y0;
    if (e.type === 'pointerup' && kind === 'swipe' && Math.abs(dx) > SWIPE && Math.abs(dx) > 2 * Math.abs(dy)) {
      step(dx < 0 ? 1 : -1);
    } else if (e.type === 'pointerup' && kind !== 'split' && Math.hypot(dx, dy) < 8) {
      onTap(e, p);
    }
    paint();
  }

  /** Two taps (or clicks) in quick succession: from fit to 1:1 about the point, or back. */
  function onTap(e, p) {
    const twice = tap && e.timeStamp - tap.t < DOUBLE_TAP_MS && Math.hypot(e.clientX - tap.x, e.clientY - tap.y) < 30;
    tap = twice ? null : { x: e.clientX, y: e.clientY, t: e.timeStamp };
    if (!twice) return;
    const fz = fitZoom();
    if (isFit()) setZoom(fz < 1 ? 1 : fz * 2, ...anchor(p, e.clientX, e.clientY));
    else setZoom(null);
  }

  function onWheel(e, p) {
    if (!current) return;
    e.preventDefault();
    // Lines (Firefox) and pages to px; a notch is then ~100 px, a trackpad pinch a few.
    const px = e.deltaY * (e.deltaMode === 1 ? 33 : e.deltaMode === 2 ? 400 : 1);
    const k = Math.exp(-Math.max(-200, Math.min(200, px)) * (e.ctrlKey ? PINCH_RATE : WHEEL_RATE));
    const z = zoomOf();
    // Zooming out stops at fit: past it the stage only gains empty space.
    setZoom(Math.max(Math.min(z, fitZoom()), z * k), ...anchor(p, e.clientX, e.clientY));
  }

  for (const p of [main, side]) {
    p.el.addEventListener('pointerdown', (e) => onDown(e, p));
    p.el.addEventListener('pointermove', (e) => onMove(e, p));
    p.el.addEventListener('pointerup', (e) => onUp(e, p));
    p.el.addEventListener('pointercancel', (e) => onUp(e, p));
    p.el.addEventListener('wheel', (e) => onWheel(e, p), { passive: false });
  }

  // --- the image shown ---------------------------------------------------------------

  async function show(artifact) {
    const changed = current?.id !== artifact.id;
    current = artifact;
    setImage(main.pic, artifact);
    main.pic.alt = artifact.metadata?.prompt || artifact.label;
    setImage(miniPic, artifact);
    sideLabel.textContent = artifact.label;
    sideKind.textContent = SOURCE_WORD[artifact.source] ?? artifact.source;
    prompt.textContent = artifact.source === 'upload' ? 'Your upload.' : artifact.metadata?.prompt || '';
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
    const all = images(c);
    reelCount.textContent = `Images · ${all.length}`;
    for (const a of all) {
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
    bar.addArtifact(current, { first: true, prefill: `Edit @${current.label}: ` });
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

  // --- keys: + − 0 1 zoom, arrows pan (or switch images at fit), C compare, I details --

  const onKey = (e) => {
    if (e.target.closest?.('input, textarea, [contenteditable], .dialog')) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const zoomed = !isFit();
    const sliding = isSlider();
    const arrows = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    if (e.key === ' ' && (e.target === document.body || e.target.closest?.('.stage, .reel'))) {
      e.preventDefault();
      if (!spaceHeld) { spaceHeld = true; paint(); }
    } else if (e.key === '+' || e.key === '=') { e.preventDefault(); zoomStep(1); }
    else if (e.key === '-' || e.key === '_') { e.preventDefault(); zoomStep(-1); }
    else if (e.key === '0') { e.preventDefault(); setZoom(null); }
    else if (e.key === '1') { e.preventDefault(); setZoom(1); }
    else if (e.key === 'c' || e.key === 'C') { e.preventDefault(); toggleCompare(); }
    else if (e.key === 'i' || e.key === 'I') { e.preventDefault(); setSide(!sideOpen); }
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
  // The stage resizes with the window, the side column, and compare's second pane.
  const sizer = new ResizeObserver(() => fit());
  sizer.observe(main.el);
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
      sizer.disconnect();
      unsubscribe.forEach((u) => u());
      bar.destroy();
      strip.destroy();
    },
  };
}
