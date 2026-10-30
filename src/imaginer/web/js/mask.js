// The mask editor (design 3a): paint the area of an image that should change, then attach
// the mask. It is uploaded as a "mask" artifact (mask-N) at the image's own size and goes
// into the bar beside its image; the message then carries it as image_options.mask, so
// edit_image uses it without the model having to name it. White changes, black is kept.
//
// Keys: B brush, E eraser, [ ] brush size, X invert, Ctrl+Z undo.

import { api, setImage } from './api.js';
import { connectionStrip, toast } from './chrome.js';
import { corners, h, icon, iconButton, clear } from './dom.js';
import { convHash } from './origin.js';
import { promptBar } from './promptbar.js';
import { byLabel, draft, ensureLoaded, running, send, stop } from './store.js';

const WORK_MAX = 1024; // the painting canvas's longer side; the upload is scaled to the image
const HISTORY = 20;
const SIZE_MIN = 6;
const SIZE_MAX = 90;

export function maskScreen({ id, label, navigate }) {
  let c = null;
  let image = null; // the artifact being masked
  let tool = 'brush';
  let size = 34; // brush diameter, screen px
  let viewMode = 'overlay'; // overlay | mask | image
  let drawing = null; // last point while a stroke is down
  let uploaded = null; // the mask artifact, once attached
  let dirty = false;
  const history = [];
  const chat = convHash(id, 'chat');
  const accent = getComputedStyle(document.documentElement).getPropertyValue('--color-accent').trim() || '#5980a6';

  // --- header ----------------------------------------------------------------------

  const title = h('span.topbar-title', `Mask for ${label}`);
  const attachBtn = h('button.btn.btn-primary.blueprint.attach-mask', {
    type: 'button', disabled: true, onclick: () => attach(),
  }, corners(), icon('check', 15), 'Attach mask');
  const header = h('header.topbar.sub',
    iconButton('back', 'Back', () => leave()),
    h('div.topbar-heading', title, h('span.topbar-meta', 'paint what should change · the rest is kept')),
    h('span.spacer'),
    h('button.btn-line', { type: 'button', onclick: () => leave() }, 'Cancel'),
    attachBtn);

  // --- tools -------------------------------------------------------------------------

  const toolBtn = (name, iconName, title, key) => {
    const b = iconButton(iconName, `${title} (${key})`, () => setTool(name));
    b.setAttribute('aria-pressed', 'false');
    return b;
  };
  const brushBtn = toolBtn('brush', 'brush', 'Brush', 'B');
  const eraseBtn = toolBtn('erase', 'eraser', 'Eraser', 'E');
  const undoBtn = iconButton('undo', 'Undo (Ctrl+Z)', () => undo());
  const tools = h('div.mask-tools.blueprint', { role: 'toolbar', 'aria-label': 'Mask tools' },
    corners(),
    brushBtn, eraseBtn,
    h('span.mask-tools-sep'),
    undoBtn,
    iconButton('invert', 'Invert (X)', () => invert()),
    iconButton('trash', 'Clear', () => clearMask()));

  const sizeOut = h('span.mono-meta.mask-size-out');
  const sizeRange = h('input.mask-size', {
    type: 'range', min: SIZE_MIN, max: SIZE_MAX, value: String(size), 'aria-label': 'Brush size',
    oninput: () => setSize(Number(sizeRange.value)),
  });
  const viewSeg = h('div.tray-seg.mask-views', { role: 'group', 'aria-label': 'View' },
    ['overlay', 'mask', 'image'].map((v) => h('button.pick', {
      type: 'button', dataset: { view: v }, onclick: () => setView(v),
    }, v === 'overlay' ? 'Overlay' : v === 'mask' ? 'Mask' : 'Image')));
  const controls = h('div.mask-controls',
    h('span.eyebrow', 'Size'), sizeRange, sizeOut, h('span.spacer'), h('span.eyebrow', 'View'), viewSeg);

  // --- stage -------------------------------------------------------------------------

  const pic = h('img.stage-pic.mask-pic', { alt: '' });
  const bw = h('canvas.mask-canvas');
  const overlay = h('canvas.mask-canvas');
  const cursor = h('div.mask-cursor', { hidden: true });
  const toolChip = h('span.kind-chip');
  const surface = h('div.mask-surface');
  const stage = h('div.stage.blueprint.mask-stage',
    corners(), pic, bw, overlay, surface, cursor,
    h('div.stage-chips', h('span.label-chip', label), toolChip));
  const center = h('div.mask-center', controls, stage);

  // --- preview -----------------------------------------------------------------------

  const thumb = h('canvas.mask-thumb');
  const coverageOut = h('span.mono-meta');
  const maskName = h('span.mono-meta.accent', 'new mask');
  const sentAs = h('span.tool-args');
  const side = h('aside.mask-side',
    h('span.eyebrow', 'Mask preview'),
    h('div.mask-thumb-frame.blueprint', corners(), thumb),
    h('div.mask-meta', maskName, coverageOut),
    h('span.eyebrow', 'Sent as'),
    h('div.tool-row.wrap', h('span.tool-name', 'edit_image'), sentAs),
    h('p.rail-hint', 'White changes and black is kept. sd-server gets the mask as an extra reference with a mask instruction, so check how well the unmasked areas hold.'));

  const bodyEl = h('div.mask-body', tools, center, side);

  // --- the bar: the image, the mask (pending until attached), and the instruction --------

  const pendingThumb = h('canvas.att-thumb', { width: 72, height: 72 });
  const bar = promptBar({
    draftKey: id,
    placeholder: `Edit ${label}: what should change in the painted area…`,
    isBusy: () => running(id),
    pending: () => (uploaded ? [] : [{ label: 'new mask', src: 'mask · pending', thumb: pendingThumb }]),
    onSubmit: async (msg) => {
      const mask = await upload();
      // One mask per message: this one replaces any attached earlier.
      const attachments = [...msg.attachments.filter((a) => a.artifact?.kind !== 'mask'), { artifact: mask }];
      await send(id, { ...msg, attachments, options: { ...msg.options, mask: mask.label } });
      navigate(chat);
    },
    onStop: () => stop(id),
  });
  const strip = connectionStrip({ turnRunning: () => running(id) });
  const dock = h('div.dock.mask-dock', bar.el);
  const el = h('div.screen.mask-editor', header, strip.el, bodyEl, dock);

  // --- canvases ------------------------------------------------------------------------

  const work = document.createElement('canvas'); // white strokes on transparent, WORK size
  const ctx = work.getContext('2d', { willReadFrequently: true });
  let view = { w: 600, h: 445 }; // stage size, screen px

  function layout() {
    if (!image) return;
    const narrow = window.innerWidth < 720;
    const maxW = narrow ? window.innerWidth - 20 : Math.min(760, window.innerWidth - 32 - 64 - 320 - 80);
    const maxH = narrow ? window.innerHeight * 0.45 : window.innerHeight - 56 - 22 - 42 - 230;
    const scale = Math.min(Math.max(240, maxW) / image.width, Math.max(200, maxH) / image.height);
    view = { w: Math.round(image.width * scale), h: Math.round(image.height * scale) };
    stage.style.width = `${view.w}px`;
    stage.style.height = `${view.h}px`;
    center.style.width = `${view.w}px`;
    // The bar sits under the tools and the stage, clear of the preview column.
    dock.style.width = narrow ? '' : `${Math.max(480, Math.min(648, 40 + 8 + 28 + view.w))}px`;
    for (const cv of [bw, overlay]) {
      cv.width = view.w;
      cv.height = view.h;
    }
    paint();
  }

  /** Redraw the overlay (accent at 60%), the black-and-white view and the thumbnails. */
  function paint() {
    const o = overlay.getContext('2d');
    o.clearRect(0, 0, overlay.width, overlay.height);
    o.globalCompositeOperation = 'source-over';
    o.drawImage(work, 0, 0, overlay.width, overlay.height);
    o.globalCompositeOperation = 'source-in';
    o.globalAlpha = 0.6;
    o.fillStyle = accent;
    o.fillRect(0, 0, overlay.width, overlay.height);
    o.globalAlpha = 1;
    o.globalCompositeOperation = 'source-over';
    for (const cv of [bw, thumb, pendingThumb]) {
      const g = cv.getContext('2d');
      g.fillStyle = '#000';
      g.fillRect(0, 0, cv.width, cv.height);
      if (cv === pendingThumb) {
        const k = Math.min(cv.width / work.width, cv.height / work.height);
        const [w, hh] = [work.width * k, work.height * k];
        g.drawImage(work, (cv.width - w) / 2, (cv.height - hh) / 2, w, hh);
      } else g.drawImage(work, 0, 0, cv.width, cv.height);
    }
  }

  function measure() {
    const data = ctx.getImageData(0, 0, work.width, work.height).data;
    let on = 0;
    let total = 0;
    for (let i = 3; i < data.length; i += 16) { // every 4th pixel is plenty
      total += 1;
      if (data[i] > 127) on += 1;
    }
    const pct = total ? Math.round((on / total) * 100) : 0;
    coverageOut.textContent = `${pct}% of image · ${image.width}×${image.height}`;
    attachBtn.disabled = pct === 0 && !uploaded;
    return pct;
  }

  function snapshot() {
    history.push(ctx.getImageData(0, 0, work.width, work.height));
    if (history.length > HISTORY) history.shift();
    undoBtn.disabled = false;
  }

  const toWork = (e) => {
    const r = surface.getBoundingClientRect();
    const k = work.width / r.width;
    return { x: (e.clientX - r.left) * k, y: (e.clientY - r.top) * k, k };
  };

  function stroke(a, b) {
    ctx.globalCompositeOperation = tool === 'erase' ? 'destination-out' : 'source-over';
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = size * a.k;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x + 0.01, b.y);
    ctx.stroke();
    ctx.globalCompositeOperation = 'source-over';
    changed();
  }

  function changed() {
    dirty = true;
    if (uploaded) {
      // Painting after attaching makes a new mask on the next attach or send.
      uploaded = null;
      maskName.textContent = 'new mask';
      renderSentAs();
      bar.refreshAtts();
    }
    paint();
  }

  surface.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    surface.setPointerCapture?.(e.pointerId);
    snapshot();
    drawing = toWork(e);
    stroke(drawing, drawing);
    e.preventDefault();
  });
  surface.addEventListener('pointermove', (e) => {
    const r = surface.getBoundingClientRect();
    cursor.hidden = false;
    cursor.style.left = `${e.clientX - r.left}px`;
    cursor.style.top = `${e.clientY - r.top}px`;
    if (!drawing) return;
    const p = toWork(e);
    stroke(drawing, p);
    drawing = p;
  });
  const endStroke = () => {
    if (!drawing) return;
    drawing = null;
    measure();
  };
  surface.addEventListener('pointerup', endStroke);
  surface.addEventListener('pointercancel', endStroke);
  surface.addEventListener('pointerleave', () => { cursor.hidden = true; });

  function undo() {
    const last = history.pop();
    if (!last) return;
    ctx.putImageData(last, 0, 0);
    undoBtn.disabled = !history.length;
    changed();
    measure();
  }

  function invert() {
    snapshot();
    const t = document.createElement('canvas');
    t.width = work.width;
    t.height = work.height;
    const g = t.getContext('2d');
    g.fillStyle = '#fff';
    g.fillRect(0, 0, t.width, t.height);
    g.globalCompositeOperation = 'destination-out';
    g.drawImage(work, 0, 0);
    ctx.clearRect(0, 0, work.width, work.height);
    ctx.drawImage(t, 0, 0);
    changed();
    measure();
  }

  function clearMask() {
    snapshot();
    ctx.clearRect(0, 0, work.width, work.height);
    changed();
    measure();
  }

  function setTool(name) {
    tool = name;
    for (const [b, n] of [[brushBtn, 'brush'], [eraseBtn, 'erase']]) {
      b.classList.toggle('on', n === tool);
      b.setAttribute('aria-pressed', String(n === tool));
    }
    toolChip.textContent = tool === 'brush' ? 'brush · paint to change' : 'eraser · restore';
  }

  function setSize(value) {
    size = Math.max(SIZE_MIN, Math.min(SIZE_MAX, Math.round(value)));
    sizeRange.value = String(size);
    sizeOut.textContent = `${size} px`;
    cursor.style.width = `${size}px`;
    cursor.style.height = `${size}px`;
  }

  function setView(v) {
    viewMode = v;
    for (const b of viewSeg.children) {
      const on = b.dataset.view === v;
      b.classList.toggle('on', on);
      b.setAttribute('aria-pressed', String(on));
    }
    overlay.hidden = v !== 'overlay';
    bw.hidden = v !== 'mask';
    pic.style.opacity = v === 'mask' ? '0' : '1';
  }

  function renderSentAs() {
    sentAs.textContent = `images=["${label}"] mask="${uploaded?.label ?? 'mask-?'}"`;
  }

  // --- upload and leave ------------------------------------------------------------------

  /** The mask as a PNG at the image's size: white where painted, black elsewhere. */
  async function upload() {
    if (uploaded) return uploaded;
    const out = document.createElement('canvas');
    out.width = image.width;
    out.height = image.height;
    const g = out.getContext('2d');
    g.fillStyle = '#000';
    g.fillRect(0, 0, out.width, out.height);
    g.drawImage(work, 0, 0, out.width, out.height);
    const blob = await new Promise((resolve) => out.toBlob(resolve, 'image/png'));
    const file = new File([blob], `${label}-mask.png`, { type: 'image/png' });
    const artifact = await api.upload(id, file, { kind: 'mask', maskFor: label });
    c.artifacts.set(artifact.id, artifact);
    uploaded = artifact;
    dirty = false;
    maskName.textContent = artifact.label;
    renderSentAs();
    bar.refreshAtts();
    return artifact;
  }

  async function attach() {
    attachBtn.disabled = true;
    try {
      const mask = await upload();
      const d = draft(id);
      const stale = d.attachments.filter((a) => a.artifact?.kind === 'mask');
      for (const a of stale) d.attachments.splice(d.attachments.indexOf(a), 1);
      bar.addArtifact(image, { first: true, prefill: `Edit ${label}: ` });
      bar.addArtifact(mask);
      navigate(chat, { focusBar: true });
    } catch (err) {
      toast(`Couldn't attach the mask: ${err.message}`, { error: true });
      attachBtn.disabled = false;
    }
  }

  function leave() {
    navigate(convHash(id, label));
  }

  // --- keys --------------------------------------------------------------------------------

  const onKey = (e) => {
    if (e.target.closest?.('input:not([type=range]), textarea, .dialog')) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); return; }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (k === 'b') setTool('brush');
    else if (k === 'e') setTool('erase');
    else if (k === 'x') invert();
    else if (e.key === '[') setSize(size - 4);
    else if (e.key === ']') setSize(size + 4);
    else return;
    e.preventDefault();
  };
  document.addEventListener('keydown', onKey);
  window.addEventListener('resize', layout);
  const beforeUnload = (e) => { if (dirty) e.preventDefault(); };
  window.addEventListener('beforeunload', beforeUnload);

  setTool('brush');
  setSize(size);
  setView(viewMode);
  undoBtn.disabled = true;
  renderSentAs();

  ensureLoaded(id).then((loaded) => {
    c = loaded;
    image = byLabel(c, label);
    if (!image || image.kind !== 'image') {
      clear(bodyEl).append(h('div.notice', `There is no image ${label} in this conversation.`));
      return;
    }
    setImage(pic, image);
    const k = Math.min(1, WORK_MAX / Math.max(image.width, image.height));
    work.width = Math.max(1, Math.round(image.width * k));
    work.height = Math.max(1, Math.round(image.height * k));
    thumb.width = 300;
    thumb.height = Math.max(1, Math.round((300 * image.height) / image.width));
    layout();
    measure();
    // The image goes into the bar first, with the instruction started.
    bar.addArtifact(image, { first: true, prefill: `Edit ${label}: ` });
    // Ready to paint: the B/E/X keys shouldn't type into the bar.
    document.activeElement?.blur();
  }, (err) => {
    clear(bodyEl).append(h('div.notice', err.status === 404 ? 'This conversation no longer exists.' : err.message));
  });

  return {
    el,
    bar,
    destroy() {
      document.removeEventListener('keydown', onKey);
      window.removeEventListener('resize', layout);
      window.removeEventListener('beforeunload', beforeUnload);
      bar.destroy();
      strip.destroy();
    },
  };
}
