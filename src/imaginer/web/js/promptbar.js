// The prompt bar (design 1a "chip row"): text, attachments, and quick-option chips that
// open a tray for aspect ratio, image count, seed and image backend. Each option maps to
// MessageCreate.image_options; "auto" leaves the choice to the model.

import { setImage } from './api.js';
import { append, corners, h, icon, clear } from './dom.js';
import { ASPECTS, truncate } from './format.js';
import {
  clearDraft, connection, defaultBackend, draft, imageBackends, imageEngine, on, options, requestOptions,
  setOption,
} from './store.js';

const STATUS_WORD = { ok: 'ready', degraded: 'loading', down: 'down', unavailable: 'off', unknown: '…' };
const SOURCE_WORD = { upload: 'upload', generated: 'generated', edited: 'edited', derived: 'derived' };

/** pending: extra attachments shown but not yet in the draft, e.g. a mask being painted:
 * () => [{label, src, thumb}] with thumb an element. */
export function promptBar({
  draftKey, placeholder, onSubmit, onStop, isBusy = () => false, onFocus, pending = () => [],
}) {
  const d = draft(draftKey);
  let trayOpen = false;
  let sending = false;
  let error = null;

  const fileInput = h('input', {
    type: 'file', accept: 'image/*', multiple: true, hidden: true,
    onchange: () => { addFiles(fileInput.files); fileInput.value = ''; },
  });
  const textarea = h('textarea.pbar-text', {
    rows: 2, placeholder, 'aria-label': 'Prompt',
    oninput: () => { d.text = textarea.value; grow(); renderActions(); },
    onkeydown: (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        submit();
      } else if (e.key === 'Escape' && trayOpen) {
        toggleTray(false);
      }
    },
    onpaste: (e) => {
      const files = [...(e.clipboardData?.files ?? [])].filter((f) => f.type.startsWith('image/'));
      if (files.length) {
        e.preventDefault();
        addFiles(files);
      }
    },
  });
  textarea.value = d.text;

  const atts = h('div.pbar-atts');
  const chips = h('div.pbar-chips');
  const optionsBtn = h('button.pbar-opt-btn', {
    type: 'button', title: 'Quick options', 'aria-label': 'Quick options',
    onclick: () => toggleTray(),
  }, icon('sliders', 17));
  const actions = h('div.pbar-actions');
  const tray = h('div.pbar-tray.blueprint', { hidden: true });
  const errorLine = h('div.pbar-error', { hidden: true });

  const bar = h('div.pbar.blueprint', corners(), atts, textarea, h('div.pbar-row',
    h('button.pbar-attach', {
      type: 'button', title: 'Attach images', 'aria-label': 'Attach images',
      onclick: () => fileInput.click(),
    }, icon('paperclip')),
    chips, optionsBtn, actions,
  ));
  const el = h('div.pbar-wrap', tray, errorLine, bar, fileInput);
  if (onFocus) el.addEventListener('focusin', () => onFocus());

  // --- pieces ---------------------------------------------------------------

  function grow() {
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 220)}px`;
  }

  function renderAtts() {
    clear(atts);
    const extra = pending();
    atts.hidden = !d.attachments.length && !extra.length;
    const after = extra.map((p) => h('div.att.pending', p.thumb,
      h('div.att-text', h('span.att-label', p.label), h('span.att-src', p.src))));
    for (const att of d.attachments) {
      const thumb = h('img.att-thumb', { alt: '' });
      if (att.artifact) setImage(thumb, att.artifact);
      else thumb.src = att.preview;
      const label = att.artifact ? att.artifact.label : truncate(att.file.name || 'image', 18);
      const maskFor = att.artifact?.metadata?.mask_for;
      const src = att.artifact?.kind === 'mask' ? `mask${maskFor ? ` · ${maskFor}` : ''}`
        : att.artifact ? SOURCE_WORD[att.artifact.source] ?? att.artifact.source : 'upload';
      atts.append(h('div.att',
        thumb,
        h('div.att-text', h('span.att-label', label), h('span.att-src', src)),
        h('button.att-x', {
          type: 'button', title: `Remove ${label}`, 'aria-label': `Remove ${label}`,
          onclick: () => removeAttachment(att),
        }, icon('x', 14)),
      ));
    }
    atts.append(...after);
  }

  function shape(ratio, scale = 0.65) {
    const [w, h0] = ratio ? ASPECTS[ratio] : [15, 15];
    return h('span.shape', {
      class: ratio ? '' : 'auto',
      style: { width: `${Math.round(w * scale)}px`, height: `${Math.round(h0 * scale)}px` },
    });
  }

  function engineName() {
    return options.backend ?? defaultBackend('t2i')?.name ?? 'image';
  }

  function renderChips() {
    clear(chips);
    const chip = (...children) => h('button.chip', { type: 'button', onclick: () => toggleTray() }, children);
    // Settings beyond the basics only get a chip when they differ from the default (design 2a).
    const pinned = (...children) => h('button.chip.pinned', { type: 'button', onclick: () => toggleTray() }, children);
    const sampler = [options.steps && `${options.steps} steps`, options.cfg_scale != null && `cfg ${options.cfg_scale}`].filter(Boolean);
    const engine = imageEngine();
    append(chips, [
      options.sizeExact
        ? pinned(icon('ruler', 14), `${options.width}×${options.height}`)
        : chip(shape(options.aspect_ratio), options.aspect_ratio ?? 'auto'),
      chip(icon('copies', 14), options.n ? `×${options.n}` : '×auto'),
      chip(icon('hash', 14), options.seedFixed ? `seed ${options.seed}` : 'seed random'),
      sampler.length > 0 && pinned(icon('steps', 14), sampler.join(' · ')),
      options.negative.trim() && pinned(icon('ban', 14), 'negative'),
      options.transparent && engine?.capabilities.includes('rgba') && pinned(h('span.shape.checker'), 'transparent'),
      maskLabel() && pinned(icon('brush', 14), maskLabel()),
      chip(h('span.dot'), engineName()),
    ].filter(Boolean));
    optionsBtn.classList.toggle('on', trayOpen);
  }

  /** The painted mask attached to this draft, if any: it rides in image_options.mask. */
  function maskLabel() {
    return d.attachments.find((a) => a.artifact?.kind === 'mask')?.artifact.label ?? null;
  }

  function pick(selected, label, onclick, kind = '') {
    return h('button.pick', {
      type: 'button', class: `${kind} ${selected ? 'on' : ''}`, 'aria-pressed': String(selected), onclick,
    }, label);
  }

  /** The size an aspect ratio gets: the backend's default area in that shape. */
  function sizeFor(aspect, defaults) {
    const mult = defaults?.size_multiple ?? 32;
    const [dw, dh] = [defaults?.width ?? 1024, defaults?.height ?? 1024];
    if (!aspect) return [dw, dh];
    const [a, b] = aspect.split(':').map(Number);
    const w = Math.sqrt((dw * dh * a) / b);
    return [Math.round(w / mult) * mult, Math.round((w * b) / a / mult) * mult];
  }

  const roundTo = (value, mult) => Math.min(4096, Math.max(256, Math.round(value / mult) * mult));

  function numberInput({ value, placeholder = '', label, step = 1, min, max, onchange }) {
    const input = h('input.seed-input', {
      type: 'number', step, min, max, value: value == null ? '' : String(value), placeholder, 'aria-label': label,
      onchange: () => onchange(input.value.trim() === '' ? null : Number(input.value), input),
      onkeydown: (e) => { if (e.key === 'Enter') { e.preventDefault(); input.blur(); } },
    });
    return input;
  }

  function renderTray() {
    tray.hidden = !trayOpen;
    if (!trayOpen) return;
    clear(tray).append(...corners());

    const aspects = h('div.tray-opts',
      pick(!options.aspect_ratio && !options.sizeExact, [shape(null, 1), h('span', 'auto')], () => setOption({ aspect_ratio: null, sizeExact: false }), 'aspect'),
      Object.keys(ASPECTS).map((k) =>
        pick(options.aspect_ratio === k && !options.sizeExact, [shape(k, 1), h('span', k)], () => setOption({ aspect_ratio: k, sizeExact: false }), 'aspect')),
    );
    const counts = h('div.tray-opts',
      pick(!options.n, 'auto', () => setOption({ n: null }), 'count wide'),
      [1, 2, 3, 4].map((k) => pick(options.n === k, String(k), () => setOption({ n: k }), 'count')),
    );
    const n = options.n ?? 1;
    const seedInput = h('input.seed-input', {
      type: 'number', min: 0, max: 2147483647, step: 1, value: String(options.seed),
      disabled: !options.seedFixed, 'aria-label': 'Seed',
      onchange: () => {
        const value = Number.parseInt(seedInput.value, 10);
        if (Number.isInteger(value) && value >= 0) setOption({ seed: Math.min(value, 2147483647) });
        else seedInput.value = String(options.seed);
      },
    });
    const engines = imageBackends();
    const engine = imageEngine();
    const defaults = engine?.image_defaults ?? null;
    const mult = defaults?.size_multiple ?? 32;

    // Size: from the aspect ratio (the backend's default area), or exact pixels.
    const [aw, ah] = sizeFor(options.aspect_ratio, defaults);
    const exact = options.sizeExact;
    const linked = options.sizeLinked !== false;
    const sizeInput = (side) => {
      const input = numberInput({
        value: exact ? options[side] : side === 'width' ? aw : ah,
        label: side === 'width' ? 'Width' : 'Height', step: mult, min: 256, max: 4096,
        onchange: (value, field) => {
          if (value == null || !Number.isFinite(value)) { field.value = String(options[side]); return; }
          const v = roundTo(value, mult);
          const other = side === 'width' ? 'height' : 'width';
          const patch = { [side]: v };
          if (linked) patch[other] = roundTo((v * options[other]) / options[side], mult);
          setOption(patch);
        },
      });
      input.disabled = !exact;
      return input;
    };
    const mp = ((exact ? options.width * options.height : aw * ah) / 1e6).toFixed(2);
    const sizeCell = h('div.tray-cell.span-2',
      h('div.cell-head', h('span.eyebrow', 'Size'), h('span.spacer'),
        h('div.tray-seg.small',
          pick(!exact, 'From aspect', () => setOption({ sizeExact: false })),
          pick(exact, 'Exact', () => setOption(exact ? {} : { sizeExact: true, width: aw, height: ah })))),
      h('div.size-row',
        h('label.seed-field', { class: exact ? '' : 'off' }, h('span.mono-meta', 'W'), sizeInput('width')),
        h('button.pick.link', {
          type: 'button', class: linked ? 'on' : '', 'aria-pressed': String(linked), disabled: !exact,
          title: 'Keep the ratio', 'aria-label': 'Keep the ratio',
          onclick: () => setOption({ sizeLinked: !linked }),
        }, icon('link', 14)),
        h('label.seed-field', { class: exact ? '' : 'off' }, h('span.mono-meta', 'H'), sizeInput('height'))),
      h('span.tray-hint', exact
        ? `rounded to ${mult} · ${mp} MP`
        : `${aw}×${ah} · ${mp} MP · ${options.aspect_ratio ? 'default area' : `${engine?.name ?? 'backend'} default`}`));

    const stepChoices = [...new Set([8, 20, 40, options.steps].filter(Boolean))].sort((a, b) => a - b);
    const stepsCell = h('div.tray-cell', h('span.eyebrow', 'Steps'),
      h('div.tray-opts',
        pick(!options.steps, 'auto', () => setOption({ steps: null }), 'count wide'),
        stepChoices.map((k) => pick(options.steps === k, String(k), () => setOption({ steps: k }), 'count'))),
      h('span.tray-hint', defaults?.steps ? `${engine.name} default ${defaults.steps}` : 'engine default'));

    const cfgDefaults = engines.filter((b) => b.image_defaults?.cfg_scale != null)
      .map((b) => `${b.name} ${b.image_defaults.cfg_scale.toFixed(1)}`);
    const cfgInput = numberInput({
      value: options.cfg_scale, placeholder: defaults?.cfg_scale != null ? defaults.cfg_scale.toFixed(1) : 'auto',
      label: 'CFG scale', step: 0.5, min: 0, max: 30,
      onchange: (value) => setOption({
        cfg_scale: value == null || !Number.isFinite(value) ? null : Math.min(30, Math.max(0, value)),
      }),
    });
    const cfgCell = h('div.tray-cell', h('span.eyebrow', 'CFG'),
      h('label.seed-field', icon('gauge', 14), cfgInput,
        options.cfg_scale == null
          ? h('span.mono-meta', 'default')
          : h('button.att-x', {
            type: 'button', title: 'Back to the default', 'aria-label': 'Reset CFG',
            onclick: () => setOption({ cfg_scale: null }),
          }, icon('x', 12))),
      h('span.tray-hint', cfgDefaults.join(' · ') || 'engine default'));

    const negInput = h('input.seed-input.text', {
      type: 'text', value: options.negative, placeholder: 'blurry, text, watermark', maxLength: 2000,
      'aria-label': 'Negative prompt',
      oninput: () => setOption({ negative: negInput.value }, { quiet: true }),
      onchange: () => setOption({ negative: negInput.value }),
      onkeydown: (e) => { if (e.key === 'Enter') { e.preventDefault(); negInput.blur(); } },
    });
    const negCell = h('div.tray-cell.span-3',
      h('div.cell-head', h('span.eyebrow', 'Negative prompt'), h('span.spacer'),
        h('span.tray-hint', 'sent with every image call this turn')),
      h('label.seed-field', icon('ban', 14), negInput));

    const rgba = !!engine?.capabilities.includes('rgba');
    const alphaOn = !!options.transparent && rgba;
    const alphaCell = h('div.tray-cell', h('span.eyebrow', 'Transparent'),
      h('div.tray-seg',
        pick(!alphaOn, 'Off', () => setOption({ transparent: false })),
        h('button.pick', {
          type: 'button', class: alphaOn ? 'on' : '', 'aria-pressed': String(alphaOn),
          disabled: !rgba, onclick: () => setOption({ transparent: true }),
        }, 'On')),
      h('span.tray-hint', rgba
        ? `rgba ✓${engine.type === 'sdcpp' ? ' · prompt-driven on sd-server' : ''}`
        : `${engine?.name ?? 'this backend'} has no rgba`));

    tray.append(
      h('div.tray-cell', h('span.eyebrow', 'Aspect'), aspects),
      h('div.tray-cell', h('span.eyebrow', 'Count'), counts,
        h('span.tray-hint', options.seedFixed && n > 1
          ? `seeds ${options.seed} – ${options.seed + n - 1}`
          : options.seedFixed ? `seed ${options.seed}` : 'random, then +1 each')),
      h('div.tray-cell', h('span.eyebrow', 'Seed'),
        h('div.tray-seg',
          pick(!options.seedFixed, 'Random', () => setOption({ seedFixed: false })),
          pick(options.seedFixed, 'Fixed', () => setOption({ seedFixed: true }))),
        h('label.seed-field', { class: options.seedFixed ? '' : 'off' }, icon('hash', 14), seedInput)),
      h('div.tray-cell', h('span.eyebrow', 'Image backend'),
        engines.length
          ? engines.map((b) => {
            const selected = engineName() === b.name;
            const isDefault = b.default_for.includes('t2i');
            return h('button.engine', {
              type: 'button', class: selected ? 'on' : '', 'aria-pressed': String(selected),
              disabled: b.status === 'unavailable',
              onclick: () => setOption({ backend: isDefault ? null : b.name }),
            },
            h('span.ring', { class: b.status === 'ok' ? 'ok' : '' }),
            h('span.engine-text',
              h('span.engine-name', b.name, isDefault && h('span.engine-default', 'default')),
              h('span.engine-meta', `${b.model ?? b.type} · ${STATUS_WORD[b.status] ?? b.status}`)));
          })
          : h('span.tray-hint', 'No image backend is configured.')),
      sizeCell, stepsCell, cfgCell,
      negCell, alphaCell,
    );
  }

  function renderActions() {
    clear(actions);
    // Design 4b: once reconnecting gives up, the draft stays and Send waits for the server.
    const down = connection().status === 'offline';
    if (isBusy() && !down) {
      actions.append(h('button.send.stop.blueprint', {
        type: 'button', title: 'Stop turn', onclick: () => onStop?.(),
      }, corners(), h('span.stop-square'), 'Stop'));
      return;
    }
    const empty = !textarea.value.trim() && !d.attachments.length;
    actions.append(h('button.send.blueprint', {
      type: 'button', title: 'Send', 'aria-label': 'Send', disabled: empty || sending || down,
      onclick: submit,
    }, corners(), icon('arrowUp')));
  }

  let offlineDismissed = false;

  function renderError() {
    const down = connection().status === 'offline';
    if (!down) offlineDismissed = false;
    const text = error ?? (down && !offlineDismissed
      ? 'Offline. Your draft is kept, and Send works again when the connection is back.' : null);
    errorLine.hidden = !text;
    if (!text) return;
    clear(errorLine).append(
      h('span', text),
      h('button.att-x', {
        type: 'button', title: 'Dismiss', 'aria-label': 'Dismiss',
        onclick: () => { if (error) error = null; else offlineDismissed = true; renderError(); },
      }, icon('x', 13)),
    );
  }

  function render() {
    renderAtts();
    renderChips();
    renderTray();
    renderActions();
    renderError();
  }

  // --- behaviour -------------------------------------------------------------

  function toggleTray(open = !trayOpen) {
    trayOpen = open;
    renderChips();
    renderTray();
  }

  function addFiles(files) {
    for (const file of files) {
      if (!file.type.startsWith('image/')) continue;
      d.attachments.push({ file, preview: URL.createObjectURL(file) });
    }
    renderAtts();
    renderActions();
    textarea.focus();
  }

  function addArtifact(artifact, { first = false, prefill = null } = {}) {
    if (!d.attachments.some((a) => a.artifact?.id === artifact.id)) {
      const att = { artifact };
      if (first) d.attachments.unshift(att);
      else d.attachments.push(att);
    }
    if (prefill && !textarea.value.trim()) {
      textarea.value = d.text = prefill;
      grow();
    }
    renderAtts();
    renderChips();
    renderActions();
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);
  }

  function removeAttachment(att) {
    d.attachments.splice(d.attachments.indexOf(att), 1);
    if (att.preview) URL.revokeObjectURL(att.preview);
    renderAtts();
    renderChips();
    renderActions();
  }

  async function submit() {
    const text = textarea.value.trim();
    if (sending || isBusy() || connection().status === 'offline' || (!text && !d.attachments.length)) return;
    sending = true;
    error = null;
    renderActions();
    renderError();
    try {
      const opts = requestOptions();
      if (maskLabel()) opts.mask = maskLabel();
      await onSubmit({ text, attachments: [...d.attachments], options: opts });
      clearDraft(draftKey);
      textarea.value = '';
      grow();
      trayOpen = false;
    } catch (err) {
      error = describe(err);
    } finally {
      sending = false;
      if (el.isConnected) render();
    }
  }

  const unsubscribe = [on('options', render), on('backends', render), on('conn', () => { renderActions(); renderError(); })];
  render();
  setTimeout(grow);

  return {
    el,
    refresh: renderActions,
    refreshAtts: () => { renderAtts(); renderChips(); },
    setPlaceholder: (text) => { textarea.placeholder = text; },
    focus: () => textarea.focus(),
    addFiles,
    addArtifact,
    destroy: () => unsubscribe.forEach((u) => u()),
  };
}

function describe(err) {
  if (err.status === 409) return 'A turn is already running here. Stop it or wait for it to finish.';
  if (err.status === 401) return 'The server needs an access token.';
  return err.message || 'Something went wrong.';
}
