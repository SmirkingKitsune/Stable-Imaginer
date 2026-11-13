// The prompt bar (design 1a "chip row"): text, attachments, and quick-option chips that
// open a tray for the prompt mode, the quality check, aspect ratio, image count, seed,
// image backend, its presets and its LoRAs. The image options map to
// MessageCreate.image_options ("auto" leaves the choice to the model); mode and qa go with
// the message itself. Typing "@" picks one of the attached images, or one of the image
// backend's LoRAs for this message only, and puts it in the text as a chip (see mentions.js).

import { setImage } from './api.js';
import { toast } from './chrome.js';
import { append, corners, h, icon, clear } from './dom.js';
import { ASPECTS, loraName, lorasWord } from './format.js';
import { loraItems, loraText, mentionEditor, uploadRef } from './mentions.js';
import {
  clearDraft, connection, defaultBackend, draft, editInfo, imageBackends, imageEngine, imageLoras, imagePreset, libraryLora,
  loadLoras, loraLibrary, on, options, requestOptions, setOption, takeQueuedLoras,
} from './store.js';

const MAX_LORAS = 8; // ImageRequestOptions.loras
const LORA_FILTER_FROM = 12; // listed LoRAs before the cell gets a filter field

const STATUS_WORD = { ok: 'ready', degraded: 'loading', down: 'down', unavailable: 'off', unknown: '…' };
const SOURCE_WORD = { upload: 'upload', generated: 'generated', edited: 'edited', derived: 'derived' };
/** How the text reaches the image model (MessageCreate.mode). */
const MODES = {
  chat: { label: 'Chat', icon: 'chat', hint: 'Llama replies, and draws when you ask for an image' },
  raw: {
    label: 'Raw', icon: 'terminal', placeholder: 'Image prompt, sent to the image model as written…',
    hint: 'no Llama: your text is the prompt; attached images are edited',
  },
  interpret: {
    label: 'Interpret', icon: 'arrowRight', placeholder: 'Describe the image…',
    hint: 'Llama turns your words into a faithful prompt and draws right away',
  },
  curate: {
    label: 'Curate', icon: 'pencil', placeholder: 'An idea to build on…',
    hint: 'Llama enriches your prompt, or asks first when it is too unclear',
  },
};

/** pending: extra attachments shown but not yet in the draft, e.g. a mask being painted:
 * () => [{label, src, thumb}] with thumb an element. compact: the viewer's one-line reply
 * field (design 2a), the draft's text alone; it opens the conversation on focus, so it
 * carries none of the bar's controls. */
export function promptBar({
  draftKey, placeholder, onSubmit, onStop, isBusy = () => false, onFocus, pending = () => [], compact = false,
}) {
  const d = draft(draftKey);
  let basePlaceholder = placeholder;
  let trayOpen = false;
  let sending = false;
  let error = null;

  const fileInput = h('input', {
    type: 'file', accept: 'image/*', multiple: true, hidden: true,
    onchange: () => { addFiles(fileInput.files); fileInput.value = ''; },
  });
  // The draft keeps the text plain, each image reference (a chip in the editor) as "@label".
  const editor = mentionEditor({
    images: () => d.attachments.map((att) => ({ ref: refOf(att), source: sourceOf(att), fill: (img) => fillThumb(img, att) })),
    loras: () => (compact ? [] : engineLoras(imageEngine())),
    className: 'pbar-text', label: 'Prompt',
    onInput: (text) => { d.text = text; renderActions(); },
    onKey: (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        submit();
      } else if (e.key === 'Escape' && trayOpen) {
        toggleTray(false);
      }
    },
    onFiles: addFiles,
  });

  const atts = compact ? h('span.mono-meta.pbar-count') : h('div.pbar-atts');
  const chips = h('div.pbar-chips');
  const optionsBtn = h('button.pbar-opt-btn', {
    type: 'button', title: 'Quick options', 'aria-label': 'Quick options',
    onclick: () => toggleTray(),
  }, icon('sliders', 17));
  const actions = h('div.pbar-actions');
  const tray = h('div.pbar-tray.blueprint', { hidden: true });
  const errorLine = h('div.pbar-error', { hidden: true });

  const bar = compact
    ? h('div.pbar.blueprint.compact', corners(), h('div.pbar-row', icon('chat', 16), editor.el, atts, actions))
    : h('div.pbar.blueprint', corners(), atts, editor.el, h('div.pbar-row',
      h('button.pbar-attach', {
        type: 'button', title: 'Attach images', 'aria-label': 'Attach images',
        onclick: () => fileInput.click(),
      }, icon('paperclip')),
      chips, optionsBtn, actions,
    ));
  const el = h('div.pbar-wrap', !compact && [tray, errorLine], bar, fileInput);
  if (onFocus) el.addEventListener('focusin', () => onFocus());

  // --- pieces ---------------------------------------------------------------

  /** What an attachment is called in the text: its label, or for an upload a name from
   * the file until the server labels it (store.send rewrites the references then). */
  const refOf = (att) => att.artifact?.label ?? att.ref;

  function sourceOf(att) {
    const maskFor = att.artifact?.metadata?.mask_for;
    return att.artifact?.kind === 'mask' ? `mask${maskFor ? ` · ${maskFor}` : ''}`
      : att.artifact ? SOURCE_WORD[att.artifact.source] ?? att.artifact.source : 'upload';
  }

  function fillThumb(img, att) {
    if (att.artifact) setImage(img, att.artifact);
    else img.src = att.preview;
  }

  function renderAtts() {
    if (compact) {
      const n = d.attachments.length + pending().length;
      atts.hidden = !n;
      atts.textContent = `${n} image${n === 1 ? '' : 's'}`;
      editor.sync();
      return;
    }
    clear(atts);
    const extra = pending();
    atts.hidden = !d.attachments.length && !extra.length;
    const after = extra.map((p) => h('div.att.pending', p.thumb,
      h('div.att-text', h('span.att-label', p.label), h('span.att-src', p.src))));
    for (const att of d.attachments) {
      const thumb = h('img.att-thumb', { alt: '' });
      fillThumb(thumb, att);
      const label = refOf(att);
      atts.append(h('div.att',
        thumb,
        h('div.att-text', h('span.att-label', label), h('span.att-src', sourceOf(att))),
        h('button.att-x', {
          type: 'button', title: `Remove ${label}`, 'aria-label': `Remove ${label}`,
          onclick: () => removeAttachment(att),
        }, icon('x', 14)),
      ));
    }
    atts.append(...after);
    editor.sync();
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

  const mode = () => MODES[options.mode] ?? MODES.chat;

  function renderPlaceholder() {
    editor.setPlaceholder(mode().placeholder ?? basePlaceholder);
  }

  function renderChips() {
    clear(chips);
    const chip = (...children) => h('button.chip', { type: 'button', onclick: () => toggleTray() }, children);
    // Settings beyond the basics only get a chip when they differ from the default (design 2a).
    const pinned = (...children) => h('button.chip.pinned', { type: 'button', onclick: () => toggleTray() }, children);
    const sampler = [options.steps && `${options.steps} steps`, options.cfg_scale != null && `cfg ${options.cfg_scale}`].filter(Boolean);
    const engine = imageEngine();
    const preset = imagePreset();
    const loras = imageLoras();
    append(chips, [
      chip(icon(mode().icon, 14), mode().label.toLowerCase()),
      chip(icon(options.qa ? 'check' : 'ban', 14), options.qa ? 'QA' : 'QA off'),
      options.sizeExact
        ? pinned(icon('ruler', 14), `${options.width}×${options.height}`)
        : chip(shape(options.aspect_ratio), options.aspect_ratio ?? 'auto'),
      chip(icon('copies', 14), options.n ? `×${options.n}` : '×auto'),
      chip(icon('hash', 14), options.seedFixed ? `seed ${options.seed}` : 'seed random'),
      preset && pinned(icon('layers', 14), preset.name),
      loras.length > 0 && pinned(icon('puzzle', 14), h('span.chip-text', lorasWord(loras))),
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
    const [dw, dh] = [defaults?.width ?? 2048, defaults?.height ?? 2048];
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

    // Presets: named LoRAs + sampler settings from the backend's config. An engine with an
    // edit card has that card's too, for edits.
    const presets = engine?.image_presets ?? [];
    const edit = editInfo();
    const editPresets = edit?.card ? edit.image_presets.filter((p) => !presets.some((q) => q.name === p.name)) : [];
    const preset = imagePreset();
    const presetCell = presets.length + editPresets.length > 0 && h('div.tray-cell.span-all', h('span.eyebrow', 'Preset'),
      h('div.tray-seg.wrap',
        pick(!preset, 'None', () => setOption({ preset: null })),
        presets.map((p) => pick(preset?.name === p.name, p.name, () => setOption({ preset: p.name }))),
        editPresets.map((p) => pick(preset?.name === p.name, `${p.name} · edits`, () => setOption({ preset: p.name })))),
      h('span.tray-hint', preset
        ? `${presetSummary(preset)}${preset.edits ? ` · for edits with ${preset.edits}` : ''}`
          // A masked edit runs the model card when the edit card takes no mask: not with this.
          + (preset.edits && maskLabel() && edit?.mask_card_name ? ` · not this masked edit (it runs ${edit.mask_card_name})` : '')
        : `${engine?.name ?? 'backend'} defaults`));
    const loraCell = engine?.available_loras?.length > 0 && lorasCell(engine, preset);

    const stepChoices = [...new Set([8, 20, 40, options.steps].filter(Boolean))].sort((a, b) => a - b);
    const stepsCell = h('div.tray-cell', h('span.eyebrow', 'Steps'),
      h('div.tray-opts',
        pick(!options.steps, 'auto', () => setOption({ steps: null }), 'count wide'),
        stepChoices.map((k) => pick(options.steps === k, String(k), () => setOption({ steps: k }), 'count'))),
      h('span.tray-hint', preset?.steps ? `${preset.name} preset ${preset.steps}`
        : defaults?.steps ? `${engine.name} default ${defaults.steps}` : 'engine default'));

    const cfgDefaults = preset?.cfg_scale != null ? [`${preset.name} preset ${preset.cfg_scale.toFixed(1)}`]
      : engines.filter((b) => b.image_defaults?.cfg_scale != null)
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

    const modeCell = h('div.tray-cell.span-3', h('span.eyebrow', 'Mode'),
      h('div.tray-seg', Object.entries(MODES).map(([key, m]) =>
        pick(options.mode === key, m.label, () => setOption({ mode: key })))),
      h('span.tray-hint', mode().hint));
    const qaCell = h('div.tray-cell', h('span.eyebrow', 'Quality check'),
      h('div.tray-seg',
        pick(!options.qa, 'Off', () => setOption({ qa: false })),
        pick(options.qa, 'On', () => setOption({ qa: true }))),
      h('span.tray-hint', options.qa
        ? 'Llama looks at each image and redoes flawed ones'
        : 'images are shown as the engine makes them'));

    tray.append(
      modeCell, qaCell,
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
      ...(presetCell ? [presetCell] : []),
      ...(loraCell ? [loraCell] : []),
      sizeCell, stepsCell, cfgCell,
      negCell, alphaCell,
    );
  }

  /** The backend's LoRAs (as it lists them) to add to this message's images, each with a
   * weight. Picks for LoRAs this backend doesn't list are kept for one that does. */
  function lorasCell(engine, preset) {
    const listed = engine.available_loras;
    const items = engineLoras(engine);
    const byPath = new Map(items.map((i) => [i.path, i]));
    const picks = imageLoras();
    const picked = new Map(picks.map((l) => [l.path, l]));
    const kept = (Array.isArray(options.loras) ? options.loras : []).filter((l) => !listed.includes(l.path));
    const put = (next) => setOption({ loras: [...kept, ...next] });
    const toggle = (path) => put(picked.has(path)
      ? picks.filter((l) => l.path !== path)
      : [...picks, { path, weight: byPath.get(path)?.weight ?? 1 }]);
    const full = picks.length >= MAX_LORAS;
    // A LoRA the user says doesn't work with this model isn't offered (one picked stays).
    const choices = items.filter((i) => !i.hidden || picked.has(i.path)).map((i) => h('button.pick.lora-pick', {
      type: 'button', class: picked.has(i.path) ? 'on' : '', 'aria-pressed': String(picked.has(i.path)),
      title: [i.path, i.words.length && `adds ${i.words.join(', ')}`, i.why].filter(Boolean).join(' · '),
      disabled: full && !picked.has(i.path), 'data-name': `${i.label} ${i.path}`.toLowerCase(),
      onclick: () => toggle(i.path),
    }, i.pinned && icon('pin', 12), i.label, i.mark && h('span.lora-mark', i.mark)));
    const labelOf = (path) => byPath.get(path)?.label ?? loraName(path);
    const filter = listed.length >= LORA_FILTER_FROM && h('input.seed-input.text', {
      type: 'search', placeholder: `Filter ${listed.length} LoRAs`, 'aria-label': 'Filter LoRAs', spellcheck: false,
      oninput: () => {
        const q = filter.value.trim().toLowerCase();
        for (const b of choices) b.hidden = !!q && !b.dataset.name.includes(q) && !b.classList.contains('on');
      },
    });
    const rows = picks.map((l) => h('div.lora-row',
      h('span.lora-name', { title: l.path }, labelOf(l.path),
        byPath.get(l.path)?.words.length > 0 && h('span.mono-meta.accent', ` + ${byPath.get(l.path).words.join(', ')}`)),
      h('label.seed-field.lora-weight', h('span.mono-meta', '×'), numberInput({
        value: l.weight, label: `${labelOf(l.path)} weight`, step: 0.1, min: -10, max: 10,
        onchange: (value, field) => {
          if (value == null || !Number.isFinite(value)) { field.value = String(l.weight); return; }
          put(picks.map((x) => (x.path === l.path ? { ...x, weight: Math.min(10, Math.max(-10, value)) } : x)));
        },
      })),
      h('button.att-x', {
        type: 'button', title: `Remove ${labelOf(l.path)}`, 'aria-label': `Remove ${labelOf(l.path)}`,
        onclick: () => toggle(l.path),
      }, icon('x', 12))));
    const under = preset?.loras ?? engine.image_defaults?.loras ?? []; // what these go on top of
    const hint = [
      under.length ? `on top of ${preset?.loras ? `${preset.name}'s` : 'the default'} ${lorasWord(under)}` : 'applied to every image this turn',
      'one already applied gets the weight here; 0 turns it off',
      'kept for later messages: type @ in the prompt for one message only',
      full && `at most ${MAX_LORAS}`,
    ].filter(Boolean).join(' · ');
    return h('div.tray-cell.span-all',
      h('div.cell-head', h('span.eyebrow', 'LoRAs'), h('span.spacer'),
        picks.length > 0 && h('button.mini', { type: 'button', onclick: () => put([]) }, 'Clear')),
      !choices.length && h('span.tray-hint', 'None of its LoRAs works with this model: see the LoRA library.'),
      filter && h('label.seed-field', icon('search', 13), filter),
      h('div.tray-seg.wrap.lora-picks', choices),
      rows.length > 0 && h('div.lora-rows', rows),
      h('span.tray-hint.wrap', hint));
  }

  /** What a preset sets, for its hint: its description, LoRAs, steps and CFG. */
  function presetSummary(p) {
    const loras = p.loras == null ? null
      : p.loras.length ? p.loras.map((l) => `${l.path.split(/[\\/]/).pop()}${l.weight === 1 ? '' : ` ×${l.weight}`}`).join(', ')
        : 'no LoRAs';
    return [p.description, loras, p.steps && `${p.steps} steps`, p.cfg_scale != null && `cfg ${p.cfg_scale}`]
      .filter(Boolean).join(' · ');
  }

  function renderActions() {
    clear(actions);
    if (compact) {
      if (isBusy()) actions.append(h('span.mono-meta.pbar-busy', h('span.dot.busy'), 'working'));
      return;
    }
    // Design 4b: once reconnecting gives up, the draft stays and Send waits for the server.
    const down = connection().status === 'offline';
    if (isBusy() && !down) {
      actions.append(h('button.send.stop.blueprint', {
        type: 'button', title: 'Stop turn', onclick: () => onStop?.(),
      }, corners(), h('span.stop-square'), 'Stop'));
      return;
    }
    // Raw mode sends the text as the prompt, so attachments alone are not enough.
    const empty = !d.text.trim() && (options.mode === 'raw' || !d.attachments.length);
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

  /** The engine's LoRAs as the pickers show them, with what the library knows of each. */
  function engineLoras(engine) {
    return engine ? loraItems(engine.available_loras ?? [], (path) => libraryLora(engine.name, path)) : [];
  }

  function render() {
    renderPlaceholder();
    renderAtts();
    if (!compact) {
      renderChips();
      renderTray();
    }
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
      if (!roomForImage()) break;
      const ref = uploadRef(file.name, new Set(d.attachments.map(refOf)));
      d.attachments.push({ file, preview: URL.createObjectURL(file), ref });
    }
    renderAtts();
    renderActions();
    editor.el.focus();
  }

  /** Whether a raw edit has room for one more image: an edit model may take only one
   * (FLUX.1 Kontext) or a few. Other modes let the chat model see more than it edits. */
  function roomForImage() {
    const max = editInfo()?.max_ref_images;
    if (options.mode !== 'raw' || !max || d.attachments.length < max) return true;
    toast(`${editInfo()?.card_name ?? 'The edit model'} edits at most ${max} ${max === 1 ? 'image' : 'images'} at a time`, { error: true });
    return false;
  }

  function addArtifact(artifact, { first = false, prefill = null } = {}) {
    const known = d.attachments.some((a) => a.artifact?.id === artifact.id);
    if (!known && !first && !roomForImage()) return;
    if (!known) {
      const att = { artifact };
      if (first) d.attachments.unshift(att);
      else d.attachments.push(att);
    }
    if (prefill && !d.text.trim()) setText(prefill);
    renderAtts();
    renderChips();
    renderActions();
    editor.focusEnd();
  }

  function setText(text) {
    d.text = text;
    editor.setText(text);
  }

  function removeAttachment(att) {
    d.attachments.splice(d.attachments.indexOf(att), 1);
    if (att.preview) URL.revokeObjectURL(att.preview);
    renderAtts();
    renderChips();
    renderActions();
  }

  async function submit() {
    const text = d.text.trim();
    if (sending || isBusy() || connection().status === 'offline') return;
    if (!text && (options.mode === 'raw' || !d.attachments.length)) return;
    sending = true;
    error = null;
    renderActions();
    renderError();
    try {
      const opts = requestOptions();
      if (maskLabel()) opts.mask = maskLabel();
      await onSubmit({ text, attachments: [...d.attachments], options: opts, mode: options.mode, qa: options.qa });
      clearDraft(draftKey);
      setText('');
      trayOpen = false;
    } catch (err) {
      error = describe(err);
    } finally {
      sending = false;
      if (el.isConnected) render();
    }
  }

  const unsubscribe = [
    on('options', render), on('backends', render), on('loras', render),
    on('conn', () => { renderActions(); renderError(); }),
  ];
  setText(d.text);
  if (!compact) {
    if (!loraLibrary()) loadLoras();
    useQueuedLoras();
  }
  render();

  /** LoRAs the library asked to use in the next prompt ({path, weight}, as the image
   * engine lists them): their mentions go at the end of the text. */
  function useQueuedLoras() {
    const queued = takeQueuedLoras();
    if (!queued.length) return;
    const engine = imageEngine();
    const items = engineLoras(engine);
    const found = queued.map((q) => [q, items.find((i) => i.path === q.path)]);
    const mentions = found.filter(([, i]) => i).map(([q, i]) => loraText(i.name, q.weight ?? i.weight));
    const missing = found.length - mentions.length;
    if (missing) {
      toast(`${engine?.name ?? 'The image engine'} doesn't list ${missing === 1 ? 'one of them' : `${missing} of them`}`, { error: true });
    }
    if (!mentions.length) return;
    const text = d.text.replace(/\s+$/, '');
    setText(`${text}${text ? ' ' : ''}${mentions.join(' ')} `);
  }

  return {
    el,
    refresh: renderActions,
    refreshAtts: () => { renderAtts(); renderChips(); },
    setPlaceholder: (text) => { basePlaceholder = text; renderPlaceholder(); },
    focus: () => editor.el.focus(),
    addFiles,
    addArtifact,
    destroy: () => { editor.close(); unsubscribe.forEach((u) => u()); },
  };
}

function describe(err) {
  if (err.status === 409) return 'A turn is already running here. Stop it or wait for it to finish.';
  if (err.status === 401) return 'The server needs an access token.';
  return err.message || 'Something went wrong.';
}
