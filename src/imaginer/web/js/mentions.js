// References in a prompt, picked by typing "@":
// - Images: "@img-3" shows as a chip with a text-sized thumbnail, and hovering one shows a
//   larger preview. The server never reads these: since one names the image by its label,
//   the model can follow it.
// - LoRAs: "@lora:ink-wash" or "@lora:ink-wash:0.6" shows as a chip with its weight, which
//   a click changes. The server reads these from the user's message and applies them to its
//   images (domain/mentions.py has the same grammar); the LLM and the engine never see them.

import { clear, h, icon } from './dom.js';
import { loraName } from './format.js';

/** A LoRA's name unquoted: letters, digits, _, - and /, with dots only inside it. */
const NAME = String.raw`[\p{L}\p{N}_\-/]+(?:\.[\p{L}\p{N}_\-/]+)*`;
/** "@lora:name[:weight]" (name maybe quoted: groups 1 or 2, weight 3), or "@ref" (group 4) for
 * an image's label; neither glued to a word before it. */
const MENTION = new RegExp(
  String.raw`(?<![\p{L}\p{N}_])(?:@lora:(?:"([^"\n]{1,1024})"|(${NAME}))(?::(-?\d+(?:\.\d+)?))?|@([\p{L}\p{N}_-]+))`, 'gu');
const BARE = new RegExp(String.raw`^${NAME}$`, 'u');

/** text as strings and pieces, one per mention accepted: {ref} where known(ref), {lora,
 * weight} where knownLora(name). Without knownLora, LoRA mentions stay text. */
export function splitMentions(text, known, knownLora = null) {
  const pieces = [];
  let last = 0;
  for (const m of text.matchAll(MENTION)) {
    let piece = null;
    if (m[4] !== undefined) {
      if (known(m[4])) piece = { ref: m[4] };
    } else if (knownLora?.(m[1] ?? m[2])) {
      piece = { lora: m[1] ?? m[2], weight: m[3] !== undefined ? Number(m[3]) : 1 };
    }
    if (!piece) continue;
    if (m.index > last) pieces.push(text.slice(last, m.index));
    pieces.push(piece);
    last = m.index + m[0].length;
  }
  if (last < text.length) pieces.push(text.slice(last));
  return pieces;
}

/** Image "@from" rewritten as "@to" for each entry of names (a Map), in one pass. */
export const renameMentions = (text, names) =>
  text.replace(MENTION, (all, quoted, bare, weight, ref) => (ref !== undefined && names.has(ref) ? `@${names.get(ref)}` : all));

/** text with each LoRA mention as the LoRA's name, for where a chip can't show (a title). */
export const loraNamesIn = (text) =>
  text.replace(MENTION, (all, quoted, bare) => (quoted ?? bare ? loraName(quoted ?? bare) : all));

/** A weight as the text and the chip show it: at most two decimals. */
export const weightText = (weight) => String(Math.round(Number(weight) * 100) / 100);

/** The text of a LoRA mention; weight 1 is left out. */
export function loraText(name, weight = 1) {
  const w = Number(weight);
  return `@lora:${BARE.test(name) ? name : `"${name}"`}${w === 1 ? '' : `:${weightText(w)}`}`;
}

/** The LoRAs a backend lists, as mentions name them: by file name without the extension,
 * or by the whole path when another has the same name. info(path), if given, is what the
 * LoRA library knows of one ({item, use}, see store.libraryLora): its display name
 * (label), default weight, trigger words, pin, and whether it works with the backend's
 * model (hidden when not; mark "?" when untested). Then pinned ones come first. */
export function loraItems(paths, info = null) {
  const count = new Map();
  for (const path of paths) count.set(loraName(path), (count.get(loraName(path)) ?? 0) + 1);
  const items = paths.map((path) => {
    const known = info?.(path) ?? null;
    const lib = known?.item ?? null;
    const compat = known?.use.compat ?? null;
    return {
      kind: 'lora',
      name: count.get(loraName(path)) === 1 ? loraName(path) : path,
      path,
      label: lib?.name ?? loraName(path),
      weight: lib?.default_weight ?? 1,
      words: (lib?.triggers ?? []).filter((t) => t.on).map((t) => t.text),
      pinned: !!lib?.pinned,
      mark: compat === 'maybe' || compat === 'unknown' ? '?' : '',
      why: compat === 'maybe' ? 'untested with this model' : compat === 'unknown' ? "can't check it against this model" : '',
      hidden: compat === 'no',
    };
  });
  return info ? items.sort((a, b) => (b.pinned - a.pinned) || a.label.localeCompare(b.label)) : items;
}

/** The one item a mention's name means: its name or path, else the path with either slash,
 * the file name, or the name without the extension (as the server matches it). */
export function findLora(items, name) {
  const want = name.replace(/\\/g, '/');
  const tests = [
    (i) => i.name === name || i.path === name,
    (i) => i.path.replace(/\\/g, '/') === want,
    (i) => i.path.split(/[\\/]/).pop() === want,
    (i) => loraName(i.path) === want,
  ];
  for (const test of tests) {
    const found = items.filter(test);
    if (found.length) return found.length === 1 ? found[0] : null;
  }
  return null;
}

/** The chip for "@lora:name": the LoRA's name and weight. In the editor it is one
 * uneditable unit, and a click on it changes the weight. */
export function loraChip(name, weight, { tag = 'span.mention.lora-mention', label = null, ...props } = {}) {
  return h(tag, { dataset: { lora: name, weight: String(weight) }, title: name, ...props },
    icon('puzzle', 12), h('span.lora-mention-name', label ?? loraName(name)),
    h('span.lora-mention-weight', `×${weightText(weight)}`));
}

/** A reference name for an upload, from its file name and unlike any in taken. */
export function uploadRef(fileName, taken) {
  const base = (fileName || '').replace(/\.[^.]*$/, '').replace(/[^\p{L}\p{N}_-]+/gu, '-')
    .slice(0, 18).replace(/^-+|-+$/g, '') || 'image';
  let ref = base;
  for (let i = 2; taken.has(ref); i++) ref = `${base}-${i}`;
  return ref;
}

/** The chip for "@ref": a text-sized thumbnail (fill(img) loads it) and the label. In the
 * editor it is one uneditable unit; in a message it is a link (tag 'a.mention'). */
export function mentionChip(ref, fill, { tag = 'span.mention', source = '', ...props } = {}) {
  const thumb = h('img.mention-thumb', { alt: '', draggable: 'false' });
  fill?.(thumb);
  return h(tag, { dataset: { ref, source }, ...props }, thumb, ref);
}

/** The plain text of an editor (or a piece of one): chips as "@ref", line breaks as \n. */
export function mentionText(root) {
  let out = '';
  const walk = (node) => {
    for (const child of node.childNodes) {
      if (child.nodeType === Node.TEXT_NODE) out += child.data;
      else if (child.dataset?.ref) out += `@${child.dataset.ref}`;
      else if (child.dataset?.lora) out += loraText(child.dataset.lora, child.dataset.weight);
      else if (child.nodeName === 'BR') out += '\n';
      else {
        if ((child.nodeName === 'DIV' || child.nodeName === 'P') && out && !out.endsWith('\n')) out += '\n';
        walk(child);
      }
    }
  };
  walk(root);
  // The browser keeps a <br> after a final line break so that the empty line shows.
  if (root.lastChild?.nodeName === 'BR') out = out.replace(/\n$/, '');
  return out.replaceAll(' ', ' ');
}

// --- editor --------------------------------------------------------------------------

/** Keys that move the caret, which can land it in (or out of) an "@query". */
const CARET_KEYS = new Set(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown']);
const LORAS_LISTED = 40; // in the picker at once: typing narrows it
let editors = 0;

/**
 * A plain-text editor in which "@" picks one of images() or loras() and puts it in as a
 * chip, where images: () => [{ref, source, fill(img)}] and loras: () => loraItems(paths).
 * Its text (text(), onInput) has each chip as its mention. onKey gets the keys the picker
 * leaves; onFiles, pasted image files.
 */
export function mentionEditor({ images, loras = () => [], className, label, onInput, onKey, onFiles }) {
  const pickerId = `mention-picker-${++editors}`;
  // Editable markup rather than a textarea, so that chips can sit in the text.
  const el = h(`div.mention-editor.${className}`, {
    contenteditable: 'true', role: 'textbox', 'aria-multiline': 'true', 'aria-label': label,
    'aria-autocomplete': 'list', 'aria-controls': pickerId, 'aria-expanded': 'false',
    oninput: () => {
      const value = text();
      el.classList.toggle('blank', !value);
      onInput?.(value);
      updatePicker();
    },
    onbeforeinput: (e) => {
      // Plain text only: no bold and the like, and line breaks rather than paragraphs.
      if (e.inputType.startsWith('format')) e.preventDefault();
      else if (e.inputType === 'insertParagraph') {
        e.preventDefault();
        document.execCommand('insertLineBreak');
      }
    },
    onkeydown: (e) => { if (!(picker && pickerKey(e))) onKey?.(e); },
    onkeyup: (e) => { if (CARET_KEYS.has(e.key)) updatePicker(); },
    onmouseup: () => updatePicker(),
    onclick: (e) => {
      const chip = e.target.closest?.('.lora-mention');
      if (chip && el.contains(chip)) editWeight(chip);
    },
    onblur: () => closePicker(),
    onscroll: () => { if (picker) placePicker(); },
    onpaste: (e) => {
      e.preventDefault();
      const files = [...(e.clipboardData?.files ?? [])].filter((f) => f.type.startsWith('image/'));
      const pasted = e.clipboardData?.getData('text/plain');
      if (files.length && onFiles) onFiles(files);
      else if (pasted) insertText(pasted);
    },
    oncopy: (e) => copySelection(e),
    oncut: (e) => { if (copySelection(e)) document.execCommand('delete'); },
  });
  previewMentions(el);
  const pickerEl = h('div.mention-picker', {
    id: pickerId, role: 'listbox', 'aria-label': 'Images and LoRAs',
    onmousedown: (e) => e.preventDefault(), // keeps the caret (and focus) in the text
  });
  let picker = null; // {node, start, end, query, items, options, index} while the picker is open
  let dismissed = null; // the "@" whose picker Escape closed: {node, start}

  const text = () => mentionText(el);
  const itemFor = (ref) => images().find((i) => i.ref === ref) ?? null;
  const loraFor = (name) => findLora(loras(), name);
  const chipFor = (item) => mentionChip(item.ref, item.fill, { contenteditable: 'false', source: item.source });
  const loraChipFor = (name, weight) => loraChip(name, weight, { contenteditable: 'false', label: loraFor(name)?.label });
  const split = (value) => splitMentions(value, (ref) => !!itemFor(ref), (name) => !!loraFor(name));
  const pieceChip = (p) => (p.ref !== undefined ? chipFor(itemFor(p.ref)) : loraChipFor(p.lora, p.weight));

  /** Replace the text, with each mention of one of the images or LoRAs as its chip. */
  function setText(value) {
    clear(el);
    for (const piece of split(value)) el.append(typeof piece === 'string' ? piece : pieceChip(piece));
    // Without a <br> after it, a final line break would not show.
    if (value.endsWith('\n')) el.append(h('br'));
    el.classList.toggle('blank', !value);
  }

  /** A chip whose image or LoRA is gone goes back to plain text (the text stays the same).
   * Mentions that have become known (the backends' LoRAs arrive after a draft is shown)
   * become chips, unless the editor is in use. */
  function sync() {
    for (const chip of el.querySelectorAll('.mention')) {
      if (chip.dataset.lora !== undefined) {
        if (!loraFor(chip.dataset.lora)) chip.replaceWith(loraText(chip.dataset.lora, chip.dataset.weight));
      } else if (!itemFor(chip.dataset.ref)) chip.replaceWith(`@${chip.dataset.ref}`);
    }
    const value = text();
    const known = split(value).filter((p) => typeof p !== 'string').length;
    if (known > el.querySelectorAll('.mention').length && document.activeElement !== el) setText(value);
  }

  /** Type text at the caret as one undoable edit, each known mention as its chip. */
  function insertText(value) {
    const pieces = split(value);
    if (pieces.every((p) => typeof p === 'string')) {
      document.execCommand('insertText', false, value);
      return;
    }
    const markup = (p) => (typeof p === 'string'
      ? h('span', p).innerHTML.replaceAll('\n', '<br>')
      : pieceChip(p).outerHTML);
    document.execCommand('insertHTML', false, pieces.map(markup).join(''));
    fillChips();
  }

  /** Load the thumbnails of chips the browser inserted as markup. */
  function fillChips() {
    for (const chip of el.querySelectorAll('.mention')) {
      const img = chip.querySelector('.mention-thumb');
      const item = itemFor(chip.dataset.ref);
      if (img && item && !img.getAttribute('src')) item.fill(img);
    }
  }

  /** Copy with chips as their mentions, so that pasting the text brings them back. */
  function copySelection(e) {
    const sel = getSelection();
    if (!sel.rangeCount || sel.isCollapsed) return false;
    const copied = h('div');
    copied.append(sel.getRangeAt(0).cloneContents());
    if (!copied.querySelector('.mention')) return false;
    e.preventDefault();
    e.clipboardData.setData('text/plain', mentionText(copied));
    return true;
  }

  // The picker.

  /** Open, filter or close the picker from the "@query" just before the caret. */
  function updatePicker() {
    const sel = getSelection();
    const node = sel.rangeCount && sel.isCollapsed ? sel.anchorNode : null;
    const end = sel.anchorOffset;
    const typed = node?.nodeType === Node.TEXT_NODE && el.contains(node)
      && node.data.slice(0, end).match(/@((?:lora:)?[\p{L}\p{N}_\-./]*)$/u);
    if (!typed) { closePicker(); return; }
    const start = end - typed[0].length;
    // An "@" inside a word (an e-mail address, say) is not a reference.
    const prev = node.previousSibling;
    const before = start > 0 ? node.data[start - 1] : prev?.nodeType === Node.TEXT_NODE ? prev.data.slice(-1) : '';
    if (/[\p{L}\p{N}_]/u.test(before) || (dismissed?.node === node && dismissed.start === start)) {
      closePicker();
      return;
    }
    const query = typed[1].toLowerCase();
    if (picker?.node === node && picker.start === start && picker.query === query) return;
    // "@lora:" asks for LoRAs only.
    const onlyLoras = query.startsWith('lora:');
    const q = onlyLoras ? query.slice(5) : query;
    const shown = onlyLoras ? [] : images().filter((i) => i.ref.toLowerCase().includes(q));
    const matching = loras().filter((l) => !l.hidden && [l.name, l.path, l.label ?? ''].some((s) => s.toLowerCase().includes(q)));
    const items = [...shown.map((i) => ({ kind: 'image', ...i })), ...matching.slice(0, LORAS_LISTED)];
    if (!items.length) { closePicker(); return; }
    picker = { node, start, end, query, items, index: 0 };
    renderPicker();
  }

  function renderPicker() {
    picker.options = picker.items.map((item, i) => {
      const option = (...children) => h('div.mention-option', {
        id: `${pickerId}-${i}`, role: 'option',
        onpointermove: () => { if (picker.index !== i) highlight(i); },
        onclick: () => choose(item),
      }, children);
      if (item.kind === 'lora') {
        return option(h('span.lora-option-icon', icon(item.pinned ? 'pin' : 'puzzle', 16)),
          h('div.att-text',
            h('span.att-label', item.label ?? loraName(item.path), item.mark && h('span.lora-mark', { title: item.why }, item.mark)),
            h('span.att-src', [item.path, item.words?.length && `+ ${item.words.join(', ')}`].filter(Boolean).join(' · '))));
      }
      const thumb = h('img.att-thumb', { alt: '' });
      item.fill(thumb);
      return option(thumb, h('div.att-text', h('span.att-label', item.ref), h('span.att-src', item.source)));
    });
    // A heading over each kind, outside the options so the arrows skip it.
    const list = [];
    picker.items.forEach((item, i) => {
      if (i === 0 || item.kind !== picker.items[i - 1].kind) {
        list.push(h('div.mention-head.eyebrow', item.kind === 'lora' ? 'LoRAs · this message only' : 'Attached images'));
      }
      list.push(picker.options[i]);
    });
    clear(pickerEl).append(
      h('div.mention-list', list),
      h('div.menu-keys', h('span', h('kbd', '↑↓'), ' choose'), h('span', h('kbd', '↵'), ' insert'),
        h('span', h('kbd', 'esc'), ' close')),
    );
    // On the page itself: inside a transformed dock a fixed position would be shifted.
    if (!pickerEl.isConnected) {
      document.body.append(pickerEl);
      window.addEventListener('hashchange', closePicker);
    }
    el.setAttribute('aria-expanded', 'true');
    placePicker();
    highlight(picker.index);
  }

  function highlight(i) {
    picker.index = i;
    picker.options.forEach((o, k) => {
      o.classList.toggle('on', k === i);
      o.setAttribute('aria-selected', String(k === i));
    });
    // Scrolls the list only: scrollIntoView would move the page behind it as well.
    const option = picker.options[i];
    const list = option.parentElement;
    // The first option of a kind scrolls its heading into view with it.
    const head = option.previousElementSibling?.classList.contains('mention-head') ? option.previousElementSibling : option;
    if (head.offsetTop < list.scrollTop) list.scrollTop = head.offsetTop;
    else if (option.offsetTop + option.offsetHeight > list.scrollTop + list.clientHeight) {
      list.scrollTop = option.offsetTop + option.offsetHeight - list.clientHeight;
    }
    el.setAttribute('aria-activedescendant', option.id);
  }

  /** Over the "@" when there is room above it (the bar sits low), under it otherwise. */
  function placePicker() {
    const range = document.createRange();
    range.setStart(picker.node, picker.start);
    range.setEnd(picker.node, picker.start + 1);
    const at = range.getBoundingClientRect();
    const box = pickerEl.getBoundingClientRect();
    const left = Math.min(Math.max(8, at.left - 8), innerWidth - box.width - 8);
    const top = at.top - box.height - 6 >= 8 ? at.top - box.height - 6 : at.bottom + 6;
    Object.assign(pickerEl.style, { left: `${left}px`, top: `${top}px` });
  }

  /** Arrows, Enter or Tab and Escape steer the open picker; other keys go on typing. */
  function pickerKey(e) {
    if (e.isComposing) return false;
    const n = picker.items.length;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') highlight((picker.index + (e.key === 'ArrowDown' ? 1 : n - 1)) % n);
    else if (e.key === 'Enter' || e.key === 'Tab') choose(picker.items[picker.index]);
    else if (e.key === 'Escape') {
      dismissed = { node: picker.node, start: picker.start };
      closePicker();
    } else return false;
    e.preventDefault();
    e.stopPropagation();
    return true;
  }

  /** Swap the "@query" for the chip and a space, as one undoable edit. */
  function choose(item) {
    const { node, start, end } = picker;
    closePicker();
    const range = document.createRange();
    range.setStart(node, start);
    range.setEnd(node, end);
    getSelection().removeAllRanges();
    getSelection().addRange(range);
    const spaced = /^\s/.test(node.data.slice(end));
    const chip = item.kind === 'lora' ? loraChipFor(item.name, item.weight ?? 1) : chipFor(item);
    document.execCommand('insertHTML', false, chip.outerHTML + (spaced ? '' : ' '));
    fillChips();
  }

  // A LoRA chip's weight.

  let weightPop = null;

  /** A small field by the chip; Enter or leaving it sets the weight, Escape keeps it. */
  function editWeight(chip) {
    closeWeight();
    closePicker();
    const input = h('input.seed-input.lora-weight-input', {
      type: 'number', step: 0.1, min: -10, max: 10, value: weightText(chip.dataset.weight),
      'aria-label': `${loraName(chip.dataset.lora)} weight`,
      onkeydown: (e) => {
        if (e.key === 'Enter') { e.preventDefault(); commit(); } else if (e.key === 'Escape') { e.preventDefault(); closeWeight(true); }
      },
      onblur: () => commit(),
    });
    const commit = () => {
      if (weightPop?.input !== input) return;
      const value = Number(input.value);
      if (input.value !== '' && Number.isFinite(value) && chip.isConnected) {
        const weight = Math.min(10, Math.max(-10, value));
        chip.dataset.weight = String(weight);
        chip.querySelector('.lora-mention-weight').textContent = `×${weightText(weight)}`;
        el.dispatchEvent(new Event('input'));
      }
      closeWeight(true);
    };
    const pop = h('div.lora-weight-pop', { onmousedown: (e) => { if (e.target !== input) e.preventDefault(); } },
      h('span.mono-meta', loraName(chip.dataset.lora)), h('label.seed-field', h('span.mono-meta', '×'), input),
      h('span.mono-meta', '0 turns it off'));
    document.body.append(pop);
    weightPop = { el: pop, input };
    const at = chip.getBoundingClientRect();
    const box = pop.getBoundingClientRect();
    const left = Math.min(Math.max(8, at.left), innerWidth - box.width - 8);
    const top = at.top - box.height - 6 >= 8 ? at.top - box.height - 6 : at.bottom + 6;
    Object.assign(pop.style, { left: `${left}px`, top: `${top}px` });
    input.focus();
    input.select();
  }

  function closeWeight(refocus = false) {
    if (!weightPop) return;
    const { el: pop } = weightPop;
    weightPop = null;
    pop.remove();
    if (refocus) caretToEnd(el);
  }

  function closePicker() {
    if (!picker) return;
    picker = null;
    pickerEl.remove();
    window.removeEventListener('hashchange', closePicker);
    el.setAttribute('aria-expanded', 'false');
    el.removeAttribute('aria-activedescendant');
  }

  return {
    el, text, setText, sync,
    focusEnd: () => caretToEnd(el),
    setPlaceholder: (value) => { el.dataset.placeholder = value; },
    close: () => { closePicker(); closeWeight(); },
  };
}

/** Focus an editor with the caret after its text, scrolled to show it. */
export function caretToEnd(el) {
  el.focus();
  getSelection().selectAllChildren(el);
  getSelection().collapseToEnd();
  el.scrollTop = el.scrollHeight;
}

// --- hover preview -----------------------------------------------------------------------

let preview = null;
let shownFor = null;

/** Show a larger preview while the pointer rests on a chip inside root. */
export function previewMentions(root) {
  root.addEventListener('mouseover', (e) => {
    const chip = e.target.closest?.('.mention');
    if (chip && chip !== shownFor) showPreview(chip);
  });
  root.addEventListener('mouseout', (e) => {
    if (shownFor && !shownFor.contains(e.relatedTarget)) hidePreview();
  });
}

function showPreview(chip) {
  const thumb = chip.querySelector('.mention-thumb');
  if (!thumb?.getAttribute('src')) return;
  if (!preview) {
    const img = h('img', { alt: '', onload: () => placePreview() });
    const caption = h('span.mono-meta');
    preview = { el: h('div.mention-preview', { 'aria-hidden': 'true' }, img, caption), img, caption };
    for (const type of ['scroll', 'pointerdown', 'keydown']) window.addEventListener(type, hidePreview, true);
    window.addEventListener('hashchange', hidePreview);
  }
  shownFor = chip;
  preview.img.src = thumb.src;
  preview.caption.textContent = [chip.dataset.ref, chip.dataset.source].filter(Boolean).join(' · ');
  document.body.append(preview.el);
  placePreview();
}

/** Centred over the chip, or under it when there is no room above. */
function placePreview() {
  if (!shownFor?.isConnected) { hidePreview(); return; }
  const at = shownFor.getBoundingClientRect();
  const box = preview.el.getBoundingClientRect();
  const left = Math.min(Math.max(8, at.left + at.width / 2 - box.width / 2), innerWidth - box.width - 8);
  const top = at.top - box.height - 8 >= 8 ? at.top - box.height - 8 : at.bottom + 8;
  Object.assign(preview.el.style, { left: `${left}px`, top: `${top}px` });
}

function hidePreview() {
  preview?.el.remove();
  shownFor = null;
}
