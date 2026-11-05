// Image references: "@img-3" in a prompt shows as a chip with a text-sized thumbnail, and
// hovering one shows a larger preview. Typing "@" in an editor picks the image from a list.
// The server never reads the "@": it is there for the user, and since it names the image
// by its label the model can follow it too.

import { clear, h } from './dom.js';

/** "@ref" where ref is a label (letters, digits, _ and -) not glued to a word before it. */
const MENTION = /(?<![\p{L}\p{N}_])@([\p{L}\p{N}_-]+)/gu;

/** text as strings and {ref} pieces, one per "@ref" that known(ref) accepts. */
export function splitMentions(text, known) {
  const pieces = [];
  let last = 0;
  for (const m of text.matchAll(MENTION)) {
    if (!known(m[1])) continue;
    if (m.index > last) pieces.push(text.slice(last, m.index));
    pieces.push({ ref: m[1] });
    last = m.index + m[0].length;
  }
  if (last < text.length) pieces.push(text.slice(last));
  return pieces;
}

/** "@from" rewritten as "@to" for each entry of names (a Map), in one pass. */
export const renameMentions = (text, names) =>
  text.replace(MENTION, (all, ref) => (names.has(ref) ? `@${names.get(ref)}` : all));

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
let editors = 0;

/**
 * A plain-text editor in which "@" picks one of images() and puts it in as a chip, where
 * images: () => [{ref, source, fill(img)}]. Its text (text(), onInput) has each chip as
 * "@ref". onKey gets the keys the picker leaves; onFiles, pasted image files.
 */
export function mentionEditor({ images, className, label, onInput, onKey, onFiles }) {
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
    id: pickerId, role: 'listbox', 'aria-label': 'Attached images',
    onmousedown: (e) => e.preventDefault(), // keeps the caret (and focus) in the text
  });
  let picker = null; // {node, start, end, query, items, options, index} while the picker is open
  let dismissed = null; // the "@" whose picker Escape closed: {node, start}

  const text = () => mentionText(el);
  const itemFor = (ref) => images().find((i) => i.ref === ref) ?? null;
  const chipFor = (item) => mentionChip(item.ref, item.fill, { contenteditable: 'false', source: item.source });

  /** Replace the text, with "@ref" of one of the images as its chip. */
  function setText(value) {
    clear(el);
    for (const piece of splitMentions(value, (ref) => !!itemFor(ref))) {
      el.append(typeof piece === 'string' ? piece : chipFor(itemFor(piece.ref)));
    }
    // Without a <br> after it, a final line break would not show.
    if (value.endsWith('\n')) el.append(h('br'));
    el.classList.toggle('blank', !value);
  }

  /** A chip whose image is gone goes back to plain "@ref" text (the text stays the same). */
  function sync() {
    for (const chip of el.querySelectorAll('.mention')) {
      if (!itemFor(chip.dataset.ref)) chip.replaceWith(`@${chip.dataset.ref}`);
    }
  }

  /** Type text at the caret as one undoable edit, "@ref" of one of the images as its chip. */
  function insertText(value) {
    const pieces = splitMentions(value, (ref) => !!itemFor(ref));
    if (pieces.every((p) => typeof p === 'string')) {
      document.execCommand('insertText', false, value);
      return;
    }
    const markup = (p) => (typeof p === 'string'
      ? h('span', p).innerHTML.replaceAll('\n', '<br>')
      : chipFor(itemFor(p.ref)).outerHTML);
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

  /** Copy with chips as "@ref", so that pasting the text brings them back. */
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
      && node.data.slice(0, end).match(/@([\p{L}\p{N}_-]*)$/u);
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
    const items = images().filter((i) => i.ref.toLowerCase().includes(query));
    if (!items.length) { closePicker(); return; }
    picker = { node, start, end, query, items, index: 0 };
    renderPicker();
  }

  function renderPicker() {
    picker.options = picker.items.map((item, i) => {
      const thumb = h('img.att-thumb', { alt: '' });
      item.fill(thumb);
      return h('div.mention-option', {
        id: `${pickerId}-${i}`, role: 'option',
        onpointermove: () => { if (picker.index !== i) highlight(i); },
        onclick: () => choose(item),
      }, thumb, h('div.att-text', h('span.att-label', item.ref), h('span.att-src', item.source)));
    });
    clear(pickerEl).append(
      h('div.mention-head.eyebrow', 'Attached images'),
      h('div.mention-list', picker.options),
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
    if (option.offsetTop < list.scrollTop) list.scrollTop = option.offsetTop;
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
    document.execCommand('insertHTML', false, chipFor(item).outerHTML + (spaced ? '' : ' '));
    fillChips();
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
    close: closePicker,
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
