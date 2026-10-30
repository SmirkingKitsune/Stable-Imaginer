// Pieces every screen shares: the brand, the engine status line, dialogs and menus.

import { corners, h, icon, clear } from './dom.js';
import { connection, defaultBackend, offline, on, reconnectNow } from './store.js';

export const brand = () => h('a.brand', { href: '#/' }, h('span.brand-mark'), 'Stable-Imaginer');

/** "● chat · qwen3-vl-8b  ● image · sdcpp"; a hollow dot when the engine is not ready,
 * "offline" for both while the server can't be reached. */
export function statusLine({ short = false } = {}) {
  // A button: it opens the engine popover (design 1d).
  const el = h('button.status-line.status-btn', {
    type: 'button', 'aria-haspopup': 'dialog', 'aria-expanded': 'false', title: 'Engines',
    onclick: () => import('./engines.js').then((m) => m.enginePopover(el)),
  });
  const render = () => {
    clear(el);
    const chat = defaultBackend('chat');
    const image = defaultBackend('t2i');
    const down = offline();
    for (const [role, b, text] of [
      ['chat', chat, chat?.model ? modelName(chat.model) : chat?.name],
      ['image', image, image?.name],
    ]) {
      const title = down ? `${role}: the server is unreachable`
        : b ? `${role}: ${b.name} (${b.status}${b.detail ? `, ${b.detail}` : ''})` : `no ${role} backend`;
      el.append(h('span.status-item', { title },
        h('span.dot', { class: b?.status === 'ok' && !down ? '' : 'off' }),
        short ? null : `${role} · ${down ? 'offline' : text ?? 'none'}`));
    }
    el.append(icon('chevronDown', 13));
    el.setAttribute('aria-label', `Engines: ${[...el.querySelectorAll('.status-item')].map((s) => s.title).join('; ')}`);
  };
  render();
  const off = on('backends', render);
  return { el, destroy: off };
}

// --- connection strip (designs 4a, 4b) -------------------------------------------------

const secondsAgo = (t) => Math.max(0, Math.round((Date.now() - t) / 1000));
const span = (ms) => (ms < 90_000 ? `${Math.max(1, Math.round(ms / 1000))} s` : `${Math.round(ms / 60_000)} min`);

/**
 * The strip under a screen's header: reconnecting (with a countdown and Retry now), retries
 * exhausted (Reconnect, and Reload when the screen has a conversation), and back online.
 * It adds .has-strip to its screen so the content below moves down.
 */
export function connectionStrip({ onReload = null, turnRunning = () => false } = {}) {
  const el = h('div.conn-strip', { hidden: true });
  let last = 'online';
  let ticker = null;

  const render = () => {
    const s = connection();
    if (s.status !== last) {
      if (s.status === 'reconnecting' && last === 'online') announce('Connection lost. Reconnecting.');
      if (s.status === 'restored') announce('Back online.');
      last = s.status;
    }
    const shown = s.status !== 'online';
    el.hidden = !shown;
    el.parentElement?.classList.toggle('has-strip', shown);
    el.className = `conn-strip ${s.status}`;
    el.setAttribute('role', s.status === 'offline' ? 'alert' : 'status');
    clearInterval(ticker);
    ticker = null;
    clear(el);
    if (!shown) return;
    if (s.status === 'reconnecting') {
      const wait = s.nextAt ? Math.max(0, Math.ceil((s.nextAt - Date.now()) / 1000)) : 0;
      const parts = [`attempt ${s.attempt} of ${s.max}`];
      if (s.lastEventId) parts.push(`last event #${s.lastEventId}, ${secondsAgo(s.lastEventAt)} s ago`);
      el.append(
        icon('wifiOff', 16),
        h('span.conn-title', wait ? `Connection lost. Reconnecting in ${wait} s` : 'Connection lost. Reconnecting…'),
        h('span.mono-meta', parts.join(' · ')),
        h('span.spacer'),
        h('span.typing', { 'aria-hidden': 'true' }, h('i'), h('i'), h('i')),
        h('button.btn-line', { type: 'button', onclick: () => reconnectNow() }, icon('retry', 14), 'Retry now'));
      ticker = setInterval(render, 1000);
    } else if (s.status === 'offline') {
      const note = [`${s.max} attempts over ${span((s.gaveUpAt ?? Date.now()) - s.startedAt)}`];
      if (turnRunning()) note.push('the turn keeps running on the server');
      el.append(
        icon('wifiOff', 16),
        h('span.conn-title', "Couldn't reconnect"),
        h('span.mono-meta', note.join(' · ')),
        h('span.spacer'),
        onReload && h('button.btn-line', { type: 'button', onclick: () => onReload() }, 'Reload conversation'),
        h('button.btn.btn-primary', { type: 'button', onclick: () => reconnectNow() }, icon('retry', 14), 'Reconnect'));
    } else if (s.status === 'restored') {
      const note = s.resumedFrom
        ? [`resumed from event #${s.resumedFrom}`, `caught up ${s.caughtUp ?? 0} event${s.caughtUp === 1 ? '' : 's'}`]
        : ['the server is reachable again'];
      el.append(
        icon('check', 15),
        h('span.conn-title', 'Back online'),
        h('span.mono-meta', note.join(' · ')));
    }
  };
  const off = on('conn', render);
  // The screen puts the strip in place after building it.
  queueMicrotask(render);
  return {
    el,
    destroy() {
      off();
      clearInterval(ticker);
    },
  };
}

/** "Qwen3VL-8B-Instruct-Q4_K_M.gguf" → "qwen3vl-8b-instruct" */
export function modelName(model) {
  if (!model) return '';
  return model.replace(/\.gguf$/i, '').replace(/[-_.]?(Q\d.*|F16|BF16|FP16)$/i, '').toLowerCase();
}

// --- dialogs (design 7a: focus trap + restore) -------------------------------------

let ids = 0;
const uid = (prefix) => `${prefix}-${(ids += 1)}`;
const modals = []; // open modal roots, topmost last
const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/** Make everything but the topmost modal inert, or nothing once none is open. */
function syncInert() {
  const top = modals[modals.length - 1] ?? null;
  for (const child of document.body.children) {
    if (child.matches('.sr-live, .toast')) continue;
    child.inert = !!top && child !== top;
  }
}

/**
 * A modal layer: the page behind it is inert, Tab cycles inside it, Esc (or a backdrop
 * click, unless `sticky`) dismisses it, and focus returns to whatever had it before.
 * build(close) returns the dialog element; close(value) removes the layer.
 */
export function modal(build, { onDismiss = () => {}, sticky = false } = {}) {
  const opener = document.activeElement;
  let open = true;
  const close = () => {
    if (!open) return;
    open = false;
    backdrop.remove();
    modals.splice(modals.indexOf(backdrop), 1);
    syncInert();
    document.removeEventListener('keydown', onKey, true);
    if (opener?.isConnected) opener.focus();
  };
  const onKey = (e) => {
    if (modals[modals.length - 1] !== backdrop) return;
    if (e.key === 'Escape') {
      e.stopPropagation();
      e.preventDefault();
      if (!sticky) { close(); onDismiss(); }
    } else if (e.key === 'Tab') {
      const items = [...box.querySelectorAll(FOCUSABLE)].filter((el) => el.offsetParent !== null);
      if (!items.length) return;
      const [first, last] = [items[0], items[items.length - 1]];
      const inside = box.contains(document.activeElement);
      if (e.shiftKey && (document.activeElement === first || !inside)) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && (document.activeElement === last || !inside)) { e.preventDefault(); first.focus(); }
    }
  };
  const box = build(close);
  const backdrop = h('div.dialog-backdrop', {
    onclick: (e) => { if (e.target === backdrop && !sticky) { close(); onDismiss(); } },
  }, box);
  document.body.append(backdrop);
  modals.push(backdrop);
  syncInert();
  document.addEventListener('keydown', onKey, true);
  return { el: box, close };
}

/** A modal over a backdrop. Resolves with the input's value (or true), or null if dismissed.
 * Focus starts on the safe action: Cancel for a dangerous one, otherwise the input or OK. */
export function dialog({ title, body, input = null, confirm = 'OK', danger = false }) {
  return new Promise((resolve) => {
    const titleId = uid('dlg-title');
    const bodyId = body ? uid('dlg-body') : null;
    const field = input
      ? h('input.input', { type: input.type ?? 'text', value: input.value ?? '', placeholder: input.placeholder ?? '', 'aria-labelledby': titleId })
      : null;
    let cancelBtn;
    let okBtn;
    const layer = modal((close) => {
      const done = (value) => { close(); resolve(value); };
      const ok = () => done(field ? field.value : true);
      field?.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.isComposing) ok(); });
      cancelBtn = h('button.btn.btn-secondary', { type: 'button', onclick: () => done(null) }, 'Cancel');
      okBtn = h('button.btn', { type: 'button', class: danger ? 'btn-danger' : 'btn-primary', onclick: ok }, confirm);
      return h('div.dialog.blueprint', {
        role: danger ? 'alertdialog' : 'dialog', 'aria-modal': 'true',
        'aria-labelledby': titleId, 'aria-describedby': bodyId,
      },
      corners(),
      h('div.dialog-title', { id: titleId }, title),
      body && h('div.dialog-body', { id: bodyId }, body),
      field,
      h('div.dialog-actions', cancelBtn, okBtn));
    }, { onDismiss: () => resolve(null) });
    (danger ? cancelBtn : field ?? okBtn).focus();
    field?.select();
    return layer;
  });
}

// --- menus (design 7a: roving focus) -----------------------------------------------

/**
 * A small menu anchored under a button: items = [{label, icon, onclick, danger, checked}].
 * An item with `checked` (true or false) is a radio item, e.g. the branch shown.
 * Arrow keys move with wrap-around, Home/End jump, a letter jumps to the next match,
 * Tab closes it and moves on; focus returns to the anchor.
 */
export function menu(anchor, items) {
  const rect = anchor.getBoundingClientRect();
  anchor.setAttribute('aria-haspopup', 'menu');
  anchor.setAttribute('aria-expanded', 'true');
  const menuId = uid('menu');
  anchor.setAttribute('aria-controls', menuId);
  let open = true;
  const close = ({ restore = true } = {}) => {
    if (!open) return;
    open = false;
    sheet.remove();
    anchor.setAttribute('aria-expanded', 'false');
    document.removeEventListener('pointerdown', outside, true);
    document.removeEventListener('keydown', onKey, true);
    window.removeEventListener('hashchange', onRoute);
    if (restore && anchor.isConnected) anchor.focus();
  };
  const onRoute = () => close({ restore: false });
  const outside = (e) => { if (!sheet.contains(e.target) && !anchor.contains(e.target)) close({ restore: false }); };
  const buttons = items.map((item) => h('button.menu-item', {
    type: 'button', tabindex: '-1',
    role: item.checked == null ? 'menuitem' : 'menuitemradio',
    'aria-checked': item.checked == null ? null : String(!!item.checked),
    class: [item.danger && 'danger', item.checked && 'checked'].filter(Boolean).join(' '),
    onclick: () => { close(); item.onclick(); },
  }, item.icon && icon(item.icon, 15), h('span.menu-label', item.label), item.checked && [h('span.spacer'), icon('check', 14)]));
  const focusAt = (i) => {
    const n = buttons.length;
    if (!n) return;
    const target = buttons[((i % n) + n) % n];
    for (const b of buttons) b.tabIndex = b === target ? 0 : -1;
    target.focus();
  };
  const current = () => buttons.indexOf(document.activeElement);
  const onKey = (e) => {
    if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(); return; }
    if (!sheet.contains(document.activeElement)) return;
    if (e.key === 'ArrowDown') { e.preventDefault(); focusAt(current() + 1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); focusAt(current() - 1); }
    else if (e.key === 'Home') { e.preventDefault(); focusAt(0); }
    else if (e.key === 'End') { e.preventDefault(); focusAt(buttons.length - 1); }
    // Focus the anchor and let the browser's Tab carry on from there.
    else if (e.key === 'Tab') { close(); }
    else if (e.key.length === 1 && /\S/.test(e.key) && !e.ctrlKey && !e.metaKey && !e.altKey) {
      const key = e.key.toLowerCase();
      const start = current();
      for (let step = 1; step <= buttons.length; step += 1) {
        const i = (start + step) % buttons.length;
        const text = buttons[i].querySelector('.menu-label').textContent.replace(/^[\s↳]+/, '').toLowerCase();
        if (text.startsWith(key)) { focusAt(i); break; }
      }
    }
  };
  const kbd = (key, word) => h('span', h('kbd', key), ` ${word}`);
  const sheet = h('div.menu.blueprint', {
    id: menuId, role: 'menu', 'aria-orientation': 'vertical',
    style: { top: `${rect.bottom + 6}px`, right: `${Math.max(8, window.innerWidth - rect.right)}px` },
  },
  corners(),
  buttons,
  h('div.menu-keys', { 'aria-hidden': 'true' }, kbd('↑↓', 'move'), kbd('↵', 'select'), kbd('esc', 'close')));
  document.body.append(sheet);
  setTimeout(() => document.addEventListener('pointerdown', outside, true));
  document.addEventListener('keydown', onKey, true);
  window.addEventListener('hashchange', onRoute);
  const checked = items.findIndex((item) => item.checked);
  focusAt(checked === -1 ? 0 : checked);
  return { close };
}

// --- toasts (design 7a: announced) -----------------------------------------------------

// Mounted once: screen readers announce what is written into a region that already exists.
const politeRegion = h('div.sr-live.sr-only', { role: 'status', 'aria-live': 'polite', 'aria-atomic': 'true' });
document.body.append(politeRegion);

/** Say something to screen readers without showing it (job finished, branch switched). */
export function announce(text) {
  politeRegion.textContent = '';
  // A fresh text node after a tick, so repeating the same words is announced again.
  setTimeout(() => { politeRegion.textContent = text; }, 30);
}

/**
 * A short message under the header. Plain toasts are announced politely and fade out.
 * `error` toasts are alerts; with an `action` ({label, onclick}) they stay until dismissed.
 * Timers pause while the toast is hovered or focused.
 */
export function toast(text, { error = false, action = null } = {}) {
  document.querySelectorAll('.toast').forEach((old) => old.remove());
  const el = h('div.toast', {
    class: error ? 'alert' : '',
    role: error ? 'alert' : null,
    'aria-hidden': error ? null : 'true',
  }, h('span', text),
  action && h('button.btn-line', {
    type: 'button', onclick: () => { dismiss(); action.onclick(); },
  }, action.label),
  action && h('button.att-x', { type: 'button', title: 'Dismiss', 'aria-label': 'Dismiss', onclick: () => dismiss() }, icon('x', 13)));
  if (!error) announce(text);
  document.body.append(el);
  let timer = null;
  let gone = false;
  const dismiss = () => {
    if (gone) return;
    gone = true;
    clearTimeout(timer);
    el.classList.add('out');
    setTimeout(() => el.remove(), 400);
  };
  const arm = () => {
    clearTimeout(timer);
    if (!action) timer = setTimeout(dismiss, 3200);
  };
  const hold = () => clearTimeout(timer);
  el.addEventListener('pointerenter', hold);
  el.addEventListener('pointerleave', () => { if (!el.contains(document.activeElement)) arm(); });
  el.addEventListener('focusin', hold);
  el.addEventListener('focusout', (e) => { if (!el.contains(e.relatedTarget)) arm(); });
  arm();
  return { dismiss };
}
