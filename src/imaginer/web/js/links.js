// Linked instances: other Imaginer instances on the network this one is paired with.
// A link is symmetric: each gallery shows the other's conversations (they stay where they
// are), and each side may use the engines the other shares, the whole model running there.
//
// Pairing takes a code: "Allow a link" shows one here, valid for ten minutes and once;
// typing it into the other instance (with this one's address) links the two. Headless
// hosts make codes with `imaginer link code`. Changes save to config/<profile>.local.yaml.

import { api } from './api.js';
import { connectionStrip, dialog, statusLine, toast } from './chrome.js';
import { append, corners, h, icon, iconButton, clear } from './dom.js';
import { ago, plural } from './format.js';
import { loadBackends } from './store.js';

const POLL_MS = 5000;
const STATUS = { ok: 'LINKED', unknown: 'CHECKING', unreachable: 'OFFLINE', incompatible: 'INCOMPATIBLE', unauthorized: 'REFUSED' };
const FIT = { ready: 'READY', cold: 'LOADS ON USE', unfit: 'WON’T FIT', unavailable: 'UNAVAILABLE' };
const GALLERY = { full: 'Full', read: 'Read only', off: 'Off' };
const gb = (mb) => (mb == null ? null : `${(mb / 1024).toFixed(1)} GB`);

export function linksScreen({ navigate }) {
  let overview = null;
  let engines = [];
  let found = [];
  let code = null; // {code, expires_at}
  let poll = null;
  let tick = null;
  let gone = false;
  let deferred = false; // a poll arrived while typing: render once focus leaves the field

  const status = statusLine();
  const meta = h('span.topbar-meta');
  const header = h('header.topbar.sub',
    iconButton('back', 'Back', () => (history.length > 1 ? history.back() : navigate('#/'))),
    h('div.topbar-heading', h('span.topbar-title', 'Linked instances'), meta),
    h('span.spacer'),
    h('div.only-wide', status.el),
    h('button.btn-line', { type: 'button', onclick: () => refresh({ probe: true }) }, icon('retry', 15), 'Check all'));
  const main = h('main.links-main');
  const strip = connectionStrip();
  const el = h('div.screen.links', header, strip.el, main);

  // Typed into the form survives the polls: the form is built once.
  const urlInput = h('input.seed-input', { type: 'url', placeholder: 'http://nas.local:8700', spellcheck: false, 'aria-label': 'Address of the other instance' });
  const codeInput = h('input.seed-input', { type: 'text', placeholder: 'XXXX-XXXX', spellcheck: false, autocomplete: 'off', 'aria-label': 'Pairing code' });
  const nameInput = h('input.seed-input', { type: 'text', placeholder: 'name here (optional)', spellcheck: false, 'aria-label': 'Name for the link' });
  const linkBtn = h('button.btn.btn-primary.blueprint', { type: 'button', onclick: () => link() }, corners(), icon('link', 14), 'Link');
  codeInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') link(); });

  async function refresh({ probe = false } = {}) {
    try {
      const [next, list, near] = await Promise.all([
        api.links(), api.engines(), api.discovered().catch(() => []),
      ]);
      if (gone) return;
      if (probe) {
        await Promise.all(next.links.map((l) => api.refreshLink(l.id).catch(() => null)));
        overview = await api.links();
      } else overview = next;
      engines = list.filter((e) => e.type !== 'linked');
      found = near;
    } catch (err) {
      if (!gone && !overview) clear(main).append(h('div.notice', err.message));
      return;
    }
    if (main.contains(document.activeElement) && document.activeElement.matches('input:not([type=checkbox])')) {
      deferred = true;
      return;
    }
    render();
  }
  main.addEventListener('focusout', () => setTimeout(() => {
    if (deferred && !main.contains(document.activeElement)) { deferred = false; render(); }
  }));

  function render() {
    const o = overview;
    meta.textContent = [o.settings_file && `saves to ${o.settings_file.split(/[\\/]/).slice(-2).join('/')}`,
      plural(o.links.length, 'link')].filter(Boolean).join(' · ');
    append(clear(main), [
      h('div.engine-head',
        h('div.engine-title',
          h('span.eyebrow.accent', o.enabled ? 'This instance' : 'Standalone'),
          h('h2.hero-title', o.name),
          h('div.engine-state',
            h('span.state', { class: o.enabled ? 'done' : 'wait' }, o.enabled ? 'LINKING ON' : 'LINKING OFF'),
            h('span.mono-meta', [o.advertise_url && `reachable at ${o.advertise_url}`, `id ${o.instance_id.slice(-12)}`,
              o.enabled && (o.discovery ? 'visible on the network' : 'not advertised')].filter(Boolean).join(' · ')))),
        o.enabled && h('button.btn.btn-secondary.engine-go', { type: 'button', onclick: () => allow() }, icon('plus', 14), 'Allow a link')),
      !o.enabled && offNote(),
      code && codeCell(),
      h('div.engine-grid',
        h('div.engine-cells', o.links.length ? o.links.map(linkCell) : h('div.tray-cell',
          h('span.eyebrow', 'Links'),
          h('span.tray-hint.wrap', o.enabled
            ? 'Not linked yet. Allow a link here and type the code on the other instance, or link with one below using the code it shows.'
            : 'Turn linking on to see other instances here.'))),
        h('div.engine-cells',
          o.enabled && addCell(),
          o.enabled && found.length > 0 && foundCell(),
          shareCell())),
    ]);
  }

  function offNote() {
    return h('div.tray-cell.links-off',
      h('span.eyebrow', 'One computer, both engines'),
      h('span.tray-hint.wrap',
        'This instance runs on its own: its engines and conversations are all here. To use another instance’s engines, '
        + 'or show its conversations in this gallery, set links.enabled: true in the profile, bind server.host to the '
        + 'network (e.g. 0.0.0.0) and set IMAGINER_SERVER__AUTH_TOKEN, then restart.'));
  }

  // --- pairing ---------------------------------------------------------------------------

  async function allow() {
    try {
      code = await api.pairingCode();
    } catch (err) {
      toast(err.message, { error: true });
      return;
    }
    render();
    clearInterval(tick);
    tick = setInterval(() => {
      if (!code || Date.parse(code.expires_at) <= Date.now()) {
        code = null;
        clearInterval(tick);
        render();
        return;
      }
      const left = main.querySelector('.code-left');
      if (left) left.textContent = leftOf(code);
    }, 1000);
  }

  function leftOf(c) {
    const s = Math.max(0, Math.round((Date.parse(c.expires_at) - Date.now()) / 1000));
    return `valid ${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')} more · single use`;
  }

  function codeCell() {
    return h('div.tray-cell.row-cell.code-cell.blueprint', corners(),
      h('div.cell-grow',
        h('span.eyebrow', 'Pairing code'),
        h('span.pair-code', code.code),
        h('span.mono-meta.wrap', `On the other instance: Linked instances › link with ${overview.advertise_url} and this code.`),
        h('span.mono-meta.code-left', leftOf(code))),
      iconButton('x', 'Hide the code', () => { code = null; clearInterval(tick); render(); }));
  }

  async function link() {
    const url = urlInput.value.trim().replace(/\/+$/, '');
    const typed = codeInput.value.trim();
    if (!url || !typed) {
      toast('Enter the other instance’s address and the code it shows.', { error: true });
      return;
    }
    const body = { url, code: typed };
    const name = nameInput.value.trim();
    if (name) body.name = name;
    linkBtn.disabled = true;
    try {
      const made = await api.link(body);
      codeInput.value = '';
      nameInput.value = '';
      document.activeElement?.blur();
      toast(`Linked with ${made.peer_name}`);
      await refresh();
    } catch (err) {
      toast(err.message, { error: true });
    } finally {
      linkBtn.disabled = false;
    }
  }

  function addCell() {
    return h('div.tray-cell',
      h('div.cell-head', h('span.eyebrow', 'Link with an instance'), h('span.spacer'),
        h('span.tray-hint', 'the code comes from “Allow a link” there')),
      h('div.link-form',
        h('label.seed-field', icon('server', 13), urlInput),
        h('label.seed-field', icon('hash', 13), codeInput),
        h('label.seed-field', nameInput),
        linkBtn));
  }

  function foundCell() {
    return h('div.tray-cell',
      h('span.eyebrow', 'On the network'),
      found.map((f) => h('div.found-row',
        h('span.ring', { class: f.link_id ? 'ok' : '' }),
        h('div.engine-text',
          h('span.engine-name', f.name),
          h('span.engine-meta', [f.url, f.link_id ? 'linked' : null].filter(Boolean).join(' · '))),
        !f.link_id && h('button.mini', {
          type: 'button',
          onclick: () => { urlInput.value = f.url; codeInput.focus(); },
        }, 'Link…'))));
  }

  // --- links -----------------------------------------------------------------------------

  function linkCell(l) {
    const ok = l.status === 'ok';
    return h('div.tray-cell.link-cell',
      h('div.pop-engine-row',
        h('span.ring', { class: ok ? 'ok' : '' }),
        h('div.engine-text',
          h('span.engine-name', l.peer_name, h('span.engine-default', l.name !== l.peer_name ? `as ${l.name}` : '')),
          h('span.engine-meta', [l.url, l.last_seen_at && `seen ${ago(l.last_seen_at)}`,
            l.gallery && l.gallery !== 'full' && `gallery ${GALLERY[l.gallery].toLowerCase()}`].filter(Boolean).join(' · '))),
        h('span.state', { class: ok ? 'done' : 'fail' }, STATUS[l.status] ?? l.status.toUpperCase())),
      !ok && l.detail && h('span.tray-hint.wrap', l.detail),
      ok && (l.engines.length
        ? l.engines.map((e) => sharedRow(l, e))
        : h('span.tray-hint', `${l.peer_name} shares no engines.`)),
      l.used_by.length > 0 && h('span.mono-meta.wrap', `used here by ${l.used_by.join(', ')}`),
      h('div.pop-actions.flush',
        h('button.mini', { type: 'button', onclick: () => check(l) }, icon('retry', 12), 'Check'),
        h('button.mini', { type: 'button', onclick: () => unlink(l) }, icon('x', 12), 'Unlink')));
  }

  function sharedRow(l, e) {
    const a = e.availability;
    const usable = a.state === 'ready' || a.state === 'cold';
    const numbers = [a.need_mb != null && `needs ~${gb(a.need_mb)}`, a.vram_free_mb != null && `${gb(a.vram_free_mb)} free`]
      .filter(Boolean).join(' · ');
    const tasks = e.kind === 'llm' ? ['chat'] : ['t2i', 'edit'];
    return h('div.shared-row',
      h('div.engine-text',
        h('span.engine-name', e.name, h('span.engine-default', e.kind === 'llm' ? 'chat' : 'image')),
        h('span.engine-meta', [e.model, a.detail, numbers, e.queue_depth ? `${e.queue_depth} queued` : null].filter(Boolean).join(' · '))),
      h('span.state', { class: a.state === 'ready' ? 'done' : usable ? 'wait' : 'fail' }, FIT[a.state] ?? a.state),
      h('button.mini.primary', {
        type: 'button', disabled: !usable,
        title: `Make ${e.name} on ${l.peer_name} this instance’s ${e.kind === 'llm' ? 'chat' : 'image'} engine`,
        onclick: () => use(l, e, tasks),
      }, e.kind === 'llm' ? 'Use for chat' : 'Use for images'));
  }

  async function use(l, e, tasks) {
    try {
      await api.useLinked(l.id, { engine: e.name, tasks });
      toast(`${tasks.includes('chat') ? 'Chat' : 'Images'} now run on ${l.peer_name} (${e.name})`);
      loadBackends();
      await refresh();
    } catch (err) {
      toast(err.message, { error: true });
    }
  }

  async function check(l) {
    try {
      await api.refreshLink(l.id);
      await refresh();
    } catch (err) {
      toast(err.message, { error: true });
    }
  }

  async function unlink(l) {
    const yes = await dialog({
      title: `Unlink ${l.peer_name}?`,
      body: `Its conversations leave this gallery and its engines stop working here${l.used_by.length ? ` (${l.used_by.join(', ')})` : ''}. `
        + `${l.peer_name} is told to forget this instance too. Linking again takes a new code.`,
      confirm: 'Unlink', danger: true,
    });
    if (!yes) return;
    try {
      await api.unlink(l.id);
      toast(`Unlinked ${l.peer_name}`);
      loadBackends();
      await refresh();
    } catch (err) {
      toast(err.message, { error: true });
    }
  }

  // --- sharing ---------------------------------------------------------------------------

  function shareCell() {
    const o = overview;
    return h('div.tray-cell',
      h('div.cell-head', h('span.eyebrow', 'Shared from here'), h('span.spacer'),
        h('span.tray-hint', 'linked instances may use these')),
      engines.length ? engines.map((e) => {
        const share = o.share[e.name];
        return h('div.share-row',
          h('label.check', h('input', { type: 'checkbox', checked: !!share, onchange: (ev) => setShare(e.name, ev.target.checked ? { remote_start: false } : null) }),
            h('span.engine-name', e.name, h('span.engine-default', e.kind === 'llm' ? 'chat' : 'image'))),
          h('label.check.small', { class: share && e.launchable ? '' : 'off', title: e.launchable ? null : 'Only engines started by this instance can be started for others' },
            h('input', {
              type: 'checkbox', checked: !!share?.remote_start, disabled: !share || !e.launchable,
              onchange: (ev) => setShare(e.name, { remote_start: ev.target.checked }),
            }), h('span.mono-meta', 'may start it')));
      }) : h('span.tray-hint', 'No local engines to share.'),
      h('div.cell-head', h('span.eyebrow', 'Conversations here'), h('span.spacer'),
        h('div.tray-seg.small', { role: 'group', 'aria-label': 'What linked instances may do with this instance’s conversations' },
          Object.entries(GALLERY).map(([key, text]) => h('button.pick', {
            type: 'button', class: o.gallery === key ? 'on' : '', 'aria-pressed': String(o.gallery === key),
            onclick: () => save({ gallery: key }),
          }, text)))),
      h('span.tray-hint.wrap', {
        full: 'Linked instances list them, and can continue them: turns run here, with these engines.',
        read: 'Linked instances list and open them, but can’t change them.',
        off: 'Linked instances don’t see them.',
      }[o.gallery]));
  }

  function setShare(name, value) {
    const share = Object.fromEntries(Object.entries(overview.share).map(([k, v]) => [k, { remote_start: v.remote_start }]));
    if (value) share[name] = value;
    else delete share[name];
    save({ share });
  }

  async function save(patch) {
    try {
      overview = await api.linkSettings(patch);
      render();
    } catch (err) {
      toast(err.message, { error: true });
      refresh();
    }
  }

  refresh();
  poll = setInterval(() => { if (!document.hidden) refresh(); }, POLL_MS);
  return {
    el,
    destroy() {
      gone = true;
      clearInterval(poll);
      clearInterval(tick);
      status.destroy();
      strip.destroy();
    },
  };
}
