// The conversation (designs 1h, 1m, 2c, 2d): the thread, bottom-anchored over a docked bar.
// Tool calls show as one-line rows with their job's state, which open to the whole call
// (every argument in full, the LoRAs and the words they added, the engine's settings); a
// running job gets a progress panel (step bar, engine settings, cancel) with lineage for
// edits; results are figures with Edit / Retry / Reference actions.
//
// Branches: the thread shows one path through the conversation's tree. "‹ 1 / 2 ›"
// switchers sit at each fork (an edited message, or a retried call's attempts), the rail
// (wide screens) and the header picker list every branch, and Edit & branch turns one of
// your messages into an editor that sends the new version on a new branch.
//
// Agents (design "Agent Mockups" 1c): the agent's portrait and name head each of its
// replies; set_portrait shows what the portrait is now and the self-portrait it becomes.

import { portrait } from './agents.js';
import { setImage } from './api.js';
import { announce, connectionStrip, dialog, menu, modelName, statusLine, toast } from './chrome.js';
import { corners, h, icon, iconButton, clear } from './dom.js';
import {
  argsLine, clock, imagesSummary, loraName, lorasWord, parseArgs, plural, ratioOf, specOf, truncate,
} from './format.js';
import {
  caretToEnd, findLora, loraChip, loraItems, mentionChip, mentionEditor, previewMentions, splitMentions,
} from './mentions.js';
import { convHash } from './origin.js';
import { promptBar } from './promptbar.js';
import {
  agentOfMessage, byLabel, cancelJob, connection, currentBranch, editEngine, ensureLoaded, imageEngine, images, libraryLora,
  loraLibrary, on, pathOf, remove, rename, retry, reloadConversation, running, select, send, setArchivedFlag, siblingsOf,
  stale, stop,
} from './store.js';

/** The image engine's LoRAs, with what the library knows of each (display names). */
function engineLoras() {
  const engine = imageEngine();
  return engine ? loraItems(engine.available_loras ?? [], (path) => libraryLora(engine.name, path)) : [];
}

const openReasoning = new Set(); // message ids, kept across visits
const openCalls = new Set(); // "<message id>:<tool call id>" of calls shown in full
const PLACEHOLDER = 'Reply, or name an image like img-2…';

export function conversationScreen({ id, navigate, focusBar = false }) {
  let c = null;
  let first = true;
  let editing = null; // id of the user message being edited
  let editText = '';

  const title = h('span.topbar-title', '…');
  const meta = h('span.topbar-meta');
  const status = statusLine();
  const archiveBtn = iconButton('archive', 'Archive', () => toggleArchive());
  const moreBtn = iconButton('more', 'More', () => menu(moreBtn, [
    { label: 'Rename', icon: 'type', onclick: askRename },
    { label: 'Delete conversation', icon: 'trash', danger: true, onclick: askDelete },
  ]));
  moreBtn.setAttribute('aria-haspopup', 'menu');
  moreBtn.setAttribute('aria-expanded', 'false');
  const pickerName = h('span.branch-pick-name');
  const picker = h('button.branch-pick', {
    type: 'button', hidden: true, title: 'Branches', 'aria-label': 'Branches',
    'aria-haspopup': 'menu', 'aria-expanded': 'false',
    onclick: () => menu(picker, c.branches.map((b) => ({
      checked: b.leaf_id === currentBranch(c)?.leaf_id,
      label: b.depth ? `${'\u00a0\u00a0'.repeat(b.depth - 1)}↳ ${b.name}` : b.name,
      icon: 'branch',
      onclick: () => showBranch(b),
    }))),
  }, icon('branch', 15), pickerName, icon('chevronDown', 14));
  const header = h('header.topbar.sub',
    iconButton('back', 'Back', () => navigate(c && images(c).length ? convHash(id) : '#/')),
    h('div.topbar-heading', title, meta),
    h('span.spacer'),
    picker,
    h('div.only-wide', status.el),
    archiveBtn,
    moreBtn,
  );

  const railList = h('div.rail-list');
  const rail = h('aside.rail', { hidden: true, 'aria-label': 'Branches' },
    h('span.eyebrow', 'Branches'),
    railList,
    h('p.rail-hint', 'Retry an image or edit any of your messages to start a branch. Each branch keeps its own images and labels.'));

  const thread = h('div.thread');
  previewMentions(thread);
  const scroller = h('div.thread-scroll', thread);
  const bar = promptBar({
    draftKey: id,
    placeholder: PLACEHOLDER,
    isBusy: () => running(id),
    onSubmit: (msg) => send(id, msg),
    onStop: () => stop(id).catch((err) => toast(err.message)),
  });
  const dock = h('div.dock', bar.el);
  const strip = connectionStrip({
    turnRunning: () => running(id),
    onReload: () => reloadConversation(id).catch((err) => toast(err.message, { error: true })),
  });
  const el = h('div.screen.conversation', header, strip.el, rail, scroller, h('div.scrim'), dock);

  // --- blocks -------------------------------------------------------------------

  const blocks = new Map(); // key -> {sig, el}

  function render() {
    if (!c) return;
    title.textContent = c.info?.title || 'Untitled';
    document.title = `${c.info?.title || 'Untitled'} · Stable-Imaginer`;
    const forked = c.branches.length > 1;
    meta.textContent = imagesSummary(images(c), c.branches.length);
    const archived = !!c.info?.archived_at;
    archiveBtn.title = archived ? 'Unarchive' : 'Archive';
    archiveBtn.setAttribute('aria-label', archiveBtn.title);
    archiveBtn.classList.toggle('on', archived);
    const branch = currentBranch(c);
    renderBranches(branch);
    bar.setPlaceholder(forked ? `Reply on branch ${branch?.name ?? 'main'}…` : PLACEHOLDER);
    bar.refresh();

    const path = pathOf(c);
    if (editing && !path.some((m) => m.id === editing)) editing = null;
    const stick = first || scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 120;
    const ctx = context(path);
    const wanted = groups(path).map((g) => block(g, ctx));
    if (!wanted.length) {
      wanted.push({ key: 'empty', el: h('div.thread-empty', h('span.eyebrow.accent', 'New conversation'), h('h2.hero-title', 'Describe an image, or drop one in.')) });
    }
    const keep = new Set(wanted.map((b) => b.key));
    for (const key of blocks.keys()) if (!keep.has(key)) blocks.delete(key);
    // While editing, what follows the edited message is only context: it stays on its branch.
    const editAt = wanted.findIndex((b) => b.key === editing);
    wanted.forEach((b, i) => {
      b.el.classList.toggle('dim', editAt !== -1 && i > editAt);
      if (thread.children[i] !== b.el) thread.insertBefore(b.el, thread.children[i] ?? null);
    });
    while (thread.children.length > wanted.length) thread.lastChild.remove();
    if (stick && editAt === -1) scroller.scrollTop = scroller.scrollHeight;
    first = false;
  }

  /** Messages grouped for display: a user message, or a run of assistant messages with
   * the tool results between them folded into their calls. */
  function groups(path) {
    const out = [];
    let run = null;
    for (const m of path) {
      if (m.role === 'user') {
        run = null;
        out.push({ kind: 'user', key: m.id, messages: [m] });
      } else if (m.role === 'assistant') {
        if (!run) out.push((run = { kind: 'assistant', key: m.id, messages: [] }));
        run.messages.push(m);
      }
    }
    if (c.error && !running(id)) out.push({ kind: 'error', key: `error-${c.error.turn_id}`, messages: [] });
    return out;
  }

  /** Tool results on this path, and every job, by "<assistant message id>:<tool call id>":
   * engines may reuse call ids across turns, so a call id alone is not unique. A call has
   * one job per attempt. */
  function context(path) {
    const results = new Map();
    let caller = null;
    for (const m of path) {
      if (m.role === 'assistant') caller = m.id;
      else if (m.role === 'tool' && caller) results.set(`${caller}:${m.tool_call_id}`, m);
    }
    const jobs = new Map();
    for (const j of [...c.jobs.values()].sort((a, b) => (a.id < b.id ? -1 : 1))) {
      if (!j.tool_call_id) continue;
      const key = `${j.request_message_id}:${j.tool_call_id}`;
      if (!jobs.has(key)) jobs.set(key, []);
      jobs.get(key).push(j);
    }
    return { results, jobs };
  }

  /** The job of the attempt shown: the one a finished result names, or the newest one
   * started for the call since the attempt began (ids are time-ordered). */
  function jobFor(ctx, key, result) {
    const list = ctx.jobs.get(key) ?? [];
    if (result && result.status !== 'streaming') {
      const named = parseArgs(result.content_text)?.job_id;
      return list.find((j) => j.id === named) ?? null;
    }
    return list.findLast((j) => !result || j.id > result.id) ?? null;
  }

  function block(g, ctx) {
    const sig = JSON.stringify(signature(g, ctx));
    const cached = blocks.get(g.key);
    if (cached?.sig === sig) return { key: g.key, el: cached.el };
    const node = g.kind === 'user' ? (g.key === editing ? editorBlock(g.messages[0]) : userBlock(g.messages[0]))
      : g.kind === 'error' ? errorBlock()
        : assistantBlock(g.messages, ctx);
    blocks.set(g.key, { sig, el: node });
    return { key: g.key, el: node };
  }

  function signature(g, ctx) {
    const art = (aid) => c.artifacts.get(aid)?.label ?? null;
    if (g.kind === 'error') return [c.error];
    if (g.kind === 'user') {
      const m = g.messages[0];
      const edit = m.id === editing && [c.branches.length, currentBranch(c)?.name];
      return [m.content_text, m.attachments.map(art), m.metadata, siblingsOf(c, m).map((s) => s.id), running(id), edit];
    }
    const agent = agentOfMessage(c, g.messages[0]);
    return [running(id), stale(id) && connection().status, agent && [agent.name, agent.portrait.url, agent.portrait.images.length],
      g.messages.map((m) => [
      m.status, m.content_text, m.reasoning_text, openReasoning.has(m.id), m.metadata?.error,
      (m.tool_calls ?? []).map((call) => {
        const key = `${m.id}:${call.id}`;
        const result = ctx.results.get(key);
        const job = jobFor(ctx, key, result);
        return [
          call.function.arguments, openCalls.has(key),
          job && [job.id, job.status, job.progress, job.seed, job.params, job.outputs.map((o) => art(o.artifact_id))],
          result && [result.id, result.status, result.content_text, result.attachments.map(art)],
          result && siblingsOf(c, result).map((s) => s.id),
          (parseArgs(call.function.arguments)?.images ?? []).map((l) => byLabel(c, l)?.id),
        ];
      }),
    ])];
  }

  function head(who, sub) {
    return h('div.msg-head', h('span.eyebrow', who), h('span.msg-sub', `· ${sub}`));
  }

  /** "‹ 2 / 3 ›" over a fork's alternatives; picking one shows its newest branch. */
  function switcher(list, current, label = null) {
    const i = list.findIndex((s) => s.id === current.id);
    const go = (step) => () => select(c, list[i + step].id);
    return h('span.switcher',
      label && h('span.switcher-label', label),
      h('button', { type: 'button', title: 'Previous', 'aria-label': `Previous ${label ?? 'version'}`, disabled: i <= 0, onclick: go(-1) }, icon('chevronLeft', 13)),
      `${i + 1} / ${list.length}`,
      h('button', { type: 'button', title: 'Next', 'aria-label': `Next ${label ?? 'version'}`, disabled: i >= list.length - 1, onclick: go(1) }, icon('chevronRight', 13)));
  }

  function attachmentChips(m) {
    const attached = m.attachments.map((aid) => c.artifacts.get(aid)).filter(Boolean);
    return attached.length > 0 && h('div.msg-atts', attached.map((a) => {
      const thumb = h('img.chip-thumb', { alt: '' });
      setImage(thumb, a);
      return h('a.att-chip', { href: convHash(id, a.label), title: `Open ${a.label}` }, thumb, h('span', a.label));
    }));
  }

  /** The text with each "@label" of an image here as a chip that opens it, and each LoRA
   * mention as a chip with its weight. */
  function withMentions(text) {
    const image = (label) => {
      const a = byLabel(c, label);
      return a?.mime?.startsWith('image/') ? a : null;
    };
    const loras = engineLoras();
    return splitMentions(text, image, () => true).map((p) => {
      if (typeof p === 'string') return p;
      if (p.lora !== undefined) return loraChip(p.lora, p.weight, { label: findLora(loras, p.lora)?.label });
      const a = image(p.ref);
      return mentionChip(a.label, (img) => setImage(img, a), {
        tag: 'a.mention', href: convHash(id, a.label), 'aria-label': `Open ${a.label}`, source: sourceWord(a),
      });
    });
  }

  function sourceWord(a) {
    return a.kind === 'mask' ? 'mask' : a.source;
  }

  function userBlock(m) {
    const opts = m.metadata?.image_options ?? {};
    const sent = [
      m.metadata?.mode && `${m.metadata.mode} mode`, m.metadata?.qa === false && 'QA off',
      opts.width ? `${opts.width}×${opts.height}` : opts.aspect_ratio, opts.n && `×${opts.n}`,
      opts.seed != null && `seed ${opts.seed}`, opts.preset && `preset ${opts.preset}`,
      opts.loras?.length && `+ ${lorasWord(opts.loras)}`, opts.steps && `${opts.steps} steps`,
      opts.cfg_scale != null && `cfg ${opts.cfg_scale}`, opts.negative_prompt && 'negative',
      opts.transparent && 'transparent', opts.mask, opts.backend,
    ].filter(Boolean);
    const alternatives = siblingsOf(c, m);
    return h('section.msg.user',
      h('div.msg-head.user-head',
        h('span.you-mark', { 'aria-hidden': 'true' }, 'YOU'),
        h('span.eyebrow', 'You'), h('span.msg-sub', `· ${clock(m.created_at)}`),
        h('span.spacer'),
        alternatives.length > 1 && switcher(alternatives, m),
        h('button.branch-btn', {
          type: 'button', title: 'Edit this message on a new branch', disabled: running(id),
          onclick: () => startEdit(m),
        }, icon('branch', 13), 'Edit & branch')),
      attachmentChips(m),
      m.content_text && h('p.msg-text', withMentions(m.content_text)),
      sent.length > 0 && h('div.mono-meta.msg-opts', icon('sliders', 12), sent.join(' · ')));
  }

  /** Design 2d: the message as an editor. Sending forks; everything below stays put. */
  function editorBlock(m) {
    // "@" picks one of the message's own images (the edit keeps its attachments) or a LoRA.
    const attached = m.attachments.map((aid) => c.artifacts.get(aid)).filter((a) => a?.mime?.startsWith('image/'));
    const input = mentionEditor({
      images: () => attached.map((a) => ({ ref: a.label, source: sourceWord(a), fill: (img) => setImage(img, a) })),
      loras: engineLoras,
      className: 'edit-text', label: 'Edit your message',
      onInput: (text) => { editText = text; },
      onKey: (e) => {
        if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
          e.preventDefault();
          submitEdit(m);
        } else if (e.key === 'Escape') {
          stopEdit();
        }
      },
    });
    input.setText(editText);
    const branch = currentBranch(c)?.name ?? 'this branch';
    return h('section.msg.user.editing.blueprint',
      corners(),
      head('Editing your message', `creates branch ${c.branches.length + 1}`),
      attachmentChips(m),
      input.el,
      h('div.edit-row',
        h('span.edit-hint', `Messages below stay on ${branch}. Nothing is deleted.`),
        h('button.btn-line', { type: 'button', onclick: stopEdit }, icon('chevronLeft', 15), 'Cancel'),
        h('button.btn-branch.blueprint', { type: 'button', disabled: running(id), onclick: () => submitEdit(m) },
          corners(), icon('branch', 15), 'Branch from here')));
  }

  function errorBlock() {
    return h('section.msg', h('div.msg-error', h('strong', c.error.title || 'The turn failed'), ' ', c.error.detail));
  }

  function assistantBlock(messages, ctx) {
    const lead = messages[0];
    const streaming = messages.some((m) => m.status === 'streaming');
    const model = lead.metadata?.mode === 'raw' ? 'raw prompt' : modelName(lead.model) || lead.backend || 'assistant';
    const parts = [];
    const agent = agentOfMessage(c, lead);
    for (const m of messages) {
      if (m.reasoning_text) parts.push(reasoning(m));
      if (m.content_text) parts.push(h('p.msg-text', m.content_text));
      if (m.status === 'streaming' && !m.content_text && !m.reasoning_text && !(m.tool_calls ?? []).length) {
        parts.push(h('span.typing', { 'aria-label': 'Thinking' }, h('i'), h('i'), h('i')));
      }
      if (m.status === 'error') {
        parts.push(h('div.msg-error', m.metadata?.error?.detail || 'This reply failed.'));
      }
      if (m.status === 'cancelled') parts.push(h('span.state.wait', 'STOPPED'));
      for (const call of m.tool_calls ?? []) parts.push(toolCall(m, call, ctx, agent));
    }
    const when = streaming ? 'streaming' : clock(lead.created_at);
    const who = agent
      ? h('div.msg-head.agent-head',
        h('a.agent-head-link', { href: `#/agents/${agent.id}`, title: `${agent.name}: its card` }, portrait(agent, { size: 26 })),
        h('span.eyebrow', agent.name), h('span.msg-sub', `· ${model} · ${when}`))
      : head('Imaginer', `${model} · ${when}`);
    return h('section.msg.assistant', who, parts);
  }

  function reasoning(m) {
    const open = openReasoning.has(m.id);
    const words = m.reasoning_text.trim().split(/\s+/).length;
    return [
      h('button.reason-toggle', {
        type: 'button', 'aria-expanded': String(open),
        onclick: () => {
          if (open) openReasoning.delete(m.id);
          else openReasoning.add(m.id);
          render();
        },
      }, h('span.chev', { class: open ? 'open' : '' }, icon('chevronRight', 14)), 'Reasoning',
      h('span.mono-meta', m.status === 'streaming' && !m.content_text ? 'thinking…' : plural(words, 'word'))),
      open && h('div.reason-box', m.reasoning_text),
    ];
  }

  function toolCall(m, call, ctx, agent = null) {
    const name = call.function.name;
    const args = parseArgs(call.function.arguments);
    const key = `${m.id}:${call.id}`;
    const result = ctx.results.get(key);
    const finished = result && result.status !== 'streaming' ? result : null;
    const job = jobFor(ctx, key, result);
    const outcome = finished ? parseArgs(finished.content_text) : null;
    const shown = job?.params?.requested ? { ...job.params.requested, seed: job.seed } : args;

    // Design 4a/4b: while the stream is down, live job data is the last thing seen.
    const live = job && (job.status === 'running' || job.status === 'queued');
    const lost = live && stale(id);
    const gaveUp = lost && connection().status === 'offline';
    const p = job?.progress;
    const stepText = p?.total_steps ? `${p.step}/${p.total_steps}` : null;
    let state;
    if (gaveUp) state = h('span.state.fail', 'UNKNOWN · OFFLINE');
    else if (lost) state = h('span.state.wait', stepText ? `LAST SEEN · ${stepText}` : 'LAST SEEN');
    else if (job?.status === 'running' && p?.phase === 'loading') {
      state = h('span.state.wait', `LOADING · ${modelName(p.model) || 'MODEL'}`.toUpperCase());
    } else if (job?.status === 'running') {
      state = h('span.state.run', stepText ? `RUNNING · ${stepText}` : 'RUNNING');
    } else if (job?.status === 'queued') state = h('span.state.wait', 'QUEUED');
    else if (job?.status === 'succeeded') state = h('span.state.done', `DONE · ${job.outputs.length}`);
    else if (job?.status === 'cancelled') state = h('span.state.wait', 'CANCELLED');
    else if (job?.status === 'failed' || outcome?.status === 'error') state = h('span.state.fail', 'FAILED');
    else if (outcome) state = h('span.state.done', 'DONE');
    else state = h('span.state.wait', running(id) ? 'CALLING' : '—');

    const attempts = result ? siblingsOf(c, result) : [];
    const attemptSwitch = attempts.length > 1 ? switcher(attempts, result, 'attempt') : null;
    const outIds = result?.attachments?.length ? result.attachments : (job?.outputs ?? []).map((o) => o.artifact_id);
    const outs = outIds.map((aid) => c.artifacts.get(aid)).filter(Boolean);

    const open = openCalls.has(key);
    const parts = [h('div.tool-row',
      h('button.tool-toggle', {
        type: 'button', 'aria-expanded': String(open), title: open ? 'Hide the call' : 'Show the whole call',
        onclick: () => {
          if (open) openCalls.delete(key);
          else openCalls.add(key);
          render();
        },
      },
      h('span.chev', { class: open ? 'open' : '' }, icon('chevronRight', 12)),
      h('span.tool-name', name),
      h('span.tool-args', argsLine(shown, call.function.arguments))),
      !outs.length && attemptSwitch,
      state)];
    if (open) parts.push(callDetail(shown ?? {}, job, call.function.arguments));
    const qa = qaLine(result?.metadata?.qa);
    if (qa) parts.push(qa);
    if (outcome?.status === 'error') {
      parts.push(h('div.tool-error',
        h('span', `${outcome.error?.code ?? 'error'}: ${outcome.error?.detail ?? ''}`),
        retryButton(finished, 'Try again on a new branch')));
    }
    const inputs = name === 'edit_image' && Array.isArray(args?.images) ? args.images : [];
    // set_portrait: what the portrait is now, and the self-portrait it becomes.
    const self = name === 'set_portrait' && outcome?.status !== 'error' ? portraitLineage(agent, outcome, job) : null;
    if (gaveUp) {
      const where = stepText ? `Progress stopped at step ${stepText.replace('/', ' / ')}.` : 'Progress stopped.';
      parts.push(h('div.tool-error', h('span', `${where} Reconnect to see the result. Nothing is lost.`)));
    } else if (live) {
      parts.push(jobPanel(job, shown, inputs, args?.mask, lost, self));
    } else if (self && !outs.length) {
      parts.push(h('div.fig-with-lineage', self));
    }
    if (outs.length) {
      const figs = h('div.figures', outs.map((a, i) => figure(a, finished, i === outs.length - 1 ? attemptSwitch : null)));
      parts.push(inputs.length ? h('div.fig-with-lineage', figs, lineage(inputs, args?.mask, outs))
        : self ? h('div.fig-with-lineage', figs, self) : figs);
    }
    return h('div.tool', parts);
  }

  /** The agent's portrait before and after a set_portrait call. */
  function portraitLineage(agent, outcome, job) {
    const done = outcome?.portrait;
    const shown = agent?.portrait;
    const nowLabel = !shown?.source ? 'Monogram' : shown.source === 'hub' ? `Hub · ${shown.hub_owner ?? ''}`.trim()
      : shown.source === 'self' ? `self-portrait v${shown.images.find((i) => i.key === shown.key)?.version ?? ''}` : 'Uploaded';
    const next = (shown?.images ?? []).filter((i) => i.source === 'self').length + 1;
    const busy = running(id) && [...c.jobs.values()].some((j) => j.id !== job?.id && (j.status === 'running' || j.status === 'queued'));
    return h('div.lineage',
      h('span.eyebrow', 'Portrait'),
      !done && h('div.lin-row', agent ? portrait(agent, { size: 22, cls: 'lin-thumb' }) : h('span.lin-thumb'),
        h('span.lin-label.plain', nowLabel), h('span.muted', 'now')),
      h('div.lin-row',
        h('span.lin-thumb.arrow', { class: done ? '' : 'pending' }, icon('arrowRight', 12)),
        h('span.lin-label', `self-portrait v${done?.version ?? next}`),
        h('span.muted', done ? 'in use' : busy ? 'after the image in progress' : job ? 'painting' : 'next')));
  }

  /** The whole call, which its row cuts to one line: each argument in full, the LoRAs
   * with the trigger words they added, and the engine's settings once the job has them. */
  function callDetail(args, job, raw) {
    const resolved = job?.params?.resolved ?? {};
    const rows = [];
    const row = (key, value, cls = '') => rows.push(h('dt', key), h('dd', { class: cls }, value));
    if (!Object.keys(args).length && raw) row('arguments', raw);
    const order = ['images', 'prompt', 'negative_prompt', 'aspect_ratio', 'width', 'height', 'n', 'seed', 'mask', 'transparent'];
    const later = new Set(['loras', 'preset', 'steps', 'cfg_scale', 'sampler']);
    const keys = [...order.filter((k) => k in args), ...Object.keys(args).filter((k) => !order.includes(k) && !later.has(k))];
    for (const k of keys) {
      const v = args[k];
      if (v == null || v === false || v === '') continue;
      if (k === 'prompt' || k === 'negative_prompt') {
        row(k, [h('span.call-text', v), h('button.mini', {
          type: 'button', title: `Copy the ${k.replace('_', ' ')}`, onclick: () => copyText(v),
        }, 'Copy')], 'call-prompt');
      } else row(k, Array.isArray(v) ? v.join(', ') : v === true ? 'yes' : typeof v === 'object' ? JSON.stringify(v) : String(v));
    }
    const loras = resolved.loras ?? args.loras ?? [];
    if (loras.length) {
      row('loras', loras.map((l) => {
        const named = job && libraryLora(job.backend, l.path)?.item?.name;
        const label = named || loraName(l.path);
        return h('div', { title: l.path }, `${label} ×${l.weight ?? 1}`,
          l.trigger && h('span.muted', ` · added "${l.trigger}"`));
      }));
    }
    if (args.preset) row('preset', args.preset);
    const size = resolved.width && resolved.height ? `${resolved.width}×${resolved.height}` : null;
    const steps = resolved.steps ?? args.steps;
    const cfg = resolved.cfg_scale ?? args.cfg_scale;
    const engine = [
      job?.backend, job?.model && modelName(job.model), size, steps != null && plural(steps, 'step'),
      cfg != null && `cfg ${cfg}`, resolved.sampler ?? args.sampler,
    ].filter(Boolean);
    if (engine.length) row('engine', engine.join(' · '));
    return h('dl.call-detail', rows);
  }

  function copyText(text) {
    if (!navigator.clipboard) return toast('Select the text to copy it: this page can\'t use the clipboard');
    navigator.clipboard.writeText(text).then(() => toast('Copied'), () => toast('Could not copy'));
  }

  /** The quality check on one attempt (metadata.qa): checking, passed, redone, flagged, skipped. */
  function qaLine(qa) {
    if (!qa || (qa.status === 'checking' && !running(id))) return null;
    const issues = (qa.issues ?? []).join('; ');
    const fixed = (qa.earlier ?? []).flatMap((e) => e.issues ?? []).join('; ');
    const [tag, kind, text] = {
      checking: ['QA · CHECKING', 'run', 'Llama is looking at the image'],
      passed: ['QA ✓', 'done', qa.attempt > 1 ? `passed on attempt ${qa.attempt}${fixed ? ` · fixed: ${fixed}` : ''}` : 'passed'],
      redone: ['QA · REDONE', 'wait', [issues, qa.changes && `→ ${changesText(qa.changes)}`].filter(Boolean).join(' ')],
      flagged: ['QA · FLAGGED', 'fail', `kept with: ${issues || 'issues the check could not fix'}`],
      skipped: ['QA · SKIPPED', 'wait', qa.detail ?? ''],
    }[qa.status] ?? ['QA', 'wait', qa.status];
    return h('div.qa-line', { title: text }, h(`span.state.${kind}`, tag), h('span.qa-text', text));
  }

  function changesText(changes) {
    return [
      changes.prompt && `prompt "${truncate(changes.prompt, 48)}"`, changes.negative_prompt && 'negative prompt',
      changes.steps && `${changes.steps} steps`, changes.cfg_scale != null && `cfg ${changes.cfg_scale}`,
      changes.new_seed && 'new seed',
    ].filter(Boolean).join(', ');
  }

  function retryButton(result, title) {
    return result && h('button.mini', {
      type: 'button', title, disabled: running(id), onclick: () => doRetry(result),
    }, icon('retry', 13), 'Retry');
  }

  /** The design's lineage list: inputs by relation, then the output (or "pending"). */
  function lineage(inputs, mask, outs = null) {
    const input = (label, relation) => {
      const a = byLabel(c, label);
      const thumb = a ? h('img.lin-thumb', { alt: '' }) : h('span.lin-thumb');
      if (a) setImage(thumb, a);
      return h('div.lin-row', thumb, h('span.lin-label', label), h('span.mono-meta', relation));
    };
    const output = (label, relation, pending) => h('div.lin-row',
      h('span.lin-thumb.arrow', { class: pending ? 'pending' : '' }, '→'),
      h('span.lin-label', { class: pending ? 'plain' : '' }, label),
      h('span.mono-meta', relation));
    return h('div.lineage',
      h('span.eyebrow', 'Lineage'),
      inputs.map((label, i) => input(label, i === 0 ? 'edit_source' : 'reference')),
      mask && input(mask, 'mask'),
      outs ? outs.map((a) => output(a.label, 'edited', false)) : output('output', 'pending', true));
  }

  function jobPanel(job, shown, inputs, mask, lost = false, aside = null) {
    const p = job.progress ?? {};
    const total = p.total_steps ?? null;
    const step = p.step ?? 0;
    const pct = total ? Math.round((step / total) * 100) : 0;
    const aspect = shown?.aspect_ratio;
    const ratio = aspect ? aspect.replace(':', ' / ') : '1 / 1';
    const of = (p.images ?? shown?.n ?? 1) > 1 ? ` · IMAGE ${p.image ?? 1}/${p.images ?? shown.n}` : '';
    const preview = h('div.job-preview.blueprint', { style: { aspectRatio: ratio } },
      corners(),
      h('span.job-chip', job.status === 'queued' ? 'QUEUED' : p.phase === 'loading' ? `LOADING ${p.model ?? ''}`.trim() : `STEP ${step}${of}`),
      h('div.scanline', { style: { top: `${pct}%` } }));
    const engine = [job.backend, modelName(job.model), `seed ${job.seed}`, aspect].filter(Boolean).join(' · ');
    return h('div.job', { class: lost ? 'stale' : '' },
      preview,
      h('div.job-side',
        aside ?? (inputs.length ? lineage(inputs, mask) : h('div')),
        h('div.job-progress',
          h('div.job-step',
            h('span.job-step-n', total ? ['Step ', String(step), h('span.muted', ` / ${total}`)] : job.status === 'queued' ? 'Queued'
              : p.phase === 'loading' ? `Loading ${p.model ?? 'the edit model'}…` : 'Starting'),
            lost ? h('span.mono-meta', 'paused · not live') : h('span.mono-meta.accent', total ? `${pct}%` : '')),
          h('div.step-bar', h('i', { style: { width: `${pct}%` } })),
          h('span.mono-meta', engine)),
        h('button.btn-line', {
          type: 'button', disabled: lost,
          onclick: () => cancelJob(c, job.id).catch((err) => toast(err.message)),
        }, h('span.stop-square.hollow'), 'Cancel job')));
  }

  function figure(a, result, attemptSwitch) {
    const pic = h('img.fig-pic', { alt: a.metadata?.prompt || a.label, loading: 'lazy' });
    setImage(pic, a);
    return h('figure.fig',
      h('div.fig-frame.blueprint', {
        style: { aspectRatio: ratioOf(a) },
        onclick: (e) => { if (!e.target.closest('button')) navigate(convHash(id, a.label)); },
      },
      corners(), pic,
      h('div.fig-hover',
        h('span.label-chip', a.label),
        figActions(a, result))),
      h('figcaption', h('span.mono-meta', specOf(a)), attemptSwitch));
  }

  /** Edit, Mask, Retry, reference: what the image engine's edits allow (an edit model may
   * take no mask, or one image only). */
  function figActions(a, result) {
    const engine = editEngine();
    const edit = engine ? engine.edit : {}; // the backends aren't loaded yet: offer them all
    const single = edit?.max_ref_images === 1;
    return h('div.fig-actions',
      edit && h('button.mini.primary', {
        type: 'button', onclick: () => bar.addArtifact(a, { first: true, prefill: `Edit @${a.label}: ` }),
      }, icon('pencil', 13), 'Edit'),
      edit?.mask !== false && edit && h('button.mini', {
        type: 'button', title: 'Paint a mask for an edit', onclick: () => navigate(convHash(id, 'mask', a.label)),
      }, icon('brush', 13), 'Mask'),
      retryButton(result, 'Run this call again on a new branch'),
      edit && !single && h('button.mini.square', {
        type: 'button', title: 'Add to the bar as a reference', 'aria-label': `Use ${a.label} as a reference`,
        onclick: () => bar.addArtifact(a),
      }, icon('plus', 13)));
  }

  // --- branches -----------------------------------------------------------------

  let railSig = null;

  function renderBranches(current) {
    const forked = c.branches.length > 1;
    el.classList.toggle('has-rail', forked);
    rail.hidden = !forked;
    picker.hidden = !forked;
    pickerName.textContent = current?.name ?? 'main';
    const sig = JSON.stringify([c.branches, current?.leaf_id]);
    if (!forked || sig === railSig) return;
    railSig = sig;
    clear(railList).append(...c.branches.map((b) => railItem(b, b.leaf_id === current?.leaf_id)));
  }

  function railItem(b, on) {
    const counts = [plural(b.message_count, 'message'), b.latest_image && `latest ${b.latest_image.label}`];
    return h('button.rail-item', {
      type: 'button', class: on ? 'on' : '', 'aria-current': on ? 'true' : null,
      style: { marginLeft: `${b.depth * 18}px` },
      onclick: () => showBranch(b),
    },
    b.depth > 0 && h('span.rail-elbow'),
    h('span.rail-icon', icon('branch', 15)),
    h('span.rail-text',
      h('span.rail-name', b.name),
      h('span.mono-meta.rail-counts', counts.filter(Boolean).flatMap((part, i) => [i ? ' · ' : '', h('span', part)])),
      h('span.mono-meta', b.origin ?? lastPrompt(b.leaf_id))));
  }

  /** The newest thing you asked on a branch, to tell main apart. */
  function lastPrompt(leafId) {
    for (let m = c.messages.get(leafId); m; m = c.messages.get(m.parent_id)) {
      if (m.role === 'user' && m.content_text) return truncate(m.content_text, 34);
    }
    return '';
  }

  // --- actions ------------------------------------------------------------------

  function showBranch(b) {
    select(c, b.leaf_id);
    announce(`Showing branch ${b.name}`);
  }

  function startEdit(m) {
    editing = m.id;
    editText = m.content_text ?? '';
    render();
    const input = thread.querySelector('.edit-text');
    if (input) caretToEnd(input);
    input?.closest('.msg')?.scrollIntoView({ block: 'nearest' });
  }

  function stopEdit() {
    editing = null;
    render();
  }

  async function submitEdit(m) {
    const text = editText.trim();
    const attachments = m.attachments.map((aid) => c.artifacts.get(aid)).filter(Boolean).map((artifact) => ({ artifact }));
    if (!text && !attachments.length) return;
    try {
      await send(id, {
        text, attachments, options: m.metadata?.image_options ?? null,
        mode: m.metadata?.mode ?? null, qa: m.metadata?.qa ?? null, editOf: m.id,
      });
      editing = null;
      render();
    } catch (err) {
      toast(err.status === 409 ? 'A turn is already running here.' : err.message);
    }
  }

  function doRetry(result) {
    retry(id, result.id).catch((err) => toast(err.status === 409 ? 'A turn is already running here.' : err.message));
  }

  async function toggleArchive() {
    const archived = !c?.info?.archived_at;
    try {
      await setArchivedFlag(id, archived);
      toast(archived ? 'Archived.' : 'Back in the gallery.');
      if (archived) navigate('#/');
    } catch (err) {
      toast(err.message);
    }
  }

  async function askRename() {
    const value = await dialog({ title: 'Rename', input: { value: c.info?.title ?? '', placeholder: 'Untitled' }, confirm: 'Save' });
    if (value == null) return;
    rename(id, value.trim()).catch((err) => toast(err.message));
  }

  async function askDelete() {
    const n = images(c).length;
    const ok = await dialog({
      title: 'Delete this conversation?',
      body: `Its messages${n ? ` and ${plural(n, 'image')}` : ''} are removed for good.`,
      confirm: 'Delete', danger: true,
    });
    if (!ok) return;
    try {
      await remove(id);
      navigate('#/');
    } catch (err) {
      toast(err.message);
    }
  }

  // --- lifecycle ----------------------------------------------------------------

  const dockSize = new ResizeObserver(() => {
    thread.style.paddingBottom = `${dock.offsetHeight + 56}px`;
    // Whatever scrolls into view (the caret while editing a message) stops above the dock.
    scroller.style.scrollPaddingBottom = `${dock.offsetHeight + 32}px`;
  });
  dockSize.observe(dock);
  // The library names LoRAs (chips, calls) by the image engine's listing: when either
  // arrives or changes, everything is drawn again.
  let named = null;
  const relabel = () => {
    const engine = imageEngine();
    const now = [loraLibrary(), engine?.name, JSON.stringify(engine?.available_loras ?? null)];
    if (named && now.every((x, i) => x === named[i])) return;
    named = now;
    blocks.clear();
    render();
  };
  const unsubscribe = [
    on(`conv:${id}`, render), on('turns', render), on('loras', relabel), on('backends', relabel), on('agents', render),
  ];

  ensureLoaded(id).then((loaded) => {
    c = loaded;
    render();
    if (focusBar) bar.focus();
  }, (err) => {
    clear(thread).append(h('div.notice', err.status === 404 ? 'This conversation no longer exists.' : err.message));
  });
  if (focusBar) bar.focus();

  return {
    el,
    bar,
    destroy() {
      dockSize.disconnect();
      unsubscribe.forEach((u) => u());
      status.destroy();
      strip.destroy();
      bar.destroy();
      document.title = 'Stable-Imaginer';
    },
  };
}
