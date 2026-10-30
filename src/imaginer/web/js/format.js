// Text formatting shared by the screens.

export const plural = (n, word, many = `${word}s`) => `${n} ${n === 1 ? word : many}`;

export function ago(iso) {
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return 'now';
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h`;
  if (s < 2 * 86400) return 'yesterday';
  if (s < 7 * 86400) return `${Math.floor(s / 86400)}d`;
  if (s < 30 * 86400) return `${Math.floor(s / (7 * 86400))}w`;
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export const clock = (iso) =>
  new Date(iso).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', hour12: false });

export function truncate(text, max) {
  const flat = String(text ?? '').replace(/\s+/g, ' ').trim();
  if (flat.length <= max) return flat;
  const cut = flat.slice(0, max);
  const space = cut.lastIndexOf(' ');
  return `${(space > max * 0.6 ? cut.slice(0, space) : cut).replace(/[\s,.;:]+$/, '')}…`;
}

/** A conversation title from the first prompt: its first line, cut at a word. */
export const titleFrom = (text) => truncate(String(text).split('\n')[0], 60) || null;

/** "4 images · img-1 → img-4" for a header; "6 images · 3 branches" once it has forked. */
export function imagesSummary(list, branches = 1) {
  if (!list.length) return '';
  if (branches > 1) return `${plural(list.length, 'image')} · ${plural(branches, 'branch', 'branches')}`;
  const span = list.length === 1 ? list[0].label : `${list[0].label} → ${list[list.length - 1].label}`;
  return `${plural(list.length, 'image')} · ${span}`;
}

export const size = (a) => (a?.width && a?.height ? `${a.width}×${a.height}` : '');

/** CSS aspect-ratio of an artifact (falls back to the design's 4/3). */
export const ratioOf = (a) => (a?.width && a?.height ? `${a.width} / ${a.height}` : '4 / 3');

export const ASPECTS = {
  '1:1': [16, 16], '4:3': [20, 15], '3:4': [15, 20], '3:2': [21, 14],
  '2:3': [14, 21], '16:9': [22, 12], '9:16': [12, 22],
};

export function seedOf(artifact) {
  const seed = artifact?.metadata?.seed;
  return seed == null ? null : seed;
}

/** "seed 41207 · 1184×888", "2048×1536" for uploads. */
export const specOf = (a) => [seedOf(a) != null && `seed ${seedOf(a)}`, size(a)].filter(Boolean).join(' · ');

/** Parse a tool call's JSON arguments; partial or broken JSON comes back as null. */
export function parseArgs(text) {
  try {
    const value = JSON.parse(text || '{}');
    return value && typeof value === 'object' ? value : null;
  } catch {
    return null;
  }
}

/** generate_image  prompt="…" aspect_ratio=4:3 n=2 seed=41207 */
export function argsLine(args, raw, overrides = {}) {
  if (!args) return raw || '';
  const merged = { ...args, ...overrides };
  const order = ['images', 'prompt', 'aspect_ratio', 'n', 'seed', 'mask', 'transparent'];
  const keys = [...order.filter((k) => k in merged), ...Object.keys(merged).filter((k) => !order.includes(k))];
  return keys
    .filter((k) => merged[k] != null && merged[k] !== false)
    .map((k) => {
      const v = merged[k];
      if (k === 'prompt' || k === 'negative_prompt') return `${k}="${truncate(v, 64)}"`;
      if (Array.isArray(v)) return `${k}=[${v.map((x) => JSON.stringify(x)).join(',')}]`;
      return `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`;
    })
    .join(' ');
}
