// Where each conversation lives: on this server, or on a linked instance. A linked
// instance's conversations are reached through this server's /v1/links/<id>/r/ proxy, so
// the browser only ever talks to (and holds the token of) its own server. Their routes
// start with #/l/<link id>.

const placed = new Map(); // conversation id -> link id

/** Remember that conversation `id` lives on link `linkId` (null: on this server). */
export function place(id, linkId) {
  if (linkId) placed.set(id, linkId);
  else placed.delete(id);
}

export const linkOf = (id) => placed.get(id) ?? null;

/** The API path prefix for conversation `id`: '' here, 'v1/links/<id>/r/' when remote. */
export function baseOf(id) {
  const link = placed.get(id);
  return link ? `v1/links/${encodeURIComponent(link)}/r/` : '';
}

/** The hash of a conversation's screen: convHash(id), convHash(id, 'chat'), convHash(id, 'img-3'). */
export function convHash(id, ...rest) {
  const link = placed.get(id);
  const head = link ? `#/l/${encodeURIComponent(link)}` : '#';
  return `${head}/c/${[id, ...rest].map(encodeURIComponent).join('/')}`;
}
