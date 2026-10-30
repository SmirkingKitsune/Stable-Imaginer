"""A conversation's branches, named the way the UI shows them.

Messages form a tree through ``parent_id``. Each leaf ends one branch. At every fork the
oldest child continues the branch it is on, and each younger child (an edited message, or
another attempt at a tool call) starts a branch of its own:

* **main** follows the oldest child from the first message down.
* Any other branch is named after its fork, the deepest message on its path that is not an
  oldest child: "message 2, edited" for a user message, "retry of img-3" for a tool result.
  Its parent branch is the one that the fork's oldest sibling continues.
"""

from collections import defaultdict

from imaginer.db import models as orm
from imaginer.domain import models as dto
from imaginer.domain.enums import Role
from imaginer.services import mappers

_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")


def describe(
    messages: list[orm.Message],
    attachments: dict[str, list[str]],
    images: dict[str, orm.Artifact],
) -> list[dto.Branch]:
    """Every branch, main first, each followed by the branches forked from it.

    ``attachments`` maps message ids to attached artifact ids; ``images`` holds the image
    artifacts among them (anything else attached is ignored).
    """
    if not messages:
        return []
    ordered = sorted(messages, key=lambda m: m.seq)
    by_id = {m.id: m for m in ordered}
    children: defaultdict[str | None, list[orm.Message]] = defaultdict(list)
    for m in ordered:
        children[m.parent_id].append(m)

    def continuation(node: orm.Message) -> orm.Message:
        while children[node.id]:
            node = children[node.id][0]
        return node

    def path(leaf: orm.Message) -> list[orm.Message]:
        nodes = [leaf]
        while nodes[-1].parent_id is not None and nodes[-1].parent_id in by_id:
            nodes.append(by_id[nodes[-1].parent_id])
        nodes.reverse()
        return nodes

    newest = ordered[-1]
    found: dict[str, dto.Branch] = {}
    forks: dict[str, int] = {}  # leaf id -> seq of its fork, for ordering
    for leaf in (m for m in ordered if not children[m.id]):
        nodes = path(leaf)
        fork_at = next(
            (
                i
                for i in range(len(nodes) - 1, -1, -1)
                if children[nodes[i].parent_id][0] is not nodes[i]
            ),
            None,
        )
        latest_image = max(
            (images[a] for m in nodes for a in attachments.get(m.id, []) if a in images),
            key=lambda a: a.id,
            default=None,
        )
        branch = dto.Branch(
            leaf_id=leaf.id,
            name="main",
            kind="main",
            fork_id=None,
            parent_leaf_id=None,
            origin=None,
            depth=0,
            message_count=_count_messages(nodes),
            latest_image=mappers.artifact(latest_image) if latest_image else None,
            latest=leaf is newest,
            updated_at=leaf.created_at,
        )
        if fork_at is not None:
            fork = nodes[fork_at]
            before = nodes[: fork_at + 1]
            branch.fork_id = fork.id
            branch.parent_leaf_id = continuation(children[fork.parent_id][0]).id
            forks[leaf.id] = fork.seq
            if fork.role is Role.TOOL:
                first = next(
                    s
                    for s in children[fork.parent_id]
                    if s.role is Role.TOOL and s.tool_call_id == fork.tool_call_id
                )
                labels = [images[a].label for a in attachments.get(first.id, []) if a in images]
                tool = (first.metadata_json or {}).get("tool", "a tool call")
                branch.kind = "retry"
                branch.name = f"retry of {', '.join(labels) or tool}"
                branch.origin = f"from assistant reply {_count_replies(before)}"
            else:
                number = sum(1 for m in before if m.role is Role.USER)
                branch.kind = "edit"
                branch.name = f"message {number}, edited"
                branch.origin = f"from your {_ordinal(number)} message"
        found[leaf.id] = branch

    # Depth-first from main: each branch, then the branches forked from it, oldest fork first.
    forked_from: defaultdict[str | None, list[dto.Branch]] = defaultdict(list)
    for branch in found.values():
        forked_from[branch.parent_leaf_id].append(branch)
    result: list[dto.Branch] = []
    stack = list(forked_from[None])  # main: the one branch without a fork
    while stack:
        branch = stack.pop()
        result.append(branch)
        below = sorted(forked_from[branch.leaf_id], key=lambda b: forks[b.leaf_id], reverse=True)
        for child in below:
            child.depth = branch.depth + 1
        stack.extend(below)
    return result


def _count_replies(nodes: list[orm.Message]) -> int:
    """Assistant replies: runs of assistant messages, with their tool results, after a user."""
    count = 0
    previous: orm.Message | None = None
    for m in nodes:
        if m.role is Role.ASSISTANT and (previous is None or previous.role is Role.USER):
            count += 1
        previous = m
    return count


def _count_messages(nodes: list[orm.Message]) -> int:
    return sum(1 for m in nodes if m.role is Role.USER) + _count_replies(nodes)


def _ordinal(n: int) -> str:
    if n <= len(_ORDINALS):
        return _ORDINALS[n - 1]
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
