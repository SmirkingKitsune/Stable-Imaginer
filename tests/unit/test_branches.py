from datetime import UTC, datetime, timedelta

from imaginer.core.ids import new_id
from imaginer.db import models as orm
from imaginer.domain.enums import ArtifactKind, ArtifactSource, Role
from imaginer.services.branches import describe

T0 = datetime(2026, 10, 28, 21, 0, tzinfo=UTC)


class Tree:
    """Messages built in creation order, with images attached to tool results."""

    def __init__(self) -> None:
        self.messages: list[orm.Message] = []
        self.attachments: dict[str, list[str]] = {}
        self.images: dict[str, orm.Artifact] = {}

    def add(
        self,
        role: Role,
        parent: orm.Message | None,
        *,
        call: str | None = None,
        image: str | None = None,
    ) -> orm.Message:
        seq = len(self.messages) + 1
        m = orm.Message(
            id=new_id(),
            conversation_id="c",
            parent_id=parent.id if parent else None,
            seq=seq,
            role=role,
            tool_call_id=call,
            metadata_json={"tool": "generate_image"} if role is Role.TOOL else None,
            created_at=T0 + timedelta(minutes=seq),
        )
        self.messages.append(m)
        if image:
            artifact = orm.Artifact(
                id=new_id(),
                conversation_id="c",
                message_id=parent.id if parent else None,
                kind=ArtifactKind.IMAGE,
                source=ArtifactSource.GENERATED,
                mime="image/png",
                storage_key="k",
                sha256="0" * 64,
                size_bytes=1,
                label=image,
                created_at=m.created_at,
            )
            self.images[artifact.id] = artifact
            self.attachments[m.id] = [artifact.id]
        return m

    def turn(self, parent: orm.Message | None, image: str) -> tuple[orm.Message, ...]:
        """user -> assistant (call) -> tool (image) -> assistant."""
        user = self.add(Role.USER, parent)
        caller = self.add(Role.ASSISTANT, user)
        result = self.add(Role.TOOL, caller, call="call_1", image=image)
        reply = self.add(Role.ASSISTANT, result)
        return user, caller, result, reply


def test_no_messages_no_branches() -> None:
    assert describe([], {}, {}) == []


def test_nested_branches_are_named_and_ordered_depth_first() -> None:
    t = Tree()
    _, a1, _, f1 = t.turn(None, "img-1")
    *_, main_leaf = t.turn(f1, "img-2")
    edited, a1e, _, edited_leaf = t.turn(None, "img-3")  # message 1, edited
    retry = t.add(Role.TOOL, a1, call="call_1", image="img-4")  # retry on main
    retry_leaf = t.add(Role.ASSISTANT, retry)
    nested = t.add(Role.TOOL, a1e, call="call_1", image="img-5")  # retry on the edit
    second = t.add(Role.USER, f1)  # message 2, edited (a reply from the middle of main)

    branches = describe(t.messages, t.attachments, t.images)
    summary = [(b.name, b.kind, b.depth, b.message_count) for b in branches]
    assert summary == [
        ("main", "main", 0, 4),
        ("message 1, edited", "edit", 1, 2),
        ("retry of img-3", "retry", 2, 2),
        ("retry of img-1", "retry", 1, 2),
        ("message 2, edited", "edit", 1, 3),
    ]
    by_name = {b.name: b for b in branches}
    assert branches[0].leaf_id == main_leaf.id and branches[0].latest_image is not None
    assert branches[0].latest_image.label == "img-2"
    assert (branches[1].leaf_id, branches[1].fork_id) == (edited_leaf.id, edited.id)
    assert branches[1].parent_leaf_id == main_leaf.id
    assert branches[2].parent_leaf_id == edited_leaf.id
    assert branches[2].origin == "from assistant reply 1"
    assert branches[3].leaf_id == retry_leaf.id
    assert branches[3].latest_image is not None and branches[3].latest_image.label == "img-4"
    reply = by_name["message 2, edited"]
    assert (reply.origin, reply.leaf_id) == ("from your second message", second.id)
    assert reply.latest_image is not None and reply.latest_image.label == "img-1"
    assert [b.latest for b in branches] == [False, False, False, False, True]
    assert branches[-1].updated_at == second.created_at
    assert branches[2].fork_id == nested.id


def test_retry_of_a_failed_call_is_named_after_the_tool() -> None:
    t = Tree()
    user = t.add(Role.USER, None)
    caller = t.add(Role.ASSISTANT, user)
    t.add(Role.TOOL, caller, call="call_1")  # failed: no image
    t.add(Role.TOOL, caller, call="call_1", image="img-1")
    names = [b.name for b in describe(t.messages, t.attachments, t.images)]
    assert names == ["main", "retry of generate_image"]
