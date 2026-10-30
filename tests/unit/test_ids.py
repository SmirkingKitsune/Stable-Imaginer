import time
import uuid

from imaginer.core.ids import new_id, uuid7, uuid7_unix_ms


def test_version_and_variant_bits() -> None:
    u = uuid7()
    assert u.version == 7
    assert u.variant == uuid.RFC_4122


def test_canonical_string_form() -> None:
    value = new_id()
    assert len(value) == 36
    assert str(uuid.UUID(value)) == value


def test_embeds_current_unix_ms() -> None:
    before = time.time_ns() // 1_000_000
    stamp = uuid7_unix_ms(new_id())
    after = time.time_ns() // 1_000_000
    # The counter may borrow one ms when a burst overflows it.
    assert before <= stamp <= after + 1


def test_strictly_increasing_within_a_burst() -> None:
    ids = [new_id() for _ in range(20_000)]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
