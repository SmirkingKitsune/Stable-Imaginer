"""A stand-in for the engine service's failure reports (EngineService.failure_report)."""

from datetime import datetime, timedelta

from imaginer.backends.base import EngineReport
from imaginer.core.clock import utcnow

ASSERT = "GGML_ASSERT(ggml_can_mul_mat(a, b)) failed"


class FakeReporter:
    """Reports the engine in ``state``; an exit ``ago`` before the question, if it exited
    earlier than the work did."""

    def __init__(self, state: str = "exited", ago: timedelta | None = None) -> None:
        self.state = state
        self.ago = ago
        self.asked: list[tuple[str, str | None, bool]] = []

    async def failure_report(
        self, engine: str, *, family: str | None, since: datetime, unreachable: bool
    ) -> EngineReport | None:
        self.asked.append((engine, family, unreachable))
        exited = self.state == "exited"
        return EngineReport(
            state=self.state,
            summary=ASSERT if exited else None,
            exit="exit code 3, aborted" if exited else None,
            exited_at=(since - self.ago if self.ago else utcnow()) if exited else None,
            details={"state": self.state, "hints": ["A tensor shape mismatch"]},
        )
