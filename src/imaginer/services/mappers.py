"""ORM row -> DTO conversions."""

from imaginer.db import models as orm
from imaginer.db.repositories.artifacts import ImageSummary
from imaginer.domain import models as dto


def conversation(row: orm.Conversation) -> dto.Conversation:
    return dto.Conversation(
        id=row.id,
        title=row.title,
        created_at=row.created_at,
        updated_at=row.updated_at,
        archived_at=row.archived_at,
        metadata=row.metadata_json,
    )


def conversation_summary(
    row: orm.Conversation, images: ImageSummary | None
) -> dto.ConversationSummary:
    return dto.ConversationSummary(
        **conversation(row).model_dump(),
        image_count=images.images if images else 0,
        cover=artifact(images.latest) if images else None,
    )


def message(row: orm.Message, attachments: list[str] | None = None) -> dto.Message:
    return dto.Message(
        id=row.id,
        conversation_id=row.conversation_id,
        parent_id=row.parent_id,
        seq=row.seq,
        role=row.role,
        status=row.status,
        content_text=row.content_text,
        reasoning_text=row.reasoning_text,
        tool_calls=row.tool_calls_json,
        tool_call_id=row.tool_call_id,
        backend=row.backend,
        model=row.model,
        attachments=attachments or [],
        metadata=row.metadata_json,
        created_at=row.created_at,
    )


def artifact(row: orm.Artifact) -> dto.Artifact:
    return dto.Artifact(
        id=row.id,
        conversation_id=row.conversation_id,
        message_id=row.message_id,
        kind=row.kind,
        source=row.source,
        label=row.label,
        mime=row.mime,
        sha256=row.sha256,
        size_bytes=row.size_bytes,
        width=row.width,
        height=row.height,
        has_alpha=row.has_alpha,
        duration_ms=row.duration_ms,
        metadata=row.metadata_json,
        created_at=row.created_at,
    )


def job(row: orm.GenerationJob, outputs: list[orm.JobOutput] | None = None) -> dto.Job:
    return dto.Job(
        id=row.id,
        conversation_id=row.conversation_id,
        request_message_id=row.request_message_id,
        tool_call_id=row.tool_call_id,
        task=row.task,
        alpha=row.alpha,
        backend=row.backend,
        model=row.model,
        params=row.params_json,
        seed=row.seed,
        status=row.status,
        error=row.error,
        engine_job_id=row.engine_job_id,
        progress=row.progress_json,
        outputs=[
            dto.JobOutputRef(artifact_id=o.artifact_id, ordinal=o.ordinal, seed=o.seed)
            for o in outputs or []
        ],
        queued_at=row.queued_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )
