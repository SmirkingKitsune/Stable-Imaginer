"""Query functions over an ``AsyncSession``. Callers own the transaction.

Every read of conversation-scoped data excludes tombstoned conversations, and artifact
labels are only ever resolved within one conversation.
"""
