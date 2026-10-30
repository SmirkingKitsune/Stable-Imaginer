"""Engine adapters behind the ``LLMBackend`` / ``ImageBackend`` protocols.

Every module in this package must import without torch installed; heavy dependencies are
optional extras imported lazily inside the adapter that needs them.
"""
