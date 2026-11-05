"""What each prompt mode tells the chat model about the user's latest message.

``chat`` adds nothing, and ``raw`` never reaches the chat model: the turn runner sends the
message to an image tool itself. ``interpret`` is also enforced: its first round must call a
tool (``tool_choice: required``).
"""

from imaginer.domain.enums import PromptMode

MODE_GUIDANCE: dict[PromptMode, str] = {
    PromptMode.INTERPRET: (
        "Mode for the user's latest message: interpreter. Treat it as a description of an "
        "image to make, or an edit when it names or attaches images. Call the image tool "
        "right away with a prompt that says exactly what the user asked for in clear, "
        "concrete English, adding only what the image model needs to get it right. Don't "
        "invent subjects, styles or details they didn't ask for. Afterwards, reply in a "
        "sentence or two."
    ),
    PromptMode.CURATE: (
        "Mode for the user's latest message: curator. Make the user's idea into the best "
        "image you can. Keep everything they asked for and enrich the prompt with fitting, "
        "concrete detail: subject, setting, composition, lighting, colour, style and mood. "
        "If the request is too vague to choose a main subject, or contradicts itself, don't "
        "draw yet: ask up to three short questions and suggest a default they can simply "
        "accept. Otherwise call the image tool right away, then say in a sentence or two "
        "what you added."
    ),
}
