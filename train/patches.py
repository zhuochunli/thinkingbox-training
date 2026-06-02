"""Monkey-patches applied before importing thinkingbox runners.

Qwen3.5's chat template enforces a single system message at index 0 and raises
``System message must be at the beginning`` otherwise. Thinkingbox's
``AgentSession`` builds a ``prefix_conversation`` of ``[msg_system, msg_bot]``
whenever ``bot_instructions`` is non-empty, which fails this check both at
vLLM-rollout time and at our local tokenization-render time.

We merge them into a single system message at the boundary by wrapping
``AgentSession.__init__`` to post-process ``prefix_conversation`` and re-seed
``self.conversation`` / ``self.llm`` accordingly.

Import this module once, early, before any rollout begins.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def _merge_leading_systems(messages: list) -> list:
    """Collapse a leading run of role=='system' messages into one."""
    if not messages:
        return messages
    head = []
    i = 0
    while i < len(messages) and getattr(messages[i], "role", None) == "system":
        head.append(messages[i])
        i += 1
    if len(head) <= 1:
        return messages
    # Concatenate string contents with a blank line; non-string contents are
    # left as-is on the first system (rare for our flows).
    first = head[0]
    parts = []
    for m in head:
        c = m.content
        if isinstance(c, str) and c:
            parts.append(c)
    merged = first.model_copy(update={"content": "\n\n".join(parts)})
    return [merged] + list(messages[i:])


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from thinkingbox.common.agent_session import AgentSession

    _orig_init = AgentSession.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        # Merge prefix_conversation systems
        new_msgs = _merge_leading_systems(list(self.prefix_conversation.messages))
        if len(new_msgs) != len(self.prefix_conversation.messages):
            logger.debug("merged %d leading system messages into 1",
                         len(self.prefix_conversation.messages) - len(new_msgs) + 1)
            self.prefix_conversation.messages = new_msgs
            # Re-seed the live conversation + llm state so the very first turn
            # already uses the merged prefix.
            self._reset_conversation()

    AgentSession.__init__ = _patched_init
    _PATCHED = True
    logger.info("AgentSession.__init__ patched: leading system messages will be merged")
