"""Ephemeral dialogue manager."""

from __future__ import annotations


class DialogueManager:
	"""Simple conversation history manager for a single session."""

	def __init__(self, max_turns: int = 8):
		self.max_turns = max_turns
		self.history = []  # list of (role, text)

	def add_user(self, text: str) -> None:
		self.history.append(("user", text))
		self._trim()

	def add_assistant(self, text: str) -> None:
		self.history.append(("assistant", text))
		self._trim()

	def _trim(self) -> None:
		if len(self.history) > self.max_turns * 2:
			self.history = self.history[-self.max_turns * 2 :]

	def get_formatted(self) -> str:
		parts = []
		for role, text in self.history:
			if role == "user":
				parts.append(f"User: {text}")
			else:
				parts.append(f"Assistant: {text}")
		return "\n".join(parts)


__all__ = ["DialogueManager"]

