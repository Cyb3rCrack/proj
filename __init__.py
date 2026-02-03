"""ACE package.

Keep this module lightweight.

During migration, many submodules still wrap the legacy monolith in aimaker.py.
Importing ACE itself can trigger heavy optional dependencies, so we only load it
on demand.
"""

__all__ = ["ACE"]


def __getattr__(name: str):
	if name == "ACE":
		from .core.ace import ACE
		return ACE
	raise AttributeError(name)
