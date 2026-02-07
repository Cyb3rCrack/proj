"""Zypherus package.

Keep this module lightweight.

During migration, many submodules still wrap the legacy monolith in aimaker.py.
Importing Zypherus itself can trigger heavy optional dependencies, so we only load it
on demand.
"""


def __getattr__(name: str):
	if name == "Zypherus":
		from .Zypherus.core.ace import ACE
		return ACE
	raise AttributeError(name)
