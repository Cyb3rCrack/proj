"""Minimal smoke test for ACE packaging/import integrity.

This is intentionally tiny: it ensures the modular architecture imports and
constructs without pulling in the old façade.
"""

from __future__ import annotations

import os
import sys


def run() -> None:
	try:
		from ace.core.ace import ACE
	except ModuleNotFoundError:
		# Allows running as a script: `python ace/selftest.py` from the repo root.
		repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		if repo_root not in sys.path:
			sys.path.insert(0, repo_root)
		from ace.core.ace import ACE

	ace = ACE()
	print("ACE self-test OK")

	# Prevent lints from treating this as unused in some configurations.
	_ = ace


if __name__ == "__main__":
	run()
