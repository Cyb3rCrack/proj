"""Claim parsing.

Preserves predicate token + tense and tries to retain modality/negation.
"""

from __future__ import annotations

import re
from typing import Any, Dict


def parse_claim(text: str) -> Dict[str, Any]:
	raw = (text or "").strip()
	s = raw

	# basic modality cues
	modality = "certain"
	if re.search(r"\b(may|might|could|possibly|likely|probably)\b", s, flags=re.IGNORECASE):
		modality = "uncertain"

	# polarity/negation
	polarity = 1
	if re.search(r"\b(not|no|never)\b", s, flags=re.IGNORECASE):
		polarity = -1

	# common connectors
	connectors = [
		" is ", " are ", " was ", " were ",
		" has ", " have ", " had ",
		" means ", " mean ", " refers to ",
		" includes ", " include ", " contains ", " contain ",
		" causes ", " cause ", " leads to ", " lead to ",
	]

	predicate = "related_to"
	predicate_token = "related_to"
	predicate_tense = "present"
	subj = ""
	obj = ""

	lowered = s.lower()
	split_at = None
	split_conn = None
	for c in connectors:
		idx = lowered.find(c)
		if idx != -1:
			split_at = idx
			split_conn = c.strip()
			break

	if split_at is not None and split_conn is not None:
		subj = s[:split_at].strip(" ,;:-")
		obj = s[split_at + len(split_conn) + 2 :].strip(" ,;:-") if split_conn not in {"refers to", "leads to", "lead to"} else s[split_at + len(split_conn) + 2 :].strip(" ,;:-")
		predicate_token = split_conn
		predicate = split_conn.replace(" ", "_")
		if split_conn in {"was", "were", "had"}:
			predicate_tense = "past"
		elif split_conn in {"is", "are", "has", "have", "means", "mean", "refers to", "includes", "include", "contains", "contain", "causes", "cause", "leads to", "lead to"}:
			predicate_tense = "present"

	# clean leading negation tokens from object if we already captured polarity
	if obj:
		obj2 = re.sub(r"^(not|no)\b\s+", "", obj, flags=re.IGNORECASE).strip()
		obj = obj2

	return {
		"subject": subj,
		"predicate": predicate,
		"predicate_token": predicate_token,
		"predicate_tense": predicate_tense,
		"object": obj,
		"polarity": polarity,
		"modality": modality,
		"raw": raw,
	}


__all__ = ["parse_claim"]

