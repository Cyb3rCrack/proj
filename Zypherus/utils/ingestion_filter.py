"""Ingestion filter to prevent polluted content from entering memory.

The ingestion filter decides: "Is this text knowledge, or just conversation?"

BLOCKS (interaction artifacts & operational noise):
  - Assistant chatter: "I'm happy to help", "Please provide"
  - Assistant self-reference: "as an AI", "I cannot", "I can help with"
  - System/UI language: "Error:", "Traceback", "Unknown command"
  - Instructions instead of facts: "You should", "Do this by", "Paste the text"

ALLOWS (knowledge):
  - Facts: "Python is a high-level programming language."
  - Definitions: "Programming refers to the process of..."
  - Explanations: "Garbage collection automatically frees memory..."
  - Relationships: "Objects are instances of classes."
  - Declarative, third-person, impersonal statements

Mental model: Models reason. Memory teaches. If memory lies, reasoning collapses.
The ingestion filter decides what your AI is allowed to believe.
"""

import re
from typing import Optional


class IngestionFilter:
	"""Filters out non-knowledge content before embedding.
	
	Runs BEFORE embeddings are created.
	Decides: Is this text knowledge, or just conversation?
	"""
	
	# === PATTERNS THAT INDICATE NON-KNOWLEDGE ===
	
	# Assistant chatter & politeness phrases
	ASSISTANT_CHATTER = [
		r"i'?m happy to help",
		r"i'?m here to help",
		r"i can help",
		r"i'?ll be happy to",
		r"feel free to",
		r"let me (?:help|explain|show|clarify)",
		r"i'?ll (?:help|explain|show|clarify)",
		r"no problem",
		r"sure thing",
		r"it seems like there is no",
		r"it seems like you",
	]
	
	# Requests for input / missing data
	INPUT_REQUESTS = [
		r"you didn'?t provide",
		r"please provide",
		r"please (?:share|paste|enter|type)",
		r"you need to (?:provide|share|paste)",
		r"without (?:more|the|your)",
		r"i (?:need|require|need to know)",
		r"provide the (?:original|text)",
		r"what went wrong",
		r"can you (?:give|tell|show|explain)",
	]
	
	# Assistant self-reference (not knowledge)
	ASSISTANT_SELF_REF = [
		r"as an ai",
		r"as a language model",
		r"i cannot",
		r"i am (?:not able|unable)",
		r"i'?m not sure",
		r"i don'?t (?:have|know)",
		r"i am designed to",
		r"my training",
	]
	
	# System/UI language (operational noise)
	SYSTEM_LANGUAGE = [
		r"error:",
		r"exception:",
		r"warning:",
		r"traceback",
		r"unknown command",
		r"invalid",
		r"failed",
		r"not found",
		r"can'?t find",
		r"does not exist",
	]
	
	# Instructions instead of facts
	INSTRUCTIONS = [
		r"^you should",
		r"^you can",
		r"do this by",
		r"^paste",
		r"^type",
		r"^click",
		r"^use (?:this|that|the)",
		r"^follow (?:these|the)",
		r"^(?:first|next|then|finally)",  # Sequential instructions
	]
	
	# Meta-language and self-reference (not knowledge)
	META_LANGUAGE = [
		r"here'?s? how",
		r"here'?s? what",
		r"for (?:example|instance)",  # Sometimes valid, but often intro to non-knowledge
		r"according to me",
		r"based on my",
		r"from my perspective",
	]
	
	# Combine all reject patterns
	ALL_REJECT_PATTERNS = (
		ASSISTANT_CHATTER + 
		INPUT_REQUESTS + 
		ASSISTANT_SELF_REF + 
		SYSTEM_LANGUAGE + 
		INSTRUCTIONS + 
		META_LANGUAGE
	)
	
	# Conversational pronouns at start (reject unless explicitly knowledge)
	CONVERSATIONAL_STARTS = [
		r"^i (?:am|can|will|would|should|think|believe|feel|don'?t|didn'?t)",
		r"^you (?:are|can|should|need|want|have|didn'?t)",
		r"^we (?:are|can|should|need)",
		r"^let'?s ",
		r"^my ",
		r"^me ",
	]
	
	# Minimum quality thresholds
	MIN_LENGTH = 10  # Characters (allow short factual inputs)
	MAX_LENGTH = 50000  # Characters (sanity limit)
	MIN_WORDS = 2  # Allow short definitions
	
	@staticmethod
	def is_valid_for_ingestion(text: str, strict: bool = True) -> tuple[bool, Optional[str]]:
		"""
		Check if text is valid for ingestion (knowledge, not noise).
		
		Args:
			text: Text to validate
			strict: If True, apply stricter checks (reject conversational starts)
		
		Returns:
			(is_valid, rejection_reason)
		"""
		if not text or not isinstance(text, str):
			return False, "Empty or non-string content"
		
		text_lower = text.lower().strip()
		
		# === BASIC QUALITY CHECKS ===
		
		# Length checks
		if len(text) < IngestionFilter.MIN_LENGTH:
			return False, f"Too short (< {IngestionFilter.MIN_LENGTH} chars)"
		
		if len(text) > IngestionFilter.MAX_LENGTH:
			return False, f"Too long (> {IngestionFilter.MAX_LENGTH} chars)"
		
		# Word count
		words = text.split()
		if len(words) < IngestionFilter.MIN_WORDS:
			return False, f"Too few words (< {IngestionFilter.MIN_WORDS})"
		
		# === REJECT PATTERNS ===
		
		# Check ALL reject patterns (comprehensive block)
		for pattern in IngestionFilter.ALL_REJECT_PATTERNS:
			if re.search(pattern, text_lower, re.IGNORECASE):
				return False, f"Matched reject pattern: {pattern}"
		
		# Check conversational starts (stricter mode)
		if strict:
			for pattern in IngestionFilter.CONVERSATIONAL_STARTS:
				if re.match(pattern, text_lower, re.IGNORECASE):
					return False, f"Conversational start: {pattern}"
		
		# If we got here, it's knowledge
		return True, None
	
	@staticmethod
	def clean_text(text: str) -> str:
		"""Clean text before ingestion (normalize whitespace, remove junk)."""
		if not text:
			return ""
		
		# Normalize whitespace
		text = re.sub(r'\s+', ' ', text)
		
		# Remove leading/trailing whitespace
		text = text.strip()
		
		# Remove multiple punctuation (don't convert "..." to "..")
		text = re.sub(r'([.!?]){3,}', r'\1\1', text)
		
		return text
	
	@staticmethod
	def classify_content_type(text: str) -> str:
		"""
		Classify content as definition, fact, or evidence.
		
		Returns: "definition", "fact", or "evidence"
		
		Priority order:
		1. Fact: Specific claims with dates, numbers, locations
		2. Definition: Statements of "what something is"
		3. Evidence: General explanations (default)
		"""
		text_lower = text.lower().strip()
		
		# === FACT PATTERNS (CHECK FIRST) ===
		# Specific claims: numbers, dates, places, discoveries, locations
		fact_patterns = [
			r'\d{4}',  # Year
			r'\d+\s*(?:million|billion|thousand|km|miles|meters|degrees)',  # Quantities
			r'(?:invented|created|discovered|founded|established|born|died)',
			r'located (?:in|at|near)',
			r'capital (?:of|city)',
			r'composed (?:of|from)',
			r'consists of',
			r'located in',
		]
		
		for pattern in fact_patterns:
			if re.search(pattern, text_lower):
				return "fact"
		
		# === DEFINITION PATTERNS ===
		# "X is/are/refers to/means/denotes..."
		definition_patterns = [
			r'^(?:a |an |the )?[\w\s]+ (?:is|are|refers to|means|represents|denotes)',
			r'^[\w\s]+ can be defined as',
			r'^definition:',
			r'^(?:the )?[\w\s]+ is (?:a |an |the )?[\w\s]+ that',
		]
		
		for pattern in definition_patterns:
			if re.search(pattern, text_lower):
				return "definition"
		
		# === DEFAULT TO EVIDENCE ===
		# General explanations, mechanisms, relationships
		return "evidence"


__all__ = ["IngestionFilter"]
