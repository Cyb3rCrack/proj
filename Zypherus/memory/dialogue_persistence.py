"""Dialogue history persistence across sessions."""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger("ZYPHERUS.DialoguePersistence")


@dataclass
class DialogueTurn:
	"""Single turn in a dialogue."""
	timestamp: str
	role: str  # "user" or "assistant"
	text: str
	metadata: Optional[Dict[str, Any]] = None
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary."""
		return {
			"timestamp": self.timestamp,
			"role": self.role,
			"text": self.text,
			"metadata": self.metadata or {}
		}


@dataclass
class DialogueSession:
	"""A complete dialogue session."""
	session_id: str
	start_time: str
	end_time: Optional[str] = None
	turns: Optional[List[DialogueTurn]] = None
	summary: Optional[str] = None
	
	def __post_init__(self):
		"""Initialize turns list if not provided."""
		if self.turns is None:
			self.turns = []
	
	def add_turn(self, role: str, text: str, metadata: Optional[Dict[str, Any]] = None):
		"""Add a turn to the session."""
		turn = DialogueTurn(
			timestamp=datetime.now().isoformat(),
			role=role,
			text=text,
			metadata=metadata
		)
		if self.turns is None:
			self.turns = []
		self.turns.append(turn)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary."""
		turns_list = self.turns if self.turns is not None else []
		return {
			"session_id": self.session_id,
			"start_time": self.start_time,
			"end_time": self.end_time,
			"turns": [t.to_dict() for t in turns_list],
			"summary": self.summary,
			"turn_count": len(turns_list)
		}


class DialoguePersistence:
	"""Manages dialogue history persistence."""
	
	def __init__(self, storage_dir: Path = Path("data/dialogues")):
		"""Initialize dialogue persistence.
		
		Args:
			storage_dir: Directory to store dialogue files
		"""
		self.storage_dir = Path(storage_dir)
		self.storage_dir.mkdir(parents=True, exist_ok=True)
		self.current_session: Optional[DialogueSession] = None
	
	def start_session(self) -> str:
		"""Start a new dialogue session.
		
		Returns:
			Session ID
		"""
		import uuid
		session_id = str(uuid.uuid4())[:8]
		self.current_session = DialogueSession(
			session_id=session_id,
			start_time=datetime.now().isoformat()
		)
		logger.debug(f"Started dialogue session: {session_id}")
		return session_id
	
	def add_user_turn(self, text: str, metadata: Optional[Dict[str, Any]] = None):
		"""Add user message to current session."""
		if self.current_session is None:
			self.start_session()
		if self.current_session is not None:
			self.current_session.add_turn("user", text, metadata)
	
	def add_assistant_turn(self, text: str, metadata: Optional[Dict[str, Any]] = None):
		"""Add assistant response to current session."""
		if self.current_session is None:
			self.start_session()
		if self.current_session is not None:
			self.current_session.add_turn("assistant", text, metadata)
	
	def end_session(self, summary: Optional[str] = None):
		"""End current dialogue session and save.
		
		Args:
			summary: Optional session summary/notes
		"""
		if self.current_session is None:
			return
		
		self.current_session.end_time = datetime.now().isoformat()
		self.current_session.summary = summary
		self.save_session(self.current_session)
		logger.debug(f"Ended dialogue session: {self.current_session.session_id}")
		self.current_session = None
	
	def save_session(self, session: DialogueSession):
		"""Save session to file.
		
		Args:
			session: Session to save
		"""
		try:
			filename = f"session_{session.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
			filepath = self.storage_dir / filename
			
			with open(filepath, 'w') as f:
				json.dump(session.to_dict(), f, indent=2)
			
			logger.info(f"Saved dialogue session: {filepath}")
		except Exception as e:
			logger.error(f"Failed to save dialogue session: {e}")
	
	def load_session(self, session_id: str) -> Optional[DialogueSession]:
		"""Load session by ID.
		
		Args:
			session_id: Session ID to load
			
		Returns:
			Loaded session or None if not found
		"""
		try:
			# Find session file
			for filepath in self.storage_dir.glob(f"session_{session_id}_*.json"):
				with open(filepath, 'r') as f:
					data = json.load(f)
				
				session = DialogueSession(
					session_id=data["session_id"],
					start_time=data["start_time"],
					end_time=data.get("end_time"),
					summary=data.get("summary")
				)
				
				# Reconstruct turns
				if session.turns is None:
					session.turns = []
				for turn_data in data.get("turns", []):
					turn = DialogueTurn(
						timestamp=turn_data["timestamp"],
						role=turn_data["role"],
						text=turn_data["text"],
						metadata=turn_data.get("metadata")
					)
					if session.turns is not None:
						session.turns.append(turn)
				
				return session
		
		except Exception as e:
			logger.error(f"Failed to load dialogue session {session_id}: {e}")
		
		return None
	
	def list_sessions(self) -> List[Dict[str, Any]]:
		"""List all saved sessions.
		
		Returns:
			List of session metadata
		"""
		sessions = []
		try:
			for filepath in sorted(self.storage_dir.glob("session_*.json")):
				with open(filepath, 'r') as f:
					data = json.load(f)
				sessions.append({
					"session_id": data["session_id"],
					"start_time": data["start_time"],
					"end_time": data.get("end_time"),
					"turn_count": data.get("turn_count", 0),
					"summary": data.get("summary"),
					"file": filepath.name
				})
		except Exception as e:
			logger.error(f"Failed to list sessions: {e}")
		
		return sessions
	
	def export_session_csv(self, session_id: str) -> Optional[str]:
		"""Export session to CSV format.
		
		Args:
			session_id: Session to export
			
		Returns:
			CSV content or None if failed
		"""
		try:
			session = self.load_session(session_id)
			if not session:
				return None
			if session.turns is None:
				return None
			
			lines = ["timestamp,role,text"]
			for turn in session.turns:
				# Escape quotes and newlines
				text = turn.text.replace('"', '""').replace('\n', ' ')
				lines.append(f'{turn.timestamp},"{turn.role}","{text}"')
			
			return "\n".join(lines)
		
		except Exception as e:
			logger.error(f"Failed to export session {session_id}: {e}")
			return None
