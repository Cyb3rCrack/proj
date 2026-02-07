"""Configuration validation and schema checking."""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("ZYPHERUS.Config")


@dataclass
class ConfigValidationError:
	"""Represents a configuration validation error."""
	field: str
	message: str
	severity: str = "error"  # "error", "warning"


class ConfigValidator:
	"""Validates ACE configuration against expected schema."""
	
	# Expected configuration schema
	SCHEMA = {
		"model": {
			"type": "object",
			"required": ["name", "backend"],
			"properties": {
				"name": {"type": "string"},
				"backend": {"type": "string", "enum": ["ollama", "openai"]},
				"temperature": {"type": "number", "min": 0, "max": 2},
				"max_tokens": {"type": "integer", "min": 1},
			}
		},
		"memory": {
			"type": "object",
			"properties": {
				"max_entries": {"type": "integer", "min": 100},
				"cache_size": {"type": "integer", "min": 1},
				"cleanup_threshold_mb": {"type": "integer", "min": 100},
			}
		},
		"ingestion": {
			"type": "object",
			"properties": {
				"mode": {"type": "string", "enum": ["safe", "max"]},
				"batch_size": {"type": "integer", "min": 1},
				"chunk_size": {"type": "integer", "min": 100},
			}
		},
		"logging": {
			"type": "object",
			"properties": {
				"level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
				"file": {"type": "string"},
			}
		},
	}
	
	def __init__(self):
		"""Initialize validator."""
		self.errors: List[ConfigValidationError] = []
	
	def validate(self, config: Dict[str, Any]) -> bool:
		"""Validate configuration dictionary.
		
		Args:
			config: Configuration dictionary to validate
			
		Returns:
			True if valid, False if errors found
		"""
		self.errors = []
		
		for section, schema in self.SCHEMA.items():
			if section not in config:
				self.errors.append(ConfigValidationError(
					field=section,
					message=f"Missing required section: {section}",
					severity="warning"
				))
				continue
			
			self._validate_section(section, config[section], schema)
		
		return len([e for e in self.errors if e.severity == "error"]) == 0
	
	def _validate_section(self, section: str, data: Dict[str, Any], schema: Dict[str, Any]):
		"""Validate a configuration section."""
		for field, field_schema in schema.get("properties", {}).items():
			if field not in data:
				if field in schema.get("required", []):
					self.errors.append(ConfigValidationError(
						field=f"{section}.{field}",
						message=f"Missing required field: {field}"
					))
				continue
			
			value = data[field]
			
			# Type checking
			if "type" in field_schema:
				if field_schema["type"] == "object" and not isinstance(value, dict):
					self.errors.append(ConfigValidationError(
						field=f"{section}.{field}",
						message=f"Expected object, got {type(value).__name__}"
					))
				elif field_schema["type"] == "integer" and not isinstance(value, int):
					self.errors.append(ConfigValidationError(
						field=f"{section}.{field}",
						message=f"Expected integer, got {type(value).__name__}"
					))
				elif field_schema["type"] == "number" and not isinstance(value, (int, float)):
					self.errors.append(ConfigValidationError(
						field=f"{section}.{field}",
						message=f"Expected number, got {type(value).__name__}"
					))
			
			# Enum validation
			if "enum" in field_schema and value not in field_schema["enum"]:
				self.errors.append(ConfigValidationError(
					field=f"{section}.{field}",
					message=f"Invalid value '{value}'. Must be one of: {field_schema['enum']}",
					severity="warning"
				))
			
			# Range validation
			if "min" in field_schema and isinstance(value, (int, float)):
				if value < field_schema["min"]:
					self.errors.append(ConfigValidationError(
						field=f"{section}.{field}",
						message=f"Value {value} is below minimum {field_schema['min']}",
						severity="warning"
					))
	
	def get_error_report(self) -> str:
		"""Get formatted error report."""
		if not self.errors:
			return "Configuration is valid."
		
		errors = [e for e in self.errors if e.severity == "error"]
		warnings = [e for e in self.errors if e.severity == "warning"]
		
		report = []
		
		if errors:
			report.append(f"❌ {len(errors)} configuration errors found:")
			for error in errors:
				report.append(f"   - {error.field}: {error.message}")
		
		if warnings:
			report.append(f"⚠️  {len(warnings)} configuration warnings:")
			for warning in warnings:
				report.append(f"   - {warning.field}: {warning.message}")
		
		return "\n".join(report)


def validate_config_file(config_path: Path) -> bool:
	"""Validate configuration file on startup.
	
	Args:
		config_path: Path to configuration file
		
	Returns:
		True if valid, False if errors found
	"""
	try:
		with open(config_path, 'r') as f:
			config = json.load(f)
		
		validator = ConfigValidator()
		is_valid = validator.validate(config)
		
		if is_valid:
			logger.info(f"✓ Configuration validated successfully: {config_path}")
		else:
			logger.warning(f"Configuration issues found:\n{validator.get_error_report()}")
		
		return is_valid
	
	except FileNotFoundError:
		logger.warning(f"Configuration file not found: {config_path}")
		return True  # Optional file
	except json.JSONDecodeError as e:
		logger.error(f"Configuration file is not valid JSON: {e}")
		return False
	except Exception as e:
		logger.error(f"Error validating configuration: {e}")
		return False
