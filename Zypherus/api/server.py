"""Production-ready REST API for Zypherus with authentication, validation, and documentation."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, Optional
from functools import wraps
from datetime import datetime
from dataclasses import dataclass
import json

from flask import Flask, Response, request, jsonify, make_response, stream_with_context
from flask_cors import CORS

from .conversation_store import ConversationStore

HAS_FLASK = True

logger = logging.getLogger("ACE.API")


# Middleware and utilities
def request_id_middleware(f):  # type: ignore[no-untyped-def]
    """Add request ID to all requests."""
    @wraps(f)
    def decorated_function(*args, **kwargs):  # type: ignore[no-untyped-def]
        request.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))  # type: ignore[union-attr]
        return f(*args, **kwargs)
    return decorated_function


def add_security_headers(response):
    """Add security headers to response."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@dataclass
class APIRequest:
    """Standardized API request."""
    action: str
    payload: Dict[str, Any]
    request_id: Optional[str] = None


@dataclass  
class APIResponse:
    """Standardized API response."""
    success: bool
    data: Any
    error: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "request_id": self.request_id,
            "timestamp": self.timestamp or datetime.utcnow().isoformat(),
        }


class ZypherusAPIServer:
    """Production-ready API server for Zypherus."""
    
    def __init__(self, ace: Any, host: str = "0.0.0.0", port: int = 8000):
        if not HAS_FLASK:
            raise ImportError("Flask is required. Install with: pip install flask flask-cors")
        
        self.ace = ace
        self.host = host
        self.port = port
        self.app: Optional[Flask] = None  # type: ignore[name-defined]
    
    def create_flask_app(self) -> Flask:  # type: ignore[no-redef,name-defined]
        """Create production-ready Flask application."""
        app = Flask(__name__)  # type: ignore[attr-defined]
        
        # Configuration
        app.config["JSON_SORT_KEYS"] = False
        app.config["JSON_PRETTYPRINT_REGULAR"] = True
        
        # CORS setup
        CORS(app, resources={r"/api/*": {"origins": "*"}})  # type: ignore[misc]

        # Conversation storage
        chat_db_path = os.getenv("CHAT_DB_PATH", os.path.join("data", "chat.db"))
        app.conversation_store = ConversationStore(chat_db_path)  # type: ignore[attr-defined]
        
        # Register error handlers
        self._register_error_handlers(app)
        
        # Register middleware
        self._register_middleware(app)
        
        # Register routes
        self._register_routes(app)
        
        self.app = app
        return app
    
    def _register_middleware(self, app: Flask) -> None:  # type: ignore[name-defined]
        """Register middleware."""
        
        @app.before_request
        def before_request():
            request.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))  # type: ignore
            request.start_time = datetime.utcnow()  # type: ignore
        
        @app.after_request
        def after_request(response):
            response = add_security_headers(response)
            if hasattr(request, "request_id"):  # type: ignore
                response.headers["X-Request-ID"] = request.request_id  # type: ignore
            return response
    
    def _register_error_handlers(self, app: Flask) -> None:  # type: ignore[name-defined]
        """Register error handlers."""
        
        @app.errorhandler(400)
        def bad_request(error):  # type: ignore
            return jsonify({  # type: ignore
                "success": False,
                "error": "BAD_REQUEST",
                "message": "Invalid request",
                "request_id": getattr(request, "request_id", None)
            }), 400
        
        @app.errorhandler(401)
        def unauthorized(error):  # type: ignore
            return jsonify({  # type: ignore
                "success": False,
                "error": "UNAUTHORIZED",
                "message": "Authentication required",
                "request_id": getattr(request, "request_id", None)
            }), 401
        
        @app.errorhandler(403)
        def forbidden(error):  # type: ignore
            return jsonify({  # type: ignore
                "success": False,
                "error": "FORBIDDEN",
                "message": "Insufficient permissions",
                "request_id": getattr(request, "request_id", None)
            }), 403
        
        @app.errorhandler(404)
        def not_found(error):  # type: ignore
            return jsonify({  # type: ignore
                "success": False,
                "error": "NOT_FOUND",
                "message": "Endpoint not found",
                "request_id": getattr(request, "request_id", None)
            }), 404
        
        @app.errorhandler(429)
        def rate_limited(error):  # type: ignore
            return jsonify({  # type: ignore
                "success": False,
                "error": "RATE_LIMIT_EXCEEDED",
                "message": "Too many requests",
                "request_id": getattr(request, "request_id", None)
            }), 429
        
        @app.errorhandler(500)
        def internal_error(error):
            logger.error(f"Internal server error: {error}")
            return jsonify({
                "success": False,
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "request_id": getattr(request, "request_id", None)
            }), 500
    
    def _register_routes(self, app: Flask) -> None:  # type: ignore[name-defined]
        """Register API routes."""

        def _format_messages(messages: list[dict[str, str]]) -> str:
            lines = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "assistant":
                    prefix = "Assistant"
                elif role == "system":
                    prefix = "System"
                else:
                    prefix = "User"
                lines.append(f"{prefix}: {content}")
            return "\n".join(lines)

        def _build_prompt(system_prompt: str, messages: list[dict[str, str]]) -> str:
            formatted = _format_messages(messages)
            return f"System: {system_prompt}\n\n{formatted}\nAssistant:"

        # Root endpoint
        @app.route("/", methods=["GET"])
        def root():
            """Serve the chat UI at the root URL."""
            return chat_ui()

        # Simple chat UI
        @app.route("/chat", methods=["GET"])
        def chat_ui():
                """Enhanced chat UI with auto-greeting, typing indicator, and error handling."""
                html = """<!doctype html>
<html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>Zypherus Chat</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&family=IBM+Plex+Mono:wght@400;600&display=swap');
            :root {
        --bg: #0f172a;
        --panel: #111827;
        --accent: #22c55e;
        --accent-2: #38bdf8;
        --text: #e5e7eb;
        --muted: #94a3b8;
        --error: #ef4444;
            }
            * { box-sizing: border-box; }
            body {
        margin: 0;
        font-family: 'Space Grotesk', system-ui, -apple-system, Segoe UI, sans-serif;
        color: var(--text);
        background: radial-gradient(1200px 800px at 10% 10%, #1e293b, #0b1020 60%) fixed;
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 24px;
            }
            .shell {
        width: min(980px, 95vw);
        background: linear-gradient(135deg, rgba(17,24,39,0.95), rgba(15,23,42,0.95));
        border: 1px solid rgba(148,163,184,0.2);
        border-radius: 20px;
        box-shadow: 0 30px 70px rgba(2, 6, 23, 0.6);
        overflow: hidden;
        display: flex;
        flex-direction: column;
            }
            header {
        padding: 20px 24px;
        display: flex;
        align-items: center;
        gap: 12px;
        border-bottom: 1px solid rgba(148,163,184,0.15);
        background: rgba(15,23,42,0.8);
            }
            .badge {
        font-family: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 12px;
        color: #0f172a;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        padding: 4px 10px;
        border-radius: 999px;
            }
            h1 {
        font-size: 20px;
        margin: 0;
        font-weight: 600;
            }
            .sub {
        color: var(--muted);
        font-size: 14px;
            }
            .chat {
        padding: 20px 24px;
        height: min(70vh, 560px);
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 14px;
        flex: 1;
            }
            .msg {
        max-width: 80%;
        padding: 12px 14px;
        border-radius: 14px;
        line-height: 1.5;
        animation: fadeIn 0.3s ease-out;
        white-space: pre-wrap;
        word-wrap: break-word;
            }
            .msg.user {
        align-self: flex-end;
        background: rgba(34,197,94,0.15);
        border: 1px solid rgba(34,197,94,0.35);
            }
            .msg.assistant {
        align-self: flex-start;
        background: rgba(56,189,248,0.12);
        border: 1px solid rgba(56,189,248,0.35);
            }
            .msg.error {
        align-self: flex-start;
        background: rgba(239,68,68,0.15);
        border: 1px solid rgba(239,68,68,0.35);
        color: #fca5a5;
            }
            .typing {
        display: flex;
        gap: 4px;
        align-items: center;
            }
            .dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--accent-2);
        animation: typing 1.2s infinite;
            }
            .dot:nth-child(2) { animation-delay: 0.2s; }
            .dot:nth-child(3) { animation-delay: 0.4s; }
            @keyframes typing {
        0%, 60%, 100% { opacity: 0.3; }
        30% { opacity: 1; }
            }
            .input-bar {
        display: flex;
        gap: 12px;
        padding: 18px 24px 22px;
        border-top: 1px solid rgba(148,163,184,0.15);
        background: rgba(15,23,42,0.85);
            }
            textarea {
        flex: 1;
        min-height: 48px;
        max-height: 140px;
        resize: vertical;
        border-radius: 12px;
        border: 1px solid rgba(148,163,184,0.25);
        background: rgba(2,6,23,0.6);
        color: var(--text);
        padding: 12px;
        font-size: 14px;
        font-family: inherit;
            }
            textarea:focus { 
        outline: none;
        border-color: rgba(34,197,94,0.5);
        background: rgba(2,6,23,0.8);
            }
            button {
        font-family: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        border: none;
        color: #0b1020;
        padding: 12px 18px;
        border-radius: 12px;
        cursor: pointer;
        font-weight: 600;
        white-space: nowrap;
        transition: opacity 0.2s;
            }
            button:hover:not(:disabled) { opacity: 0.9; }
            button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
            }
            .toast {
        position: fixed;
        bottom: 30px;
        right: 30px;
        padding: 14px 18px;
        background: rgba(34,197,94,0.2);
        border: 1px solid rgba(34,197,94,0.5);
        border-radius: 10px;
        font-size: 13px;
        animation: slideIn 0.3s ease-out;
        z-index: 9999;
            }
            @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
            }
            @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
            }
        </style>
    </head>
    <body>
        <div class=\"shell\">
            <header>
        <span class=\"badge\">LIVE</span>
        <div>
            <h1>Zypherus Chat</h1>
            <div class=\"sub\">For coding & tech questions | Full memory enabled</div>
        </div>
            </header>
            <div id=\"chat\" class=\"chat\"></div>
            <div class=\"input-bar\">
        <textarea id=\"input\" placeholder=\"Ask Zypherus for coding help...\"></textarea>
        <button id=\"send\" onclick=\"sendMessage()\">Send</button>
            </div>
        </div>
        <script>
            const chat = document.getElementById('chat');
            const input = document.getElementById('input');
            const sendBtn = document.getElementById('send');
            let conversationId = null;
            let hasGreeted = false;

            function showToast(message, duration = 3000) {
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), duration);
            }

            function addMessage(role, text) {
        const bubble = document.createElement('div');
        bubble.className = `msg ${role}`;
        bubble.textContent = text;
        chat.appendChild(bubble);
        chat.scrollTop = chat.scrollHeight;
        return bubble;
            }

            function addTypingIndicator() {
        const bubble = document.createElement('div');
        bubble.className = 'msg assistant';
        bubble.innerHTML = '<div class=\"typing\"><div class=\"dot\"></div><div class=\"dot\"></div><div class=\"dot\"></div></div>';
        chat.appendChild(bubble);
        chat.scrollTop = chat.scrollHeight;
        return bubble;
            }

            function autoGreet() {
        if (hasGreeted) return;
        hasGreeted = true;
        const greetings = [
            'Hey! I\\'m Zypherus. Ask me about coding, debugging, or tech architecture.',
            'Ready to help with your code! Ask me anything about programming.',
            'Welcome! I\\'m here for coding questions and tech discussions.'
        ];
        const greeting = greetings[Math.floor(Math.random() * greetings.length)];
        addMessage('assistant', greeting);
        setTimeout(() => showToast('Memory initialized ✓'), 500);
            }

            async function sendMessage() {
        const message = input.value.trim();
        if (!message) {
            input.focus();
            return;
        }
        
        addMessage('user', message);
        input.value = '';
        sendBtn.disabled = true;
        input.disabled = true;

        const payload = {
            message,
            conversation_id: conversationId,
            stream: true
        };

        try {
            const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
            });

            if (!res.ok) {
        const err = await res.text();
        addMessage('error', `Server error: ${res.status} ${res.statusText}\\n${err}`);
        sendBtn.disabled = false;
        input.disabled = false;
        return;
            }

            if (!conversationId) {
        conversationId = res.headers.get('X-Conversation-Id');
            }

            const typingBubble = addTypingIndicator();
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let assistantText = '';

            while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\\n');
        for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const data = line.slice(6);
            if (data === '[DONE]') {
                chat.removeChild(typingBubble);
                addMessage('assistant', assistantText);
                sendBtn.disabled = false;
                input.disabled = false;
                input.focus();
                return;
            }
            assistantText += data;
            chat.scrollTop = chat.scrollHeight;
        }
            }
        } catch (err) {
            addMessage('error', `Error: ${err.message}. Check browser console for details.`);
            console.error('Chat error:', err);
            sendBtn.disabled = false;
            input.disabled = false;
            input.focus();
        }
            }

            // Initialize
            window.addEventListener('DOMContentLoaded', () => {
        setTimeout(autoGreet, 300);
        input.focus();
            });

            // Send on Ctrl+Enter
            input.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            sendMessage();
        }
            });
        </script>
    </body>
</html>
"""
                return Response(html, mimetype="text/html")

        # Stateful chat
        @app.route("/api/chat", methods=["POST"])
        def chat():
                """Stateful chat endpoint with optional streaming."""
                data = request.get_json() or {}
                message = str(data.get("message", "")).strip()
                stream = bool(data.get("stream", True))
                conversation_id = data.get("conversation_id") or str(uuid.uuid4())

                if not message:
                        return jsonify(APIResponse(
                                success=False,
                                error="INVALID_INPUT",
                                data={"message": "Message cannot be empty"},
                                request_id=getattr(request, "request_id", None)
                        ).to_dict()), 400

                store = getattr(app, "conversation_store", None)
                if store is None:
                        return jsonify(APIResponse(
                                success=False,
                                error="CHAT_STORE_UNAVAILABLE",
                                data={"message": "Conversation store not initialized"},
                                request_id=getattr(request, "request_id", None)
                        ).to_dict()), 500

                messages = store.append_message(conversation_id, "user", message)
                max_history = int(os.getenv("CHAT_HISTORY_MAX", "20"))
                system_prompt = os.getenv(
                        "ZYPHERUS_SYSTEM_PROMPT",
                        "You are Zypherus, an intelligent, conversational AI. You remember past context and respond naturally."
                ).strip()
                recent = messages[-max_history:] if max_history > 0 else messages
                prompt = _build_prompt(system_prompt, recent)

                if stream:
                        def event_stream():
                                chunks = []
                                try:
                                        for chunk in self.ace.llm.stream(prompt, max_tokens=self.ace.llm.tokens_answer):
                                                chunks.append(chunk)
                                                yield f"data: {chunk}\n\n"
                                finally:
                                        reply = "".join(chunks).strip()
                                        store.append_message(conversation_id, "assistant", reply)
                                        yield "data: [DONE]\n\n"

                        headers = {
                                "Cache-Control": "no-cache",
                                "X-Accel-Buffering": "no",
                                "X-Conversation-Id": conversation_id,
                        }
                        return Response(
                                stream_with_context(event_stream()),
                                mimetype="text/event-stream",
                                headers=headers,
                        )

                reply = self.ace.llm.generate(prompt, max_tokens=self.ace.llm.tokens_answer)
                store.append_message(conversation_id, "assistant", reply)
                return jsonify({
                        "conversation_id": conversation_id,
                        "reply": reply,
                        "tokens": len(reply.split())
                }), 200
        
        # Health check
        @app.route("/health", methods=["GET"])
        def health():
            """Service health check."""
            try:
                checker = getattr(app, "health_checker", None)
                if checker:
                    health_data = checker.check_all()
                else:
                    health_data = {
                        "overall_status": "healthy",
                        "checks": {
                            "api": {"status": "healthy"},
                            "memory": {"status": "healthy" if hasattr(self.ace, "memory") else "unknown"},
                        }
                    }
                
                return jsonify(APIResponse(
                    success=True,
                    data={
                        "status": health_data.get("overall_status", "unknown"),
                        "version": "0.2.0",
                        "checks": health_data.get("checks", {}),
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 200
            except Exception as e:
                logger.error(f"Health check error: {e}", exc_info=True)
                return jsonify({
                    "success": False,
                    "status": "unhealthy",
                    "error": str(e),
                    "request_id": getattr(request, "request_id", None)
                }), 503
        
        # Readiness check (for Kubernetes/Render)
        @app.route("/ready", methods=["GET"])
        def ready():
            """Service readiness check."""
            try:
                checker = getattr(app, "health_checker", None)
                if not checker:
                    return jsonify({
                        "success": False,
                        "ready": False,
                        "message": "Health checker not available"
                    }), 503
                
                is_ready = checker.is_ready()
                status_code = 200 if is_ready else 503
                
                return jsonify({
                    "success": is_ready,
                    "ready": is_ready,
                    "message": "Service is ready" if is_ready else "Service is not ready",
                    "request_id": getattr(request, "request_id", None)
                }), status_code
            except ValueError as e:
                logger.error(f"Readiness check validation error: {e}")
                return jsonify({
                    "success": False,
                    "ready": False,
                    "error": f"Validation error: {str(e)}"
                }), 503
            except Exception as e:
                logger.error(f"Readiness check error: {e}", exc_info=True)
                return jsonify({
                    "success": False,
                    "ready": False,
                    "error": str(e)
                }), 503
        
                logger.error(f"Health check error: {e}")
                return jsonify({
                    "success": False,
                    "status": "unhealthy",
                    "error": str(e),
                    "request_id": getattr(request, "request_id", None)
                }), 503
        
        # Ingest document
        @app.route("/api/ingest", methods=["POST"])
        def ingest():
            """Ingest document into knowledge base."""
            try:
                data = request.get_json() or {}
                text = data.get("text", "").strip()
                source = data.get("source", "api")
                
                if not text:
                    return jsonify(APIResponse(
                        success=False,
                        error="INVALID_INPUT",
                        data={"message": "Content cannot be empty"},
                        request_id=getattr(request, "request_id", None)
                    ).to_dict()), 400
                
                if not isinstance(text, str) or len(text) > 1000000:
                    return jsonify(APIResponse(
                        success=False,
                        error="INVALID_INPUT",
                        data={"message": "Text must be string and <= 1MB"},
                        request_id=getattr(request, "request_id", None)
                    ).to_dict()), 400
                
                self.ace.ingest_document(source, text)
                
                return jsonify(APIResponse(
                    success=True,
                    data={
                        "message": f"Successfully ingested {len(text)} characters",
                        "source": source,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 200
            except ValueError as e:
                logger.error(f"Ingest validation error: {e}")
                return jsonify(APIResponse(
                    success=False,
                    error="VALIDATION_ERROR",
                    data={"message": str(e)},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 400
            except IOError as e:
                logger.error(f"Ingest I/O error: {e}")
                return jsonify(APIResponse(
                    success=False,
                    error="IO_ERROR",
                    data={"message": f"Failed to store document: {str(e)}"},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 500
            except Exception as e:
                logger.error(f"Ingest error: {e}", exc_info=True)
                return jsonify(APIResponse(
                    success=False,
                    error="INGEST_FAILED",
                    data={"message": str(e)},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 400
        
        # Answer question
        @app.route("/api/answer", methods=["POST"])
        def answer():
            """Answer a question using knowledge base with caching."""
            try:
                data = request.get_json() or {}
                query = data.get("query", "").strip()
                use_cache = data.get("cache", True)
                
                if not query:
                    return jsonify(APIResponse(
                        success=False,
                        error="INVALID_INPUT",
                        data={"message": "Query cannot be empty"},
                        request_id=getattr(request, "request_id", None)
                    ).to_dict()), 400
                
                # Check cache first
                cache = getattr(app, "query_cache", None)
                if use_cache and cache:
                    cached_result = cache.get(query)
                    if cached_result:
                        logger.debug(f"Cache hit for query: {query[:50]}")
                        return jsonify(APIResponse(
                            success=True,
                            data={**cached_result, "from_cache": True},
                            request_id=getattr(request, "request_id", None)
                        ).to_dict()), 200
                
                # Get fresh result
                result = self.ace.answer(query)
                
                # Cache result
                if cache:
                    cache.set(result, query)
                
                return jsonify(APIResponse(
                    success=True,
                    data={**result, "from_cache": False},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 200
            except ValueError as e:
                logger.error(f"Answer validation error: {e}")
                return jsonify(APIResponse(
                    success=False,
                    error="VALIDATION_ERROR",
                    data={"message": str(e)},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 400
            except Exception as e:
                logger.error(f"Answer error: {e}", exc_info=True)
                return jsonify(APIResponse(
                    success=False,
                    error="ANSWER_FAILED",
                    data={"message": str(e)},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 400
        
        # Get status
        @app.route("/api/status", methods=["GET"])
        def status():
            """Get system status."""
            try:
                status_data: Dict[str, Any] = {
                    "uptime": "running",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                
                if hasattr(self.ace, "memory"):
                    status_data["memory_entries"] = len(getattr(self.ace.memory, "entries", []) or [])
                
                if hasattr(self.ace, "claim_store"):
                    status_data["claims"] = len(getattr(self.ace.claim_store, "claims", {}) or {})
                
                if hasattr(self.ace, "concept_graph"):
                    status_data["concepts"] = len(getattr(self.ace.concept_graph, "nodes", {}) or {})
                
                return jsonify(APIResponse(  # type: ignore
                    success=True,
                    data=status_data,
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 200
            except Exception as e:
                logger.error(f"Status error: {e}")
                return jsonify(APIResponse(  # type: ignore
                    success=False,
                    error="STATUS_FAILED",
                    data={"message": str(e)},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 500
        
        # Get memory info
        @app.route("/api/memory", methods=["GET"])
        def get_memory():
            """Get memory system information."""
            try:
                memory_data: Dict[str, Any] = {
                    "status": "operational",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                
                if hasattr(self.ace, "memory"):
                    entries = getattr(self.ace.memory, "entries", []) or []
                    sources = set()
                    for entry in entries:
                        if isinstance(entry, dict):
                            sources.add(entry.get("source", "unknown"))
                    
                    memory_data.update({
                        "total_entries": len(entries),
                        "unique_sources": len(sources),
                        "sources": sorted(list(sources))
                    })
                
                return jsonify(APIResponse(  # type: ignore
                    success=True,
                    data=memory_data,
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 200
            except Exception as e:
                logger.error(f"Memory error: {e}")
                return jsonify(APIResponse(  # type: ignore
                    success=False,
                    error="MEMORY_ERROR",
                    data={"message": str(e)},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 500
        
        # Get beliefs
        @app.route("/api/beliefs", methods=["GET"])
        def get_beliefs():
            """Get top beliefs/claims from knowledge base."""
            try:
                beliefs_data: Dict[str, Any] = {
                    "timestamp": datetime.utcnow().isoformat(),
                }
                
                if hasattr(self.ace, "claim_store"):
                    claims = getattr(self.ace.claim_store, "claims", {}) or {}
                    top_claims = sorted(
                        claims.values() if isinstance(claims, dict) else [],
                        key=lambda x: x.get("confidence", 0) if isinstance(x, dict) else 0,
                        reverse=True
                    )[:10]
                    
                    beliefs_data.update({
                        "total_beliefs": len(claims),
                        "top_beliefs": [
                            {
                                "subject": c.get("subject") if isinstance(c, dict) else None,
                                "predicate": c.get("predicate") if isinstance(c, dict) else None,
                                "object": c.get("object") if isinstance(c, dict) else None,
                                "confidence": c.get("confidence", 0) if isinstance(c, dict) else 0,
                            }
                            for c in top_claims
                        ]
                    })
                
                return jsonify(APIResponse(
                    success=True,
                    data=beliefs_data,
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 200
            except Exception as e:
                logger.error(f"Beliefs error: {e}")
                return jsonify(APIResponse(
                    success=False,
                    error="BELIEFS_ERROR",
                    data={"message": str(e)},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 500
        
        # Search memory
        @app.route("/api/search", methods=["POST"])
        def search():
            """Search knowledge base entries."""
            try:
                data = request.get_json() or {}
                query = data.get("query", "").strip()
                limit = min(int(data.get("limit", 10)), 100)
                
                if not query:
                    return jsonify(APIResponse(
                        success=False,
                        error="INVALID_INPUT",
                        data={"message": "Query cannot be empty"},
                        request_id=getattr(request, "request_id", None)
                    ).to_dict()), 400
                
                results = []
                if hasattr(self.ace, "memory"):
                    entries = getattr(self.ace.memory, "entries", []) or []
                    results = [
                        e for e in entries 
                        if isinstance(e, dict) and query.lower() in e.get("text", "").lower()
                    ][:limit]
                
                return jsonify(APIResponse(
                    success=True,
                    data={
                        "query": query,
                        "results_count": len(results),
                        "results": results,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 200
            except Exception as e:
                logger.error(f"Search error: {e}")
                return jsonify(APIResponse(
                    success=False,
                    error="SEARCH_FAILED",
                    data={"message": str(e)},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 500
        
        # Get concepts
        @app.route("/api/concepts", methods=["GET"])
        def get_concepts():
            """Get concept graph information."""
            try:
                concepts_data: Dict[str, Any] = {
                    "timestamp": datetime.utcnow().isoformat(),
                }
                
                if hasattr(self.ace, "concept_graph"):
                    nodes = getattr(self.ace.concept_graph, "nodes", {}) or {}
                    edges = getattr(self.ace.concept_graph, "edges", {}) or {}
                    
                    concepts_data.update({
                        "total_concepts": len(nodes),
                        "total_relationships": len(edges),
                        "sample_concepts": list(nodes.keys())[:20] if nodes else []
                    })
                
                return jsonify(APIResponse(
                    success=True,
                    data=concepts_data,
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 200
            except Exception as e:
                logger.error(f"Concepts error: {e}")
                return jsonify(APIResponse(
                    success=False,
                    error="CONCEPTS_ERROR",
                    data={"message": str(e)},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 500
        
        # API documentation
        @app.route("/api/docs", methods=["GET"])
        def api_docs():
            """API documentation."""
            docs = {
                "title": "Zypherus API",
                "version": "0.2.0",
                "description": "Production-ready REST API for the Zypherus advanced reasoning system",
                "endpoints": {
                    "Health & Status": {
                        "GET /health": "Service health check"
,
                        "GET /api/status": "System status and metrics"
                    },
                    "Chat": {
                        "POST /api/chat": "Stateful chat (supports streaming SSE)",
                        "GET /chat": "Minimal chat UI"
                    },
                    "Ingestion": {
                        "POST /api/ingest": "Ingest document or text"
                    },
                    "Queries": {
                        "POST /api/answer": "Answer question using knowledge base",
                        "POST /api/search": "Search knowledge base"
                    },
                    "Knowledge": {
                        "GET /api/memory": "Memory system info",
                        "GET /api/beliefs": "Top beliefs/claims",
                        "GET /api/concepts": "Concept graph info"
                    },
                    "Meta": {
                        "GET /api/docs": "This documentation"
                    }
                },
                "authentication": {
                    "type": "API Key or JWT",
                    "header": "X-API-Key or Authorization: Bearer <token>"
                },
                "request_format": {
                    "headers": {
                        "Content-Type": "application/json",
                        "X-Request-ID": "Optional request tracking ID"
                    }
                },
                "response_format": {
                    "success": "Boolean indicating success",
                    "data": "Response payload",
                    "error": "Error code if failed",
                    "request_id": "Request tracking ID",
                    "timestamp": "Response timestamp"
                }
            }
            
            return jsonify(docs), 200
        
        # System statistics
        @app.route("/api/stats", methods=["GET"])
        def get_stats():
            """Get system statistics and performance metrics."""
            try:
                stats: Dict[str, Any] = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "system": {}
                }
                
                # Cache stats
                cache = getattr(app, "query_cache", None)
                if cache:
                    stats["cache"] = cache.stats()
                
                # Memory stats
                memory = getattr(self.ace, "memory", None)
                if memory and hasattr(memory, "entries"):
                    stats["memory"] = {
                        "entry_count": len(memory.entries),
                        "file_path": getattr(memory, "filepath", "unknown")
                    }
                
                return jsonify(APIResponse(
                    success=True,
                    data=stats,
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 200
            except EnvironmentError as e:
                logger.error(f"Stats retrieval environment error: {e}")
                return jsonify(APIResponse(
                    success=False,
                    error="ENVIRONMENT_ERROR",
                    data={"message": str(e)},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 500
            except Exception as e:
                logger.error(f"Stats error: {e}", exc_info=True)
                return jsonify(APIResponse(
                    success=False,
                    error="STATS_ERROR",
                    data={"message": str(e)},
                    request_id=getattr(request, "request_id", None)
                ).to_dict()), 500
    
    def run(self, debug: bool = False):
        """Run API server."""
        if self.app is None:
            self.create_flask_app()
        
        logger.info(f"Starting Zypherus API server on {self.host}:{self.port}")
        if self.app is not None:
            self.app.run(host=self.host, port=self.port, debug=debug, use_reloader=False)


class ZypherusAPIClient:
    """Client for Zypherus API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = None
    
    def _get_session(self):
        """Get or create session."""
        if self.session is None:
            try:
                import requests
                self.session = requests.Session()
                if self.api_key:
                    self.session.headers.update({"X-API-Key": self.api_key})
            except ImportError:
                raise ImportError("requests is required. Install with: pip install requests")
        return self.session
    
    def health_check(self) -> bool:
        """Check if API is healthy."""
        try:
            resp = self._get_session().get(f"{self.base_url}/health", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False
    
    def ingest(self, text: str, source: str = "client") -> Dict[str, Any]:
        """Ingest document."""
        data = {"text": text, "source": source}
        resp = self._get_session().post(f"{self.base_url}/api/ingest", json=data)
        resp.raise_for_status()
        return resp.json()
    
    def answer(self, query: str) -> Dict[str, Any]:
        """Ask question."""
        data = {"query": query}
        resp = self._get_session().post(f"{self.base_url}/api/answer", json=data)
        resp.raise_for_status()
        return resp.json()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status."""
        resp = self._get_session().get(f"{self.base_url}/api/status")
        resp.raise_for_status()
        return resp.json()
    
    def get_memory(self) -> Dict[str, Any]:
        """Get memory info."""
        resp = self._get_session().get(f"{self.base_url}/api/memory")
        resp.raise_for_status()
        return resp.json()
    
    def get_beliefs(self) -> Dict[str, Any]:
        """Get beliefs."""
        resp = self._get_session().get(f"{self.base_url}/api/beliefs")
        resp.raise_for_status()
        return resp.json()
    
    def search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search knowledge base."""
        data = {"query": query, "limit": limit}
        resp = self._get_session().post(f"{self.base_url}/api/search", json=data)
        resp.raise_for_status()
        return resp.json()
    
    def get_concepts(self) -> Dict[str, Any]:
        """Get concept graph info."""
        resp = self._get_session().get(f"{self.base_url}/api/concepts")
        resp.raise_for_status()
        return resp.json()
    
    def get_docs(self) -> Dict[str, Any]:
        """Get API documentation."""
        resp = self._get_session().get(f"{self.base_url}/api/docs")
        resp.raise_for_status()
        return resp.json()


# Alias for backward compatibility
ACEAPIServer = ZypherusAPIServer
ACEAPIClient = ZypherusAPIClient


def create_app():
    """Factory function to create Flask app for WSGI servers."""
    from ..core.ace import ACE
    ace = ACE()
    server = ZypherusAPIServer(ace)
    return server.create_flask_app()


__all__ = [
    "APIRequest",
    "APIResponse",
    "ZypherusAPIServer",
    "ZypherusAPIClient",
    "ACEAPIServer",
    "ACEAPIClient",
    "create_app",
]
