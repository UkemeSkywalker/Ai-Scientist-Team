import logging
import structlog
from typing import Any, Dict


def setup_logger(log_level: str = "INFO") -> structlog.stdlib.BoundLogger:
    """Setup structured logging for the application."""
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=None,
        level=getattr(logging, log_level.upper()),
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger()


class AgentLogger:
    """Logger wrapper for agents with context."""
    
    def __init__(self, agent_name: str, session_id: str):
        self.logger = setup_logger()
        self.agent_name = agent_name
        self.session_id = session_id
    
    def _add_context(self, **kwargs) -> Dict[str, Any]:
        """Add agent context to log entries."""
        return {
            "agent": self.agent_name,
            "session_id": self.session_id,
            **kwargs
        }
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, **self._add_context(**kwargs))
    
    def error(self, message: str, **kwargs):
        self.logger.error(message, **self._add_context(**kwargs))
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, **self._add_context(**kwargs))
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, **self._add_context(**kwargs))