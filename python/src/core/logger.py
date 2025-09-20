import logging
import structlog
from typing import Any, Dict, Optional
import traceback


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

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance"""
    return structlog.get_logger(name)

class StrandsErrorHandler:
    """Error handler for Strands agent operations"""
    
    def __init__(self, logger: structlog.stdlib.BoundLogger):
        self.logger = logger
    
    def handle_agent_error(self, agent_name: str, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle Strands agent errors with comprehensive logging"""
        error_info = {
            "agent_name": agent_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        
        self.logger.error(
            "Strands agent error occurred",
            **error_info
        )
        
        return error_info
    
    def handle_tool_error(self, tool_name: str, error: Exception, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle Strands tool errors"""
        error_info = {
            "tool_name": tool_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "parameters": parameters or {},
            "traceback": traceback.format_exc()
        }
        
        self.logger.error(
            "Strands tool error occurred",
            **error_info
        )
        
        return error_info

class AgentLogger:
    """Logger wrapper for agents with context and Strands integration."""
    
    def __init__(self, agent_name: str, session_id: str):
        self.logger = setup_logger()
        self.agent_name = agent_name
        self.session_id = session_id
        self.error_handler = StrandsErrorHandler(self.logger)
    
    def _add_context(self, **kwargs) -> Dict[str, Any]:
        """Add agent context to log entries."""
        return {
            "agent": self.agent_name,
            "session_id": self.session_id,
            **kwargs
        }
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, **self._add_context(**kwargs))
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception handling"""
        log_data = self._add_context(**kwargs)
        
        if error:
            log_data.update({
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc()
            })
        
        self.logger.error(message, **log_data)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, **self._add_context(**kwargs))
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, **self._add_context(**kwargs))
    
    def log_strands_interaction(self, interaction_type: str, details: Dict[str, Any]):
        """Log Strands-specific interactions"""
        self.logger.info(
            f"Strands {interaction_type}",
            **self._add_context(
                interaction_type=interaction_type,
                **details
            )
        )
    
    def log_tool_execution(self, tool_name: str, parameters: Dict[str, Any], result: Any = None, error: Exception = None):
        """Log tool execution with results or errors"""
        log_data = self._add_context(
            tool_name=tool_name,
            parameters=parameters
        )
        
        if error:
            log_data.update(self.error_handler.handle_tool_error(tool_name, error, parameters))
            self.logger.error("Tool execution failed", **log_data)
        else:
            if result is not None:
                log_data["result_type"] = type(result).__name__
                log_data["result_size"] = len(str(result)) if result else 0
            self.logger.info("Tool executed successfully", **log_data)