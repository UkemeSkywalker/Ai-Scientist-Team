"""
Strands SDK configuration and setup for AI Scientist Team
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import structlog

# Load environment variables
load_dotenv()

logger = structlog.get_logger(__name__)

class StrandsConfig:
    """Configuration class for Strands SDK settings"""
    
    def __init__(self):
        self.model_id = os.getenv("STRANDS_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
        self.region = os.getenv("STRANDS_REGION", "us-east-1")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for Strands agents"""
        return {
            "model_id": self.model_id,
            "region": self.region
        }
    
    def validate_config(self) -> bool:
        """Validate that required configuration is present"""
        required_vars = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "STRANDS_REGION"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error("Missing required environment variables", missing_vars=missing_vars)
            return False
        
        logger.info("Strands configuration validated successfully")
        return True

# Global configuration instance
strands_config = StrandsConfig()