import os
from typing import Any

from src.core.config import ConfigManager, ModelConfig
from src.core.logger import setup_logger

logger = setup_logger(__name__)


def create_google_llm(config: ModelConfig, **override_params) -> Any:
    """Create Google ChatGoogleGenerativeAI instance from configuration."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as e:
        logger.error("langchain-google-genai not installed")
        raise ImportError(
            "langchain-google-genai is required for Google models. "
            "Install with: uv add langchain-google-genai"
        ) from e

    # Get API key from environment
    api_key = os.getenv(config.api_key_env)
    if not api_key:
        logger.warning(
            f"Environment variable '{config.api_key_env}' not set. "
            f"Set it in your environment or .env file."
        )
        # Don't raise error - let langchain-google-genai handle it

    # Merge configuration with overrides
    params = {
        "model": config.model_name,
        "temperature": config.temperature,
        **config.additional_params,
        **override_params,  # Overrides take precedence
    }

    # Add max_tokens if specified
    if config.max_tokens is not None:
        params["max_tokens"] = config.max_tokens

    # Add API key if found
    if api_key:
        params["google_api_key"] = api_key

    logger.info(
        f"Creating ChatGoogleGenerativeAI: {config.model_name} "
        f"(temperature={params['temperature']})"
    )

    try:
        llm = ChatGoogleGenerativeAI(**params)
        logger.debug(f"ChatGoogleGenerativeAI instance created successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to create ChatGoogleGenerativeAI instance: {e}")
        raise
