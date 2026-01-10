import os
from typing import Any

from src.core.config import ConfigManager, ModelConfig
from src.core.logger import setup_logger

logger = setup_logger(__name__)


def create_openai_llm(config: ModelConfig, **override_params) -> Any:
    # Check for langchain-openai package
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        logger.error("langchain-openai not installed")
        raise ImportError(
            "langchain-openai is required for OpenAI models. "
            "Install with: uv add langchain-openai"
        ) from e

    # Get API key from environment
    api_key = os.getenv(config.api_key_env)
    if not api_key:
        logger.warning(
            f"Environment variable '{config.api_key_env}' not set. "
            f"Set it in your environment or .env file."
        )
        # Don't raise error - let langchain-openai handle it
        # This allows for better error messages from the library

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
        params["api_key"] = api_key

    logger.info(
        f"Creating ChatOpenAI: {config.model_name} "
        f"(temperature={params['temperature']})"
    )

    try:
        llm = ChatOpenAI(**params)
        logger.debug(f"ChatOpenAI instance created successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to create ChatOpenAI instance: {e}")
        raise
