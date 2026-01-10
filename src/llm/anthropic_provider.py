import os
from typing import Any

from src.core.config import ModelConfig
from src.core.logger import setup_logger

logger = setup_logger(__name__)


def create_anthropic_llm(config: ModelConfig, **override_params) -> Any:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        logger.error("langchain-anthropic not installed")
        raise ImportError(
            "langchain-anthropic is required for Anthropic models. "
            "Install with: uv add langchain-anthropic"
        ) from e

    # get api key
    api_key = os.getenv(config.api_key_env)
    if not api_key:
        logger.warning(f"Environment variable '{config.api_key_env}' not set. ")

    # merge configuration with overrides
    params = {
        "model": config.model_name,
        "temperature": config.temperature,
        **config.additional_params,
        **override_params,  # Overrides take precedence
    }

    if config.max_tokens is not None:
        params["max_tokens"] = config.max_tokens

    if api_key:
        params["anthropic_api_key"] = api_key

    logger.info(
        f"Creating ChatAnthropic: {config.model_name} "
        f"(temperature={params['temperature']})"
    )

    try:
        llm = ChatAnthropic(**params)
        return llm
    except Exception as e:
        logger.error(f"Failed to create ChatAnthropic instance: {e}")
        raise
