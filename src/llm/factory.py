from enum import Enum
from typing import Any, Optional

from src.core.config import ConfigManager, ModelConfig
from src.core.logger import setup_logger

logger = setup_logger(__name__)


class LLMProvider(str, Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"


class LLMFactory:
    """Factory for creating LLM instances."""

    @staticmethod
    def create_llm(
        model_name: str, provider: Optional[LLMProvider] = None, **override_params
    ) -> Any:
        """Create LLM instance from configuration."""
        config_manager = ConfigManager()

        # Load model configuration
        try:
            config = config_manager.get_model_config(model_name)
        except KeyError as e:
            logger.error(f"Model '{model_name}' not found in configuration")
            raise

        # Determine provider
        if provider is None:
            try:
                provider = LLMProvider(config.provider)
            except ValueError:
                raise ValueError(
                    f"Unsupported provider '{config.provider}' for model '{model_name}'. "
                    f"Supported providers: {[p.value for p in LLMProvider]}"
                )

        logger.info(f"Creating LLM: {model_name} (provider: {provider.value})")

        # Route to provider-specific creation
        if provider == LLMProvider.OPENAI:
            from src.llm.openai_provider import create_openai_llm

            return create_openai_llm(config, **override_params)

        elif provider == LLMProvider.GOOGLE:
            from src.llm.google_provider import create_google_llm

            return create_google_llm(config, **override_params)

        elif provider == LLMProvider.ANTHROPIC:
            from src.llm.anthropic_provider import create_anthropic_llm

            return create_anthropic_llm(config, **override_params)

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def create_adk_agent(
        name: str,
        model_name: str,
        instruction: str,
        description: str = "",
        tools: Optional[list] = None,
        sub_agents: Optional[list] = None,
        **agent_params,
    ) -> Any:
        """Create Google ADK Agent instance."""
        try:
            from google.adk.agents import Agent
        except ImportError as e:
            logger.error("google-adk not installed. Install with: uv add google-adk")
            raise ImportError(
                "google-adk is required for ADK agents. Install with: uv add google-adk"
            ) from e

        logger.info(f"Creating ADK Agent: {name} (model: {model_name})")

        # Build agent parameters
        params = {
            "name": name,
            "model": model_name,
            "instruction": instruction,
            **agent_params,
        }

        if description:
            params["description"] = description
        if tools:
            params["tools"] = tools
        if sub_agents:
            params["sub_agents"] = sub_agents

        return Agent(**params)

    @staticmethod
    def get_available_models() -> list[str]:
        config_manager = ConfigManager()
        models = config_manager.list_models()
        aliases = list(config_manager.list_aliases().keys())
        return models + aliases

    @staticmethod
    def get_model_info(model_name: str) -> dict:
        config_manager = ConfigManager()
        config = config_manager.get_model_config(model_name)

        return {
            "provider": config.provider,
            "model_name": config.model_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "api_key_env": config.api_key_env,
        }
