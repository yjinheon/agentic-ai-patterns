import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ModelConfig:
    provider: str
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_key_env: str = ""
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptConfig:
    name: str
    template: str
    description: str = ""


class ConfigManager:
    """Singleton configuration manager.

    Loads and manages configurations from YAML files:
    - config/models.yaml: LLM model configurations
    - config/prompts/*.yaml: Prompt templates

    """

    _instance: Optional["ConfigManager"] = None

    def __new__(cls) -> "ConfigManager":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._config_dir = Path(__file__).parent.parent.parent / "config"
            self._models: Dict[str, ModelConfig] = {}
            self._prompts: Dict[str, PromptConfig] = {}
            self._aliases: Dict[str, str] = {}
            self._load_models()
            self._load_prompts()
            self._initialized = True

    def _load_models(self) -> None:
        models_file = self._config_dir / "models.yaml"

        if not models_file.exists():
            # Create empty config if file doesn't exist
            self._models = {}
            self._aliases = {}
            return

        with open(models_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Load model config
        if "models" in data:
            for name, config in data["models"].items():
                self._models[name] = ModelConfig(
                    provider=config.get("provider", ""),
                    model_name=config.get("model_name", name),
                    temperature=config.get("temperature", 0.7),
                    max_tokens=config.get("max_tokens"),
                    api_key_env=config.get("api_key_env", ""),
                    additional_params=config.get("additional_params", {}),
                )

        # Load aliases
        if "aliases" in data:
            self._aliases = data["aliases"]

    def _load_prompts(self) -> None:
        prompts_dir = self._config_dir / "prompts"

        if not prompts_dir.exists():
            self._prompts = {}
            return

        # Load all YAML files in prompts directory
        for yaml_file in prompts_dir.glob("*.yaml"):
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if "prompts" in data:
                for name, config in data["prompts"].items():
                    self._prompts[name] = PromptConfig(
                        name=name,
                        template=config.get("template", ""),
                        description=config.get("description", ""),
                    )

    def get_model_config(self, name: str) -> ModelConfig:
        resolved_name = self._aliases.get(name, name)

        if resolved_name not in self._models:
            raise KeyError(f"Model '{name}' not found in configuration")

        return self._models[resolved_name]

    def get_prompt_template(self, name: str) -> str:
        if name not in self._prompts:
            raise KeyError(f"Prompt template '{name}' not found in configuration")

        return self._prompts[name].template

    def get_prompt_config(self, name: str) -> PromptConfig:
        if name not in self._prompts:
            raise KeyError(f"Prompt template '{name}' not found in configuration")

        return self._prompts[name]

    def get_api_key(self, env_var: str) -> str:
        api_key = os.getenv(env_var)

        if not api_key:
            raise ValueError(
                f"Environment variable '{env_var}' not set. "
                f"Please set it in your environment or .env file."
            )

        return api_key

    def list_models(self) -> list[str]:
        return list(self._models.keys())

    def list_prompts(self) -> list[str]:
        return list(self._prompts.keys())

    def list_aliases(self) -> Dict[str, str]:
        return self._aliases.copy()
