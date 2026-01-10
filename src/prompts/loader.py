from typing import Any, Dict, Optional

from jinja2 import Template, TemplateSyntaxError, UndefinedError

from src.core.config import ConfigManager
from src.core.logger import setup_logger

logger = setup_logger(__name__)


class PromptLoader:
    def __init__(self):
        self._cache: Dict[str, Template] = {}
        self._config_manager = ConfigManager()
        logger.debug("PromptLoader initialized")

    def get_template(self, name: str) -> Template:
        # Check cache first
        if name in self._cache:
            logger.debug(f"Template '{name}' loaded from cache")
            return self._cache[name]

        # Load template string from config
        try:
            template_str = self._config_manager.get_prompt_template(name)
        except KeyError as e:
            logger.error(f"Prompt template '{name}' not found")
            raise

        # Create Jinja2 template
        try:
            template = Template(template_str)
            self._cache[name] = template
            logger.info(f"Template '{name}' loaded and cached")
            return template
        except TemplateSyntaxError as e:
            logger.error(f"Invalid Jinja2 syntax in template '{name}': {e}")
            raise

    def render(
        self, name: str, variables: Dict[str, Any], validate: bool = True
    ) -> str:
        template = self.get_template(name)

        try:
            # Render template with variables
            rendered = template.render(**variables)
            logger.debug(f"Template '{name}' rendered successfully")
            return rendered

        except UndefinedError as e:
            if validate:
                logger.error(
                    f"Missing variable in template '{name}': {e}\n"
                    f"Available variables: {list(variables.keys())}"
                )
                raise
            else:
                logger.warning(f"Undefined variable in template '{name}': {e}")
                from jinja2 import Environment

                env = Environment()
                template_lenient = env.from_string(template.source)
                return template_lenient.render(**variables)

    def render_string(self, template_str: str, variables: Dict[str, Any]) -> str:
        template = Template(template_str)
        return template.render(**variables)

    def list_templates(self) -> list[str]:
        return self._config_manager.list_prompts()

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.info("Template cache cleared")

    def reload_templates(self) -> None:
        self.clear_cache()
        # Force config manager to reload
        self._config_manager._load_prompts()
        logger.info("Templates reloaded from configuration")
