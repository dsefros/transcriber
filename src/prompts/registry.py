from pathlib import Path
import yaml


class PromptNotFound(Exception):
    pass


class PromptRegistry:
    """
    Loads and renders versioned prompts.
    """

    def __init__(self, base_dir: str = "src/prompts"):
        self.base_dir = Path(base_dir)

    def load(self, relative_path: str) -> dict:
        """
        Example: analysis/v1.yaml
        """
        path = self.base_dir / relative_path

        if not path.exists():
            raise PromptNotFound(f"Prompt not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def render(self, relative_path: str, **variables) -> str:
        prompt = self.load(relative_path)

        template = prompt.get("template")
        if not template:
            raise ValueError("Prompt template missing")

        for key, value in variables.items():
            template = template.replace(f"{{{{ {key} }}}}", value)

        return template
