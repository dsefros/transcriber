from __future__ import annotations

import pytest

from src.prompts.registry import PromptNotFound, PromptRegistry

pytestmark = pytest.mark.unit


def test_render_substitutes_transcript(tmp_path):
    prompt_dir = tmp_path / "prompts" / "analysis"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "v1.yaml").write_text("template: 'Summary: {{ transcript }}'\n", encoding="utf-8")

    rendered = PromptRegistry(base_dir=str(tmp_path / "prompts")).render("analysis/v1.yaml", transcript="hello")

    assert rendered == "Summary: hello"


def test_render_missing_prompt_fails(tmp_path):
    with pytest.raises(PromptNotFound):
        PromptRegistry(base_dir=str(tmp_path)).render("analysis/v1.yaml", transcript="hello")


def test_render_malformed_yaml_fails(tmp_path):
    prompt_dir = tmp_path / "prompts" / "analysis"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "v1.yaml").write_text("template: [unterminated\n", encoding="utf-8")

    with pytest.raises(Exception):
        PromptRegistry(base_dir=str(tmp_path / "prompts")).render("analysis/v1.yaml", transcript="hello")
