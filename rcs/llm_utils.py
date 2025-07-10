"""Utility classes for working with local language models."""

import asyncio
from typing import Any

from transformers import pipeline


class LocalChatModel:
    """Simple async wrapper around a transformers text-generation pipeline."""

    def __init__(self, model_name: str = "distilgpt2", temperature: float = 0.7) -> None:
        self.pipeline = pipeline(
            "text-generation", model=model_name, tokenizer=model_name, temperature=temperature
        )

    async def apredict(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.pipeline(prompt, max_new_tokens=150)[0]["generated_text"]
        )
        return result


class SimpleLLMChain:
    """Minimal replacement for LangChain's LLMChain."""

    def __init__(self, llm: LocalChatModel, prompt_template: str) -> None:
        self.llm = llm
        self.prompt_template = prompt_template

    async def arun(self, **kwargs: Any) -> str:
        prompt = self.prompt_template.format(**kwargs)
        return await self.llm.apredict(prompt)
