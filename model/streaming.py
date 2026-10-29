"""Small streamer compatible with WH's custom generate loop.

Transformers' TextStreamer is not used because its behavior varies across
Transformers 5.x releases and can trigger backend-specific failures with custom
causal-LM generate loops. This class only depends on tokenizer.decode.
"""
from __future__ import annotations

from typing import Callable, Optional


class TokenChunkStreamer:
    def __init__(
        self,
        tokenizer,
        callback: Optional[Callable[[str], None]] = None,
        skip_prompt: bool = True,
        skip_special_tokens: bool = True,
        end_callback: Optional[Callable[[], None]] = None,
    ):
        self.tokenizer = tokenizer
        self.callback = callback or (lambda text: print(text, end="", flush=True))
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.end_callback = end_callback
        self.next_tokens_are_prompt = True
        self.cache = []
        self.emitted = 0

    def put(self, value):
        if self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            if self.skip_prompt:
                return
        self.cache.extend(value.flatten().tolist())
        text = self.tokenizer.decode(
            self.cache,
            skip_special_tokens=self.skip_special_tokens,
        )
        chunk = text[self.emitted:]
        self.emitted = len(text)
        if chunk:
            self.callback(chunk)

    def end(self):
        if self.end_callback is not None:
            self.end_callback()
        self.next_tokens_are_prompt = True
        self.cache = []
        self.emitted = 0
