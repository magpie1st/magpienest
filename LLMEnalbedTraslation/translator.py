"""Domain logic for English→Korean translation via Ollama."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    _current_dir = Path(__file__).resolve().parent
    if str(_current_dir) not in sys.path:
        sys.path.insert(0, str(_current_dir))

    from client import OllamaClient, OllamaClientError  # type: ignore
    from config import JetsonConnectionConfig  # type: ignore
    from text_utils import TokenEstimator, chunk_text, is_garbage_translation  # type: ignore
else:
    from .client import OllamaClient, OllamaClientError
    from .config import JetsonConnectionConfig
    from .text_utils import TokenEstimator, chunk_text, is_garbage_translation


PROMPT_TEMPLATE = """You are a professional English→Korean literary translator.
    Task: Translate the following English text into Korean.
    Intent & Rules:
        - Genre: Romance novel.
        - Narration: Use formal written past tense in Korean (e.g. "...였다", "...했다").
        - Dialogue: Keep quotation marks and translate into natural spoken Korean 
          that matches the character’s tone.
        - Preserve paragraph breaks, names, and places.
        - Do not add explanations or notes; only output the translation.
    English text:\n\n{source}"""
DEFAULT_MAX_TOKENS_PER_CHUNK = 3800
DEFAULT_MAX_SENTENCES_PER_CHUNK = 5
DEFAULT_RETRY_ATTEMPTS = 1
THINK_TAG_PATTERN = re.compile(r"<\s*think\s*>(.*?)<\s*/\s*think\s*>", re.IGNORECASE | re.DOTALL)


@dataclass
class TranslationResult:
    """Represents a completed translation."""

    source_text: str
    translated_text: str
    model: str


class TranslationService:
    """Coordinator that exposes higher-level translation operations."""

    def __init__(
        self,
        client: OllamaClient | None = None,
        *,
        token_estimator: TokenEstimator | None = None,
        max_tokens_per_chunk: int = DEFAULT_MAX_TOKENS_PER_CHUNK,
        max_sentences_per_chunk: int = DEFAULT_MAX_SENTENCES_PER_CHUNK,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    ) -> None:
        self._client = client or OllamaClient()
        self._token_estimator = token_estimator or TokenEstimator()
        self._max_tokens_per_chunk = max_tokens_per_chunk
        self._max_sentences_per_chunk = max_sentences_per_chunk
        self._retry_attempts = retry_attempts

    @property
    def client(self) -> OllamaClient:
        return self._client

    @property
    def max_tokens_per_chunk(self) -> int:
        return self._max_tokens_per_chunk

    @property
    def max_sentences_per_chunk(self) -> int:
        return self._max_sentences_per_chunk

    @property
    def retry_attempts(self) -> int:
        return self._retry_attempts

    @property
    def timeout(self) -> int:
        return self._client.timeout

    def available_models(
        self,
        extra_models: Iterable[str] | None = None,
        *,
        host: str | None = None,
        port: int | None = None,
        timeout: int | None = None,
    ) -> list[str]:
        client = self._get_client(host=host, port=port, timeout=timeout)
        return client.list_models(extra_models)

    def update_connection(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        timeout: int | None = None,
    ) -> JetsonConnectionConfig:
        self._client.update_config(host=host, port=port, timeout=timeout)
        return self._client.config

    def translate(
        self,
        english_text: str,
        model: str,
        *,
        host: str | None = None,
        port: int | None = None,
        prompt_template: str | None = None,
        stream: bool = False,
        options: Optional[dict[str, object]] = None,
        max_tokens_per_chunk: int | None = None,
        max_sentences_per_chunk: int | None = None,
        retry_attempts: int | None = None,
        timeout: int | None = None,
        strip_think_tag: bool = False,
    ) -> TranslationResult:
        iterator, _ = self.iter_translate_chunks(
            english_text,
            model,
            host=host,
            port=port,
            prompt_template=prompt_template,
            stream=stream,
            options=options,
            max_tokens_per_chunk=max_tokens_per_chunk,
            max_sentences_per_chunk=max_sentences_per_chunk,
            retry_attempts=retry_attempts,
            timeout=timeout,
            strip_think_tag=strip_think_tag,
        )
        aggregate: list[str] = []
        for _, _, chunk_text in iterator:
            aggregate.append(chunk_text)
        translated = "\n\n".join(part for part in aggregate if part)
        source_text = (english_text or "").strip()
        return TranslationResult(source_text=source_text, translated_text=translated.strip(), model=model)

    def iter_translate_chunks(
        self,
        english_text: str,
        model: str,
        *,
        host: str | None = None,
        port: int | None = None,
        prompt_template: str | None = None,
        stream: bool = False,
        options: Optional[dict[str, object]] = None,
        max_tokens_per_chunk: int | None = None,
        max_sentences_per_chunk: int | None = None,
        retry_attempts: int | None = None,
        timeout: int | None = None,
        strip_think_tag: bool = False,
    ) -> tuple[Iterable[tuple[int, int, str]], int]:
        text = (english_text or "").strip()
        if not text:
            raise ValueError("영어 텍스트를 입력해 주세요.")
        if not model:
            raise ValueError("번역에 사용할 모델을 선택해 주세요.")

        tokens_limit = max_tokens_per_chunk or self._max_tokens_per_chunk
        sentences_limit = max_sentences_per_chunk or self._max_sentences_per_chunk
        retries = self._retry_attempts if retry_attempts is None else retry_attempts

        chunks = chunk_text(
            text,
            estimator=self._token_estimator,
            max_tokens=tokens_limit,
            max_sentences=sentences_limit,
        )
        if not chunks:
            raise ValueError("번역할 문장을 찾지 못했습니다.")

        client = self._get_client(host=host, port=port, timeout=timeout)
        total_chunks = len(chunks)
        stream_enabled = stream and total_chunks == 1

        def generator() -> Iterable[tuple[int, int, str]]:
            for idx, chunk in enumerate(chunks):
                chunk_prompt = self._build_prompt(template=prompt_template, source=chunk)
                if total_chunks > 1:
                    chunk_prompt = (
                        f"{chunk_prompt}\n\n(전체 텍스트 중 {idx + 1}/{total_chunks} 번째 조각입니다. 앞뒤 문맥과 톤을 유지해 자연스럽게 번역하세요.)"
                    )

                translated_chunk = self._translate_chunk(
                    client,
                    model=model,
                    prompt=chunk_prompt,
                    stream=stream_enabled,
                    options=options,
                )

                if is_garbage_translation(translated_chunk) and retries > 0:
                    retry_prompt = (
                        f"{chunk_prompt}\n\n(이전 번역이 비정상적이었습니다. 문맥과 의미를 유지해 다시 번역하세요.)"
                    )
                    for _ in range(retries):
                        translated_chunk = self._translate_chunk(
                            client,
                            model=model,
                            prompt=retry_prompt,
                            stream=stream_enabled,
                            options=options,
                        )
                        if not is_garbage_translation(translated_chunk):
                            break

                if is_garbage_translation(translated_chunk):
                    raise OllamaClientError(
                        f"번역 결과가 비정상적으로 감지되었습니다. {idx + 1}/{total_chunks} 번째 조각을 다시 시도해 주세요."
                    )

                yield idx, total_chunks, self._post_process_chunk(translated_chunk, strip_think_tag)

        return generator(), total_chunks

    # Internal helpers -------------------------------------------------
    def _build_config(self, *, host: str | None, port: int | None) -> JetsonConnectionConfig:
        current = self._client.config
        new_host = host if host is not None else current.host
        new_port = port if port is not None else current.port
        return JetsonConnectionConfig(host=new_host, port=new_port)

    def _build_prompt(self, *, template: str | None, source: str) -> str:
        tpl = (template or PROMPT_TEMPLATE).strip()
        if "{source}" in tpl:
            return tpl.format(source=source)
        if tpl:
            return f"{tpl}\n\n{source}"
        return source

    @property
    def default_prompt(self) -> str:
        return PROMPT_TEMPLATE

    def _translate_chunk(
        self,
        client: OllamaClient,
        *,
        model: str,
        prompt: str,
        stream: bool,
        options: Optional[dict[str, object]] = None,
    ) -> str:
        opts = dict(options or {})
        response = client.generate(model=model, prompt=prompt, stream=stream, options=opts)
        return response.response.strip()

    def _post_process_chunk(self, text: str, strip_think_tag: bool) -> str:
        cleaned = text.strip()
        if not strip_think_tag:
            return cleaned

        removed = THINK_TAG_PATTERN.sub("", cleaned).strip()
        if removed:
            return removed

        # If everything was inside <think>, keep the inner content instead of returning empty.
        return THINK_TAG_PATTERN.sub(lambda m: (m.group(1) or "").strip(), cleaned).strip()

    def _get_client(
        self,
        *,
        host: str | None,
        port: int | None,
        timeout: int | None,
    ) -> OllamaClient:
        if host is None and port is None and timeout is None:
            return self._client
        cfg = self._build_config(host=host, port=port)
        effective_timeout = timeout if timeout is not None else self._client.timeout
        return OllamaClient(cfg, timeout=effective_timeout)
