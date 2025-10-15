"""FastAPI application that exposes the translation UI and APIs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    _current_dir = Path(__file__).resolve().parent
    if str(_current_dir) not in sys.path:
        sys.path.insert(0, str(_current_dir))

    from client import OllamaClientError  # type: ignore
    from config import DEFAULT_JETSON_CONFIG  # type: ignore
    from translator import PROMPT_TEMPLATE, TranslationService  # type: ignore
    from tts import CoquiTTSService, TTSResult, TTSSettings  # type: ignore
else:
    from .client import OllamaClientError
    from .config import DEFAULT_JETSON_CONFIG
    from .translator import PROMPT_TEMPLATE, TranslationService
    from .tts import CoquiTTSService, TTSResult, TTSSettings

ROOT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = ROOT_DIR / "templates"
STATIC_DIR = ROOT_DIR / "static"

app = FastAPI(title="LLM Enabled Translation")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
service = TranslationService()
tts_service = CoquiTTSService(settings=TTSSettings())

logger = logging.getLogger("LLMEnalbedTraslation.server")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logging.getLogger("LLMEnalbedTraslation.translator").setLevel(logging.DEBUG)
logging.getLogger("LLMEnalbedTraslation.tts").setLevel(logging.DEBUG)


class TranslateRequest(BaseModel):
    text: str = Field(..., description="Source English text to translate")
    model: str = Field(..., description="Ollama model name")
    host: Optional[str] = Field(None, description="Jetson host override")
    port: Optional[int] = Field(None, description="Jetson port override")
    prompt_template: Optional[str] = Field(None, description="Prompt template that wraps the source text")
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature passed to Ollama options",
    )
    stream: Optional[bool] = Field(False, description="Whether to enable streaming responses")
    max_tokens_per_chunk: Optional[int] = Field(
        None,
        ge=128,
        le=6000,
        description="Maximum tokens per chunk when splitting long text",
    )
    max_sentences_per_chunk: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Maximum sentence count per chunk",
    )
    retry_garbage_attempts: Optional[int] = Field(
        None,
        ge=0,
        le=5,
        description="Retry count when translation output looks corrupted",
    )
    timeout: Optional[int] = Field(
        None,
        ge=10,
        le=600,
        description="Request timeout in seconds",
    )
    progress_updates: Optional[bool] = Field(False, description="Stream chunk-level progress events")
    live_text: Optional[bool] = Field(False, description="Include translated text in progress events")
    strip_think_tag: Optional[bool] = Field(False, description="Remove <think>...</think> segments from output")


class ConnectionRequest(BaseModel):
    host: Optional[str] = Field(None)
    port: Optional[int] = Field(None, ge=1, le=65535)
    extra_models: Optional[List[str]] = Field(None, description="Optional extra models to merge")
    timeout: Optional[int] = Field(None, ge=10, le=600)


class TTSRequest(BaseModel):
    text: str = Field(..., description="English text to narrate")
    language: Optional[str] = Field(None, description="Language code supported by XTTS (default en)")
    speaker_wav: Optional[str] = Field(
        None,
        description="Optional filesystem path to a reference audio file for voice cloning",
    )
    speaker: Optional[str] = Field(None, description="Speaker name/ID for multi-speaker models (optional)")
    save_output: Optional[bool] = Field(False, description="Persist synthesized MP3 to the configured output directory")
    filename: Optional[str] = Field(None, description="Optional file name when saving to disk")


class TTSSettingsPayload(BaseModel):
    ffmpeg_path: Optional[str] = Field(None, description="Absolute path to ffmpeg executable or folder")
    output_dir: Optional[str] = Field(None, description="Directory where synthesized files should be stored")
    default_language: Optional[str] = Field(None, description="Default narration language (e.g. en)")
    default_speaker: Optional[str] = Field(None, description="Default speaker name to use when none is provided")
    use_cuda: Optional[bool] = Field(None, description="Use GPU acceleration if available")


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request) -> HTMLResponse:
    logger.debug("[server] GET / | client=%s", request.client)
    models = service.available_models()
    config = service.client.config
    context = {
        "request": request,
        "default_host": config.host or DEFAULT_JETSON_CONFIG.host,
        "default_port": config.port or DEFAULT_JETSON_CONFIG.port,
        "models": models,
        "default_prompt": service.default_prompt,
        "default_temperature": 0.0,
        "default_stream": False,
        "default_max_tokens": service.max_tokens_per_chunk,
        "default_max_sentences": service.max_sentences_per_chunk,
        "default_retry_attempts": service.retry_attempts,
        "default_timeout": service.timeout,
        "default_progress": False,
        "default_live_text": False,
        "default_strip_think": False,
    }
    return templates.TemplateResponse("index.html", context)


@app.get("/api/tts/settings")
def get_tts_settings() -> JSONResponse:
    logger.debug("[server] GET /api/tts/settings")
    settings = tts_service.get_settings()
    payload = settings.to_dict(
        include_available=True,
        available_speakers=tts_service.available_speakers(),
    )
    return JSONResponse(payload)


@app.post("/api/tts/settings")
def update_tts_settings(payload: TTSSettingsPayload = Body(...)) -> JSONResponse:
    logger.debug(
        "[server] POST /api/tts/settings | ffmpeg=%s output=%s lang=%s speaker=%s cuda=%s",
        payload.ffmpeg_path,
        payload.output_dir,
        payload.default_language,
        payload.default_speaker,
        payload.use_cuda,
    )
    settings = tts_service.update_settings(
        ffmpeg_path=payload.ffmpeg_path,
        output_dir=payload.output_dir,
        default_language=payload.default_language,
        default_speaker=payload.default_speaker,
        use_cuda=payload.use_cuda,
    )
    response = settings.to_dict(
        include_available=True,
        available_speakers=tts_service.available_speakers(),
    )
    return JSONResponse(response)


@app.post("/api/translate")
def translate(payload: TranslateRequest = Body(...)) -> JSONResponse:
    logger.debug(
        "[server] POST /api/translate | model=%s len=%d stream=%s progress=%s",
        payload.model,
        len((payload.text or "").strip()),
        payload.stream,
        payload.progress_updates,
    )
    if payload.progress_updates:
        return _translate_with_progress(payload)

    try:
        options: dict[str, object] | None = None
        if payload.temperature is not None:
            options = {"temperature": payload.temperature}
        result = service.translate(
            payload.text,
            payload.model,
            host=payload.host,
            port=payload.port,
            prompt_template=payload.prompt_template,
            stream=bool(payload.stream),
            options=options,
            max_tokens_per_chunk=payload.max_tokens_per_chunk,
            max_sentences_per_chunk=payload.max_sentences_per_chunk,
            retry_attempts=payload.retry_garbage_attempts,
            timeout=payload.timeout,
            strip_think_tag=bool(payload.strip_think_tag),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OllamaClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    logger.debug("[server] translation finished | model=%s", result.model)
    return JSONResponse(
        {
            "translation": result.translated_text,
            "model": result.model,
        }
    )


@app.post("/api/models")
def fetch_models(payload: Optional[ConnectionRequest] = Body(default=None)) -> JSONResponse:
    logger.debug("[server] POST /api/models | host=%s port=%s", payload.host, payload.port)
    payload = payload or ConnectionRequest()
    try:
        models = service.available_models(
            payload.extra_models,
            host=payload.host,
            port=payload.port,
            timeout=payload.timeout,
        )
    except OllamaClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return JSONResponse({"models": models})


@app.post("/api/config")
def update_config(payload: ConnectionRequest = Body(...)) -> JSONResponse:
    logger.debug("[server] POST /api/config | host=%s port=%s", payload.host, payload.port)
    config = service.update_connection(host=payload.host, port=payload.port, timeout=payload.timeout)
    return JSONResponse({"host": config.host, "port": config.port, "timeout": service.timeout})


def _translate_with_progress(payload: TranslateRequest) -> StreamingResponse:
    logger.debug("[server] streaming translation start | model=%s", payload.model)
    import json

    options: dict[str, object] | None = None
    if payload.temperature is not None:
        options = {"temperature": payload.temperature}

    try:
        iterator, total = service.iter_translate_chunks(
            payload.text,
            payload.model,
            host=payload.host,
            port=payload.port,
            prompt_template=payload.prompt_template,
            stream=bool(payload.stream),
            options=options,
            max_tokens_per_chunk=payload.max_tokens_per_chunk,
            max_sentences_per_chunk=payload.max_sentences_per_chunk,
            retry_attempts=payload.retry_garbage_attempts,
            timeout=payload.timeout,
            strip_think_tag=bool(payload.strip_think_tag),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OllamaClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    live_text = bool(payload.live_text)

    def event_stream() -> Iterable[str]:
        try:
            aggregated: list[str] = []
            yield json.dumps({"type": "meta", "total": total}, ensure_ascii=False) + "\n"
            logger.debug("[server] streaming translation meta | total=%s", total)
            for index, total_chunks, chunk_text in iterator:
                aggregated.append(chunk_text)
                event: dict[str, object] = {
                    "type": "chunk",
                    "index": index,
                    "total": total_chunks,
                    "percent": int(((index + 1) / total_chunks) * 100),
                }
                if live_text:
                    event["text"] = chunk_text
                logger.debug("[server] streaming chunk %s/%s", index + 1, total_chunks)
                yield json.dumps(event, ensure_ascii=False) + "\n"
            translation = "\n\n".join(part for part in aggregated if part).strip()
            logger.debug("[server] streaming translation done | chars=%d", len(translation))
            yield json.dumps({"type": "done", "translation": translation}, ensure_ascii=False) + "\n"
        except OllamaClientError as exc:
            logger.exception("[server] streaming translation error: %s", exc)
            yield json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/api/tts")
def synthesize_tts(payload: TTSRequest = Body(...)) -> StreamingResponse:
    logger.debug(
        "[server] POST /api/tts | len=%d lang=%s speaker=%s wav=%s save=%s",
        len((payload.text or "").strip()),
        payload.language,
        payload.speaker,
        bool(payload.speaker_wav),
        bool(payload.save_output),
    )
    def event_stream() -> Iterable[str]:
        import json

        text = (payload.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="낭독할 영어 문장을 입력해 주세요.")

        try:
            logger.debug("[server] TTS stream started | len=%d", len(text))
            yield json.dumps({"type": "meta", "total": 100}, ensure_ascii=False) + "\n"
            yield json.dumps(
                {"type": "progress", "percent": 15, "message": "모델 준비 중..."}, ensure_ascii=False
            ) + "\n"

            result: TTSResult = tts_service.synthesize_to_mp3(
                text,
                language=payload.language,
                speaker_wav=payload.speaker_wav,
                speaker=payload.speaker,
                save_to_disk=bool(payload.save_output),
                filename=payload.filename,
            )

            yield json.dumps(
                {"type": "progress", "percent": 80, "message": "MP3 변환 중..."}, ensure_ascii=False
            ) + "\n"

            encoded = tts_service.encode_base64(result.audio_bytes)
            yield json.dumps(
                {
                    "type": "done",
                    "percent": 100,
                    "audio_base64": encoded,
                    "mime": "audio/mpeg",
                    "language": result.language,
                    "speaker": result.speaker,
                    "saved_path": str(result.saved_path) if result.saved_path else None,
                    "filename": result.filename,
                },
                ensure_ascii=False,
            ) + "\n"
            logger.debug("[server] TTS stream completed | speaker=%s saved=%s", result.speaker, result.saved_path)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - runtime error path
            logger.exception("[server] TTS stream error: %s", exc)
            yield json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


def get_app() -> FastAPI:
    """Expose app instance for ASGI servers."""

    return app


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    import uvicorn

    uvicorn.run("LLMEnabledTranslation.server:get_app", host="0.0.0.0", port=8000, reload=False)
