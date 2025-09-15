import os
import re
import tempfile
from pathlib import Path
from typing import List, Tuple

import gradio as gr

# Prefer higher-quality translator (offline-capable MarianMT EN↔KO)
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Optional online TTS (better quality than pyttsx3)
try:
    from gtts import gTTS
    _has_gtts = True
except Exception:
    _has_gtts = False

# Offline TTS fallback
try:
    import pyttsx3
    _has_pyttsx3 = True
except Exception:
    _has_pyttsx3 = False


# ---------------------------- Translation (EN↔KO) ----------------------------
_device = "cuda" if torch.cuda.is_available() else "cpu"
_tok_en_ko = None
_mdl_en_ko = None
_tok_ko_en = None
_mdl_ko_en = None

MODEL_EN_KO = "Helsinki-NLP/opus-mt-en-ko"
MODEL_KO_EN = "Helsinki-NLP/opus-mt-ko-en"


def _load_pair(dir_model: str) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    tok = AutoTokenizer.from_pretrained(dir_model)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(dir_model).to(_device)
    return tok, mdl


def _ensure_models():
    global _tok_en_ko, _mdl_en_ko, _tok_ko_en, _mdl_ko_en
    if _tok_en_ko is None or _mdl_en_ko is None:
        _tok_en_ko, _mdl_en_ko = _load_pair(MODEL_EN_KO)
    if _tok_ko_en is None or _mdl_ko_en is None:
        _tok_ko_en, _mdl_ko_en = _load_pair(MODEL_KO_EN)


def _smart_split(text: str, target_chunk_chars: int = 900) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= target_chunk_chars:
        return [text]
    parts = re.split(r"(?<=[.!?\n])\s+", text)
    chunks, buf, cur = [], [], 0
    for p in parts:
        if cur + len(p) + 1 > target_chunk_chars and buf:
            chunks.append(" ".join(buf).strip())
            buf, cur = [p], len(p)
        else:
            buf.append(p)
            cur += len(p) + 1
    if buf:
        chunks.append(" ".join(buf).strip())
    return chunks


@torch.inference_mode()
def translate_en_to_ko(text: str) -> str:
    if not text.strip():
        return ""
    _ensure_models()
    outs: List[str] = []
    for chunk in _smart_split(text, 600):
        inp = _tok_en_ko(chunk, return_tensors="pt", truncation=True).to(_device)
        gen = _mdl_en_ko.generate(**inp, max_new_tokens=600)
        outs.append(_tok_en_ko.batch_decode(gen, skip_special_tokens=True)[0])
    return "\n\n".join(outs)


@torch.inference_mode()
def translate_ko_to_en(text: str) -> str:
    if not text.strip():
        return ""
    _ensure_models()
    outs: List[str] = []
    for chunk in _smart_split(text, 600):
        inp = _tok_ko_en(chunk, return_tensors="pt", truncation=True).to(_device)
        gen = _mdl_ko_en.generate(**inp, max_new_tokens=600)
        outs.append(_tok_ko_en.batch_decode(gen, skip_special_tokens=True)[0])
    return "\n\n".join(outs)


def looks_korean(text: str, threshold: float = 0.2) -> bool:
    if not text:
        return False
    total = 0
    hangul = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if "\uAC00" <= ch <= "\uD7A3":
            hangul += 1
    return (hangul / max(total, 1)) >= threshold


# ----------------------------- TTS (force EN) -------------------------------
def tts_gtts_en(text: str, out_path: str) -> str:
    if not _has_gtts:
        raise RuntimeError("gTTS 미설치: pip install gTTS")
    tts = gTTS(text=text, lang="en")
    tts.save(out_path)
    return out_path


def tts_pyttsx3_en(text: str, out_path: str) -> str:
    if not _has_pyttsx3:
        raise RuntimeError("pyttsx3 미설치: pip install pyttsx3")
    engine = pyttsx3.init()
    # Try to pick an English voice if available
    try:
        voices = engine.getProperty("voices")
        for v in voices:
            name = getattr(v, "name", "").lower()
            lang = "".join(getattr(v, "languages", [])).lower() if hasattr(v, "languages") else ""
            if "en" in name or "en" in lang:
                engine.setProperty("voice", v.id)
                break
    except Exception:
        pass
    engine.save_to_file(text, out_path)
    engine.runAndWait()
    return out_path if os.path.exists(out_path) else None


# ------------------------------- Pipeline ----------------------------------
def process(text: str, do_translate: bool, target_lang: str, tts_backend: str):
    text = (text or "").strip()
    if not text:
        return "", None, None

    # 1) Prepare display translation
    translated_display = text
    if do_translate:
        if target_lang == "ko":
            translated_display = translate_en_to_ko(text)
        else:
            # target_lang == "en"
            translated_display = translate_ko_to_en(text) if looks_korean(text) else text

    # 2) Prepare TTS text (must be English)
    tts_text_en = text if not looks_korean(text) else translate_ko_to_en(text)

    # 3) Chunk for stable synthesis
    chunks = _smart_split(tts_text_en, target_chunk_chars=1500)

    # 4) Synthesize and optionally concatenate
    tmpdir = tempfile.mkdtemp(prefix="tts_")
    parts: List[str] = []
    for i, ch in enumerate(chunks):
        out_file = os.path.join(tmpdir, f"part_{i:03d}.mp3")
        if tts_backend == "gTTS (online)":
            made = tts_gtts_en(ch, out_file)
        else:
            made = tts_pyttsx3_en(ch, out_file)
        parts.append(made)

    final_mp3 = os.path.join(tmpdir, "output.mp3")
    try:
        from pydub import AudioSegment
        combined = AudioSegment.empty()
        for p in parts:
            combined += AudioSegment.from_file(p)
        combined.export(final_mp3, format="mp3")
        audio_path = final_mp3
    except Exception:
        # No pydub/ffmpeg; return first part
        audio_path = parts[0]

    # 5) Outputs: translated text for UI, audio for preview + download
    return translated_display, audio_path, audio_path


# --------------------------------- UI --------------------------------------
with gr.Blocks(title="Translate + TTS (EN speech)") as demo:
    gr.Markdown(
        """
        # Translate + TTS
        - 번역 품질 개선: MarianMT EN↔KO 사용(오프라인 가능)
        - TTS는 항상 영어로 읽습니다 (한국어 입력 시 자동 번역)
        - 기본 TTS는 gTTS(온라인, 더 자연스러움), 오프라인은 pyttsx3 백업
        """
    )
    with gr.Row():
        with gr.Column():
            raw_text = gr.Textbox(label="Paste text", lines=10)
            do_translate = gr.Checkbox(label="Translate? (display only)", value=True)
            target_lang = gr.Dropdown(choices=["ko", "en"], value="ko", label="Target language (display)")
            tts_backend = gr.Radio(["gTTS (online)", "pyttsx3 (offline)"], value="gTTS (online)", label="TTS backend")
            go = gr.Button("Run")
        with gr.Column():
            translated_out = gr.Textbox(label="Translated text", lines=20)
            audio_out = gr.Audio(label="TTS (English)", type="filepath")
            mp3_file = gr.File(label="Download MP3")

    go.click(
        process,
        inputs=[raw_text, do_translate, target_lang, tts_backend],
        outputs=[translated_out, audio_out, mp3_file],
    )


if __name__ == "__main__":
    demo.launch()
