"""
Translate + TTS Agent (Gradio)
--------------------------------
Lightweight app to translate and synthesize audio for text files.

What’s new (Meta models)
- Translation engines now include:
  * HF Opus-MT (default, offline-capable)
  * HF NLLB-200 600M (quality↑, larger)
  * HF NLLB-200 3.3B (quality↑↑, very large)
  * Meta SeamlessM4T v2 (text→text; optional)
- TTS backends now include:
  * Meta SeamlessM4T v2 (text→speech; optional)
  * gTTS (Google, online)
  * pyttsx3 (offline basic)

Notes / setup
- NLLB-200 3.3B and SeamlessM4T v2 are large models. Expect high VRAM/CPU/RAM usage.
- Install requirements as needed:
  pip install -U gradio transformers torch sentencepiece gTTS pyttsx3 pydub soundfile
  # for SeamlessM4T v2 via Transformers (recommended):
  pip install -U transformers>=4.41 torch torchaudio sentencepiece soundfile
  # (Optional alternative) via Meta’s seamless_communication
  pip install -U fairseq2 torchaudio soundfile seamless-communication

Tips
- The app auto-chunks long text for TTS stability.
- SeamlessM4T v2 tasks will only load when selected.
"""

import io
import os
import re
import time
import tempfile
import json
import csv
import difflib
import math
from pathlib import Path
from typing import Tuple, Optional, List
import threading

import gradio as gr

# Translation (offline-capable) ---------------------------------------------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Default EN->KO translation model
DEFAULT_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-ko"
ALT_TRANSLATION_MODEL_KO_EN = "Helsinki-NLP/opus-mt-ko-en"
NLLB_MODEL = "facebook/nllb-200-distilled-600M"      # optional, larger but better
NLLB_MODEL_33B = "facebook/nllb-200-3.3B"            # very large, best quality (VRAM heavy)
SEAMLESS_M4T_MODEL = "facebook/seamless-m4t-v2-large" # Meta SeamlessM4T v2

_tokenizer = None
_model = None
_tokenizer_ko_en = None
_model_ko_en = None
_tokenizer_nllb = None
_model_nllb = None
_tokenizer_nllb_33b = None
_model_nllb_33b = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

# Global abort event for user-initiated stop
_ABORT_EVENT = threading.Event()

# Runtime-tunable TTS settings
_CURRENT_TTS_BACKEND = "SeamlessM4T v2 (TTS)"
_TTS_ENABLED = True

# Simple translation cache for repeated sentences
_translation_cache = {}

# Translation defaults / fallback controls
_PREFER_NLLB_DEFAULT = False
_FALLBACK_TO_NLLB = False

def _abort_requested() -> bool:
    return _ABORT_EVENT.is_set()

def _abort_set():
    _ABORT_EVENT.set()

def _abort_clear():
    _ABORT_EVENT.clear()


def _set_tts_backend(backend: str):
    global _CURRENT_TTS_BACKEND
    _CURRENT_TTS_BACKEND = backend or _CURRENT_TTS_BACKEND
    return f"TTS backend: {_CURRENT_TTS_BACKEND}"


def _set_tts_enabled(enabled: bool):
    global _TTS_ENABLED
    _TTS_ENABLED = bool(enabled)
    # If toggled off during synthesis, request stop
    if not _TTS_ENABLED:
        _abort_set()
    return f"TTS enabled: {_TTS_ENABLED}"


def _apply_default_engine(selection: str):
    """Map a simple default-engine selection to the translation backend radio."""
    try:
        mapping = {
            "Opus-MT (default)": "HF: Opus-MT (default)",
            "NLLB-200 600M": "HF: NLLB-200 600M (better)",
            "NLLB-200 3.3B": "HF: NLLB-200 3.3B (best)",
        }
        val = mapping.get(selection)
        if val:
            return gr.update(value=val)
    except Exception:
        pass
    return None


def _set_fallback_to_nllb(enabled: bool):
    global _FALLBACK_TO_NLLB
    _FALLBACK_TO_NLLB = bool(enabled)
    return f"Fallback to NLLB: {_FALLBACK_TO_NLLB}"

# Optional Argos Translate fallback (offline)
try:
    import argostranslate.package as _argos_pkg
    import argostranslate.translate as _argos_tr
    _has_argos = True
except Exception:
    _has_argos = False


def load_translation(model_name: str = DEFAULT_TRANSLATION_MODEL):
    global _tokenizer, _model
    if _tokenizer is None or _model is None or getattr(_model, "name_or_path", None) != model_name:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(_device)
    return _tokenizer, _model

def load_translation_ko_en():
    global _tokenizer_ko_en, _model_ko_en
    if _tokenizer_ko_en is None or _model_ko_en is None:
        _tokenizer_ko_en = AutoTokenizer.from_pretrained(ALT_TRANSLATION_MODEL_KO_EN)
        _model_ko_en = AutoModelForSeq2SeqLM.from_pretrained(ALT_TRANSLATION_MODEL_KO_EN).to(_device)
    return _tokenizer_ko_en, _model_ko_en


def load_translation_nllb():
    """Lazy-load NLLB-200 for higher-quality EN↔KO."""
    global _tokenizer_nllb, _model_nllb
    if _tokenizer_nllb is None or _model_nllb is None:
        _tokenizer_nllb = AutoTokenizer.from_pretrained(NLLB_MODEL)
        _model_nllb = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL).to(_device)
    return _tokenizer_nllb, _model_nllb


def load_translation_nllb_33b():
    """Lazy-load NLLB-200 3.3B (very large)."""
    global _tokenizer_nllb_33b, _model_nllb_33b
    if _tokenizer_nllb_33b is None or _model_nllb_33b is None:
        _tokenizer_nllb_33b = AutoTokenizer.from_pretrained(NLLB_MODEL_33B)
        _model_nllb_33b = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_33B).to(_device)
    return _tokenizer_nllb_33b, _model_nllb_33b


@torch.inference_mode()
def translate_text(text: str, model_name: str = DEFAULT_TRANSLATION_MODEL, max_len: int = 600,
                   num_beams: int = 5, no_repeat_ngram_size: int = 3) -> str:
    if not text.strip():
        return ""
    try:
        tok, mdl = load_translation(model_name)
        sentences = smart_split(text, target_chunk_chars=max_len)
        outputs: List[str] = []
        for s in sentences:
            if _abort_requested():
                break
            inputs = tok(s, return_tensors="pt", truncation=True).to(_device)
            gen = mdl.generate(
                **inputs,
                max_new_tokens=600,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=1.0,
            )
            out = tok.batch_decode(gen, skip_special_tokens=True)[0]
            outputs.append(out)
        result = join_with_spacing(outputs)
        if result.strip():
            return result
    except Exception:
        pass

    # Fallback: Argos Translate EN->KO if available
    if _has_argos:
        try:
            return translate_argos(text, from_code="en", to_code="ko")
        except Exception:
            return ""
    return ""


# -------------------- Argos fallback helpers --------------------
def ensure_argos_model_installed(from_code: str, to_code: str):
    if not _has_argos:
        return
    installed = _argos_tr.get_installed_languages()
    has_from = any(l.code == from_code for l in installed)
    has_to = any(l.code == to_code for l in installed)
    if has_from and has_to:
        return
    # Try to download the specific pair if available
    packages = _argos_pkg.get_available_packages()
    for p in packages:
        if p.from_code == from_code and p.to_code == to_code:
            path = p.download()
            _argos_pkg.install_from_path(path)
            break


def translate_argos(text: str, from_code: str, to_code: str) -> str:
    ensure_argos_model_installed(from_code, to_code)
    langs = _argos_tr.get_installed_languages()
    from_lang = next(l for l in langs if l.code == from_code)
    to_lang = next(l for l in langs if l.code == to_code)
    tr = from_lang.get_translation(to_lang)
    return tr.translate(text)


@torch.inference_mode()
def translate_ko_to_en(text: str, max_len: int = 600,
                       num_beams: int = 5, no_repeat_ngram_size: int = 3) -> str:
    if not text.strip():
        return ""
    tok, mdl = load_translation_ko_en()
    sentences = smart_split(text, target_chunk_chars=max_len)
    outputs: List[str] = []
    for s in sentences:
        if _abort_requested():
            break
        inputs = tok(s, return_tensors="pt", truncation=True).to(_device)
        gen = mdl.generate(
            **inputs,
            max_new_tokens=600,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=1.0,
        )
        out = tok.batch_decode(gen, skip_special_tokens=True)[0]
        outputs.append(out)
    return join_with_spacing(outputs)


@torch.inference_mode()
def translate_nllb(text: str, from_code: str, to_code: str,
                   max_len: int = 600, num_beams: int = 5) -> str:
    """Translate via NLLB-200. from_code/to_code: 'eng_Latn'/'kor_Hang'."""
    if not text.strip():
        return ""
    tok, mdl = load_translation_nllb()
    sentences = smart_split(text, target_chunk_chars=max_len)
    outs: List[str] = []
    # Ensure tokenizer knows the source language
    try:
        tok.src_lang = from_code
    except Exception:
        pass
    forced_bos = tok.lang_code_to_id[to_code]
    for s in sentences:
        if _abort_requested():
            break
        inputs = tok(s, return_tensors="pt", truncation=True).to(_device)
        gen = mdl.generate(
            **inputs,
            max_new_tokens=600,
            num_beams=num_beams,
            early_stopping=True,
            forced_bos_token_id=forced_bos,
        )
        outs.append(tok.batch_decode(gen, skip_special_tokens=True)[0])
    return join_with_spacing(outs)


@torch.inference_mode()
def translate_nllb_33b(text: str, from_code: str, to_code: str,
                       max_len: int = 600, num_beams: int = 5) -> str:
    """Translate via NLLB-200 3.3B. from_code/to_code: 'eng_Latn'/'kor_Hang'."""
    if not text.strip():
        return ""
    tok, mdl = load_translation_nllb_33b()
    sentences = smart_split(text, target_chunk_chars=max_len)
    outs: List[str] = []
    # Ensure tokenizer knows the source language
    try:
        tok.src_lang = from_code
    except Exception:
        pass
    forced_bos = tok.lang_code_to_id[to_code]
    for s in sentences:
        if _abort_requested():
            break
        inputs = tok(s, return_tensors="pt", truncation=True).to(_device)
        gen = mdl.generate(
            **inputs,
            max_new_tokens=600,
            num_beams=num_beams,
            early_stopping=True,
            forced_bos_token_id=forced_bos,
        )
        outs.append(tok.batch_decode(gen, skip_special_tokens=True)[0])
    return join_with_spacing(outs)


# -------------------- SeamlessM4T v2 (optional) --------------------
def _load_seamless_t2t():
    """Load SeamlessM4T v2 for Text→Text with best available class.
    Returns (processor, model)."""
    try:
        # Try task-specific class first
        try:
            from transformers import SeamlessM4Tv2ForTextToText as _T2T
        except Exception:
            _T2T = None
        try:
            from transformers import SeamlessM4TProcessor as _Proc
        except Exception:
            _Proc = None
        if _T2T is not None and _Proc is not None:
            processor = _Proc.from_pretrained(SEAMLESS_M4T_MODEL)
            model = _T2T.from_pretrained(SEAMLESS_M4T_MODEL).to(_device)
        else:
            # Fallback to AutoProcessor + generic model
            from transformers import AutoProcessor
            try:
                from transformers import SeamlessM4Tv2Model as _SeamlessModel
            except Exception:
                from transformers import SeamlessM4TModel as _SeamlessModel
            processor = AutoProcessor.from_pretrained(SEAMLESS_M4T_MODEL)
            model = _SeamlessModel.from_pretrained(SEAMLESS_M4T_MODEL).to(_device)
        # Disable cache for stability on some builds
        try:
            if hasattr(model, 'config'):
                model.config.use_cache = False
        except Exception:
            pass
        return processor, model
    except Exception as e:
        raise RuntimeError(
            "SeamlessM4T v2 not available. Update transformers/torchaudio/sentencepiece.\n"
            "pip install -U transformers torch torchaudio sentencepiece soundfile"
        ) from e


def _load_seamless_t2s():
    """Load SeamlessM4T v2 for Text→Speech with best available class.
    Returns (processor, model)."""
    try:
        try:
            from transformers import SeamlessM4Tv2ForTextToSpeech as _T2S
        except Exception:
            _T2S = None
        try:
            from transformers import SeamlessM4TProcessor as _Proc
        except Exception:
            _Proc = None
        if _T2S is not None and _Proc is not None:
            processor = _Proc.from_pretrained(SEAMLESS_M4T_MODEL)
            model = _T2S.from_pretrained(SEAMLESS_M4T_MODEL).to(_device)
        else:
            # Fallback to AutoProcessor + generic model
            from transformers import AutoProcessor
            try:
                from transformers import SeamlessM4Tv2Model as _SeamlessModel
            except Exception:
                from transformers import SeamlessM4TModel as _SeamlessModel
            processor = AutoProcessor.from_pretrained(SEAMLESS_M4T_MODEL)
            model = _SeamlessModel.from_pretrained(SEAMLESS_M4T_MODEL).to(_device)
        # Disable cache for stability on some builds
        try:
            if hasattr(model, 'config'):
                model.config.use_cache = False
        except Exception:
            pass
        return processor, model
    except Exception as e:
        raise RuntimeError(
            "SeamlessM4T v2 not available. Update transformers/torchaudio/sentencepiece.\n"
            "pip install -U transformers torch torchaudio sentencepiece soundfile"
        ) from e


@torch.inference_mode()
def translate_seamless_m4t(text: str, from_lang: str, to_lang: str, max_len: int = 600) -> str:
    """Text→Text via SeamlessM4T v2 using Transformers.
    from_lang/to_lang: ISO-3 codes like 'eng', 'kor'.
    """
    if not text.strip():
        return ""
    processor, model = _load_seamless_t2t()
    sentences = smart_split(text, target_chunk_chars=max_len)
    outs: List[str] = []
    for s in sentences:
        if _abort_requested():
            break
        inputs = processor(text=s, src_lang=from_lang, return_tensors="pt").to(_device)
        try:
            gen = model.generate(**inputs, tgt_lang=to_lang, use_cache=False)
        except TypeError:
            # older API may require keyword names slightly differ; fallback to positional
            gen = model.generate(**inputs)
        # Decode
        out = None
        try:
            out = processor.decode(gen[0].tolist(), skip_special_tokens=True)
        except Exception:
            try:
                out = processor.batch_decode(gen, skip_special_tokens=True)[0]
            except Exception:
                pass
        if out is None:
            # As ultimate fallback, return raw ids as string
            out = str(gen[0].tolist())
        outs.append(out)
    return join_with_spacing(outs)


@torch.inference_mode()
def text_to_mp3_seamless_m4t(text: str, src_lang: str = "eng", tgt_lang: str = "eng", out_path: str = "tts_seamless.mp3") -> str:
    """Text→Speech via SeamlessM4T v2 using Transformers. Returns path to audio (mp3 if possible, else wav).
    Note: Requires transformers with SeamlessM4T v2 support and soundfile/pydub for writing.
    """
    processor, model = _load_seamless_t2s()
    inputs = processor(text=text, src_lang=src_lang, return_tensors="pt").to(_device)
    # Generate audio
    try:
        gen = model.generate(**inputs, tgt_lang=tgt_lang, generate_speech=True, use_cache=False)
    except TypeError:
        # Some builds expose a separate flag or attribute; try without explicit flag
        gen = model.generate(**inputs)

    # Extract waveform
    audio = None
    try:
        if isinstance(gen, (list, tuple)):
            audio = gen[0]
        elif hasattr(gen, "audio_values"):
            audio = gen.audio_values[0]
        elif torch.is_tensor(gen):
            audio = gen.squeeze(0).detach().cpu().numpy()
    except Exception:
        pass
    if audio is None:
        raise RuntimeError("Could not obtain audio from SeamlessM4T v2 output.")

    # Write WAV
    wav_path = out_path.replace(".mp3", ".wav")
    wrote_wav = False
    try:
        import soundfile as sf
        sf.write(wav_path, audio, 16000)
        wrote_wav = True
    except Exception:
        try:
            from scipy.io import wavfile
            import numpy as np
            arr = audio if hasattr(audio, "__array__") else np.array(audio)
            wavfile.write(wav_path, 16000, arr)
            wrote_wav = True
        except Exception:
            pass
    if not wrote_wav:
        # As a last resort, return any temp path and let caller handle
        return wav_path

    # Convert to MP3 if possible
    try:
        from pydub import AudioSegment
        AudioSegment.from_wav(wav_path).export(out_path, format="mp3")
        try:
            os.remove(wav_path)
        except Exception:
            pass
        return out_path
    except Exception:
        return wav_path


def translate_with_backend(text: str, backend: str, target_lang: str) -> str:
    """Switchable translation like TTS backend selection."""
    if not text.strip():
        return ""
    target_lang = (target_lang or "ko").lower()
    is_src_ko = looks_korean(text)
    try:
        if backend.startswith("HF: Opus-MT"):
            if target_lang.startswith("ko") and not is_src_ko:
                try:
                    return translate_text(text, DEFAULT_TRANSLATION_MODEL)
                except Exception:
                    if _FALLBACK_TO_NLLB:
                        return translate_nllb(text, from_code="eng_Latn", to_code="kor_Hang")
                    raise
            elif target_lang.startswith("en") and is_src_ko:
                try:
                    return translate_ko_to_en(text)
                except Exception:
                    if _FALLBACK_TO_NLLB:
                        return translate_nllb(text, from_code="kor_Hang", to_code="eng_Latn")
                    raise
            else:
                return text
        elif backend.startswith("HF: NLLB"):
            # NLLB language tags
            from_code = "kor_Hang" if is_src_ko else "eng_Latn"
            to_code = "eng_Latn" if is_src_ko else ("kor_Hang" if target_lang.startswith("ko") else "eng_Latn")
            if (is_src_ko and target_lang.startswith("en")) or (not is_src_ko and target_lang.startswith("ko")):
                # Distinguish between 600M and 3.3B
                if "3.3B" in backend:
                    return translate_nllb_33b(text, from_code=from_code, to_code=to_code)
                return translate_nllb(text, from_code=from_code, to_code=to_code)
            else:
                return text
        elif backend.startswith("Meta: SeamlessM4T v2"):
            from_lang = "kor" if is_src_ko else "eng"
            to_lang = "eng" if is_src_ko else ("kor" if target_lang.startswith("ko") else "eng")
            if (is_src_ko and target_lang.startswith("en")) or (not is_src_ko and target_lang.startswith("ko")):
                return translate_seamless_m4t(text, from_lang=from_lang, to_lang=to_lang)
            else:
                return text
        elif backend.startswith("Both: NLLB 3.3B + Seamless v2"):
            from_code = "kor_Hang" if is_src_ko else "eng_Latn"
            to_code = "eng_Latn" if is_src_ko else ("kor_Hang" if target_lang.startswith("ko") else "eng_Latn")
            from_lang = "kor" if is_src_ko else "eng"
            to_lang = "eng" if is_src_ko else ("kor" if target_lang.startswith("ko") else "eng")
            nllb_txt = translate_nllb_33b(text, from_code=from_code, to_code=to_code) if ((is_src_ko and target_lang.startswith("en")) or (not is_src_ko and target_lang.startswith("ko"))) else text
            try:
                sm4t_txt = translate_seamless_m4t(text, from_lang=from_lang, to_lang=to_lang)
            except Exception as _:
                sm4t_txt = "(SeamlessM4T v2 unavailable — see setup notes)"
            return f"[NLLB-200 3.3B]\n{nllb_txt}\n\n[SeamlessM4T v2]\n{sm4t_txt}"
        elif backend.startswith("Argos") and _has_argos:
            if target_lang.startswith("ko") and not is_src_ko:
                return translate_argos(text, from_code="en", to_code="ko")
            elif target_lang.startswith("en") and is_src_ko:
                return translate_argos(text, from_code="ko", to_code="en")
            else:
                return text
    except Exception:
        # Fallback to Opus-MT if something failed
        try:
            if target_lang.lower().startswith("ko") and not is_src_ko:
                return translate_text(text, DEFAULT_TRANSLATION_MODEL)
            elif target_lang.lower().startswith("en") and is_src_ko:
                return translate_ko_to_en(text)
        except Exception:
            pass
    return text


# ===================== Advanced translation utilities =====================
def _adv_signature(cfg: dict) -> tuple:
    keys = (
        'num_beams','no_repeat_ngram_size','length_penalty','repetition_penalty',
        'n_best','protect_numbers','protect_caps_code','apply_glossary','bt_check','bt_min_sim',
        'style','domain','cache_repeats'
    )
    return tuple((k, cfg.get(k)) for k in keys)


def _load_glossary(glossary_file: Optional[gr.File]) -> list:
    if glossary_file is None:
        return []
    path = glossary_file.name
    items = []
    try:
        if path.lower().endswith('.json'):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = data.get('items', [])
            for it in data:
                src = it.get('src') or it.get('source')
                tgt = it.get('tgt') or it.get('target')
                mode = (it.get('mode') or 'fixed').lower()
                regex = bool(it.get('regex', False))
                ci = bool(it.get('ci', True))
                if src:
                    items.append((src, tgt, mode, regex, ci))
        else:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    src = row.get('src') or row.get('source')
                    tgt = row.get('tgt') or row.get('target')
                    mode = (row.get('mode') or 'fixed').lower()
                    regex = (row.get('regex') or '').strip().lower() in ('1','true','yes')
                    ci = (row.get('ci') or 'true').strip().lower() in ('1','true','yes')
                    if src:
                        items.append((src, tgt, mode, regex, ci))
    except Exception:
        pass
    return items


def _protect_text(text: str, protect_numbers: bool, protect_caps_code: bool, glossary_rules: list):
    """Protect tokens from translation using placeholders. Returns (text, restore_map).
    restore_map maps placeholder -> replacement_after_translation.
    """
    restore = {}
    idx = {'NUM':0,'CAP':0,'CODE':0,'TERM':0}

    def put(kind: str, val: str, replacement: Optional[str]=None):
        idx[kind] += 1
        ph = f"[[{kind}{idx[kind]}]]"
        restore[ph] = replacement if (replacement is not None) else val
        return ph

    out = text

    if protect_numbers:
        # numbers with optional unit, percents, currency, ranges
        pattern = re.compile(r"(?i)(\b\d{1,3}(?:[\d,\.]*\d)?(?:\s?(?:%|percent|pct|kg|g|mg|km|m|cm|mm|mb|gb|tb|kb|hz|khz|mhz|ghz|usd|eur|krw|won|$|¥|€))?\b)")
        def repl_num(m):
            return put('NUM', m.group(0))
        out = pattern.sub(repl_num, out)

    if protect_caps_code:
        # inline code blocks: `code`
        code_pat = re.compile(r"`([^`]+)`")
        def repl_code(m):
            return put('CODE', m.group(1))
        out = code_pat.sub(lambda m: put('CODE', m.group(1)), out)
        # ALL CAPS tokens (2+ letters)
        caps_pat = re.compile(r"\b[A-Z]{2,}(?:[A-Z0-9_-]{0,})\b")
        out = caps_pat.sub(lambda m: put('CAP', m.group(0)), out)

    # Glossary protection
    for src, tgt, mode, is_regex, ci in glossary_rules or []:
        flags = re.IGNORECASE if ci else 0
        try:
            if is_regex:
                pat = re.compile(src, flags)
                out = pat.sub(lambda m: put('TERM', m.group(0), replacement=(tgt if mode != 'keep' else m.group(0))), out)
            else:
                esc = re.escape(src)
                pat = re.compile(esc, flags)
                out = pat.sub(lambda m: put('TERM', m.group(0), replacement=(tgt if mode != 'keep' else m.group(0))), out)
        except Exception:
            continue

    return out, restore


def _restore_placeholders(text: str, restore_map: dict) -> str:
    out = text
    for ph, real in restore_map.items():
        out = out.replace(ph, real)
    return out


def _heuristic_score(src_en: str, tgt: str) -> float:
    # digits preservation
    def digits(s):
        return re.findall(r"\d", s)
    d_src = len(digits(src_en))
    d_tgt = len(digits(tgt))
    digit_score = 1.0 - min(1.0, abs(d_src - d_tgt)/max(1, d_src or d_tgt))

    # bracket/quotes balance
    def balance_ok(s):
        pairs = [('(',')'),('[',']'),('{','}'),('“','”'),('"','"'),("'","'")]
        score = 1.0
        for a,b in pairs:
            score -= 0.05 * abs(s.count(a) - s.count(b))
        return max(0.0, score)
    bal = balance_ok(tgt)

    # length ratio tolerance
    lr = len(tgt.strip())/max(1, len(src_en.strip()))
    len_score = 1.0 - min(1.0, abs(lr-1.0)) * 0.5

    return 0.4*digit_score + 0.3*bal + 0.3*len_score


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _nbest_generate(tok, mdl, inputs, n_best: int, gen_kwargs: dict) -> List[str]:
    n_best = max(1, int(n_best or 1))
    out = mdl.generate(
        **inputs,
        num_beams=max(n_best, int(gen_kwargs.get('num_beams', 5) or 5)),
        num_return_sequences=n_best,
        early_stopping=True,
        **{k:v for k,v in gen_kwargs.items() if k not in ('num_beams', 'num_return_sequences')},
    )
    return tok.batch_decode(out, skip_special_tokens=True)


@torch.inference_mode()
def translate_with_backend_enhanced(text: str, backend: str, target_lang: str, cfg: dict,
                                    glossary_file: Optional[gr.File]) -> str:
    if not text.strip():
        return ""
    target_lang = (target_lang or "ko").lower()
    is_src_ko = looks_korean(text)

    # Cache key
    cache_key = (backend, target_lang, _adv_signature(cfg), text)
    if cfg.get('cache_repeats') and cache_key in _translation_cache:
        return _translation_cache[cache_key]

    # Load glossary rules (once per call)
    glossary_rules = _load_glossary(glossary_file) if cfg.get('apply_glossary') else []

    # Build style/domain hint prefix if requested
    hint = ""
    if cfg.get('style') and cfg.get('style') != 'Neutral':
        hint += f"Style: {cfg['style']}. "
    if cfg.get('domain') and cfg.get('domain') != 'General':
        hint += f"Domain: {cfg['domain']}. "
    if hint:
        if not is_src_ko and target_lang.startswith('ko'):
            hint = "Translate to Korean. " + hint
        elif is_src_ko and target_lang.startswith('en'):
            hint = "Translate to English. " + hint

    # Sentence chunks
    chunks = smart_split(text, target_chunk_chars=int(cfg.get('max_len', 600) or 600))
    outs: List[str] = []

    # Generation kwargs
    gen_kwargs = dict(
        max_new_tokens=600,
        num_beams=int(cfg.get('num_beams', 5) or 5),
        no_repeat_ngram_size=int(cfg.get('no_repeat_ngram_size', 3) or 3),
        length_penalty=float(cfg.get('length_penalty', 1.0) or 1.0),
        repetition_penalty=float(cfg.get('repetition_penalty', 1.0) or 1.0),
    )
    n_best = int(cfg.get('n_best', 1) or 1)
    do_bt = bool(cfg.get('bt_check'))
    bt_min = float(cfg.get('bt_min_sim', 0.82) or 0.82)

    for s in chunks:
        if _abort_requested():
            break
        # Protection + glossary
        protected, restore_map = _protect_text(s, cfg.get('protect_numbers'), cfg.get('protect_caps_code'), glossary_rules)
        enc_inp = (hint + protected).strip() if hint else protected

        # Build inputs per backend and produce candidates
        cands: List[str] = []
        try:
            if backend.startswith("HF: Opus-MT"):
                try:
                    tok, mdl = load_translation(DEFAULT_TRANSLATION_MODEL if (target_lang.startswith('ko') and not is_src_ko) else ALT_TRANSLATION_MODEL_KO_EN)
                    inputs = tok(enc_inp, return_tensors="pt", truncation=True).to(_device)
                    cands = _nbest_generate(tok, mdl, inputs, n_best=n_best, gen_kwargs=gen_kwargs)
                except Exception:
                    if _FALLBACK_TO_NLLB:
                        # Use NLLB 600M with proper language ids
                        tok, mdl = load_translation_nllb()
                        from_code = "kor_Hang" if is_src_ko else "eng_Latn"
                        to_code = "eng_Latn" if is_src_ko else ("kor_Hang" if target_lang.startswith("ko") else "eng_Latn")
                        try:
                            tok.src_lang = from_code
                        except Exception:
                            pass
                        inputs = tok(enc_inp, return_tensors="pt", truncation=True).to(_device)
                        forced_bos = tok.lang_code_to_id[to_code]
                        cands = _nbest_generate(tok, mdl, inputs, n_best=n_best, gen_kwargs={**gen_kwargs, 'forced_bos_token_id': forced_bos})
                    else:
                        raise
            elif backend.startswith("HF: NLLB-200 3.3B"):
                tok, mdl = load_translation_nllb_33b()
                from_code = "kor_Hang" if is_src_ko else "eng_Latn"
                to_code = "eng_Latn" if is_src_ko else ("kor_Hang" if target_lang.startswith("ko") else "eng_Latn")
                # set source language for tokenizer
                try:
                    tok.src_lang = from_code
                except Exception:
                    pass
                inputs = tok(enc_inp, return_tensors="pt", truncation=True).to(_device)
                forced_bos = tok.lang_code_to_id[to_code]
                cands = _nbest_generate(tok, mdl, inputs, n_best=n_best, gen_kwargs={**gen_kwargs, 'forced_bos_token_id': forced_bos})
            elif backend.startswith("HF: NLLB-200 600M") or backend.startswith("HF: NLLB-200 (better)"):
                tok, mdl = load_translation_nllb()
                from_code = "kor_Hang" if is_src_ko else "eng_Latn"
                to_code = "eng_Latn" if is_src_ko else ("kor_Hang" if target_lang.startswith("ko") else "eng_Latn")
                try:
                    tok.src_lang = from_code
                except Exception:
                    pass
                inputs = tok(enc_inp, return_tensors="pt", truncation=True).to(_device)
                forced_bos = tok.lang_code_to_id[to_code]
                cands = _nbest_generate(tok, mdl, inputs, n_best=n_best, gen_kwargs={**gen_kwargs, 'forced_bos_token_id': forced_bos})
            elif backend.startswith("Meta: SeamlessM4T v2"):
                # Single best candidate; still apply protections
                out1 = translate_seamless_m4t(enc_inp, from_lang=("kor" if is_src_ko else "eng"), to_lang=("eng" if is_src_ko else "kor"), max_len=int(cfg.get('max_len', 600) or 600))
                cands = [out1]
            elif backend.startswith("Both: NLLB 3.3B + Seamless v2"):
                nllb_txt = translate_nllb_33b(enc_inp, from_code=("kor_Hang" if is_src_ko else "eng_Latn"), to_code=("eng_Latn" if is_src_ko else "kor_Hang"))
                try:
                    sm4t_txt = translate_seamless_m4t(enc_inp, from_lang=("kor" if is_src_ko else "eng"), to_lang=("eng" if is_src_ko else "kor"))
                except Exception:
                    sm4t_txt = "(SeamlessM4T v2 unavailable)"
                # Restore placeholders in both then join
                nllb_txt = _restore_placeholders(nllb_txt, restore_map)
                sm4t_txt = _restore_placeholders(sm4t_txt, restore_map)
                out = f"[NLLB-200 3.3B]\n{nllb_txt}\n\n[SeamlessM4T v2]\n{sm4t_txt}"
                outs.append(out)
                continue
            elif backend.startswith("Argos") and _has_argos:
                if target_lang.startswith("ko") and not is_src_ko:
                    out1 = translate_argos(enc_inp, from_code="en", to_code="ko")
                elif target_lang.startswith("en") and is_src_ko:
                    out1 = translate_argos(enc_inp, from_code="ko", to_code="en")
                else:
                    out1 = enc_inp
                cands = [out1]
            else:
                cands = [s]
        except Exception:
            # fallback to basic path
            cands = [translate_with_backend(s, backend, target_lang)]

        # Restore placeholders for each candidate
        cands = [_restore_placeholders(c, restore_map) for c in cands]

        # Re-ranking heuristics
        if len(cands) > 1:
            scored = []
            for c in cands:
                base = _heuristic_score(text if not is_src_ko else translate_ko_to_en(s), c)
                scored.append((base, c))
            cands = [c for _, c in sorted(scored, key=lambda x: x[0], reverse=True)]

        best = cands[0]

        # Back-translation quality check
        if do_bt:
            try:
                if not is_src_ko and target_lang.startswith('ko'):
                    bt = translate_ko_to_en(best)
                    sim = _similarity(text, bt)
                elif is_src_ko and target_lang.startswith('en'):
                    # EN backtranslation via Opus-MT EN<-KO
                    bt = translate_text(best, ALT_TRANSLATION_MODEL_KO_EN)
                    sim = _similarity(s, bt)
                else:
                    sim = 1.0
            except Exception:
                sim = 0.0

            if sim < bt_min and len(cands) > 1:
                # Try next best candidate
                for alt in cands[1:]:
                    try:
                        if not is_src_ko and target_lang.startswith('ko'):
                            sim_alt = _similarity(text, translate_ko_to_en(alt))
                        else:
                            sim_alt = _similarity(s, translate_text(alt, ALT_TRANSLATION_MODEL_KO_EN))
                    except Exception:
                        sim_alt = 0.0
                    if sim_alt > sim:
                        best = alt
                        sim = sim_alt

        outs.append(best)

    out_text = join_with_spacing(outs)
    if cfg.get('cache_repeats'):
        _translation_cache[cache_key] = out_text
        # naive pruning
        if len(_translation_cache) > 2048:
            for _ in range(256):
                try:
                    _translation_cache.pop(next(iter(_translation_cache)))
                except Exception:
                    break
    return out_text


# TTS backends ---------------------------------------------------------------
# 1) gTTS (online)
from gtts import gTTS

# 2) pyttsx3 (offline)
try:
    import pyttsx3
    _has_pyttsx3 = True
except Exception:
    _has_pyttsx3 = False


def text_to_mp3_gtts(text: str, lang_code: str = "en", out_path: str = "tts_gtts.mp3") -> str:
    tts = gTTS(text=text, lang=lang_code)
    tts.save(out_path)
    return out_path


def text_to_mp3_pyttsx3(text: str, voice_index: Optional[int] = None, rate: Optional[int] = None, out_path: str = "tts_pyttsx3.mp3") -> str:
    if not _has_pyttsx3:
        raise RuntimeError("pyttsx3 is not installed. Install it or switch to gTTS.")
    # pyttsx3 natively writes to WAV; we'll convert to mp3 using pydub if available.
    import platform
    engine = pyttsx3.init()
    if rate:
        engine.setProperty('rate', rate)
    if voice_index is not None:
        voices = engine.getProperty('voices')
        if 0 <= voice_index < len(voices):
            engine.setProperty('voice', voices[voice_index].id)
    wav_tmp = out_path.replace('.mp3', '.wav')
    engine.save_to_file(text, wav_tmp)
    engine.runAndWait()
    # Convert wav->mp3 if pydub & ffmpeg exist, else return wav
    try:
        from pydub import AudioSegment
        AudioSegment.from_wav(wav_tmp).export(out_path, format='mp3')
        os.remove(wav_tmp)
        return out_path
    except Exception:
        return wav_tmp


# Utilities -----------------------------------------------------------------

def smart_split(text: str, target_chunk_chars: int = 1200) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= target_chunk_chars:
        return [text]
    # Split by sentence-ish boundaries, then pack
    parts = re.split(r"(?<=[.!?\n])\s+", text)
    chunks, buf = [], []
    cur = 0
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


def join_with_spacing(chunks: List[str]) -> str:
    return "\n\n".join([c.strip() for c in chunks if c and c.strip()])

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


def ensure_text(inp_file: Optional[gr.File], inp_text: str) -> str:
    if inp_file is not None:
        with open(inp_file.name, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    return inp_text or ""


def translate_only(text_file, raw_text, target_lang, do_translate, translation_backend,
                   num_beams, no_repeat_ngram_size, length_penalty, repetition_penalty,
                   n_best, protect_numbers, protect_caps_code, apply_glossary, glossary_file,
                   bt_check, bt_min_sim, style, domain, cache_repeats):
    """Run translation only with advanced controls; no TTS."""
    source_text = ensure_text(text_file, raw_text)
    if not source_text.strip():
        return "", source_text, ""

    if do_translate:
        cfg = dict(
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            n_best=n_best,
            protect_numbers=protect_numbers,
            protect_caps_code=protect_caps_code,
            apply_glossary=apply_glossary,
            bt_check=bt_check,
            bt_min_sim=bt_min_sim,
            style=style,
            domain=domain,
            cache_repeats=cache_repeats,
            max_len=600,
        )
        translated = translate_with_backend_enhanced(source_text, translation_backend, target_lang, cfg, glossary_file)
    else:
        translated = ""

    translated_text = translated if translated else "(No translation requested)"
    return translated_text, source_text, translated


def synthesize_only(state_source_text: str, state_translated_text: str, read_source_or_translated: str):
    """Run TTS only using current runtime settings (_TTS_ENABLED, _CURRENT_TTS_BACKEND)."""
    # Clear previous abort
    _abort_clear()

    if not _TTS_ENABLED:
        return None, None

    # Choose text to read
    base_text = (state_source_text or "") if (read_source_or_translated or "").lower().startswith("source") else (state_translated_text or "")
    if not base_text.strip():
        return None, None

    # TTS reads English; convert if needed
    chosen_text = base_text if not looks_korean(base_text) else translate_ko_to_en(base_text)
    lang_code_for_tts = 'en'

    # Chunking
    chunks = smart_split(chosen_text, target_chunk_chars=1800)

    # Synthesis with runtime-tunable backend
    tmpdir = tempfile.mkdtemp()
    out_paths = []
    aborted = False
    for i, ch in enumerate(chunks):
        if _abort_requested() or not _TTS_ENABLED:
            aborted = True
            break
        piece_base = os.path.join(tmpdir, f"part_{i:03d}.mp3")
        backend = _CURRENT_TTS_BACKEND  # read live backend selection
        if backend == "gTTS (online)":
            piece = text_to_mp3_gtts(ch, lang_code=lang_code_for_tts, out_path=piece_base)
        elif backend == "SeamlessM4T v2 (TTS)":
            try:
                piece = text_to_mp3_seamless_m4t(ch, src_lang=lang_code_for_tts if lang_code_for_tts == 'en' else 'eng', tgt_lang='eng', out_path=piece_base)
            except Exception:
                piece = text_to_mp3_gtts(ch, lang_code=lang_code_for_tts, out_path=piece_base)
        elif backend == "pyttsx3 (offline)":
            piece = text_to_mp3_pyttsx3(ch, out_path=piece_base)
        else:
            # Unknown backend; fallback
            piece = text_to_mp3_gtts(ch, lang_code=lang_code_for_tts, out_path=piece_base)
        out_paths.append(piece)

    # Concatenate
    final_mp3 = os.path.join(tmpdir, "output.mp3")
    try:
        from pydub import AudioSegment
        combined = AudioSegment.empty()
        for p in out_paths:
            seg = AudioSegment.from_file(p)
            combined += seg
        combined.export(final_mp3, format='mp3')
        mp3_path = final_mp3
    except Exception:
        mp3_path = out_paths[0] if out_paths else None

    # If aborted, still return partial audio
    return mp3_path, mp3_path


# Gradio UI -----------------------------------------------------------------
def _trigger_stop():
    _abort_set()
    return "Stopping..."


with gr.Blocks(title="Translate + TTS Agent") as demo:
    gr.Markdown("# Translate + TTS Agent")
    gr.Markdown(
        "Use the top section to translate your text, then optionally generate audio below."
    )

    # Shared state between steps
    state_source_text = gr.State("")
    state_translated_text = gr.State("")

    # ===================== Translation (Top) =====================
    with gr.Group():
        gr.Markdown("## 1) Translation")
        gr.Markdown("Upload a .txt file or paste text. Choose engine and target language.")
        text_file = gr.File(label="Upload .txt (UTF-8)")
        raw_text = gr.Textbox(label="Paste text", lines=12, placeholder="Paste or type your text here...")
        with gr.Row():
            target_lang = gr.Dropdown(
                choices=["ko (Korean)", "en (English)"],
                value="ko (Korean)",
                label="Translation target",
            )
            do_translate = gr.Checkbox(label="Enable translation", value=True)
            default_engine = gr.Dropdown(
                ["Opus-MT (default)", "NLLB-200 600M", "NLLB-200 3.3B"],
                value="Opus-MT (default)",
                label="Default engine",
            )
        translation_backend = gr.Radio(
            [
                "HF: Opus-MT (default)",
                "HF: NLLB-200 600M (better)",
                "HF: NLLB-200 3.3B (best)",
                "Meta: SeamlessM4T v2 (T2TT)",
                "Both: NLLB 3.3B + Seamless v2",
                "Argos (offline)",
            ],
            value="HF: Opus-MT (default)",
            label="Translation engine",
        )
        with gr.Accordion("Advanced settings", open=False):
            with gr.Row():
                num_beams = gr.Slider(1, 10, value=5, step=1, label="num_beams")
                n_best = gr.Slider(1, 5, value=1, step=1, label="n_best (return candidates)")
            with gr.Row():
                no_repeat_ngram_size = gr.Slider(0, 5, value=3, step=1, label="no_repeat_ngram_size")
                length_penalty = gr.Slider(0.7, 1.5, value=1.0, step=0.05, label="length_penalty")
                repetition_penalty = gr.Slider(1.0, 1.5, value=1.0, step=0.05, label="repetition_penalty")
            with gr.Row():
                protect_numbers = gr.Checkbox(value=True, label="Protect numbers/units")
                protect_caps_code = gr.Checkbox(value=True, label="Protect ALL CAPS / `code`")
                cache_repeats = gr.Checkbox(value=True, label="Cache repeated sentences")
                fallback_to_nllb = gr.Checkbox(value=True, label="Fallback to NLLB if Opus fails")
            with gr.Row():
                apply_glossary = gr.Checkbox(value=False, label="Apply glossary")
                glossary_file = gr.File(file_count="single", file_types=[".json", ".csv"], label="Glossary (JSON/CSV)")
            with gr.Row():
                bt_check = gr.Checkbox(value=False, label="Back-translation quality check")
                bt_min_sim = gr.Slider(0.5, 0.95, value=0.82, step=0.01, label="BT min similarity")
            with gr.Row():
                style = gr.Dropdown(["Neutral","Formal","Casual","Technical","Medical","Legal","Marketing"], value="Neutral", label="Style")
                domain = gr.Dropdown(["General","Technical","Medical","Legal","Marketing"], value="General", label="Domain")

        translate_btn = gr.Button("Translate", variant="primary")
        translated_out = gr.Textbox(label="Translated text", lines=18)

    # ===================== Speech / TTS (Bottom) =====================
    with gr.Group():
        gr.Markdown("## 2) Speech (TTS)")
        gr.Markdown("Select what to read, enable TTS, and choose the backend. Changes apply live.")
        read_source_or_translated = gr.Radio(
            ["Source (English only)", "Translated (English only)"],
            value="Source (English only)",
            label="Read from",
        )
        with gr.Row():
            tts_enabled = gr.Checkbox(label="Enable TTS", value=True)
            tts_backend = gr.Radio(
                ["SeamlessM4T v2 (TTS)", "gTTS (online)", "pyttsx3 (offline)"],
                value="SeamlessM4T v2 (TTS)",
                label="TTS backend (live)",
            )
        with gr.Row():
            synth_btn = gr.Button("Synthesize")
            stop_btn = gr.Button("Stop", variant="stop")
        audio_out = gr.Audio(label="MP3 preview", type="filepath")
        mp3_file = gr.File(label="Download MP3")

    # Actions
    # 1) Translation
    translate_btn.click(
        translate_only,
        inputs=[
            text_file, raw_text, target_lang, do_translate, translation_backend,
            num_beams, no_repeat_ngram_size, length_penalty, repetition_penalty,
            n_best, protect_numbers, protect_caps_code, apply_glossary, glossary_file,
            bt_check, bt_min_sim, style, domain, cache_repeats
        ],
        outputs=[translated_out, state_source_text, state_translated_text],
    )

    # 2) TTS: live settings via change events
    default_engine.change(_apply_default_engine, inputs=[default_engine], outputs=[translation_backend])
    fallback_to_nllb.change(_set_fallback_to_nllb, inputs=[fallback_to_nllb], outputs=[])
    tts_backend.change(_set_tts_backend, inputs=[tts_backend], outputs=[])
    tts_enabled.change(_set_tts_enabled, inputs=[tts_enabled], outputs=[])

    # 3) Synthesize and stop
    synth_evt = synth_btn.click(
        synthesize_only,
        inputs=[state_source_text, state_translated_text, read_source_or_translated],
        outputs=[audio_out, mp3_file],
    )
    try:
        stop_btn.click(None, cancels=[synth_evt])
    except Exception:
        pass
    stop_btn.click(_trigger_stop, inputs=[], outputs=[])


if __name__ == "__main__":
    demo.launch()
