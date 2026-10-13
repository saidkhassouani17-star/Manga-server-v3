"""
Manga Translation Server V3 — Server-does-everything.
- POST /translate: OCR -> translate -> inpaint -> render -> final image
- POST /test-provider: verify DeepL/Groq/Gemini API keys
- GET /health: liveness check
"""
import base64
import io
import json
import logging
import os
from typing import List, Optional, Tuple

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("manga-server")

app = FastAPI(title="Manga Translation Server")

API_TOKEN = os.environ.get("API_TOKEN", "")

# ---------------------------------------------------------------- OCR ----
_ocr_reader = None
_ocr_backend = None  # "paddle" or "easyocr"


def get_ocr():
    """Lazy-load OCR engine. Prefer PaddleOCR, fall back to EasyOCR."""
    global _ocr_reader, _ocr_backend
    if _ocr_reader is not None:
        return _ocr_reader, _ocr_backend
    try:
        from paddleocr import PaddleOCR
        log.info("Loading PaddleOCR (multilingual)...")
        _ocr_reader = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        _ocr_backend = "paddle"
        log.info("PaddleOCR ready")
    except Exception as e:
        log.warning("PaddleOCR failed (%s), falling back to EasyOCR", e)
        import easyocr
        # EasyOCR: ch_sim only works with en, can't mix with ja/ko
        # Use English-only for fallback (PaddleOCR handles multilingual)
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
        _ocr_backend = "easyocr"
        log.info("EasyOCR ready")
    return _ocr_reader, _ocr_backend


def run_ocr(image_bytes: bytes) -> List[dict]:
    """Return [{'text': str, 'box': [x1,y1,x2,y2], 'score': float}, ...]."""
    reader, backend = get_ocr()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    results = []
    if backend == "paddle":
        out = reader.ocr(arr, cls=False)
        # PaddleOCR returns [[ [points], (text, score) ], ...]
        pages = out[0] if out and out[0] else []
        for item in pages:
            pts, (text, score) = item[0], item[1]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            results.append({
                "text": text.strip(),
                "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                "score": float(score),
            })
    else:
        out = reader.readtext(arr)
        for box, text, score in out:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            results.append({
                "text": text.strip(),
                "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                "score": float(score),
            })
    # Drop empties and very low confidence
    results = [r for r in results if r["text"] and r["score"] >= 0.25]
    log.info("OCR (%s): %d regions", backend, len(results))
    return results


# ------------------------------------------------------- translation ----
async def translate_deepl(text: str, target: str, key: str,
                          endpoint: str = "auto") -> str:
    base = ("https://api-free.deepl.com" if key.endswith(":fx") and endpoint == "auto"
            else "https://api.deepl.com" if endpoint in ("auto", "paid")
            else "https://api-free.deepl.com")
    if endpoint == "free":
        base = "https://api-free.deepl.com"
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{base}/v2/translate",
                         headers={"Authorization": f"DeepL-Auth-Key {key}"},
                         json={"text": [text],
                               "target_lang": target.upper()})
        r.raise_for_status()
        return r.json()["translations"][0]["text"]


async def translate_groq(text: str, target: str, key: str,
                         model: str = "llama-3.3-70b-versatile",
                         glossary: dict = None) -> str:
    sys = (f"You are a manga translator. Translate the following text to "
           f"{target}. Return ONLY the translation, no explanations.")
    if glossary:
        pairs = "; ".join(f'"{k}" -> "{v}"' for k, v in glossary.items())
        sys += f" Use this glossary: {pairs}."
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model,
                  "messages": [{"role": "system", "content": sys},
                               {"role": "user", "content": text}],
                  "temperature": 0.3})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


async def translate_gemini(text: str, target: str, key: str,
                           model: str = "gemini-2.0-flash",
                           glossary: dict = None) -> str:
    prompt = (f"Translate the following manga text to {target}. "
              f"Return ONLY the translation.")
    if glossary:
        pairs = "; ".join(f'"{k}" -> "{v}"' for k, v in glossary.items())
        prompt += f" Glossary: {pairs}."
    prompt += f"\n\n{text}"
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={key}",
            json={"contents": [{"parts": [{"text": prompt}]}]})
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


async def translate_text(text: str, target: str, priority: List[str],
                         keys: dict, models: dict,
                         glossary: dict) -> Tuple[str, str]:
    """Try providers in order. Returns (translated, provider_used)."""
    # Apply glossary first (exact replacements)
    for k, v in (glossary or {}).items():
        text = text.replace(k, v)
    last_err = "no provider configured"
    for p in priority:
        p = p.strip().lower()
        try:
            if p == "deepl" and keys.get("deepl"):
                t = await translate_deepl(text, target, keys["deepl"])
                return t, "deepl"
            if p == "groq" and keys.get("groq"):
                t = await translate_groq(text, target, keys["groq"],
                                         models.get("groq", "llama-3.3-70b-versatile"),
                                         glossary)
                return t, "groq"
            if p == "gemini" and keys.get("gemini"):
                t = await translate_gemini(text, target, keys["gemini"],
                                           models.get("gemini", "gemini-2.0-flash"),
                                           glossary)
                return t, "gemini"
        except Exception as e:
            last_err = f"{p}: {e}"
            log.warning("Provider %s failed: %s", p, e)
    raise RuntimeError(f"All providers failed. Last error: {last_err}")


# ----------------------------------------------------------- render ----
RTL_LANGS = {"ar", "he", "fa", "ur"}


def shape_rtl(text: str) -> str:
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text


def find_font(size: int) -> ImageFont.FreeTypeFont:
    # Try common fonts that support Arabic
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_translations(image_bytes: bytes, items: List[dict],
                        target_lang: str) -> bytes:
    """Erase original text and draw translations inside boxes."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    rtl = target_lang.lower() in RTL_LANGS

    # 1. Inpaint original text areas
    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    for it in items:
        x1, y1, x2, y2 = it["box"]
        pad = 3
        cv2.rectangle(mask,
                      (max(0, x1 - pad), max(0, y1 - pad)),
                      (min(arr.shape[1], x2 + pad), min(arr.shape[0], y2 + pad)),
                      255, -1)
    arr = cv2.inpaint(arr, mask, 5, cv2.INPAINT_TELEA)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    # 2. Draw translated text
    for it in items:
        text = it["translated"]
        if not text:
            continue
        if rtl:
            text = shape_rtl(text)
        x1, y1, x2, y2 = it["box"]
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        # Auto-fit font size
        size = max(8, bh // 2)
        font = find_font(size)
        # Shrink until it fits
        while size > 8:
            bbox = draw.multiline_textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if tw <= bw * 0.95 and th <= bh * 0.95:
                break
            size -= 2
            font = find_font(size)
        bbox = draw.multiline_textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = x1 + (bw - tw) // 2
        ty = y1 + (bh - th) // 2
        # Black outline + white fill (scanlation style)
        for ox, oy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            draw.multiline_text((tx + ox, ty + oy), text, font=font, fill="black")
        draw.multiline_text((tx, ty), text, font=font, fill="white")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


# -------------------------------------------------------- endpoints ----
def check_token(token: str):
    if API_TOKEN and token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API token")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/test-provider")
async def test_provider(payload: dict, x_api_token: str = Header(default="")):
    check_token(x_api_token)
    provider = payload.get("provider", "").lower()
    key = payload.get("api_key", "")
    model = payload.get("model", "")
    if not key:
        return {"working": False, "error": "No API key provided"}
    try:
        if provider == "deepl":
            t = await translate_deepl("Hello", "AR", key)
        elif provider == "groq":
            t = await translate_groq("Hello", "Arabic", key,
                                     model or "llama-3.3-70b-versatile")
        elif provider == "gemini":
            t = await translate_gemini("Hello", "Arabic", key,
                                       model or "gemini-2.0-flash")
        else:
            return {"working": False, "error": f"Unknown provider: {provider}"}
        return {"working": True, "sample": t[:80]}
    except Exception as e:
        return {"working": False, "error": str(e)[:200]}


@app.post("/list-models")
async def list_models(request: Request, x_api_token: str = Header(None)):
    check_token(x_api_token or "")
    payload = await request.json()
    provider = payload.get("provider", "").lower()
    key = payload.get("api_key", "")
    if not key:
        return {"models": [], "error": "No API key provided"}
    try:
        if provider == "gemini":
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models?key={key}")
                r.raise_for_status()
                data = r.json()
                models = []
                for m in data.get("models", []):
                    name = m.get("name", "").replace("models/", "")
                    # Only text generation models
                    methods = m.get("supportedGenerationMethods", [])
                    if "generateContent" in methods:
                        models.append({
                            "id": name,
                            "name": m.get("displayName", name),
                        })
                return {"models": models}
        elif provider == "groq":
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {key}"})
                r.raise_for_status()
                data = r.json()
                models = [{"id": m["id"], "name": m["id"]}
                          for m in data.get("data", [])]
                return {"models": models}
        else:
            return {"models": [], "error": f"Model listing not supported for: {provider}"}
    except Exception as e:
        return {"models": [], "error": str(e)[:200]}


@app.post("/translate")
async def translate(
    image: UploadFile = File(...),
    target_lang: str = Form("ar"),
    source_lang: str = Form("auto"),
    provider_priority: str = Form("groq,deepl,gemini"),
    glossary: str = Form("{}"),
    deepl_key: str = Form(""),
    groq_key: str = Form(""),
    gemini_key: str = Form(""),
    groq_model: str = Form("llama-3.3-70b-versatile"),
    gemini_model: str = Form("gemini-2.0-flash"),
    deepl_endpoint: str = Form("auto"),
    x_api_token: str = Header(default=""),
):
    check_token(x_api_token)
    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise ValueError("Empty image")
        # 1. OCR
        regions = run_ocr(image_bytes)
        if not regions:
            raise ValueError("No text detected in image")
        # 2. Translate each region
        try:
            glossary_dict = json.loads(glossary) if glossary else {}
        except Exception:
            glossary_dict = {}
        keys = {"deepl": deepl_key, "groq": groq_key, "gemini": gemini_key}
        models = {"groq": groq_model, "gemini": gemini_model}
        priority = [p for p in provider_priority.split(",") if p.strip()]
        provider_used = None
        for r in regions:
            translated, provider_used = await translate_text(
                r["text"], target_lang, priority, keys, models, glossary_dict)
            r["translated"] = translated
        # 3. Render
        out_bytes = render_translations(image_bytes, regions, target_lang)
        b64 = base64.b64encode(out_bytes).decode()
        return {"translated_image": b64, "provider_used": provider_used,
                "regions": len(regions)}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("translate failed")
        # NEVER return the original image on failure — error only
        return JSONResponse(status_code=500, content={"error": str(e)[:300]})
