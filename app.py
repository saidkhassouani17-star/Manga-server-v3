# -*- coding: utf-8 -*-
"""
Manga Translation Server — v5.0 (PaddleOCR 3.x + LaMa Only Edition)
===================================================================
- OCR: PaddleOCR 3.x (Thread-safe + Kaggle PIR/MKLDNN/PDX/LangChain fixes)
- Inpainting: SimpleLama (LaMa PyTorch model)
- Translation: Groq / DeepL / Google Translate fallback
- Arabic Rendering: Pillow + arabic_reshaper + python-bidi (Bubble-aware word wrap)
"""

import os
import sys
import types
import threading
import io
import re
import time
import base64
import logging
import urllib.request
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# 1. KAGGLE & PADDLEX PRE-INITIALIZATION FIXES (MUST RUN BEFORE ANY IMPORTS)
# ---------------------------------------------------------------------------
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["USE_LAMA"] = "true"

# Mock deprecated langchain submodules required by paddlex on Kaggle (Python 3.12/3.13)
for _mod in [
    "langchain",
    "langchain.docstore",
    "langchain.docstore.document",
    "langchain.text_splitter",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)
sys.modules["langchain.docstore.document"].Document = object
sys.modules["langchain.text_splitter"].RecursiveCharacterTextSplitter = object

# ---------------------------------------------------------------------------
# 2. STANDARD IMPORTS
# ---------------------------------------------------------------------------
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import httpx
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("manga-server")

# RTL Arabic shaping support
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    RTL_AVAILABLE = True
    logger.info("RTL shaping: available (arabic_reshaper + python-bidi)")
except ImportError:
    RTL_AVAILABLE = False
    logger.warning("RTL shaping libraries not found! Arabic text may appear disconnected.")

# ---------------------------------------------------------------------------
# 3. GLOBAL SINGLETONS & THREAD LOCKS (PaddleOCR + LaMa ONLY)
# ---------------------------------------------------------------------------
_ocr_reader = None
_ocr_backend = "paddle"
_ocr_lock = threading.Lock()

_lama = None
_lama_lock = threading.Lock()


def _patch_paddlex_reinit():
    """Prevent RuntimeError: PDX has already been initialized."""
    try:
        import paddlex.repo_manager.core as _pdx_core
        _orig_init = _pdx_core.initialize

        def _safe_init(*args, **kwargs):
            if _pdx_core._GlobalContext.is_initialized():
                return
            return _orig_init(*args, **kwargs)

        _pdx_core.initialize = _safe_init
    except Exception:
        pass


def get_ocr():
    """Initialize and return PaddleOCR 3.x reader (strictly no fallback)."""
    global _ocr_reader, _ocr_backend
    if _ocr_reader is not None:
        return _ocr_reader, _ocr_backend
    with _ocr_lock:
        if _ocr_reader is not None:
            return _ocr_reader, _ocr_backend
        _patch_paddlex_reinit()
        from paddleocr import PaddleOCR
        logger.info("Initializing PaddleOCR 3.x engine (enable_mkldnn=False)...")
        _ocr_reader = PaddleOCR(
            lang="en",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        _ocr_backend = "paddle"
        logger.info("PaddleOCR 3.x engine loaded successfully.")
    return _ocr_reader, _ocr_backend


class DirectLaMaFallback:
    """Direct TorchScript big-lama.pt runner if simple_lama_inpainting package has issues."""
    def __init__(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_url = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
        model_path = "/tmp/big-lama.pt"
        if not os.path.exists(model_path):
            logger.info("Downloading LaMa weights (big-lama.pt)...")
            urllib.request.urlretrieve(model_url, model_path)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.torch = torch

    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        torch = self.torch
        img_np = np.array(image.convert("RGB"))
        mask_np = np.array(mask.convert("L"))
        h, w = img_np.shape[:2]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            img_np = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            mask_np = np.pad(mask_np, ((0, pad_h), (0, pad_w)), mode="reflect")
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        mask_t = (torch.from_numpy(mask_np).float().div(255.0) > 0.5).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            inpainted = self.model(img_t, mask_t)
        res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
        res = np.clip(res * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(res[:h, :w])


def get_lama():
    """Initialize and return LaMa inpainting model."""
    global _lama
    if _lama is not None:
        return _lama
    with _lama_lock:
        if _lama is not None:
            return _lama
        logger.info("Initializing LaMa inpainting model...")
        try:
            from simple_lama_inpainting import SimpleLama
            _lama = SimpleLama()
            logger.info("SimpleLama loaded successfully.")
        except Exception as e:
            logger.warning(f"SimpleLama import failed ({e}), loading DirectLaMa (big-lama.pt)...")
            _lama = DirectLaMaFallback()
            logger.info("DirectLaMa loaded successfully.")
    return _lama


# ---------------------------------------------------------------------------
# 4. OCR & BUBBLE GROUPING PIPELINE
# ---------------------------------------------------------------------------
def run_ocr(image: Image.Image) -> List[Dict[str, Any]]:
    """Run PaddleOCR 3.x on PIL image and return list of {text, box, score}."""
    reader, _ = get_ocr()
    arr = np.array(image.convert("RGB"))
    results = []
    with _ocr_lock:
        # Support both PaddleOCR 3.x .predict() and 2.x .ocr()
        if hasattr(reader, "predict"):
            out = list(reader.predict(arr))
            d = out[0] if out else {}
            rec_texts = d.get("rec_texts", []) if hasattr(d, "get") else getattr(d, "rec_texts", [])
            rec_scores = d.get("rec_scores", []) if hasattr(d, "get") else getattr(d, "rec_scores", [])
            dt_polys = d.get("dt_polys", []) if hasattr(d, "get") else getattr(d, "dt_polys", [])
            for t, s, poly in zip(rec_texts, rec_scores, dt_polys):
                text_str = str(t).strip()
                if not text_str:
                    continue
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                results.append({
                    "text": text_str,
                    "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                    "score": float(s),
                })
        else:
            out = reader.ocr(arr, cls=False)
            lines = out[0] if (out and out[0]) else []
            for line in lines:
                poly, (t, s) = line[0], line[1]
                text_str = str(t).strip()
                if not text_str:
                    continue
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                results.append({
                    "text": text_str,
                    "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                    "score": float(s),
                })
    return results


def merge_nearby_boxes(items: List[Dict[str, Any]], img_w: int, img_h: int) -> List[Dict[str, Any]]:
    """Group nearby OCR lines inside the same speech bubble into paragraphs."""
    if not items:
        return []
    # Sort top-to-bottom, then left-to-right
    items = sorted(items, key=lambda x: (x["box"][1], x["box"][0]))
    clusters: List[List[Dict[str, Any]]] = []

    for item in items:
        x1, y1, x2, y2 = item["box"]
        line_h = max(12, y2 - y1)
        placed = False
        for cluster in clusters:
            cx1 = min(c["box"][0] for c in cluster)
            cy1 = min(c["box"][1] for c in cluster)
            cx2 = max(c["box"][2] for c in cluster)
            cy2 = max(c["box"][3] for c in cluster)

            # Vertical gap and horizontal overlap check
            v_gap = y1 - cy2
            h_overlap = min(x2, cx2) - max(x1, cx1)
            min_w = min(x2 - x1, cx2 - cx1)
            center_dist_x = abs(((x1 + x2) / 2) - ((cx1 + cx2) / 2))

            if -line_h * 0.5 <= v_gap <= line_h * 1.4 and (
                h_overlap > -line_h * 0.8 or center_dist_x < max(cx2 - cx1, x2 - x1) * 0.65
            ):
                cluster.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])

    merged = []
    for cluster in clusters:
        cluster = sorted(cluster, key=lambda x: (x["box"][1], x["box"][0]))
        full_text = " ".join(c["text"] for c in cluster).strip()
        bx1 = max(0, min(c["box"][0] for c in cluster))
        by1 = max(0, min(c["box"][1] for c in cluster))
        bx2 = min(img_w, max(c["box"][2] for c in cluster))
        by2 = min(img_h, max(c["box"][3] for c in cluster))
        avg_score = sum(c.get("score", 1.0) for c in cluster) / len(cluster)
        merged.append({
            "text": full_text,
            "box": [bx1, by1, bx2, by2],
            "lines": [c["box"] for c in cluster],
            "score": float(avg_score),
        })
    return merged


# ---------------------------------------------------------------------------
# 5. INPAINTING WITH LAMA ONLY
# ---------------------------------------------------------------------------
def inpaint_bubbles_lama(image: Image.Image, bubbles: List[Dict[str, Any]]) -> Image.Image:
    """Erase text regions using LaMa inpainting model."""
    if not bubbles:
        return image
    lama_model = get_lama()
    w, h = image.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    pad = 6
    for b in bubbles:
        sub_boxes = b.get("lines") or [b["box"]]
        for sb in sub_boxes:
            x1, y1, x2, y2 = sb
            draw.rectangle(
                [max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)],
                fill=255,
            )

    with _lama_lock:
        cleaned = lama_model(image.convert("RGB"), mask)
    return cleaned


# ---------------------------------------------------------------------------
# 6. TRANSLATION ENGINE (Groq / DeepL / Google GTX Fallback)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DEEPL_API_KEY = os.environ.get("DEEPL_API_KEY", "")


async def translate_texts(
    texts: List[str],
    target_lang: str = "ar",
    source_lang: str = "auto",
    groq_key: Optional[str] = None,
    deepl_key: Optional[str] = None,
) -> List[str]:
    if not texts:
        return []

    g_key = groq_key or GROQ_API_KEY
    d_key = deepl_key or DEEPL_API_KEY

    # 1. Try Groq LLM batch translation if key is provided
    if g_key:
        try:
            numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
            prompt = (
                f"Translate the following manga/comic dialogue lines into natural Arabic. "
                f"Return ONLY the translated lines numbered 1 to {len(texts)}, one per line, without extra commentary:\n\n{numbered}"
            )
            async with httpx.AsyncClient(timeout=25.0) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {g_key}"},
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                    },
                )
                if r.status_code == 200:
                    content = r.json()["choices"][0]["message"]["content"].strip()
                    lines = [
                        re.sub(r"^\d+[\.)\-:\s]+", "", line).strip()
                        for line in content.splitlines()
                        if line.strip()
                    ]
                    if len(lines) == len(texts):
                        return lines
        except Exception as e:
            logger.warning(f"Groq translation fallback triggered: {e}")

    # 2. Try DeepL if key is provided
    if d_key:
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.post(
                    "https://api-free.deepl.com/v2/translate",
                    headers={"Authorization": f"DeepL-Auth-Key {d_key}"},
                    data={"text": texts, "target_lang": target_lang.upper()},
                )
                if r.status_code == 200:
                    data = r.json()
                    return [item["text"] for item in data.get("translations", [])]
        except Exception as e:
            logger.warning(f"DeepL translation fallback triggered: {e}")

    # 3. Reliable Free Google Translate GTX endpoint
    translated = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        for t in texts:
            try:
                r = await client.get(
                    "https://translate.googleapis.com/translate_a/single",
                    params={
                        "client": "gtx",
                        "sl": source_lang if source_lang != "auto" else "auto",
                        "tl": target_lang,
                        "dt": "t",
                        "q": t,
                    },
                )
                if r.status_code == 200:
                    segs = r.json()[0]
                    translated.append("".join(s[0] for s in segs if s and s[0]))
                else:
                    translated.append(t)
            except Exception:
                translated.append(t)
    return translated


# ---------------------------------------------------------------------------
# 7. ARABIC FONT & TEXT RENDERING
# ---------------------------------------------------------------------------
def _find_font_paths() -> List[str]:
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    found = [p for p in candidates if os.path.exists(p)]
    logger.info(f"Font files found: {found}")
    return found


FONT_PATHS = _find_font_paths()


def get_font(size: int) -> ImageFont.FreeTypeFont:
    for fp in FONT_PATHS:
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def shape_arabic(text: str) -> str:
    if not text:
        return ""
    if RTL_AVAILABLE:
        try:
            reshaped = arabic_reshaper.reshape(text)
            return get_display(reshaped)
        except Exception:
            return text
    return text


def draw_translated_bubbles(image: Image.Image, bubbles: List[Dict[str, Any]]) -> Image.Image:
    """Render translated Arabic text centered inside each cleaned bubble."""
    draw = ImageDraw.Draw(image)
    for b in bubbles:
        raw_text = b.get("translated") or b.get("text", "")
        if not raw_text:
            continue
        x1, y1, x2, y2 = b["box"]
        bw = max(24, x2 - x1)
        bh = max(20, y2 - y1)

        words = raw_text.split()
        if not words:
            continue

        best_font = get_font(16)
        best_lines = [raw_text]

        for font_size in range(min(36, max(14, int(bh * 0.45))), 11, -2):
            font = get_font(font_size)
            lines = []
            current = ""
            for w in words:
                test_line = f"{current} {w}".strip()
                shaped_test = shape_arabic(test_line)
                bbox = draw.textbbox((0, 0), shaped_test, font=font)
                tw = bbox[2] - bbox[0]
                if tw <= bw * 1.08 or not current:
                    current = test_line
                else:
                    lines.append(current)
                    current = w
            if current:
                lines.append(current)

            line_h = int(font_size * 1.28)
            total_h = line_h * len(lines)
            if total_h <= bh * 1.25:
                best_font = font
                best_lines = lines
                break
            best_font = font
            best_lines = lines

        font_size = getattr(best_font, "size", 16)
        line_h = int(font_size * 1.25)
        total_h = line_h * len(best_lines)
        curr_y = y1 + (bh - total_h) / 2

        for line in best_lines:
            shaped = shape_arabic(line)
            bbox = draw.textbbox((0, 0), shaped, font=best_font)
            tw = bbox[2] - bbox[0]
            curr_x = x1 + (bw - tw) / 2
            # White stroke outline + black text for crisp readability on any background
            draw.text(
                (curr_x, curr_y),
                shaped,
                font=best_font,
                fill=(0, 0, 0),
                stroke_width=3,
                stroke_fill=(255, 255, 255),
            )
            curr_y += line_h

    return image


# ---------------------------------------------------------------------------
# 8. FASTAPI APPLICATION & ENDPOINTS
# ---------------------------------------------------------------------------
app = FastAPI(title="Manga Server v5.0 (PaddleOCR + LaMa)", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_preload():
    """Preload PaddleOCR and LaMa once at startup so first request never times out."""
    logger.info("Preloading PaddleOCR 3.x and LaMa at server startup...")
    try:
        get_ocr()
        get_lama()
        logger.info("Startup preload completed: PaddleOCR + LaMa ready!")
    except Exception as e:
        logger.error(f"Startup preload warning: {e}")


@app.get("/")
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "version": "5.0.0",
        "ocr_backend": "paddleocr-3.x",
        "inpaint_backend": "lama",
        "rtl": RTL_AVAILABLE,
    }


async def _extract_image_from_request(
    request: Request,
    file: Optional[UploadFile] = None,
    image: Optional[UploadFile] = None,
) -> Image.Image:
    upload = file or image
    if upload is not None:
        raw = await upload.read()
        return Image.open(io.BytesIO(raw)).convert("RGB")

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        b64_str = body.get("image") or body.get("file") or body.get("data") or body.get("base64")
        if b64_str:
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            raw = base64.b64decode(b64_str)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        img_url = body.get("url") or body.get("image_url")
        if img_url:
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.get(img_url)
                return Image.open(io.BytesIO(r.content)).convert("RGB")

    form = await request.form()
    for key in ("file", "image", "data", "base64"):
        if key in form:
            val = form[key]
            if hasattr(val, "read"):
                raw = await val.read()
                return Image.open(io.BytesIO(raw)).convert("RGB")
            elif isinstance(val, str):
                b64_str = val.split(",", 1)[1] if "," in val else val
                raw = base64.b64decode(b64_str)
                return Image.open(io.BytesIO(raw)).convert("RGB")

    raise HTTPException(status_code=400, detail="No valid image found in request.")


@app.post("/ocr")
async def ocr_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
):
    try:
        pil_img = await _extract_image_from_request(request, file, image)
        w, h = pil_img.size
        raw_items = run_ocr(pil_img)
        bubbles = merge_nearby_boxes(raw_items, w, h)
        return {"status": "ok", "bubbles": bubbles, "raw": raw_items}
    except Exception as e:
        logger.exception("OCR error")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/translate")
@app.post("/translate-image")
@app.post("/process")
async def translate_image_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    target_lang: str = Form("ar"),
    source_lang: str = Form("auto"),
    response_format: str = Form("auto"),
    groq_api_key: Optional[str] = Form(None),
    deepl_api_key: Optional[str] = Form(None),
):
    t0 = time.time()
    try:
        pil_img = await _extract_image_from_request(request, file, image)
        w, h = pil_img.size

        # Parse JSON options if sent as JSON
        if "application/json" in request.headers.get("content-type", ""):
            try:
                body = await request.json()
                target_lang = body.get("target_lang", target_lang)
                source_lang = body.get("source_lang", source_lang)
                response_format = body.get("response_format", body.get("format", response_format))
                groq_api_key = body.get("groq_api_key", groq_api_key)
                deepl_api_key = body.get("deepl_api_key", deepl_api_key)
            except Exception:
                pass

        # 1. OCR (PaddleOCR 3.x)
        raw_items = run_ocr(pil_img)
        bubbles = merge_nearby_boxes(raw_items, w, h)

        if bubbles:
            # 2. Translate
            texts_to_translate = [b["text"] for b in bubbles]
            translations = await translate_texts(
                texts_to_translate,
                target_lang=target_lang,
                source_lang=source_lang,
                groq_key=groq_api_key,
                deepl_key=deepl_api_key,
            )
            for b, tr in zip(bubbles, translations):
                b["translated"] = tr

            # 3. Inpaint (LaMa)
            cleaned_img = inpaint_bubbles_lama(pil_img, bubbles)

            # 4. Draw Arabic text
            final_img = draw_translated_bubbles(cleaned_img, bubbles)
        else:
            final_img = pil_img

        buf = io.BytesIO()
        final_img.save(buf, format="JPEG", quality=92)
        img_bytes = buf.getvalue()
        b64_img = base64.b64encode(img_bytes).decode("utf-8")

        elapsed = round(time.time() - t0, 2)
        accept = request.headers.get("accept", "")
        if response_format == "image" or ("image/" in accept and response_format == "auto"):
            return Response(content=img_bytes, media_type="image/jpeg")

        return {
            "status": "ok",
            "image": b64_img,
            "image_base64": f"data:image/jpeg;base64,{b64_img}",
            "bubbles": bubbles,
            "count": len(bubbles),
            "elapsed_sec": elapsed,
            "ocr": "paddleocr-3.x",
            "inpaint": "lama",
        }
    except Exception as e:
        logger.exception("Translation pipeline error")
        return JSONResponse(status_code=500, content={"error": str(e)})
