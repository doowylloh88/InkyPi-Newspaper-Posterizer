import base64
import json
import logging
import os
import re
import tempfile
import time
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

import requests
from flask import abort, jsonify, request
from groq import Groq
from openai import BadRequestError, OpenAI
from PIL import Image, ImageColor, ImageOps

from blueprints.plugin import plugin_bp
from plugins.base_plugin.base_plugin import BasePlugin
from utils.image_utils import pad_image_blur

from .constants import NEWSPAPERS
from .poster_rules import POSTER_RULES

logger = logging.getLogger(__name__)


FREEDOM_FORUM_URL = "https://cdn.freedomforum.org/dfp/jpg{}/lg/{}.jpg"

_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_SECRET")

VIBES_FILE = Path(__file__).parent / "vibes.json"

# Model registry - Image = Open Ai, Front Page Analysis = llama or GPT-4o
MODEL_CATALOG = {
    "image": {
        "gpt-image-1-mini": {"label": "Image 1 Mini (OpenAI)", "provider": "openai"},
        "gpt-image-1":      {"label": "Image 1 (OpenAI)",      "provider": "openai"},
        "gpt-image-1.5":    {"label": "Image 1.5 (OpenAI)",    "provider": "openai"},
    },
    "analysis": {
        "meta-llama/llama-4-scout-17b-16e-instruct": {"label": "Llama-4 (Groq)",       "provider": "groq"},
        "gpt-4o":                                   {"label": "ChatGPT-4o (OpenAI)",  "provider": "openai"},
    },
}

DEFAULT_MODELS = {
    "image": "gpt-image-1-mini",
    # default headline analysis to ChatGPT-4o (OpenAI) - But llama is better at pulling headlines.
    "analysis": os.getenv("VISION_MODEL", "gpt-4o").strip(),
}

# Fixes Issue with the api key appearing blank and not rendering the drop-downs properly 
def _has_key(v) -> bool:
    return bool(v and str(v).strip())

# Choose the user-selected image/analysis model from settings
def _pick_model(settings: dict, kind: str):
    key = "imageModel" if kind == "image" else "analysisModel"
    model_id = (settings.get(key) or DEFAULT_MODELS.get(kind, "") or "").strip()

    if model_id not in MODEL_CATALOG[kind]:
        logger.warning(f"Unknown {kind} model '{model_id}', defaulting to {DEFAULT_MODELS.get(kind)}")
        model_id = (DEFAULT_MODELS.get(kind) or "").strip()

    if model_id not in MODEL_CATALOG[kind]:
        model_id = next(iter(MODEL_CATALOG[kind].keys()))
        logger.warning(f"{kind} default also invalid; falling back to first catalog entry: {model_id}")

    return model_id, MODEL_CATALOG[kind][model_id]

# Load some vibes and get their descriptions
def _read_vibes() -> list:
    try:
        if not VIBES_FILE.exists():
            # create empty file once
            _atomic_write_json(VIBES_FILE, [])
            return []
        data = json.loads(VIBES_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []

def _get_vibe_description(vibe_id: str) -> str:
    vibe_id = (vibe_id or "").strip()
    if not vibe_id:
        return ""
    for v in _read_vibes():   # <-- use the one reader
        if (v.get("id") or "").strip() == vibe_id:
            return (v.get("description") or "").strip()
    return ""

# Write into the vibes.json using atomic write
def _atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)  # atomic swap
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# Vibe list house-keeping
def _slugify(label: str) -> str:
    s = (label or "").strip().lower()
    s = re.sub(r"[\"']", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"^_+|_+$", "", s)
    return s or "vibe"

def _sorted(vibes: list) -> list:
    return sorted(
        [v for v in vibes if isinstance(v, dict) and (v.get("id") or "").strip()],
        key=lambda v: str(v.get("label") or v.get("id") or "").casefold()
    )

# Define a GET endpoint and read me some vibes
@plugin_bp.get("/plugin/<plugin_id>/vibes/list")
def vibes_list(plugin_id):
    if plugin_id != "newspaper_poster":
        abort(404)
    vibes = _sorted(_read_vibes())
    resp = jsonify({"ok": True, "vibes": vibes})
    # discourage caching so UI always reflects disk
    resp.headers["Cache-Control"] = "no-store"
    return resp

# Add vibe
@plugin_bp.post("/plugin/<plugin_id>/vibes/add")
def vibes_add(plugin_id):
    if plugin_id != "newspaper_poster":
        abort(404)

    payload = request.get_json(silent=True) or {}
    label = (payload.get("label") or "").strip()
    description = (payload.get("description") or "").strip()

    if not label or not description:
        return jsonify({"ok": False, "error": "label_and_description_required"}), 400

    vibes = _read_vibes()

    # Reject duplicate LABEL (case-insensitive, trimmed)
    label_norm = re.sub(r"\s+", " ", label).strip().casefold()
    existing_labels = {
        re.sub(r"\s+", " ", (v.get("label") or "")).strip().casefold()
        for v in vibes
        if isinstance(v, dict)
    }
    if label_norm in existing_labels:
        return jsonify({
            "ok": False,
            "error": "duplicate_vibe_label",
            "message": "That vibe name already exists. Please pick another name."
        }), 409

    # You can still slugify into an ID (IDs can collide even if labels don't; rare, but safe)
    base_id = _slugify(label)
    existing_ids = {str(v.get("id") or "") for v in vibes if isinstance(v, dict)}
    if base_id in existing_ids:
        # If slug collides but label doesn't, require a different name to avoid confusion
        return jsonify({
            "ok": False,
            "error": "duplicate_vibe_id",
            "message": "That vibe name would create a duplicate ID. Please pick vibe name."
        }), 409

    vibe_id = base_id

    vibes.append({"id": vibe_id, "label": label, "description": description})
    vibes = _sorted(vibes)

    _atomic_write_json(VIBES_FILE, vibes)
    return jsonify({"ok": True, "vibes": vibes, "added_id": vibe_id})

# Delete a vibe - No confirmation, you better be sure! 
@plugin_bp.post("/plugin/<plugin_id>/vibes/delete")
def vibes_delete(plugin_id):
    if plugin_id != "newspaper_poster":
        abort(404)

    payload = request.get_json(silent=True) or {}
    vibe_id = (payload.get("id") or "").strip()

    if not vibe_id:
        return jsonify({"ok": False, "error": "id_required"}), 400

    vibes = [v for v in _read_vibes() if isinstance(v, dict) and (v.get("id") or "").strip() != vibe_id]
    vibes = _sorted(vibes)

    _atomic_write_json(VIBES_FILE, vibes)
    return jsonify({"ok": True, "vibes": vibes, "deleted_id": vibe_id})

# Start the AI magic
class NewspaperPoster(BasePlugin):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._groq_client = Groq(api_key=_GROQ_API_KEY) if _GROQ_API_KEY else None

    # Grab that headline and article - or at least try by analyzing front page 
    def analyze_front_page(self, image_url: str, model_id: str, model_meta: dict):
        prompt_text = (
            "Look at the front page image and do TWO tasks.\n"
            "1) Extract the single MAIN banner headline.\n"
            "2) Find the matching article blurb on the front page and rewrite it as ONE paragraph.\n\n"
            "OUTPUT FORMAT (follow exactly):\n"
            "HEADLINE: <headline text>\n"
            "ARTICLE: <one-paragraph article blurb>\n\n"
            "RULES:\n"
            "- Headline must be ONLY the headline words (no colon, no extra text).\n"
            "- ARTICLE must NOT repeat the headline.\n"
            "- Do not include the newspaper name, date, bylines, section labels, or subheadlines.\n"
            )

        provider = (model_meta or {}).get("provider")

        # ---- Local helper: detect refusals / blocks / non-useful analysis ----
        def _looks_blocked_or_useless(parsed: dict, raw_text: str) -> bool:
            headline = (parsed.get("headline") or "").strip()
            article = (parsed.get("article") or "").strip()
            raw = (raw_text or "").strip()

            # Must have a headline to proceed
            if not headline:
                # If raw is a refusal, treat as blocked; otherwise it's just malformed
                return True

            # Common refusal / copyright / can't assist signals.  NYT, WP, WSJ have issues with GPT.  Llama is better
            refusal_markers = [
                "i'm sorry",
                "i'm unable",
                "i cannot assist",
                "i can’t assist",
                "i can't assist",
                "i can't help",
                "i cannot help",
                "can't help with that",
                "cannot help with that",
                "unable to comply",
                "copyright",
            ]

            low = article.lower()
            if any(m in low for m in refusal_markers):
                return True

            # Also catch when the whole response is basically a refusal
            low_raw = raw.lower()
            if any(m in low_raw for m in refusal_markers) and len(raw) < 300:
                return True

            return False

        # ---- OpenAI path (ChatGPT-4o) ----
        if provider == "openai":
            if not _OPENAI_API_KEY:
                logger.warning("OpenAI analysis selected but OPENAI key not set; cannot analyze front page.")
                return None

            client = OpenAI(api_key=_OPENAI_API_KEY)

            try:
                resp = client.responses.create(
                    model=model_id,  # e.g. "gpt-4o"
                    input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt_text},
                            {"type": "input_image", "image_url": image_url},
                        ],
                    }],
                    max_output_tokens=600,
                )

                text = (resp.output_text or "").strip()
                if not text:
                    logger.warning("OpenAI returned empty output_text for image analysis.")
                    return None

                parsed = self._parse_headline_article(text)

                print("\n" + "=" * 60, flush=True)
                print("DEBUG ANALYSIS RESULT", flush=True)
                print("HEADLINE:", (parsed.get("headline") or "").strip(), flush=True)
                print("ARTICLE:",  (parsed.get("article")  or "").strip(), flush=True)
                print("=" * 60 + "\n", flush=True)

                # If the model refused / was blocked / returned unusable output, treat as failure
                if _looks_blocked_or_useless(parsed, text):
                    logger.warning("Analysis blocked/refused or unusable. Raw output:\n%s", text)
                    return None

                return parsed

            except Exception as e:
                logger.exception(f"OpenAI image analysis failed: {e}")
                return None

        # ---- Groq path (Llama-4) ----
        if provider == "groq":
            if not self._groq_client:
                logger.warning("Groq analysis selected but GROQ_API_KEY not set; cannot analyze front page.")
                return None

            try:
                resp = self._groq_client.chat.completions.create(
                    model=model_id,  # e.g. "meta-llama/llama-4-scout-17b-16e-instruct"
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        }
                    ],
                    max_tokens=512,
                )
                text = (resp.choices[0].message.content or "").strip()
                if not text:
                    logger.warning("Groq returned empty content for image analysis.")
                    return None

                parsed = self._parse_headline_article(text)

                print("\n" + "=" * 60, flush=True)
                print("DEBUG ANALYSIS RESULT (GROQ)", flush=True)
                print("HEADLINE:", (parsed.get("headline") or "").strip(), flush=True)
                print("ARTICLE:",  (parsed.get("article")  or "").strip(), flush=True)
                print("=" * 60 + "\n", flush=True)

                if _looks_blocked_or_useless(parsed, text):
                    logger.warning("Groq analysis blocked/refused or unusable. Raw output:\n%s", text)
                    return None

                return parsed

            except Exception as e:
                logger.exception(f"Groq image analysis failed: {e}")
                return None

        logger.warning(f"Unknown analysis provider '{provider}' for model '{model_id}'")
        return None

    # Parse the headline and article from the above raw
    def _parse_headline_article(self, text: str) -> dict:
        headline = ""
        article_lines = []
        in_article = False

        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line:
                continue

            if line.startswith("HEADLINE:"):
                headline = line.replace("HEADLINE:", "", 1).strip()
                in_article = False
                continue

            if line.startswith("ARTICLE:"):
                rest = line.replace("ARTICLE:", "", 1).strip()
                if rest:
                    article_lines.append(rest)
                in_article = True
                continue

            if in_article:
                article_lines.append(line)

        article = " ".join(article_lines).strip()

        if not headline and not article:
            return {"headline": "", "article": text or ""}

        return {"headline": headline, "article": article}

    # Build prompt to sends to image model: Headline, article, poster rules, text safety rules, and vibes
    def build_ai_prompt(self, vibe_id: str, analysis: dict) -> str:
        analysis = analysis or {}
        headline = (analysis.get("headline") or "").strip()
        article  = (analysis.get("article")  or "").strip()

        if not headline and not article:
            headline = "UNKNOWN HEADLINE"
            article = "No article text extracted from the front page."

        if isinstance(POSTER_RULES, dict):
            rules_text = "\n".join(f"- {k}: {v}" for k, v in POSTER_RULES.items())
        else:
            rules_text = str(POSTER_RULES).strip()

        vibe_text = _get_vibe_description(vibe_id)

       
        story_block = (
            "STORY CONTENT TO USE:\n"
            f"HEADLINE: {headline}\n"
            f"ARTICLE (summary / blurb): {article}\n"
        )

        final_prompt = (
            f"{(vibe_text + '\n\n') if vibe_text else ''}"
            f"{rules_text}\n\n"
            f"{story_block}\n"
          
        ).strip()

        #print("\n==== FINAL IMAGE PROMPT ====\n" + final_prompt + "\n==== END FINAL IMAGE PROMPT ====\n", flush=True)
        return final_prompt

   # Decide what to generate (find front page → analyze → build prompt)
    def generate_image(self, settings, device_config):

    # 1. Resolve Settings
        newspaper_slug = settings.get("newspaper_id") or settings.get("newspaperSlug")
        if not newspaper_slug:
            raise RuntimeError("Newspaper ID not provided in settings.")

        newspaper_slug = newspaper_slug.upper()
        today = datetime.today()

        # 2. Date Cycling: Try Next Day, Today, then 2 Days Prior
        days = [today + timedelta(days=diff) for diff in [1, 0, -1, -2]]

        analysis = None
        for date in days:
            image_url = FREEDOM_FORUM_URL.format(date.day, newspaper_slug)
            print(f"GROQ_IMAGE_URL: {image_url}")

            head_timeout = (5, 15)
            head_retries = 1

            for attempt in range(head_retries + 1):
                try:
                    check = requests.head(
                        image_url,
                        timeout=head_timeout,
                        allow_redirects=True,
                    )
                    if check.status_code == 200:
                        name = next(
                            (n["name"] for n in NEWSPAPERS if n.get("slug", "").upper() == newspaper_slug),
                            newspaper_slug
                        )
                        print(f"SELECTED: {name} ({newspaper_slug}) | {date.strftime('%Y-%m-%d')} | day={date.day}")
                        logger.info(f"Found {newspaper_slug} for day {date.day}")

                        analysis_model_id, analysis_meta = _pick_model(settings, "analysis")
                        analysis = self.analyze_front_page(image_url, analysis_model_id, analysis_meta)

                        # IMPORTANT: if analyze_front_page returns None (blocked/refusal), keep searching other days
                        if analysis:
                            break
                    else:
                        logger.info(f"HEAD status={check.status_code} for {image_url}")
                        break
                except Exception as e:
                    logger.warning(f"HEAD attempt {attempt+1}/{head_retries+1} failed for {image_url}: {e}")
                    if attempt < head_retries:
                        time.sleep(1.0)
                    else:
                        logger.error(f"Failed to check URL {image_url}: {e}")

            if analysis:
                break

        # If analysis is still None, we NEVER generate an image. This triggers the UI error modal.
        if not analysis:
            raise RuntimeError(
                "Could not extract headline/article text from that front page (likely blocked/refused). "
                "Try switching Analysis Model to Groq, or choose a different newspaper/day."
            )

        vibe_id = settings.get("vibe_id")

        ai_prompt = self.build_ai_prompt(vibe_id, analysis)
        image = self.generate_openai_image(ai_prompt, settings, device_config)
        return image

    #Render (prompt → OpenAI image → fit to screen).
    def generate_openai_image(self, ai_prompt: str, settings, device_config) -> Image.Image:
            if not _OPENAI_API_KEY:
                raise RuntimeError("OpenAI API key not set.")

            # 1. Resolve Model and Orientation
            image_model_id, image_meta = _pick_model(settings, "image")
            IMAGE_MODEL = image_model_id 

            orientation = (device_config.get_config("orientation") or "horizontal").lower()
            w, h = device_config.get_resolution()

            # Normalize target dims to match orientation
            if orientation == "vertical" and w > h:
                w, h = h, w
            elif orientation == "horizontal" and h > w:
                w, h = h, w

            # Map to OpenAI supported sizes
            size = "1536x1024" if orientation == "horizontal" else "1024x1536"

            print(f"\n--- GENERATING IMAGE ---")
            print(f"Model: {IMAGE_MODEL} | Size: {size} | Target: {w}x{h}")
            
            client = OpenAI(api_key=_OPENAI_API_KEY)

            # 2. Call OpenAI with Safety Error Handling
            try:
                resp = client.images.generate(
                    model=IMAGE_MODEL,
                    prompt=ai_prompt,
                    size=size,
                    n=1,
                )
            except BadRequestError as e:
                # Handle Moderation / Safety blocks specifically
                err = getattr(e, "body", None) or {}
                err_obj = (err or {}).get("error", {}) if isinstance(err, dict) else {}
                if err_obj.get("code") == "moderation_blocked":
                    logger.error(f"Moderation blocked: {err_obj.get('message')}")
                    raise RuntimeError("OpenAI safety system blocked this headline/prompt.")
                raise RuntimeError(f"OpenAI Image Error: {e}")
            except Exception as e:
                raise RuntimeError(f"OpenAI image generation failed: {e}")

            # 3. Process the Result
            img0 = resp.data[0] if (resp.data and len(resp.data) > 0) else None
            if not img0:
                raise RuntimeError("OpenAI returned no image data.")

            if getattr(img0, "b64_json", None):
                img = Image.open(BytesIO(base64.b64decode(img0.b64_json))).convert("RGB")
            elif getattr(img0, "url", None):
                r = requests.get(img0.url, timeout=20)
                r.raise_for_status()
                img = Image.open(BytesIO(r.content)).convert("RGB")
            else:
                raise RuntimeError("OpenAI returned neither b64_json nor url.")

            # 4. Final Scale and Pad
            target_dimensions = (w, h)
            if settings.get("padImage") == "true":
                if settings.get("backgroundOption") == "blur":
                    return pad_image_blur(img, target_dimensions)
                else:
                    bg_hex = settings.get("backgroundColor") or "#ffffff"
                    background_color = ImageColor.getcolor(bg_hex, "RGB")
                    return ImageOps.pad(img, target_dimensions, color=background_color, method=Image.Resampling.LANCZOS)

            return img

    #Let's talk to the HTML
    def generate_settings_template(self):
        template_params = super().generate_settings_template()

        settings_obj = template_params.get("plugin_settings") or {}
        if not isinstance(settings_obj, dict):
            settings_obj = {}
        template_params["plugin_settings"] = settings_obj

        # Re-read keys at request time (don’t trust module globals)
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_SECRET")

        # API Required 
        template_params["api_key"] = {
            "required": True,
            "service": "OpenAI",
            "expected_key": "OPEN_AI_SECRET"
        }

        # ---- filter IMAGE models (all are OpenAI in your catalog) ----
        image_models = []
        for model_id, meta in MODEL_CATALOG["image"].items():
            provider = (meta or {}).get("provider")
            if provider == "openai" and not _has_key(openai_key):
                continue
            image_models.append({"id": model_id, "label": meta.get("label", model_id)})
        template_params["image_models"] = image_models  # always present (maybe [])

        # ---- filter ANALYSIS models ----
        analysis_models = []
        for model_id, meta in MODEL_CATALOG["analysis"].items():
            provider = (meta or {}).get("provider")
            if provider == "groq" and not _has_key(groq_key):
                continue
            if provider == "openai" and not _has_key(openai_key):
                continue
            analysis_models.append({"id": model_id, "label": meta.get("label", model_id)})
        template_params["analysis_models"] = analysis_models  # always present (maybe [])

        template_params["newspapers"] = NEWSPAPERS
        return template_params