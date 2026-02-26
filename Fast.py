from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
from PIL import Image, ImageChops
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
import asyncio
from io import BytesIO
import tempfile
import os
from scipy import stats

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Models (loaded at startup)
# ──────────────────────────────────────────────
deepfake_processor = None
deepfake_model     = None
ai_image_detector  = None

DEEPFAKE_MODEL = "dima806/deepfake_vs_real_image_detection"
AI_GEN_MODEL   = "umm-maybe/AI-image-detector"   # detects AI-generated images (SD, MJ, DALL-E, etc.)


@app.on_event("startup")
async def load_models():
    global deepfake_processor, deepfake_model, ai_image_detector

    deepfake_processor = AutoImageProcessor.from_pretrained(DEEPFAKE_MODEL)
    deepfake_model     = AutoModelForImageClassification.from_pretrained(DEEPFAKE_MODEL)
    deepfake_model.eval()

    ai_image_detector = pipeline(
        "image-classification",
        model=AI_GEN_MODEL,
        device=0 if torch.cuda.is_available() else -1,
    )

    print("✅ All models loaded.")


ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/quicktime", "video/x-matroska"}


# ──────────────────────────────────────────────
# Signal 1 – Error Level Analysis (ELA)
# Detects spliced / locally manipulated regions
# ──────────────────────────────────────────────
def error_level_analysis(img: Image.Image, quality: int = 90) -> float:
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    ela_array = np.array(ImageChops.difference(img.convert("RGB"), recompressed), dtype=np.float32)
    max_val   = ela_array.max()
    if max_val == 0:
        return 0.0

    ela_norm = ela_array / max_val
    score    = float(np.mean(ela_norm) + np.std(ela_norm))
    return min(round(score, 4), 1.0)


# ──────────────────────────────────────────────
# Signal 2 – Frequency / noise inconsistency (DFT)
# AI-generated images have unnatural high-freq artefacts
# ──────────────────────────────────────────────
def noise_inconsistency_score(img: Image.Image) -> float:
    gray  = np.array(img.convert("L"), dtype=np.float32)
    shift = np.fft.fftshift(np.fft.fft2(gray))
    mag   = np.log1p(np.abs(shift))

    h, w   = mag.shape
    cx, cy = h // 2, w // 2
    radius = min(h, w) // 8
    mask   = np.zeros_like(mag)
    cv2.circle(mask, (cy, cx), radius, 1, -1)

    low_energy   = np.sum(mag * mask)
    total_energy = np.sum(mag)
    if total_energy == 0:
        return 0.0

    # Low low-frequency dominance → synthetic / AI-generated
    return round(float(np.clip(1.0 - low_energy / total_energy, 0, 1)), 4)


# ──────────────────────────────────────────────
# Signal 3 – Face-region blending inconsistency
# Face-swap / deepfake composites show noise mismatch
# ──────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def face_region_inconsistency(img: Image.Image) -> float:
    gray_np = np.array(img.convert("L"))
    faces   = face_cascade.detectMultiScale(gray_np, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return 0.0

    face_pixels, bg_pixels = [], []
    for y_idx in range(gray_np.shape[0]):
        for x_idx in range(gray_np.shape[1]):
            in_face = any(
                x <= x_idx <= x + w and y <= y_idx <= y + h
                for (x, y, w, h) in faces
            )
            (face_pixels if in_face else bg_pixels).append(float(gray_np[y_idx, x_idx]))

    if not face_pixels or not bg_pixels:
        return 0.0

    face_std = float(np.std(face_pixels))
    bg_std   = float(np.std(bg_pixels))
    diff     = abs(face_std - bg_std) / (max(face_std, bg_std) + 1e-6)
    return round(float(np.clip(diff, 0, 1)), 4)


# ──────────────────────────────────────────────
# Signal helpers – image style analysis
# ──────────────────────────────────────────────
def analyse_style(img: Image.Image) -> dict:
    """
    Returns style signals used to distinguish:
      - Real photos
      - Digital art / illustrations / paintings / wallpapers
      - AI-generated images

    All signals use only cv2, numpy, PIL — no new installs.
    """
    rgb = np.array(img.convert("RGB"), dtype=np.float32)
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

    sat      = hsv[:, :, 1]   # 0-255
    val      = hsv[:, :, 2]   # 0-255 (brightness)
    avg_sat  = float(np.mean(sat))
    std_sat  = float(np.std(sat))
    avg_val  = float(np.mean(val))
    std_val  = float(np.std(val))

    # ── Edge density (Canny) ──
    # Illustrations / line-art have very sharp, clean edges.
    # Real photos have softer, more distributed edges.
    gray      = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges     = cv2.Canny(gray, threshold1=50, threshold2=150)
    edge_dens = float(np.count_nonzero(edges)) / float(edges.size)

    # ── Colour palette size (quantised) ──
    # Digital art / flat illustrations use fewer distinct colours.
    # Real photos have millions of subtle colour variations.
    small     = img.convert("RGB").resize((64, 64))
    quantised = small.quantize(colors=32).convert("RGB")
    palette   = len(set(quantised.getdata()))

    # ── Gradient smoothness ──
    # AI-generated and rendered images often have unnaturally smooth gradients.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag     = np.sqrt(gx**2 + gy**2)
    grad_std     = float(np.std(grad_mag))
    grad_mean    = float(np.mean(grad_mag))
    smoothness   = round(1.0 - min(grad_std / (grad_mean + 1e-6), 1.0), 4)  # higher = smoother

    return {
        "avg_saturation": round(avg_sat, 2),
        "std_saturation": round(std_sat, 2),
        "avg_brightness": round(avg_val, 2),
        "std_brightness": round(std_val, 2),
        "edge_density":   round(edge_dens, 4),
        "palette_size":   palette,
        "smoothness":     smoothness,
    }


def is_digital_art(style: dict, face_score: float) -> tuple:
    """
    Returns (is_art: bool, art_confidence: float).

    Digital art indicators:
      - Very high saturation with low std  → flat / cel-shaded illustration
      - High edge density                  → line-art / illustration
      - Small colour palette               → flat design / cartoon
      - Very high smoothness               → 3D render / CGI
      - No face detected OR cartoon face   → reduces photo likelihood

    Real photo indicators (any of these → NOT art):
      - Moderate saturation with high std  → natural photo variance
      - Low smoothness                     → real-world texture/noise
      - Large colour palette               → photographic colour range
    """
    score = 0.0

    # Strong art signals
    if style["avg_saturation"] > 170 and style["std_saturation"] < 40:
        score += 0.35   # flat / cel-shaded

    if style["edge_density"] > 0.12:
        score += 0.20   # lots of sharp outlines → illustration

    if style["palette_size"] < 18:
        score += 0.25   # very limited palette → cartoon / flat art

    if style["smoothness"] > 0.75:
        score += 0.20   # unnaturally smooth → CGI / render

    # Moderate art signals
    if style["avg_saturation"] > 150:
        score += 0.10

    if style["std_brightness"] < 25:
        score += 0.10   # flat lighting → illustration

    # Reduce score if image looks like a real photo
    if style["palette_size"] > 28:
        score -= 0.20   # rich palette → real photo

    if style["std_saturation"] > 60:
        score -= 0.15   # high sat variance → real photo

    if face_score > 0.30:
        score -= 0.15   # real face blending detected → probably a photo

    art_confidence = round(float(np.clip(score, 0.0, 1.0)), 3)
    return art_confidence >= 0.40, art_confidence


# ──────────────────────────────────────────────
# Core multi-signal predictor
# ──────────────────────────────────────────────
def predict_image(img: Image.Image) -> dict:
    """
    Returns one of FIVE labels:
      REAL               – authentic photograph or video frame
      FAKE_DEEPFAKE      – face-swapped or identity-manipulated real photo
      FAKE_AI_GENERATED  – fully synthetic image (SD, MJ, DALL-E) with real-looking people/scenes
      FAKE_MANIPULATED   – real photo that has been spliced, edited, or composited
      DIGITAL_ART        – illustration, painting, anime, 3D render, wallpaper, concept art

    Priority order (no conflicts):
      1. DIGITAL_ART        – checked first; art cannot also be a deepfake
      2. FAKE_AI_GENERATED  – AI model strongly says synthetic + not clearly art
      3. FAKE_DEEPFAKE      – deepfake model fires + face inconsistency present
      4. FAKE_MANIPULATED   – two manipulation signals fire simultaneously
      5. REAL               – none of the above
    """
    signals = {}

    # ── Model 1: Deepfake / face-swap detector ──
    inputs = deepfake_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = deepfake_model(**inputs)
    probs    = torch.nn.functional.softmax(outputs.logits, dim=1)
    df_label = deepfake_model.config.id2label[probs.argmax().item()].lower()
    df_conf  = round(probs.max().item(), 4)
    signals["deepfake_label"]      = df_label
    signals["deepfake_confidence"] = df_conf

    # ── Model 2: AI-generated image detector ──
    ai_results = ai_image_detector(img)
    ai_map     = {r["label"].lower(): round(r["score"], 4) for r in ai_results}
    ai_score   = max(
        ai_map.get("artificial", 0),
        ai_map.get("ai", 0),
        ai_map.get("fake", 0),
        ai_map.get("generated", 0),
    )
    signals["ai_generated_score"] = ai_score

    # ── Signal 3: ELA – detects spliced / pasted regions ──
    ela_score = error_level_analysis(img)
    signals["ela_score"] = ela_score

    # ── Signal 4: Frequency noise – detects GAN / diffusion artefacts ──
    noise_score = noise_inconsistency_score(img)
    signals["noise_score"] = noise_score

    # ── Signal 5: Face-region blending inconsistency ──
    face_score = face_region_inconsistency(img)
    signals["face_inconsistency_score"] = face_score

    # ── Signal 6: Style analysis – digital art vs photo ──
    style              = analyse_style(img)
    art_flag, art_conf = is_digital_art(style, face_score)
    smoothness         = style["smoothness"]   # extracted for use in rules below
    signals.update(style)
    signals["art_confidence"] = art_conf

    # ─────────────────────────────────────────────────────────────────────
    # RULE 1 – DIGITAL_ART
    # Checked FIRST and takes priority over AI/deepfake models because
    # AI detectors often misclassify illustrations as "AI-generated".
    # Art cannot simultaneously be a deepfake or manipulation.
    # ─────────────────────────────────────────────────────────────────────
    if art_flag:
        # Sub-case: if the AI model also fires strongly, it's AI-generated art
        # (e.g. a Midjourney illustration), not hand-drawn art.
        # We still label it DIGITAL_ART but note it in signals.
        signals["ai_assisted_art"] = ai_score >= 0.65
        label      = "DIGITAL_ART"
        confidence = round(art_conf, 3)

    # ─────────────────────────────────────────────────────────────────────
    # RULE 2 – FAKE_AI_GENERATED
    # Three ways to trigger (handles photorealistic AI like Gemini/DALL-E 3):
    #   a) AI model score >= 0.55 alone         (Gemini scores ~0.60, real photos ~0.35)
    #   b) AI >= 0.40 + deepfake model agrees   (two weak signals = one strong conclusion)
    #   c) AI >= 0.40 + extreme noise artefacts (GAN/diffusion fingerprint)
    # ─────────────────────────────────────────────────────────────────────
    elif (
        ai_score >= 0.55                                                           # (a) primary – covers Gemini, SD, MJ
        or (ai_score >= 0.40 and df_label.startswith("fake") and df_conf >= 0.55)  # (b) two models weakly agree
        or (ai_score >= 0.40 and noise_score >= 0.85)                              # (c) moderate AI + extreme noise artefacts
    ):
        label      = "FAKE_AI_GENERATED"
        confidence = round(ai_score, 3)

    # ─────────────────────────────────────────────────────────────────────
    # RULE 3 – FAKE_DEEPFAKE
    # Deepfake model fires confidently AND face inconsistency is present.
    # Requiring BOTH prevents the model from flagging real celeb photos
    # where the model is uncertain.
    # ─────────────────────────────────────────────────────────────────────
    elif df_label.startswith("fake") and df_conf >= 0.70 and face_score >= 0.30:
        label      = "FAKE_DEEPFAKE"
        confidence = df_conf

    # ─────────────────────────────────────────────────────────────────────
    # RULE 4 – FAKE_MANIPULATED
    # Requires at least TWO of three manipulation signals to fire.
    # Single-signal threshold is intentionally high to avoid
    # flagging re-compressed or HDR-processed real photos.
    # ─────────────────────────────────────────────────────────────────────
    elif sum([
        ela_score   >= 0.45,   # social-media recompression ≈ 0.20–0.35
        noise_score >= 0.80,   # phone HDR ≈ 0.65–0.75
        face_score  >= 0.55,   # natural face/bg lighting diff ≈ 0.30–0.45
    ]) >= 2:
        label  = "FAKE_MANIPULATED"
        scores = [s for s, t in [
            (ela_score, 0.45),
            (noise_score, 0.80),
            (face_score, 0.55),
        ] if s >= t]
        confidence = round(float(np.mean(scores)), 3)

    # ─────────────────────────────────────────────────────────────────────
    # RULE 5 – REAL
    # NOTE: noise_score is excluded from fake_peak here because it is
    # chronically high on real phone photos (HDR, compression) and was
    # already used only as a gating signal in RULE 4. Including it here
    # would unfairly tank confidence on genuine images.
    # ─────────────────────────────────────────────────────────────────────
    else:
        label     = "REAL"
        fake_peak = max(
            ai_score,
            df_conf if df_label.startswith("fake") else 0.0,
            ela_score,
            face_score,
        )
        confidence = round(max(1.0 - fake_peak, 0.55), 3)

    return {"label": label, "confidence": confidence, "signals": signals}


# ──────────────────────────────────────────────
# Video helpers
# ──────────────────────────────────────────────
def sample_frames(video_path: str, num_frames: int = 16) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise ValueError("Video has no frames.")

    interval = max(total // num_frames, 1)
    frames   = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    cap.release()
    return frames


def aggregate_video_results(results: list) -> dict:
    """
    Aggregates per-frame predictions into a single video-level verdict.

    Label groups:
      NEUTRAL  → REAL, DIGITAL_ART      (not deceptive content)
      HARMFUL  → FAKE_*                 (deceptive / synthetic content)

    DIGITAL_ART videos (e.g. animated films, motion graphics) are NOT
    treated as fake — they get their own label at the video level too.
    """
    label_counts: dict = {}
    for r in results:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    total = len(results)

    real_count = label_counts.get("REAL", 0)
    art_count  = label_counts.get("DIGITAL_ART", 0)
    fake_labels_found = [
        lbl for lbl in label_counts
        if lbl.startswith("FAKE_")
    ]
    fake_count = sum(label_counts[l] for l in fake_labels_found)

    neutral_count = real_count + art_count

    # ── Majority decision ──────────────────────────────────────
    if neutral_count > fake_count:
        # Most frames are real or art — pick the dominant neutral label
        if real_count >= art_count:
            video_label = "REAL"
        else:
            video_label = "DIGITAL_ART"
    else:
        # Most frames are fake — pick the most common fake sub-type
        all_fake_frame_labels = [r["label"] for r in results if r["label"].startswith("FAKE_")]
        video_label = (
            stats.mode(all_fake_frame_labels, keepdims=True).mode[0]
            if all_fake_frame_labels else "FAKE_MANIPULATED"
        )

    return {
        "video_label":        video_label,
        "real_frames":        real_count,
        "art_frames":         art_count,
        "fake_frames":        fake_count,
        "total_frames":       total,
        "average_confidence": round(float(np.mean([r["confidence"] for r in results])), 3),
        "label_breakdown":    label_counts,
    }


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Allowed: {ALLOWED_IMAGE_TYPES}",
        )
    try:
        img = Image.open(BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    try:
        result = await asyncio.get_running_loop().run_in_executor(None, predict_image, img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return result


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Allowed: {ALLOWED_VIDEO_TYPES}",
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            frames = sample_frames(tmp_path, num_frames=16)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if not frames:
            raise HTTPException(status_code=400, detail="No frames could be extracted.")

        loop    = asyncio.get_running_loop()
        results = list(await asyncio.gather(
            *[loop.run_in_executor(None, predict_image, f) for f in frames]
        ))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    summary          = aggregate_video_results(results)
    summary["frames"] = results
    return summary


@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "models_loaded": deepfake_model is not None and ai_image_detector is not None,
    }