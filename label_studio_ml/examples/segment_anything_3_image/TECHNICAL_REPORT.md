# SAM3 ML Backend Technical Report

## Overview

This document describes the SAM3 (Segment Anything Model 3) ML backend for Label Studio, a custom implementation developed in the Glidance fork of `label-studio-ml-backend`. This backend provides intelligent image segmentation for blind navigation robot training data, specifically optimized for walkable surface detection.

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     Label Studio UI                              │
│   (User clicks point / draws box on image)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SAM3 ML Backend                               │
│  ┌─────────────────┐    ┌──────────────────────────────────┐    │
│  │  Sam3Model      │    │  Sam3TrackerModel                │    │
│  │  (Text+Point)   │    │  (Point-only fallback)           │    │
│  │  FP16 enabled   │    │  FP16 enabled                    │    │
│  └─────────────────┘    └──────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Domain-Specific Prompt Ensemble              │   │
│  │  "road" → ["roadway", "street", "driving lane", ...]     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Segmentation Mask (RLE encoded)                     │
│              Returned to Label Studio                            │
└─────────────────────────────────────────────────────────────────┘
```

### Model Loading

The backend loads two SAM3 model variants at startup:

1. **Sam3Model** (Primary) - Text-capable model that understands semantic prompts
2. **Sam3TrackerModel** (Fallback) - Point-only model for cases where text prompts fail

Both models are loaded with FP16 precision for ~10-20% faster inference:

```python
model = Sam3Model.from_pretrained(MODEL_NAME, token=HF_TOKEN).to(DEVICE)
model = model.half()  # FP16 quantization
model.eval()
```

---

## Enhancements Over Original Label Studio ML Backends

### 1. New SAM3 Integration (Not in Upstream)

The upstream `label-studio-ml-backend` repository only includes `grounding_sam` (SAM1-based). This fork adds a completely new `segment_anything_3_image` backend that integrates Meta's latest SAM3 model via Hugging Face Transformers.

**Key Difference:**
| Feature | Upstream (grounding_sam) | Fork (segment_anything_3_image) |
|---------|-------------------------|--------------------------------|
| Model | SAM1 | SAM3 |
| Text Prompts | Via GroundingDINO | Native SAM3 text encoder |
| Installation | Requires manual model download | Auto-downloads from HuggingFace |
| Precision | FP32 | FP16 (optimized) |

### 2. Domain-Specific Multi-Prompt Ensemble (Method 6)

Instead of using a single text prompt per label, the backend tries multiple semantically-related prompts and selects the best result:

```python
DOMAIN_PROMPTS = {
    "road": [
        "roadway",
        "street",
        "driving lane",
        "carriageway",
    ],
    "paved path": [
        "sidewalk",
        "paved footpath",
        "pedestrian walkway",
        "paved path",
    ],
    # ... 8 label categories total
}
```

**How it works:**
1. User clicks on a "road" - backend receives label "road"
2. Backend tries all 4 prompts: "roadway", "street", "driving lane", "carriageway"
3. Each prompt generates candidate masks with confidence scores
4. The mask with highest score (that contains the click point) is selected

**Benefits:**
- Robust to SAM3's prompt sensitivity
- Different prompts work better for different image contexts
- Automatically selects the best interpretation

### 3. Confidence-Based Hybrid Fallback (Method 4)

When text prompts produce low-confidence results, the backend falls back to point-only segmentation:

```python
# If presence score is below threshold, fall back to point-only
if best_mask is None or max_presence < PRESENCE_THRESHOLD:
    if point_coords and len(point_coords) > 0:
        return self._point_only_predict(image, point_coords, point_labels)
```

**Fallback Logic:**
1. Try text-prompted segmentation with all domain prompts
2. Check `presence_logits` - SAM3's confidence that the concept exists in the image
3. If presence < 0.5 (configurable), switch to Sam3TrackerModel
4. Sam3TrackerModel uses only the click point, ignoring text entirely

**Why this matters:**
- Text prompts can fail on unusual images or edge cases
- Point-only segmentation is more reliable but less semantically aware
- Hybrid approach gets best of both worlds

### 4. Batch Prediction Mode

Enables automatic pre-annotation when images are imported, without user interaction:

```python
# Environment variable enables batch mode
SAM3_BATCH_LABELS="road,paved path,marked crossing,unpaved path,driveway"

# On image import, backend automatically segments all configured labels
for label in labels:
    prompts = DOMAIN_PROMPTS.get(label.lower(), [label])
    # Try all prompts, keep best result above confidence threshold
    if best_score >= BATCH_MIN_SCORE:
        task_results.append(mask_result)
```

**Configuration:**
| Variable | Default | Description |
|----------|---------|-------------|
| `SAM3_BATCH_LABELS` | `""` | Comma-separated labels to auto-segment |
| `SAM3_BATCH_MIN_SCORE` | `0.51` | Minimum confidence to include prediction |

### 5. FP16 Inference Optimization

Both models run in half-precision (FP16) for faster inference:

```python
model = model.half()  # Converts weights from FP32 to FP16
tracker_model = tracker_model.half()
```

**Performance Impact:**
- ~10-20% faster inference on GPU
- ~50% reduction in GPU memory usage
- Minimal impact on mask quality (SAM3 is designed for FP16)

### 6. Configurable Thresholds

Multiple thresholds are exposed as environment variables for tuning:

| Variable | Default | Description |
|----------|---------|-------------|
| `MASK_THRESHOLD` | `0.5` | Probability threshold for mask binarization (higher = tighter masks) |
| `PRESENCE_THRESHOLD` | `0.5` | Below this, fall back to point-only prediction |
| `SAM3_BATCH_MIN_SCORE` | `0.51` | Minimum confidence for batch predictions |

---

## Inference Pipeline

### Interactive Mode (User Clicks)

```
1. User clicks point on image with label "road"
                    │
                    ▼
2. Backend receives: point_coords=[[x,y]], label="road"
                    │
                    ▼
3. Multi-prompt ensemble runs 4 forward passes:
   - "roadway"      → mask1, score=0.72, presence=0.85
   - "street"       → mask2, score=0.68, presence=0.82
   - "driving lane" → mask3, score=0.81, presence=0.89  ◄── BEST
   - "carriageway"  → mask4, score=0.65, presence=0.78
                    │
                    ▼
4. Select mask3 (highest score that contains click point)
                    │
                    ▼
5. If presence < 0.5: Fall back to point-only prediction
                    │
                    ▼
6. Return RLE-encoded mask to Label Studio
```

### Batch Mode (Auto-Segmentation)

```
1. Image imported to Label Studio
                    │
                    ▼
2. Backend receives task with no context (no user clicks)
                    │
                    ▼
3. Check SAM3_BATCH_LABELS environment variable
                    │
                    ▼
4. For each label in ["road", "paved path", "marked crossing", ...]:
   - Run multi-prompt ensemble (4 prompts × 6 labels = 24 forward passes)
   - Keep masks with score >= BATCH_MIN_SCORE
                    │
                    ▼
5. Return all detected masks as pre-annotations
```

---

## Performance Characteristics

### Current Performance (FP16 Only)

| Metric | Value |
|--------|-------|
| Interactive prediction | ~2-3 seconds per click |
| Batch prediction (6 labels) | ~15-20 seconds per image |
| GPU Memory | ~4-6 GB |
| Forward passes per click | 4 (one per prompt) |
| Forward passes per batch image | 24 (4 prompts × 6 labels) |

### Optimization Attempts

| Optimization | Status | Result |
|--------------|--------|--------|
| FP16 quantization | Deployed | ~10-20% faster |
| torch.compile() | Reverted | CUDA graph errors with dynamic inputs |
| Prompt batching | Not implemented | Would reduce forward passes |
| Selective label prediction | Not implemented | Would skip irrelevant labels |

---

## File Structure

```
segment_anything_3_image/
├── model.py               # Core SAM3 implementation (728 lines)
│   ├── DOMAIN_PROMPTS     # Multi-prompt definitions
│   ├── Model loading      # FP16, dual model setup
│   ├── _sam_predict()     # Interactive inference
│   ├── _batch_predict()   # Batch auto-segmentation
│   └── _point_only_predict()  # Fallback method
├── _wsgi.py               # Flask/Gunicorn server
├── Dockerfile             # Container build
├── docker-compose.yml     # Local development
├── requirements.txt       # Python dependencies
└── TECHNICAL_REPORT.md    # This document
```

---

## Commit History (Key Enhancements)

| Commit | Description |
|--------|-------------|
| `215bb28` | Remove torch.compile() - causes CUDA graph errors |
| `06f124b` | Add FP16 quantization for faster inference |
| `10b9b22` | Add minimum confidence threshold for batch predictions |
| `bb93440` | Add batch prediction mode for auto-segmentation |
| `7377134` | Method 4+6 hybrid: multi-prompt ensemble with point-only fallback |
| `23c7bc1` | Method 4: Confidence-based hybrid (text → point fallback) |
| `4b4349b` | Method 1: Domain-specific text prompts |
| `c058088` | Initial Docker build workflow |

---

## Future Optimization Opportunities

1. **Prompt Batching** - Process all 4 prompts in single forward pass instead of 4 separate passes
2. **Selective Label Prediction** - Pre-classify image type to skip irrelevant labels
3. **Caching** - Cache image embeddings when multiple labels are predicted for same image
4. **Reduced Prompts** - Use only top 2 most effective prompts per label

---

## Deployment

The backend is deployed to AWS EKS via GitHub Actions:

1. Push to `sam3-method4-hybrid` branch
2. GitHub Actions builds Docker image
3. Image pushed to ECR: `767397759867.dkr.ecr.us-east-1.amazonaws.com/label-studio-ml-backend:sam3-method4-hybrid`
4. Kubernetes deployment restarted to pull new image

```bash
# Manual deployment
kubectl rollout restart deployment/sam3-ml-backend -n label-studio
```
