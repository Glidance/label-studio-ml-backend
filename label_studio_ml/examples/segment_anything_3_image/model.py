"""
SAM3 (Segment Anything Model 3) backend for Label Studio ML Backend.

This module provides interactive segmentation using SAM3 via the Hugging Face Transformers library.
It supports both point prompts (KeypointLabels) and box prompts (RectangleLabels) from Label Studio.
"""
import os
import logging
import numpy as np
import torch
from typing import List, Dict, Optional
from uuid import uuid4
from PIL import Image

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.converter import brush
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path

logger = logging.getLogger(__name__)

# Domain-specific prompts for blind navigation robot segmentation
# SAM3 likes short noun phrases (concepts). Use 1-4 prompts per class.
# Method 6: Multi-prompt ensemble - try all prompts and pick the best result.
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
    "marked crossing": [
        "crosswalk",
        "pedestrian crossing",
        "zebra crossing",
        "painted crossing markings",
    ],
    "unpaved path": [
        "trail",
        "dirt path",
        "gravel path",
        "unpaved footpath",
    ],
    "driveway": [
        "driveway",
        "vehicle entrance",
        "garage approach",
        "parking access lane",
    ],
    "staircase": [
        "stairs",
        "staircase",
        "steps",
        "stairwell",
    ],
    "mixed use": [
        "shared space",
        "shared street",
        "pedestrian and vehicle shared area",
        "plaza with traffic access",
    ],
    "walkable space": [
        "pedestrian area",
        "walkable open space",
        "courtyard",
        "public square",
    ],
}

# Environment configuration
DEVICE = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = os.getenv('MODEL_NAME', 'facebook/sam3')
HF_TOKEN = os.getenv('HF_TOKEN', os.getenv('HUGGING_FACE_HUB_TOKEN', None))
MASK_THRESHOLD = float(os.getenv('MASK_THRESHOLD', '0.5'))  # Higher = tighter masks

logger.info(f"Using device: {DEVICE}")
logger.info(f"Loading model: {MODEL_NAME}")
logger.info(f"Mask threshold: {MASK_THRESHOLD} (higher = tighter masks)")

# Initialize model and processor
# Try multiple model/processor combinations for compatibility
processor = None
model = None

# Method 1: Try Sam3Processor/Model FIRST (supports TEXT prompts + boxes)
# This is critical for domain-specific text prompting
try:
    from transformers import Sam3Processor, Sam3Model
    processor = Sam3Processor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = Sam3Model.from_pretrained(MODEL_NAME, token=HF_TOKEN).to(DEVICE)
    model.eval()
    MODEL_TYPE = "sam3"
    logger.info("SAM3 model (text-capable) loaded successfully")
except Exception as e:
    logger.warning(f"Sam3 (text-capable) failed: {e}")
    # Fallback to Sam3TrackerProcessor/Model (supports points and boxes, NO text)
    try:
        from transformers import Sam3TrackerProcessor, Sam3TrackerModel
        processor = Sam3TrackerProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        model = Sam3TrackerModel.from_pretrained(MODEL_NAME, token=HF_TOKEN).to(DEVICE)
        model.eval()
        MODEL_TYPE = "sam3_tracker"
        logger.warning("SAM3 Tracker loaded - TEXT PROMPTS NOT AVAILABLE")
    except Exception as e2:
        logger.warning(f"Sam3Tracker failed: {e2}")
        # Fallback to AutoModel/AutoProcessor
        try:
            from transformers import AutoProcessor, AutoModel
            processor = AutoProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
            model = AutoModel.from_pretrained(MODEL_NAME, token=HF_TOKEN).to(DEVICE)
            model.eval()
            MODEL_TYPE = "auto"
            logger.info("SAM3 model loaded via AutoModel")
        except Exception as e3:
            logger.error(f"Failed to load SAM3 model: {e3}")
            MODEL_TYPE = None


class SAM3Model(LabelStudioMLBase):
    """SAM3 ML Backend for Label Studio interactive segmentation.

    Supports:
    - Point prompts via KeypointLabels
    - Box prompts via RectangleLabels
    - Outputs brush masks via BrushLabels
    """

    def setup(self):
        """Initialize the model on first request."""
        self.set("model_version", f"sam3-{MODEL_NAME.split('/')[-1]}")

    def get_results(self, masks: List[np.ndarray], probs: List[float],
                    width: int, height: int, from_name: str, to_name: str,
                    label: str) -> List[Dict]:
        """Convert SAM3 masks to Label Studio format.

        Args:
            masks: List of binary masks from SAM3
            probs: List of confidence scores
            width: Original image width
            height: Original image height
            from_name: Label Studio from_name
            to_name: Label Studio to_name
            label: Selected label name

        Returns:
            List of prediction dictionaries in Label Studio format
        """
        results = []
        total_prob = 0

        for mask, prob in zip(masks, probs):
            label_id = str(uuid4())[:4]
            # Convert mask to RLE format for Label Studio
            mask_uint8 = (mask * 255).astype(np.uint8)
            rle = brush.mask2rle(mask_uint8)
            total_prob += prob

            results.append({
                'id': label_id,
                'from_name': from_name,
                'to_name': to_name,
                'original_width': width,
                'original_height': height,
                'image_rotation': 0,
                'value': {
                    'format': 'rle',
                    'rle': rle,
                    'brushlabels': [label],
                },
                'score': prob,
                'type': 'brushlabels',
                'readonly': False
            })

        return [{
            'result': results,
            'model_version': self.get('model_version'),
            'score': total_prob / max(len(results), 1)
        }]

    def load_image(self, image_url: str, task_id: int) -> Image.Image:
        """Load and prepare image from URL or local path.

        Args:
            image_url: URL or path to image
            task_id: Label Studio task ID

        Returns:
            PIL Image in RGB format
        """
        image_path = get_local_path(image_url, task_id=task_id)
        image = Image.open(image_path).convert("RGB")
        return image

    def _sam_predict(self, img_url: str, point_coords: Optional[List] = None,
                     point_labels: Optional[List] = None, input_box: Optional[List] = None,
                     task: Optional[Dict] = None, label: Optional[str] = None) -> Dict:
        """Run SAM3 inference with given prompts.

        Args:
            img_url: Image URL or path
            point_coords: List of [x, y] point coordinates
            point_labels: List of point labels (1=positive, 0=negative)
            input_box: Bounding box as [x1, y1, x2, y2]
            task: Label Studio task dictionary
            label: Text label for Sam3Model (used as text prompt)

        Returns:
            Dictionary with 'masks' and 'probs' keys
        """
        if model is None or processor is None:
            raise RuntimeError("SAM3 model not loaded. Check HF_TOKEN and model availability.")

        # Load image
        task_id = task.get('id') if task else None
        image = self.load_image(img_url, task_id)

        # Different processing based on model type
        if MODEL_TYPE == "sam3_tracker":
            # Sam3TrackerProcessor format:
            # input_points: [batch, objects, points_per_object, coords]
            # input_labels: [batch, objects, points_per_object]
            # input_boxes: [batch, objects, 4]
            input_points_tensor = None
            input_labels_tensor = None
            input_boxes_tensor = None

            if point_coords and len(point_coords) > 0:
                input_points_tensor = [[point_coords]]
                input_labels_tensor = [[point_labels]]

            if input_box is not None:
                input_boxes_tensor = [[input_box]]

            inputs = processor(
                images=image,
                input_points=input_points_tensor,
                input_labels=input_labels_tensor,
                input_boxes=input_boxes_tensor,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                outputs = model(**inputs, multimask_output=True)

        elif MODEL_TYPE == "sam3":
            # METHOD 6: MULTI-PROMPT ENSEMBLE
            # Try all prompts for a label and pick the best result
            import torch.nn.functional as F

            # Get list of prompts for this label
            prompts = DOMAIN_PROMPTS.get(label.lower(), [label]) if label else ["surface"]
            if isinstance(prompts, str):
                prompts = [prompts]  # Handle legacy single-string prompts

            logger.info(f"Multi-prompt ensemble: trying {len(prompts)} prompts for label '{label}'")

            # Get image dimensions
            img_w, img_h = image.size

            # Track best result across all prompts
            best_mask = None
            best_score = 0.0
            best_prompt = None

            for text_prompt in prompts:
                logger.info(f"  Trying prompt: '{text_prompt}'")

                # Run text-prompted inference
                inputs = processor(
                    images=image,
                    text=text_prompt,
                    return_tensors="pt"
                ).to(DEVICE)

                with torch.no_grad():
                    outputs = model(**inputs)

                # Extract outputs
                pred_masks = outputs.pred_masks[0]  # [200, H, W]
                pred_logits = torch.sigmoid(outputs.pred_logits[0])  # [200]

                # Get presence score
                presence = 1.0
                if hasattr(outputs, 'presence_logits') and outputs.presence_logits is not None:
                    presence = torch.sigmoid(outputs.presence_logits[0, 0]).item()

                # Compute final scores
                final_scores = pred_logits * presence
                threshold = 0.05
                keep_indices = torch.where(final_scores > threshold)[0]

                logger.info(f"    Presence: {presence:.3f}, kept {len(keep_indices)} masks")

                if len(keep_indices) == 0:
                    continue

                mask_h, mask_w = pred_masks.shape[1], pred_masks.shape[2]

                # Find mask containing click point (if provided)
                if point_coords and len(point_coords) > 0:
                    click_x, click_y = int(point_coords[0][0]), int(point_coords[0][1])
                    scaled_x = max(0, min(int(click_x * mask_w / img_w), mask_w - 1))
                    scaled_y = max(0, min(int(click_y * mask_h / img_h), mask_h - 1))

                    for idx in keep_indices:
                        idx = idx.item()
                        mask_prob = torch.sigmoid(pred_masks[idx])
                        point_activation = mask_prob[scaled_y, scaled_x].item()

                        if point_activation > 0.5:
                            score = final_scores[idx].item()
                            if score > best_score:
                                best_score = score
                                best_prompt = text_prompt
                                # Resize and store mask
                                mask_resized = F.interpolate(
                                    mask_prob.unsqueeze(0).unsqueeze(0),
                                    size=(img_h, img_w),
                                    mode='bilinear',
                                    align_corners=False
                                )[0, 0]
                                best_mask = (mask_resized > MASK_THRESHOLD).cpu().numpy().astype(np.uint8)
                else:
                    # No click point - use highest scoring mask for this prompt
                    idx = keep_indices[torch.argmax(final_scores[keep_indices])].item()
                    score = final_scores[idx].item()
                    if score > best_score:
                        best_score = score
                        best_prompt = text_prompt
                        mask_prob = torch.sigmoid(pred_masks[idx])
                        mask_resized = F.interpolate(
                            mask_prob.unsqueeze(0).unsqueeze(0),
                            size=(img_h, img_w),
                            mode='bilinear',
                            align_corners=False
                        )[0, 0]
                        best_mask = (mask_resized > MASK_THRESHOLD).cpu().numpy().astype(np.uint8)

            if best_mask is not None:
                logger.info(f"Best result: prompt='{best_prompt}', score={best_score:.3f}, pixels={best_mask.sum()}")
                return {'masks': [best_mask], 'probs': [best_score]}
            else:
                logger.warning(f"No masks found across {len(prompts)} prompts")
                return {'masks': [], 'probs': []}

        else:
            # Generic AutoModel fallback
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)

        # Post-process based on model type
        if MODEL_TYPE == "sam3_tracker":
            # Sam3Tracker: post_process_masks returns list of tensors
            masks = processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"]
            )[0]  # Get first batch item

            # Get scores - outputs.iou_scores has shape [batch, objects, num_masks]
            scores = outputs.iou_scores.cpu().numpy()[0, 0]  # [num_masks]

            # Select best mask (highest IoU score)
            best_idx = np.argmax(scores)
            best_mask = masks[0, best_idx].numpy().astype(np.uint8)
            best_prob = float(scores[best_idx])

        # Note: MODEL_TYPE == "sam3" returns early in the inference block above

        else:
            # Fallback for AutoModel - try different output formats
            if hasattr(outputs, 'pred_masks'):
                masks = outputs.pred_masks.cpu().numpy()
                best_mask = masks[0, 0].astype(np.uint8)
                best_prob = 0.5
            else:
                logger.warning("Unknown output format")
                return {'masks': [], 'probs': []}

        logger.info(f"Generated mask with score {best_prob:.3f}")

        return {
            'masks': [best_mask],
            'probs': [best_prob]
        }

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None,
                **kwargs) -> ModelResponse:
        """Generate predictions for Label Studio tasks.

        This method is called when the user places keypoints or rectangles
        on an image in interactive mode.

        Args:
            tasks: List of Label Studio tasks
            context: Interactive annotation context with current selections
            **kwargs: Additional arguments

        Returns:
            ModelResponse with predicted masks
        """
        # If no context, no interaction has happened yet - return early
        if not context or not context.get('result'):
            return ModelResponse(predictions=[])

        from_name, to_name, value = self.get_first_tag_occurence('BrushLabels', 'Image')

        # Get image dimensions from context
        image_width = context['result'][0]['original_width']
        image_height = context['result'][0]['original_height']

        # Collect prompts from context
        point_coords = []
        point_labels = []
        input_box = None
        selected_label = None

        for ctx in context['result']:
            # Convert percentage coordinates to pixel coordinates
            x = ctx['value']['x'] * image_width / 100
            y = ctx['value']['y'] * image_height / 100
            ctx_type = ctx['type']
            selected_label = ctx['value'][ctx_type][0]

            if ctx_type == 'keypointlabels':
                # Point prompt - is_positive indicates if it's a positive (1) or negative (0) point
                point_labels.append(int(ctx.get('is_positive', 1)))
                point_coords.append([int(x), int(y)])
            elif ctx_type == 'rectanglelabels':
                # Box prompt - convert to [x1, y1, x2, y2] format
                box_width = ctx['value']['width'] * image_width / 100
                box_height = ctx['value']['height'] * image_height / 100
                input_box = [int(x), int(y), int(x + box_width), int(y + box_height)]

        logger.info(f"Point coords: {point_coords}, Point labels: {point_labels}, Box: {input_box}")

        # Get image URL
        img_url = tasks[0]['data'][value]

        # Run SAM3 prediction
        try:
            predictor_results = self._sam_predict(
                img_url=img_url,
                point_coords=point_coords if point_coords else None,
                point_labels=point_labels if point_labels else None,
                input_box=input_box,
                task=tasks[0],
                label=selected_label
            )
        except Exception as e:
            logger.error(f"SAM3 prediction failed: {e}")
            return ModelResponse(predictions=[])

        # Convert results to Label Studio format
        predictions = self.get_results(
            masks=predictor_results['masks'],
            probs=predictor_results['probs'],
            width=image_width,
            height=image_height,
            from_name=from_name,
            to_name=to_name,
            label=selected_label
        )

        return ModelResponse(predictions=predictions)


# Export model class
NewModel = SAM3Model
