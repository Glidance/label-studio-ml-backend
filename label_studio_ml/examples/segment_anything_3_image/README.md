# SAM3 (Segment Anything Model 3) for Label Studio

Interactive image segmentation using Meta's SAM3 model via Hugging Face Transformers.

## Overview

SAM3 (Segment Anything Model 3) is Meta's latest foundation model for promptable segmentation. This ML backend enables interactive segmentation in Label Studio using point clicks and bounding boxes.

**Key Features:**
- Point prompts via `KeypointLabels` (positive/negative clicks)
- Box prompts via `RectangleLabels`
- Outputs segmentation masks as `BrushLabels`
- Uses Hugging Face Transformers (no separate model download required)

## Prerequisites

- Docker and Docker Compose installed
- Hugging Face account with access to `facebook/sam3` (gated model)
- GPU recommended for faster inference (CPU supported but slower)

## Quick Start

### 1. Get Hugging Face Token

SAM3 is a gated model. You need to:

1. Go to [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
2. Accept the model terms
3. Get your access token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 2. Set Up Authentication

**Option A: Environment variable (recommended)**
```bash
export HF_TOKEN="your_huggingface_token_here"
```

**Option B: Token file**
```bash
# Create a token file (DO NOT commit this file!)
echo "your_token_here" > ~/.hf_token
chmod 600 ~/.hf_token
```

### 3. Start the Backend

```bash
cd label_studio_ml/examples/segment_anything_3_image

# With environment variable
HF_TOKEN=your_token docker-compose up

# Or if you have HF_TOKEN exported
docker-compose up
```

### 4. Verify the Backend

```bash
# Quick health check
curl http://localhost:9090/health
# Expected: {"status":"UP"}

# Run full verification
./verify.sh --hf-token-file ~/.hf_token
```

## Configuration

### Environment Variables

Set these in `docker-compose.yml` or pass via environment:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | (required) | Hugging Face access token |
| `MODEL_NAME` | `facebook/sam3` | Hugging Face model ID |
| `DEVICE` | `cpu` | Device: `cpu` or `cuda` |
| `LOG_LEVEL` | `INFO` | Logging level |
| `WORKERS` | `1` | Gunicorn workers |
| `THREADS` | `4` | Threads per worker |
| `PORT` | `9090` | Server port |

### GPU Support

To enable GPU acceleration, uncomment the `deploy` section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Also set `DEVICE=cuda` in the environment.

## Label Studio Configuration

### Labeling Config

Use this labeling configuration for interactive SAM3 segmentation:

```xml
<View>
  <Style>
    .container { display: flex; gap: 20px; margin-bottom: 20px; }
    .column { flex: 1; padding: 10px; background: #f5f5f5; border-radius: 5px; }
    .title { font-weight: bold; margin-bottom: 10px; }
  </Style>

  <View className="container">
    <View className="column">
      <View className="title">Output Labels</View>
      <BrushLabels name="tag" toName="image">
        <Label value="Object" background="#FF6B6B"/>
        <Label value="Background" background="#4ECDC4"/>
      </BrushLabels>
    </View>

    <View className="column">
      <View className="title">Point Prompts</View>
      <KeyPointLabels name="keypoints" toName="image" smart="true">
        <Label value="Object" background="#FF6B6B"/>
        <Label value="Background" background="#4ECDC4"/>
      </KeyPointLabels>
    </View>

    <View className="column">
      <View className="title">Box Prompts</View>
      <RectangleLabels name="boxes" toName="image" smart="true">
        <Label value="Object" background="#FFE66D"/>
      </RectangleLabels>
    </View>
  </View>

  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
</View>
```

### Connecting to Label Studio

**IMPORTANT: Docker Networking**

When Label Studio runs in Docker, `localhost` refers to the container itself, not your host machine. You must use the host's actual IP address.

**Find your IP:**
```bash
# macOS/Linux
ifconfig | grep "inet " | grep -v 127.0.0.1

# Windows
ipconfig
```

**Connect the backend:**

1. Go to your Label Studio project: **Settings > Machine Learning**
2. Click **Add Model**
3. Enter the URL using your host IP:
   - **Correct:** `http://192.168.1.100:9090`
   - **Wrong:** `http://localhost:9090`
4. Enable **Interactive preannotations**

### Example Workflow

1. Create a project with the labeling config above
2. Import images
3. Connect the SAM3 ML backend
4. Open an image for labeling
5. Select a label (e.g., "Object")
6. Click on the object using the Keypoint tool - SAM3 generates a mask
7. Add more points to refine, or draw a box to constrain the region
8. Accept the mask and save

## API Endpoints

### GET /health
Health check endpoint.
```bash
curl http://localhost:9090/health
# {"status":"UP"}
```

### POST /setup
Configure the model for a Label Studio project.
```bash
curl -X POST http://localhost:9090/setup \
  -H "Content-Type: application/json" \
  -d '{"project": "1", "schema": "<View>...</View>"}'
```

### POST /predict
Generate predictions for interactive annotation.
```bash
curl -X POST http://localhost:9090/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [{"id": 1, "data": {"image": "https://example.com/image.jpg"}}],
    "context": {
      "result": [{
        "original_width": 800,
        "original_height": 600,
        "value": {"x": 50, "y": 50, "keypointlabels": ["Object"]},
        "type": "keypointlabels",
        "is_positive": 1
      }]
    }
  }'
```

## Running Verification Tests

The verification script tests all endpoints:

```bash
# Full verification with Docker build
./verify.sh --hf-token-file ~/.hf_token

# Skip build if image already exists
./verify.sh --skip-build --hf-token-file ~/.hf_token

# Keep container running after tests
./verify.sh --keep-running

# Manual Python test
python verify_endpoints.py --url http://localhost:9090 --verbose
```

## Troubleshooting

### "401 Unauthorized" or Model Access Denied
- Verify you've accepted the model terms at huggingface.co/facebook/sam3
- Check your HF_TOKEN is correct and has read access
- Try: `huggingface-cli login` locally to verify credentials

### Container Exits Immediately
Check logs:
```bash
docker-compose logs sam3-backend
```

Common issues:
- Missing HF_TOKEN
- Out of memory (try reducing WORKERS/THREADS or use smaller batch)
- CUDA not available (set DEVICE=cpu)

### Slow Inference
- Enable GPU (see GPU Support section)
- First request downloads model weights (subsequent requests are faster)
- Reduce image size in Label Studio settings

### Label Studio Can't Connect
- Verify backend is running: `curl http://localhost:9090/health`
- Use host IP, not localhost (see Docker Networking section)
- Check firewall allows port 9090
- Verify no auth mismatch (BASIC_AUTH_USER/PASS)

### Out of Memory (GPU)
- Use `DEVICE=cpu` as fallback
- Reduce `WORKERS` to 1
- The model requires ~4-8GB GPU memory

## Development

### Running Without Docker

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-base.txt
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN="your_token"
export DEVICE="cpu"

# Start server
python _wsgi.py --port 9090 --debug
```

### Project Structure

```
segment_anything_3_image/
├── model.py              # SAM3 model implementation
├── _wsgi.py             # Flask/Gunicorn entry point
├── Dockerfile           # Docker build configuration
├── docker-compose.yml   # Docker Compose configuration
├── requirements.txt     # SAM3-specific dependencies
├── requirements-base.txt # Base Label Studio ML dependencies
├── requirements-test.txt # Test dependencies
├── start.sh             # Container start script
├── verify.sh            # Verification script
├── verify_endpoints.py  # Python endpoint tests
└── README.md            # This file
```

## Security Notes

- **Never commit** your HF_TOKEN or any API keys
- The token file should have restricted permissions (`chmod 600`)
- Don't log or print tokens in debugging
- Use environment variables or mounted secrets, never bake tokens into images

## License

This example follows the same license as the label-studio-ml-backend repository.

SAM3 model weights are subject to Meta's license terms - see the model card at huggingface.co/facebook/sam3.
