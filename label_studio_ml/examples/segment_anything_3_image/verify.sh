#!/bin/bash
#
# SAM3 ML Backend Verification Script
#
# This script builds the Docker image, starts the container,
# and runs endpoint verification tests.
#
# Usage:
#   ./verify.sh [--hf-token-file /path/to/token] [--skip-build] [--keep-running]
#
# Requirements:
#   - Docker and docker-compose installed
#   - HF_TOKEN environment variable set OR --hf-token-file provided
#
# The script will exit with 0 on success, non-zero on failure.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default settings
SKIP_BUILD=false
KEEP_RUNNING=false
HF_TOKEN_FILE=""
BACKEND_URL="http://localhost:9090"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --hf-token-file)
            HF_TOKEN_FILE="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --keep-running)
            KEEP_RUNNING=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --hf-token-file FILE  Path to file containing HF token"
            echo "  --skip-build          Skip Docker build step"
            echo "  --keep-running        Keep container running after tests"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "SAM3 ML Backend Verification"
echo "========================================"
echo ""

# Check for HF token
if [ -n "$HF_TOKEN_FILE" ]; then
    if [ -f "$HF_TOKEN_FILE" ]; then
        export HF_TOKEN=$(cat "$HF_TOKEN_FILE" | tr -d '\n')
        echo -e "${GREEN}Loaded HF token from file${NC}"
    else
        echo -e "${RED}ERROR: HF token file not found: $HF_TOKEN_FILE${NC}"
        exit 1
    fi
fi

if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}WARNING: HF_TOKEN not set. SAM3 requires authentication.${NC}"
    echo "Set HF_TOKEN environment variable or use --hf-token-file"
    echo ""
    echo "For CPU-only testing without the model, you can continue,"
    echo "but /predict with context will fail."
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Cleanup function
cleanup() {
    if [ "$KEEP_RUNNING" = false ]; then
        echo ""
        echo "Stopping containers..."
        docker-compose down 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Step 1: Build Docker image
if [ "$SKIP_BUILD" = false ]; then
    echo ""
    echo "Step 1: Building Docker image..."
    echo "----------------------------------------"
    if docker-compose build; then
        echo -e "${GREEN}Docker build successful${NC}"
    else
        echo -e "${RED}Docker build failed${NC}"
        exit 1
    fi
else
    echo ""
    echo "Step 1: Skipping Docker build (--skip-build)"
fi

# Step 2: Start the container
echo ""
echo "Step 2: Starting container..."
echo "----------------------------------------"
docker-compose down 2>/dev/null || true
docker-compose up -d

# Step 3: Wait for the server to be ready
echo ""
echo "Step 3: Waiting for server to start..."
echo "----------------------------------------"
MAX_WAIT=300
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s -f "$BACKEND_URL/health" > /dev/null 2>&1; then
        echo -e "${GREEN}Server is ready!${NC}"
        break
    fi
    echo "  Waiting... ($WAITED/$MAX_WAIT seconds)"
    sleep 10
    WAITED=$((WAITED + 10))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo -e "${RED}Server failed to start within ${MAX_WAIT}s${NC}"
    echo "Checking container logs:"
    docker-compose logs --tail=50
    exit 1
fi

# Step 4: Run verification tests
echo ""
echo "Step 4: Running endpoint verification..."
echo "----------------------------------------"

# Install requests if needed
pip install requests -q 2>/dev/null || true

if python verify_endpoints.py --url "$BACKEND_URL" --verbose; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}All verification tests PASSED!${NC}"
    echo -e "${GREEN}========================================${NC}"
    EXIT_CODE=0
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Some verification tests FAILED${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Container logs:"
    docker-compose logs --tail=100
    EXIT_CODE=1
fi

if [ "$KEEP_RUNNING" = true ]; then
    echo ""
    echo "Container is still running (--keep-running)"
    echo "To stop: docker-compose down"
    echo "To view logs: docker-compose logs -f"
fi

exit $EXIT_CODE
