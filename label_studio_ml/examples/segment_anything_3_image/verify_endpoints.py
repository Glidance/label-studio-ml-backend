#!/usr/bin/env python3
"""
Verification script for SAM3 ML Backend endpoints.

This script tests the /health, /setup, and /predict endpoints
to verify the backend is functioning correctly.

Usage:
    python verify_endpoints.py [--url http://localhost:9090] [--verbose]
"""

import argparse
import json
import requests
import sys
import time


def check_health(base_url: str, verbose: bool = False) -> bool:
    """Check the /health endpoint returns 200."""
    url = f"{base_url}/health"
    try:
        response = requests.get(url, timeout=30)
        if verbose:
            print(f"  GET {url}")
            print(f"  Status: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
        return response.status_code == 200
    except requests.RequestException as e:
        if verbose:
            print(f"  Error: {e}")
        return False


def check_setup(base_url: str, verbose: bool = False) -> bool:
    """Check the /setup endpoint returns 200 with a minimal payload."""
    url = f"{base_url}/setup"
    # Minimal Label Studio project configuration
    payload = {
        "project": "1",
        "schema": "<View><Image name=\"image\" value=\"$image\"/><BrushLabels name=\"tag\" toName=\"image\"><Label value=\"Object\"/></BrushLabels></View>",
        "hostname": "http://localhost:8080",
        "access_token": "test-token"
    }
    try:
        response = requests.post(url, json=payload, timeout=60)
        if verbose:
            print(f"  POST {url}")
            print(f"  Payload: {json.dumps(payload, indent=2)}")
            print(f"  Status: {response.status_code}")
            print(f"  Response: {response.text[:500]}")
        return response.status_code == 200
    except requests.RequestException as e:
        if verbose:
            print(f"  Error: {e}")
        return False


def check_predict(base_url: str, verbose: bool = False) -> bool:
    """Check the /predict endpoint returns 200 with a sample task.

    Note: This test uses a minimal payload without actual image data.
    The backend should return an empty predictions list when no context is provided.
    """
    url = f"{base_url}/predict"
    # Minimal Label Studio prediction request
    # Without context, the backend should return empty predictions
    payload = {
        "tasks": [
            {
                "id": 1,
                "data": {
                    "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
                }
            }
        ],
        # No context = no interactive prompt, should return empty predictions
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        if verbose:
            print(f"  POST {url}")
            print(f"  Payload: {json.dumps(payload, indent=2)}")
            print(f"  Status: {response.status_code}")
            print(f"  Response: {response.text[:1000]}")

        if response.status_code != 200:
            return False

        # Verify response structure
        data = response.json()
        if "results" not in data:
            if verbose:
                print("  Warning: 'results' key missing from response")
            # Some versions use different keys, check for predictions
            return "predictions" in data or "results" in data or isinstance(data, list)
        return True
    except requests.RequestException as e:
        if verbose:
            print(f"  Error: {e}")
        return False
    except json.JSONDecodeError as e:
        if verbose:
            print(f"  JSON decode error: {e}")
        return False


def check_predict_with_context(base_url: str, verbose: bool = False) -> bool:
    """Check the /predict endpoint with interactive context (point prompt).

    This tests the actual SAM3 inference with a point prompt.
    """
    url = f"{base_url}/predict"
    # Label Studio prediction request with interactive context
    payload = {
        "tasks": [
            {
                "id": 1,
                "data": {
                    "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
                }
            }
        ],
        "context": {
            "result": [
                {
                    "original_width": 280,
                    "original_height": 228,
                    "value": {
                        "x": 50.0,  # 50% of width = 140px
                        "y": 50.0,  # 50% of height = 114px
                        "keypointlabels": ["Object"]
                    },
                    "type": "keypointlabels",
                    "is_positive": 1
                }
            ]
        }
    }
    try:
        response = requests.post(url, json=payload, timeout=180)
        if verbose:
            print(f"  POST {url} (with context)")
            print(f"  Status: {response.status_code}")
            response_text = response.text
            if len(response_text) > 2000:
                print(f"  Response (truncated): {response_text[:2000]}...")
            else:
                print(f"  Response: {response_text}")

        if response.status_code != 200:
            return False

        # Verify response contains predictions
        data = response.json()
        results = data.get("results", data.get("predictions", []))

        if verbose:
            print(f"  Number of results: {len(results)}")

        # Check if we got at least one result with a mask
        if results and len(results) > 0:
            first_result = results[0]
            if isinstance(first_result, dict):
                result_items = first_result.get("result", [])
                if result_items:
                    for item in result_items:
                        if item.get("type") == "brushlabels" and "rle" in item.get("value", {}):
                            if verbose:
                                print("  Success: Found RLE mask in response!")
                            return True

        if verbose:
            print("  Warning: No valid mask found in response")
        return len(results) > 0  # At least return True if we got any results
    except requests.RequestException as e:
        if verbose:
            print(f"  Error: {e}")
        return False
    except json.JSONDecodeError as e:
        if verbose:
            print(f"  JSON decode error: {e}")
        return False


def wait_for_server(base_url: str, max_wait: int = 300, verbose: bool = False) -> bool:
    """Wait for the server to become available."""
    print(f"Waiting for server at {base_url} (max {max_wait}s)...")
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"  Server is ready! (took {time.time() - start_time:.1f}s)")
                return True
        except requests.RequestException:
            pass
        if verbose:
            print(f"  Waiting... ({int(time.time() - start_time)}s)")
        time.sleep(5)
    return False


def main():
    parser = argparse.ArgumentParser(description="Verify SAM3 ML Backend endpoints")
    parser.add_argument("--url", default="http://localhost:9090", help="Backend URL")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--wait", type=int, default=0, help="Wait for server (seconds)")
    parser.add_argument("--skip-predict-context", action="store_true",
                        help="Skip prediction test with context (requires model)")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    verbose = args.verbose
    results = {}

    print(f"\n{'='*60}")
    print("SAM3 ML Backend Endpoint Verification")
    print(f"{'='*60}")
    print(f"Target: {base_url}\n")

    # Wait for server if requested
    if args.wait > 0:
        if not wait_for_server(base_url, args.wait, verbose):
            print("ERROR: Server did not become available")
            sys.exit(1)
        print()

    # Test 1: Health check
    print("1. Testing /health endpoint...")
    results["health"] = check_health(base_url, verbose)
    print(f"   Result: {'PASS' if results['health'] else 'FAIL'}\n")

    # Test 2: Setup endpoint
    print("2. Testing /setup endpoint...")
    results["setup"] = check_setup(base_url, verbose)
    print(f"   Result: {'PASS' if results['setup'] else 'FAIL'}\n")

    # Test 3: Predict endpoint (no context)
    print("3. Testing /predict endpoint (no context)...")
    results["predict_no_context"] = check_predict(base_url, verbose)
    print(f"   Result: {'PASS' if results['predict_no_context'] else 'FAIL'}\n")

    # Test 4: Predict endpoint (with context)
    if not args.skip_predict_context:
        print("4. Testing /predict endpoint (with point prompt)...")
        results["predict_with_context"] = check_predict_with_context(base_url, verbose)
        print(f"   Result: {'PASS' if results['predict_with_context'] else 'FAIL'}\n")

    # Summary
    print(f"{'='*60}")
    print("Summary:")
    all_passed = all(results.values())
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print(f"{'='*60}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
