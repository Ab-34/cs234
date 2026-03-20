"""
extract_scores.py

Reads all videos in the `playback/` folder, extracts frames, scores each frame
(-100 to 100) using Claude via Amazon Bedrock, saves frames to `img/`, and writes
a JSON file mapping frame paths to their scores.
"""

import os
import json
import base64
import glob
import time
import boto3
import cv2
from botocore.config import Config

# ──────────────────────────────────────────────
# Configuration — edit these as needed
# ──────────────────────────────────────────────
PLAYBACK_DIR = "playback"
IMG_DIR = "img"
OUTPUT_JSON = "frame_scores.json"

MODEL_ID = "global.anthropic.claude-sonnet-4-6"
REGION = "us-east-1"
config = Config(read_timeout=1000)

# How many frames to sample per video (None = every frame)
FRAMES_PER_VIDEO = 101

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries

SCORING_PROMPT_INTRO = """\
Act as a precise robotic vision evaluator for a "Lift" task.

**Task Definition:**
1. Approach: Move gripper to the cube's location.
2. Grasp: Close fingers securely around the cube.
3. Lift: Raise the cube off the table surface.

**Scoring Rubric (-100 to 100):**

Negative scores (penalize bad gripper states):
- -100 to -50: Gripper is severely misaligned (e.g., approaching the cube sideways, fingers splayed
               perpendicular to the grasp axis, or colliding with the table/environment).
- -50 to -1:   Gripper is moderately misaligned or poorly oriented (e.g., tilted at an angle that
               would prevent a clean grasp, approaching from above when a side grasp is needed,
               or fingers not parallel to the cube faces).

Positive scores (reward good progress):
- 0-25:  Gripper at home position or actively moving toward the cube but >5 cm away, with
         reasonable approach orientation.
- 25-50: Gripper positioned around the cube with good alignment, ready to grasp.
- 50-75: Gripper fingers closed; contact made, but cube still touching the table.
- 75-100: Grippers fingers closed and cube clearly lifted off the table.

**Key alignment signals that trigger negative scores:**
- Fingers not bracketing the cube (gripper off-center or rotated >45° from optimal grasp axis).
- Gripper approaching from a direction that would knock the cube away rather than grasp it.
- Wrist/arm posture that would make a stable grasp geometrically impossible.

You will be shown up to 6 reference frames to give you temporal context:
  # Initial frame  — the very first frame of the episode (baseline/home state).
  # Frame t-2      — two frames before the current frame.
  # Frame t-1      — one frame before the current frame.
  # CURRENT frame  — the frame you must score.
  # Frame t+1      — one frame after the current frame.
  # Frame t+2      — two frames after the current frame.

Use the surrounding frames only as context to understand the trajectory.
Score ONLY the CURRENT frame.

**Constraint:** Reply with ONLY a single integer between -100 and 100. No text, units, or explanations.\
"""

SCORING_PROMPT_FINAL = "Based on all the frames above, what is the score for the CURRENT frame? Reply with a single integer from -100 to 100 only."


# ──────────────────────────────────────────────
# Bedrock helpers
# ──────────────────────────────────────────────

def _make_client():
    return boto3.client("bedrock-runtime", region_name=REGION, config=config)


def _image_block(img_path: str) -> dict:
    """Build a Bedrock image content block from a file path."""
    ext = os.path.splitext(img_path)[1].lower()
    media_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
    }.get(ext, "image/png")
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def score_frame(current_path: str, client, *,
                initial_path: str = None,
                prev2_path: str = None,
                prev1_path: str = None,
                next1_path: str = None,
                next2_path: str = None) -> int:
    """
    Return a -100 to 100 reward score for current_path using surrounding frames as context.

    Args:
        current_path:  The frame to score.
        client:        Boto3 Bedrock runtime client.
        initial_path:  Frame 0 of the episode (baseline state).
        prev2_path:    Frame at t-2 (None if unavailable).
        prev1_path:    Frame at t-1 (None if unavailable).
        next1_path:    Frame at t+1 (None if unavailable).
        next2_path:    Frame at t+2 (None if unavailable).
    """
    content_blocks = [_text_block(SCORING_PROMPT_INTRO)]

    if initial_path:
        content_blocks += [_text_block("--- Initial frame (home/baseline state) ---"),
                           _image_block(initial_path)]
    if prev2_path:
        content_blocks += [_text_block("--- Frame t-2 ---"), _image_block(prev2_path)]
    if prev1_path:
        content_blocks += [_text_block("--- Frame t-1 ---"), _image_block(prev1_path)]

    content_blocks += [_text_block("--- CURRENT frame (score this one) ---"),
                       _image_block(current_path)]

    if next1_path:
        content_blocks += [_text_block("--- Frame t+1 ---"), _image_block(next1_path)]
    if next2_path:
        content_blocks += [_text_block("--- Frame t+2 ---"), _image_block(next2_path)]

    content_blocks.append(_text_block(SCORING_PROMPT_FINAL))

    body = json.dumps({
        "messages": [{"role": "user", "content": content_blocks}],
        "max_tokens": 16,
        "temperature": 0.0,
        "anthropic_version": "bedrock-2023-05-31",
    })

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.invoke_model(
                modelId=MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            result = json.loads(response["body"].read())
            text = result["content"][0]["text"].strip()
            # Preserve leading minus sign for negative scores
            score = int("".join(c for c in text if c.isdigit() or c in "-").rstrip("-"))
            return max(-100, min(100, score))
        except Exception as e:
            print(f"    [attempt {attempt}/{MAX_RETRIES}] Error scoring {current_path}: {e} {text}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    print(f"    Failed to score {current_path} after {MAX_RETRIES} attempts — defaulting to 0.")
    return 0  # neutral fallback on failure


# ──────────────────────────────────────────────
# Frame extraction
# ──────────────────────────────────────────────

def extract_frames(video_path: str, out_dir: str, n_frames=None) -> list[str]:
    """
    Extract frames from a video and save as PNGs.

    Args:
        video_path: Path to the .mp4 file.
        out_dir:    Directory where frames are saved.
        n_frames:   Number of evenly-spaced frames to extract. None = every frame.

    Returns:
        List of saved frame file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Warning: cannot open {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        print(f"  Warning: 0 frames in {video_path}")
        cap.release()
        return []

    if n_frames is None or n_frames >= total:
        indices = list(range(total))
    else:
        step = total / n_frames
        indices = [int(i * step) for i in range(n_frames)]

    stem = os.path.splitext(os.path.basename(video_path))[0]
    saved = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        fname = f"{stem}_frame{idx:06d}.png"
        fpath = os.path.join(out_dir, fname)
        cv2.imwrite(fpath, frame)
        saved.append(fpath)

    cap.release()
    return saved


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    os.makedirs(IMG_DIR, exist_ok=True)

    video_paths = sorted(glob.glob(os.path.join(PLAYBACK_DIR, "*.mp4")))
    if not video_paths:
        print(f"No .mp4 files found in '{PLAYBACK_DIR}'. Exiting.")
        return

    print(f"Found {len(video_paths)} video(s) in '{PLAYBACK_DIR}'.")

    client = _make_client()
    results = {}

    # Load existing results so we can resume interrupted runs
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} frames already scored.")

    for v_idx, video_path in enumerate(video_paths, 1):
        print(f"\n[{v_idx}/{len(video_paths)}] Processing: {video_path}")

        video_out_dir = os.path.join(IMG_DIR, os.path.splitext(os.path.basename(video_path))[0])
        frame_paths = extract_frames(video_path, video_out_dir, n_frames=FRAMES_PER_VIDEO)

        if not frame_paths:
            print("  No frames extracted — skipping.")
            continue

        print(f"  Extracted {len(frame_paths)} frame(s).")

        n = len(frame_paths)
        for i, frame_path in enumerate(frame_paths):
            f_idx = i + 1
            # Normalize to forward slashes for consistent JSON keys
            key = frame_path.replace("\\", "/")

            if key in results:
                print(f"  [{f_idx}/{n}] Already scored: {key} = {results[key]}")
                continue

            score = score_frame(
                frame_path, client,
                initial_path=frame_paths[0] if i > 0 else None,
                prev2_path=frame_paths[i - 2] if i >= 2 else None,
                prev1_path=frame_paths[i - 1] if i >= 1 else None,
                next1_path=frame_paths[i + 1] if i + 1 < n else None,
                next2_path=frame_paths[i + 2] if i + 2 < n else None,
            )
            results[key] = score
            print(f"  [{f_idx}/{n}] {key} → {score}")

            # Save after every frame so progress isn't lost on interruption
            with open(OUTPUT_JSON, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} frames scored. Results saved to '{OUTPUT_JSON}'.")


if __name__ == "__main__":
    main()
