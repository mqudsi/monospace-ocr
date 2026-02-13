#!/usr/bin/env -S uv.exe run
from concurrent.futures import ProcessPoolExecutor
import os
import argparse
import cv2
import json
import numpy as np
import random
import subprocess
import sys
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from ultralytics import YOLO

# --- CONFIGURATION ---
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
CHAR_TO_IDX = {char: i for i, char in enumerate(ALPHABET)}
IDX_TO_CHAR = {i: char for char, i in CHAR_TO_IDX.items()}

_FALLBACK_FONT_PATHS = [
    "/usr/share/fonts/truetype/msttcorefonts/Courier_New.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/cour.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]


def _discover_font_paths():
    wanted = {
        "courier new",
        "liberation mono",
        "dejavu sans mono",
    }

    try:
        p = subprocess.run(
            ["fc-list", "-f", "%{family}\t%{file}\n"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        return list(_FALLBACK_FONT_PATHS)

    paths = []
    seen = set()
    for line in p.stdout.splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        fam, fpath = parts[0].strip().lower(), parts[1].strip()
        if not fpath or fpath in seen:
            continue
        if any(w in fam for w in wanted):
            paths.append(fpath)
            seen.add(fpath)

    return paths if paths else list(_FALLBACK_FONT_PATHS)


FONT_PATHS = _discover_font_paths()
FONT_SIZE = 16
CANVAS_W, CANVAS_H = 800, 64
DATASET_DIR = "ocr_dataset"
# MODEL_NAME = "yolo11n.pt" # Latest YOLO version
MODEL_NAME = "yolo26n.pt"  # Latest YOLO version


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _generate_ocr_sample(args):
    i, split, img_dir, lbl_dir, fine_tune, debug = args
    font = None
    for fp in random.sample(FONT_PATHS, k=len(FONT_PATHS)):
        try:
            font = ImageFont.truetype(fp, FONT_SIZE)
            break
        except Exception:
            pass

    if font is None:
        print("Font file not found. Please check FONT_PATHS.")
        return

    # Create base canvas
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Generate random text, oversampling tricky characters
    hard_chars = "ijlI1t/f"
    text_len = int(80 - (random.random() ** 2 * 60))

    if random.random() < 0.021:
        text_len = 1

    text_len = max(1, text_len)

    # 40% of the time, fill at least 30% of the slots with confusable characters
    if True and random.random() < 0.65:
        num_hard = int(text_len * 0.3)
        num_normal = text_len - num_hard

        # Create a mixed pool and shuffle it
        pool = random.choices(hard_chars, k=num_hard) + random.choices(
            ALPHABET, k=num_normal
        )
        random.shuffle(pool)
        text = "".join(pool)
    else:
        # Standard random distribution
        text = "".join(random.choices(ALPHABET, k=text_len))

    # Calculate text dimensions for centering
    total_w = font.getlength(text)
    curr_x = (CANVAS_W - total_w) // 2
    curr_y = (CANVAS_H - FONT_SIZE) // 2

    # Introduce a slight shift to curr_y and curr_x
    curr_y += random.randint(-8, 8)
    if text_len < 70:
        curr_x -= random.randint(0, 60)

    labels = []

    # Draw all the text at once so that kerning is properly applied,
    # to try and mimic how real-world inputs rendered with Microsoft
    # Office's GPOS text shaping engine might look.
    draw.text((curr_x, curr_y), text, font=font, fill=(0, 0, 0))

    # Extract the character bounding boxes for training/val data
    for j, char in enumerate(text):
        if char.isspace():
            continue

        # Assume the position of character x is the length of the
        # entire string up to and including x, minus the length
        # of x itself.
        prefix_len = font.getlength(text[: j + 1])
        char_len = font.getlength(char)
        char_start = curr_x + (prefix_len - char_len)

        # Get character bounding box (left, top, right, bottom)
        bbox = font.getbbox(char)

        # Get absolute bbox coordinates
        left = char_start + bbox[0]
        top = curr_y + bbox[1]
        right = char_start + bbox[2]
        bottom = curr_y + bbox[3]

        char_w = right - left
        char_h = bottom - top

        # Calculate normalized (relative) bounding box
        x_center = (left + char_w / 2) / CANVAS_W
        y_center = (top + char_h / 2) / CANVAS_H
        nw = char_w / CANVAS_W
        nh = char_h / CANVAS_H
        labels.append(
            f"{CHAR_TO_IDX[char]} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}"
        )

        if debug:
            draw.rectangle(
                [left - 1, top - 1, right + 1, bottom + 1],
                outline="red",
                width=1,
            )

    if debug:
        # Show bounding boxes on line
        img.show()
        sys.exit()

    # Save (at slightly lower resolution when fine-tuning)
    name = f"{split}_{i:05d}"
    img_path = os.path.join(img_dir, f"{name}.jpg")
    if not fine_tune:
        # default quality is 75!?
        img.save(img_path, "JPEG", quality=95)
    else:
        img.save(img_path, "JPEG", quality=random.randint(85, 95))

    with open(os.path.join(lbl_dir, f"{name}.txt"), "w") as f:
        f.write("\n".join(labels))


class YOLO_OCR:
    def __init__(self, model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_path:
            if not os.path.exists(model_path):
                raise Exception(f"Model {model_path} not found")
            self.model = YOLO(model_path)
            self.fine_tune = True
        else:
            self.model = YOLO(MODEL_NAME)
            self.fine_tune = False

    # --- PART 1: DATA GENERATION ---
    def generate_data(self, count=1000, split="train"):
        print(f"Generating {count} samples for {split}...")
        img_dir = os.path.join(DATASET_DIR, split, "images")
        lbl_dir = os.path.join(DATASET_DIR, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        debug = False
        with ProcessPoolExecutor() as executor:
            jobs = ((i, split, img_dir, lbl_dir, self.fine_tune, debug) for i in range(count))
            for _ in executor.map(_generate_ocr_sample, jobs):
                pass

    # --- PART 2: TRAINING ---
    def train(self, epochs=100, patience=30):
        # Create YAML
        yaml_path = os.path.join(DATASET_DIR, "data.yaml")
        with open(yaml_path, "w") as f:
            f.write(f"path: {os.path.abspath(DATASET_DIR)}\n")
            f.write("train: train/images\nval: val/images\n")
            f.write(f"names:\n")
            for i, c in IDX_TO_CHAR.items():
                f.write(f"  {i}: '{c}'\n")

        train_args = {
            "device": self.device,
            "data": yaml_path,
            "epochs": int(epochs),
            "patience": int(patience),
            "batch": 48,
            "imgsz": 1600,
            "rect": True,
            "mosaic": 0.0,
            "close_mosaic": 0,
            "fliplr": 0.0,
            "flipud": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "degrees": 0.0,
            "scale": 0.2,
            "perspective": 0.0,
            "cache": False,
            "erasing": 0.0,
        }

        # Override certain args when fine tuning
        fine_tune_args = (
            {
                "epochs": min(int(epochs), 50),  # or 30
                "patience": int(patience),
                "warmup_epochs": 0,
                "pretrained": True,
            }
            if self.fine_tune
            else {}
        )

        self.model.train(**(train_args | fine_tune_args))

    # --- PART 3: INFERENCE & DOCUMENT PROCESSING ---
    def process_document(self, image_path):
        img = cv2.imread(image_path)
        # Slightly boost contrast to account for screenshot color drift
        # alpha (1.2) = contrast, beta (0) = brightness
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)

        # Slightly denoise input and use that to find text margins
        _, mask = cv2.threshold(inv, 30, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask)
        if coords is None:
            return "No text found"

        gx, gy, gw, gh = cv2.boundingRect(coords)
        text_area = inv[gy : gy + gh, gx : gx + gw]

        # Use horizontal ink projection to detect line boundaries
        line_sums = np.sum(text_area, axis=1)
        line_indices = np.where(line_sums > 255 * 4)[0]

        if len(line_indices) == 0:
            return ""

        # Group indices into contiguous blocks (the lines)
        lines = []
        if len(line_indices) > 0:
            start = line_indices[0]
            for i in range(1, len(line_indices)):
                if line_indices[i] > line_indices[i - 1] + 6:  # 6px gap threshold
                    lines.append((start, line_indices[i - 1]))
                    start = line_indices[i]
            lines.append((start, line_indices[-1]))

        # Perform predictions line-by-line
        full_text = []
        for y1, y2 in lines:
            line_img = text_area[y1:y2, :]
            # Convert back to black on white
            line_img = cv2.bitwise_not(line_img)

            h, w = line_img.shape
            # Extract and sort by X-coordinate
            raw_boxes = []

            # If the line is wider than the canvas, run sliding-window inference to avoid
            # shrinking characters (which hurts l/1 disambiguation).
            tile_overlap = 64
            tile_stride = max(1, CANVAS_W - tile_overlap)
            x_starts = [0]
            if w > CANVAS_W:
                x_starts = list(range(0, w, tile_stride))
                if x_starts and x_starts[-1] + CANVAS_W < w:
                    x_starts.append(w - CANVAS_W)

            for x0 in x_starts:
                crop = line_img[:, x0 : x0 + CANVAS_W]
                ch, cw = crop.shape

                # Center crop into 800x64 canvas (height might exceed 64 in some scans)
                canvas = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
                if ch > CANVAS_H:
                    scale = CANVAS_H / ch
                    new_h = CANVAS_H
                    new_w = max(1, int(round(cw * scale)))
                    crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    ch, cw = crop.shape

                offset_x = (CANVAS_W - cw) // 2
                offset_y = (CANVAS_H - ch) // 2
                canvas[offset_y : offset_y + ch, offset_x : offset_x + cw] = crop
                canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

                # Predict, returing even low confidence items
                results = self.model.predict(
                    canvas_bgr,
                    imgsz=1600,
                    conf=0.05,
                    verbose=False,
                    end2end=False,
                    iou=0.1,
                )

                for box in results[0].boxes:
                    # Map x coordinate back into full line coordinate system
                    x = box.xywh[0][0].item() - offset_x + x0
                    raw_boxes.append(
                        {
                            "char": IDX_TO_CHAR[int(box.cls[0].item())],
                            "conf": box.conf.item(),
                            "x": x,
                        }
                    )

            if not raw_boxes:
                continue

            raw_boxes.sort(key=lambda b: b["x"])
            # eprint(json.dumps(boxes))

            line_str = "".join([b["char"] for b in raw_boxes])
            # eprint(f"Original line: {line_str}")

            # Now try to filter out bad overlaps. We know characters never truly overlap,
            # so if two characters are located in roughly the same position, only take
            # the higher confidence one.
            filtered = [raw_boxes[0]]
            for current in raw_boxes[1:]:
                prev = filtered[-1]

                # Check if this box overlaps significantly with the previous
                if current["x"] - prev["x"] < 3.3:
                    # Replace the last one if the current is more confident
                    if current["conf"] > prev["conf"]:
                        filtered[-1] = current
                    # Otherwise, we just ignore 'current' and keep 'prev'
                else:
                    # Not overlapping sufficiently
                    filtered.append(current)

            line_str = "".join([b["char"] for b in filtered])
            full_text.append(line_str)

        return "\n".join(full_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate", action="store_true", help="Generate synthetic data"
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=5000,
        help="Number of synthetic training samples to generate",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=1000,
        help="Number of synthetic validation samples to generate",
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (use with --train)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early-stopping patience in epochs (use with --train)",
    )
    parser.add_argument("--predict", type=str, help="Path to image for inference")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/train/weights/best.pt",
        help="Path to weights",
    )
    parser.add_argument("--resume", type=str, help="Training run number to resume from")
    args = parser.parse_args()

    # Determine which model to load
    m_path = args.model if os.path.exists(args.model) else None
    if m_path is None and args.resume is not None:
        m_path = f"runs/detect/train{args.resume}/weights/best.pt"
    ocr = YOLO_OCR(m_path)

    if args.generate:
        ocr.generate_data(count=args.train_count, split="train")
        ocr.generate_data(count=args.val_count, split="val")

    if args.train:
        ocr.train(epochs=args.epochs, patience=args.patience)

    if args.predict:
        result = ocr.process_document(args.predict)
        eprint("\n--- OCR RESULTS ---\n")
        print(result)
