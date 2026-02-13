#!/usr/bin/env -S uv.exe run
from concurrent.futures import ProcessPoolExecutor
import os
import argparse
import cv2
import json
import numpy as np
import random
import sys
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from ultralytics import YOLO

# --- CONFIGURATION ---
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
CHAR_TO_IDX = {char: i for i, char in enumerate(ALPHABET)}
IDX_TO_CHAR = {i: char for char, i in CHAR_TO_IDX.items()}

FONT_PATH = "/usr/local/lib/python3.12/dist-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSansMono.ttf"  # Update path for Linux/Mac if necessary
FONT_SIZE = 16
CANVAS_W, CANVAS_H = 800, 64
DATASET_DIR = "ocr_dataset"
# MODEL_NAME = "yolo11n.pt" # Latest YOLO version
MODEL_NAME = "yolo26n.pt"  # Latest YOLO version


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _generate_ocr_sample(args):
    i, split, img_dir, lbl_dir, fine_tune, debug = args
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except Exception:
        print("Font file not found. Please check FONT_PATH.")
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
    def train(self):
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
            "epochs": 100,
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
                "epochs": 50,  # or 30
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
        _, mask = cv2.threshold(inv, 10, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask)
        if coords is None:
            return "No text found"

        gx, gy, gw, gh = cv2.boundingRect(coords)
        text_area = inv[gy : gy + gh, gx : gx + gw]

        # Use horizontal ink projection to detect line boundaries
        line_sums = np.sum(text_area, axis=1)
        line_indices = np.where(line_sums > 0)[0]

        if len(line_indices) == 0:
            return ""

        # Group indices into contiguous blocks (the lines)
        lines = []
        if len(line_indices) > 0:
            start = line_indices[0]
            for i in range(1, len(line_indices)):
                if line_indices[i] > line_indices[i - 1] + 2:  # 2px gap threshold
                    lines.append((start, line_indices[i - 1]))
                    start = line_indices[i]
            lines.append((start, line_indices[-1]))

        # Perform predictions line-by-line
        full_text = []
        for y1, y2 in lines:
            line_img = text_area[y1:y2, :]
            # Convert back to black on white
            line_img = cv2.bitwise_not(line_img)

            # Center into 800x64 canvas
            canvas = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
            h, w = line_img.shape
            # Scale down if too wide, otherwise just center
            if w > CANVAS_W:
                w = CANVAS_W
                line_img = line_img[:, :CANVAS_W]
            if h > CANVAS_H:
                h = CANVAS_H
                line_img = line_img[:CANVAS_H, :]

            offset_x = (CANVAS_W - w) // 2
            offset_y = (CANVAS_H - h) // 2
            canvas[offset_y : offset_y + h, offset_x : offset_x + w] = line_img
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

            # Debug
            # cv2.imwrite(f"debug_canvas-{y1}.png", canvas)

            # Predict, returing even low confidence items
            results = self.model.predict(
                canvas_bgr,
                imgsz=1600,
                conf=0.05,
                verbose=False,
                end2end=False,
                iou=0.1,
            )

            # # This creates a BGR image with boxes and labels drawn on it
            # annotated_frame = results[0].plot()
            #
            # # Save or display it
            # cv2.imwrite(f"detected_line.jpg", annotated_frame)

            # print(results[0].probs)
            # Extract and sort by X-coordinate
            raw_boxes = []
            for box in results[0].boxes:
                raw_boxes.append(
                    {
                        "char": IDX_TO_CHAR[int(box.cls[0].item())],
                        "conf": box.conf.item(),
                        "x": box.xywh[0][0].item(),
                    }
                )

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
    parser.add_argument("--train", action="store_true", help="Train the model")
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
        ocr.generate_data(count=5000, split="train")
        ocr.generate_data(count=1000, split="val")

    if args.train:
        ocr.train()

    if args.predict:
        result = ocr.process_document(args.predict)
        eprint("\n--- OCR RESULTS ---\n")
        print(result)
