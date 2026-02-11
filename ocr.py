import os
import argparse
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from ultralytics import YOLO

# --- CONFIGURATION ---
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
CHAR_TO_IDX = {char: i for i, char in enumerate(ALPHABET)}
IDX_TO_CHAR = {i: char for char, i in CHAR_TO_IDX.items()}

FONT_PATH = "times.ttf"  # Update path for Linux/Mac if necessary
FONT_SIZE = 16
CANVAS_W, CANVAS_H = 800, 64
DATASET_DIR = "ocr_dataset"
# MODEL_NAME = "yolo11n.pt" # Latest YOLO version
MODEL_NAME = "yolo26n.pt" # Latest YOLO version

class YOLO_OCR:
    def __init__(self, model_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model_path:
            if not os.path.exists(model_path):
                raise Exception(f"Model {model_path} not found")
            self.model = YOLO(model_path)
            self.fine_tune = True
        else:
            self.model = YOLO(MODEL_NAME)
            self.fine_tune = False

    # --- PART 1: DATA GENERATION ---
    def generate_data(self, count=1000, split='train'):
        print(f"Generating {count} samples for {split}...")
        img_dir = os.path.join(DATASET_DIR, split, "images")
        lbl_dir = os.path.join(DATASET_DIR, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        try:
            font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        except:
            print("Font file not found. Please check FONT_PATH.")
            return

        for i in range(count):
            # Create base canvas
            img = Image.new('RGB', (CANVAS_W, CANVAS_H), (255, 255, 255))
            draw = ImageDraw.Draw(img)

            # Generate random text, oversampling tricky characters
            hard_chars = "ijlI1t/f"
            # text_len = random.randint(10, 80)
            # random.triangular(low, high, mode)
            text_len = int(random.triangular(10, 85, 72))

            # 40% of the time, fill at least 30% of the slots with confusable characters
            if True and random.random() < 0.65:
                num_hard = int(text_len * 0.3)
                num_normal = text_len - num_hard

                # Create a mixed pool and shuffle it
                pool = random.choices(hard_chars, k=num_hard) + random.choices(ALPHABET, k=num_normal)
                random.shuffle(pool)
                text = "".join(pool)
            else:
                # Standard random distribution
                text = "".join(random.choices(ALPHABET, k=text_len))

            # Calculate text dimensions for centering
            total_w = font.getlength(text)
            curr_x = (CANVAS_W - total_w) // 2
            curr_y = (CANVAS_H - FONT_SIZE) // 2

            labels = []
            ascent, descent = font.getmetrics()
            font_height = ascent + descent

            for j, char in enumerate(text):
                char_w = font.getlength(char)
                # Use minimum bbox width for thin chars like 1l/i
                # They seem to default to ~4px
                char_w = max(char_w, 6)

                # Bounding Box (Normalized)
                x_center = (curr_x + char_w/2) / CANVAS_W
                y_center = (curr_y + font_height/2) / CANVAS_H
                nw = char_w / CANVAS_W
                nh = font_height / CANVAS_H
                labels.append(f"{CHAR_TO_IDX[char]} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}")

                draw.text((curr_x, curr_y), char, font=font, fill=(0,0,0))
                curr_x += char_w

            # Add some synthetic noise and blur to help with jpeg detection
            # if self.fine_tune:
            #     radius = random.uniform(0.0, 0.2)
            #     img = img.filter(ImageFilter.GaussianBlur(radius))

            # Save (at slightly lower resolution when fine-tuning)
            name = f"{split}_{i:05d}"
            img_path = os.path.join(img_dir, f"{name}.jpg")
            if not self.fine_tune:
                # default quality is 75!?
                img.save(img_path, "JPEG", quality=95)
            else:
                img.save(img_path, "JPEG", quality=random.randint(85, 95))

            with open(os.path.join(lbl_dir, f"{name}.txt"), "w") as f:
                f.write("\n".join(labels))

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
        fine_tune_args = {
            "epochs": 50, # or 30
            "warmup_epochs": 0,
            "pretrained": True,
        } if self.fine_tune else {}

        self.model.train(**(train_args | fine_tune_args))

    # --- PART 3: INFERENCE & DOCUMENT PROCESSING ---
    def process_document(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)

        # 1. Use provided Denoise/Bounding Box logic
        _, mask = cv2.threshold(inv, 10, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask)
        if coords is None:
            return "No text found"

        gx, gy, gw, gh = cv2.boundingRect(coords)
        text_area = inv[gy:gy+gh, gx:gx+gw]

        # 2. Line Splitting (Horizontal Projection)
        line_sums = np.sum(text_area, axis=1)
        line_indices = np.where(line_sums > 0)[0]

        if len(line_indices) == 0: return ""

        # Group indices into contiguous blocks (lines)
        lines = []
        if len(line_indices) > 0:
            start = line_indices[0]
            for i in range(1, len(line_indices)):
                if line_indices[i] > line_indices[i-1] + 2: # 2px gap threshold
                    lines.append((start, line_indices[i-1]))
                    start = line_indices[i]
            lines.append((start, line_indices[-1]))

        # 3. Process each line
        full_text = []
        for (y1, y2) in lines:
            line_img = text_area[y1:y2, :]
            # Convert back to black on white
            line_img = cv2.bitwise_not(line_img)

            # Center into 800x64 canvas
            canvas = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
            h, w = line_img.shape
            # Scale down if too wide, otherwise just center
            if w > CANVAS_W: w = CANVAS_W; line_img = line_img[:, :CANVAS_W]
            if h > CANVAS_H: h = CANVAS_H; line_img = line_img[:CANVAS_H, :]

            offset_x = (CANVAS_W - w) // 2
            offset_y = (CANVAS_H - h) // 2
            canvas[offset_y:offset_y+h, offset_x:offset_x+w] = line_img
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

            # Debug
            cv2.imwrite(f"debug_canvas-{y1}.png", canvas)

            # Predict
            results = self.model.predict(canvas_bgr, imgsz=1600, conf=0.10, verbose=False)

            # This creates a BGR image with boxes and labels drawn on it
            annotated_frame = results[0].plot()

            # Save or display it
            cv2.imwrite(f"detected_line.jpg", annotated_frame)

            # Extract and sort by X-coordinate
            boxes = []
            for box in results[0].boxes:
                boxes.append({
                    "x": box.xywh[0][0].item(),
                    "char": IDX_TO_CHAR[int(box.cls[0].item())]
                })

            boxes.sort(key=lambda b: b["x"])
            line_str = "".join([b["char"] for b in boxes])
            full_text.append(line_str)

        return "\n".join(full_text)

if __name__ == "__main__":
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Generate synthetic data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", type=str, help="Path to image for inference")
    parser.add_argument("--model", type=str, default="runs/detect/train/weights/best.pt", help="Path to weights")
    parser.add_argument("--resume", type=str, help="Training run number to resume from")
    args = parser.parse_args()

    # Determine which model to load
    m_path = args.model if os.path.exists(args.model) else None
    if m_path is None and args.resume is not None:
        m_path = f"runs/detect/train{args.resume}/weights/best.pt"
    ocr = YOLO_OCR(m_path)

    if args.generate:
        ocr.generate_data(count=5000, split='train')
        ocr.generate_data(count=1000, split='val')

    if args.train:
        ocr.train()

    if args.predict:
        result = ocr.process_document(args.predict)
        print("\n--- OCR RESULTS ---\n")
        print(result)
