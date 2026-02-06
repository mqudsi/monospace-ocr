#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import argparse

# --- DETERMINISM ---
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/> "
CLUSTERS = len(ALPHABET)
EXPECTED_COLS = 78
TARGET_CELL_SIZE = (32, 32)
WEIGHTS_PATH = "./weights"

def log(msg):
    print(msg, file=sys.stderr, flush=True)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Input is 1 (gray) + 2 (coords) = 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # We stop pooling here to keep an 8x8 grid for high-detail spatial features
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.2), # Generalizes better
            nn.Linear(512, num_classes)
        )

    def add_coords(self, x):
        # Generates X and Y coordinate maps from -1 to 1
        bs, _, h, w = x.size()
        xx = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(bs, 1, h, w)
        yy = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(bs, 1, h, w)
        return torch.cat([x, xx, yy], dim=1)

    def forward(self, x):
        x = self.add_coords(x)
        return self.fc(self.conv(x))

def normalize_character_soft(raw_cell, h_step):
    if raw_cell.size == 0: return np.zeros(TARGET_CELL_SIZE, dtype=np.uint8)
    clean_cell = raw_cell.copy()
    clean_cell[:, 0] = 0
    _, mask = cv2.threshold(clean_cell, 25, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    canvas = np.zeros(TARGET_CELL_SIZE, dtype=np.uint8)
    if coords is None: return canvas
    bx, by, bw, bh = cv2.boundingRect(coords)
    char_crop = clean_cell[by:by+bh, bx:bx+bw]
    scale = 28.0 / h_step
    nw, nh = max(1, int(bw * scale)), max(1, int(bh * scale))
    if nw > 30 or nh > 30:
        f = 30.0 / max(nw, nh)
        nw, nh = int(nw * f), int(nh * f)
    resized = cv2.resize(char_crop, (nw, nh), interpolation=cv2.INTER_AREA)
    rel_y_offset = by / h_step
    target_y = int(rel_y_offset * 32)
    target_x = (32 - nw) // 2
    if target_y + nh > 32: target_y = 32 - nh
    if target_y < 0: target_y = 0
    canvas[target_y:target_y+nh, target_x:target_x+nw] = resized
    return canvas

def solve_grid_2d(img, gx, gy, gw, gh, num_lines):
    def score_axis(projection, n_segments, start_guess, dim_guess):
        best_cost, best_params = float('inf'), (start_guess, dim_guess)
        for s_try in range(start_guess - 4, start_guess + 5):
            for d_try in range(dim_guess - 8, dim_guess + 9):
                step = d_try / n_segments
                gutter_ink = 0
                for i in range(n_segments + 1):
                    p = int(s_try + i * step)
                    if 0 <= p < len(projection): gutter_ink += projection[p]
                if gutter_ink < best_cost:
                    best_cost, best_params = gutter_ink, (s_try, d_try)
        return best_params
    y_proj = np.sum(img[:, gx:gx+gw], axis=1)
    y_s, h_t = score_axis(y_proj, num_lines, gy, gh)
    x_proj = np.sum(img[y_s:y_s+h_t, :], axis=0)
    x_s, w_t = score_axis(x_proj, EXPECTED_COLS, gx, gw)
    return x_s, y_s, w_t, h_t

def extract_cells(image_path, num_lines, debug=False):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    bg = np.median(img)
    inv = cv2.absdiff(img, int(bg))
    _, mask = cv2.threshold(inv, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None: return None
    gx, gy, gw, gh = cv2.boundingRect(coords)
    xs, ys, wt, ht = solve_grid_2d(inv, gx, gy, gw, gh, num_lines)
    w_step, h_step = wt / EXPECTED_COLS, ht / num_lines
    if debug:
        dbg = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
        for r in range(num_lines + 1):
            y = int(ys + r * h_step)
            cv2.line(dbg, (int(xs), y), (int(xs + wt), y), (0, 255, 0), 1)
        for c in range(EXPECTED_COLS + 1):
            x = int(xs + c * w_step)
            cv2.line(dbg, (x, int(ys)), (x, int(ys + ht)), (0, 255, 0), 1)
        plt.figure(figsize=(8, 8)); plt.imshow(dbg); plt.title("Grid Overlay"); plt.show()
    cells = []
    for r in range(num_lines):
        for c in range(EXPECTED_COLS):
            y1, y2 = int(ys + r * h_step), int(ys + (r + 1) * h_step)
            x1, x2 = int(xs + c * w_step), int(xs + (c + 1) * w_step)
            cell_raw = inv[max(0,y1):y2, max(0,x1):x2]
            cells.append(normalize_character_soft(cell_raw, h_step))
    return np.array(cells)

def parse_training_file(path, line_offset):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.read().splitlines()

    while raw_lines and not raw_lines[-1].strip():
        raw_lines.pop()

    labels = []
    char_to_idx = {c: i for i, c in enumerate(ALPHABET)}

    for r_idx, line in enumerate(raw_lines):
        if len(line) != EXPECTED_COLS:
            raise ValueError(
                f"Format Error in {path}: Line {r_idx + 1} (Grid Line {r_idx + 1 + line_offset}) "
                f"has length {len(line)}, but expected {EXPECTED_COLS}."
            )

        for c_idx, char in enumerate(line):
            if char not in char_to_idx:
                raise ValueError(
                    f"Alphabet Error in {path}: Invalid character '{char}' "
                    f"at Grid Coordinate ({r_idx + 1 + line_offset}, {c_idx + 1})."
                )
            labels.append(char_to_idx[char])

    return np.array(labels), len(raw_lines)

def calculate_bucket_averages(visuals, labels):
    averages = np.zeros((CLUSTERS, 32, 32), dtype=np.uint8)
    for i in range(CLUSTERS):
        mask = (labels == i)
        if np.any(mask):
            averages[i] = np.mean(visuals[mask], axis=0).astype(np.uint8)
    return averages

def show_outliers(visuals, labels, ref_averages, title, model_preds=None):
    fig, axes = plt.subplots(6, 11, figsize=(18, 11))
    ax_f = axes.flatten()
    for i, char in enumerate(ALPHABET):
        mask = (labels == i)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            if i < len(ax_f): ax_f[i].axis('off')
            continue

        avg_img = ref_averages[i]
        subset_cells = visuals[indices].astype(float)
        ref_float = avg_img.astype(float)
        mse = np.mean((subset_cells - ref_float)**2, axis=(1, 2))
        worst_idx_in_group = np.argmax(mse)
        global_idx = indices[worst_idx_in_group]
        outlier_img = visuals[global_idx]

        line, col = (global_idx // EXPECTED_COLS) + 1, (global_idx % EXPECTED_COLS) + 1
        meta = ""
        if model_preds is not None:
            pred_char = ALPHABET[model_preds[global_idx]]
            meta = f" P='{pred_char}'"

        combined = np.zeros((32, 65), dtype=np.uint8)
        combined[:, :32] = avg_img
        combined[:, 33:65] = outlier_img
        ax_f[i].imshow(combined, cmap='magma')
        ax_f[i].set_title(f"'{char}' ({line},{col}){meta}", fontsize=7)
        ax_f[i].axis('off')
    plt.suptitle(title, fontsize=16); plt.tight_layout(); plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("train_path", nargs='?', default=None, help="Top N lines of ground truth")
    parser.add_argument("bottom_train_path", nargs='?', default=None, help="Optional: Bottom N lines of ground truth")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("-o", "--output", help="Path to write output instead of stdout")
    parser.add_argument("--lines", type=int, default=65, help="Total grid lines in image")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(CLUSTERS).to(device)

    visuals = extract_cells(args.image, args.lines, debug=args.debug)
    if visuals is None: return

    gt_labels_list = []
    gt_visuals_list = []

    if args.train_path:
        labels_top, n_lines_top = parse_training_file(args.train_path, 0)
        gt_labels_list.append(labels_top)
        gt_visuals_list.append(visuals[:n_lines_top * EXPECTED_COLS])
        log(f"Loaded {n_lines_top} lines from top training file.")

        if args.bottom_train_path:
            with open(args.bottom_train_path, "r", encoding="utf-8") as f:
                n_lines_bot = len([l for l in f.read().splitlines() if l.strip()])

            offset = args.lines - n_lines_bot
            if offset < n_lines_top:
                log("Warning: Top and Bottom training sets overlap.")

            labels_bot, _ = parse_training_file(args.bottom_train_path, offset)
            gt_labels_list.append(labels_bot)
            gt_visuals_list.append(visuals[offset * EXPECTED_COLS : (offset + n_lines_bot) * EXPECTED_COLS])
            log(f"Loaded {n_lines_bot} lines from bottom training file (Offset: {offset}).")

        all_gt_labels = np.concatenate(gt_labels_list)
        all_gt_visuals = np.concatenate(gt_visuals_list)

        ground_truth_averages = calculate_bucket_averages(all_gt_visuals, all_gt_labels)
        if args.debug:
            show_outliers(all_gt_visuals, all_gt_labels, ground_truth_averages,
                          "TRAINING DATA: Avg vs Most Deviant")

        X = torch.tensor(all_gt_visuals, dtype=torch.float32).unsqueeze(1) / 255.0
        Y = torch.tensor(all_gt_labels, dtype=torch.long)
        train_idx, val_idx = train_test_split(np.arange(len(Y)), test_size=0.1, random_state=42)
        train_loader = DataLoader(TensorDataset(X[train_idx], Y[train_idx]), batch_size=64, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        for epoch in range(30):
            model.train()
            l_sum = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(); loss = criterion(model(xb), yb)
                loss.backward(); optimizer.step(); l_sum += loss.item()
            log(f"Epoch {epoch+1:02d} | Loss: {l_sum/len(train_loader):.4f}")

        torch.save(model.state_dict(), WEIGHTS_PATH)
    else:
        if not os.path.exists(WEIGHTS_PATH):
            log("Error: Weights not found. Provide a training file to generate them."); return
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))

    # --- INFERENCE ---
    model.eval()
    X_all = torch.tensor(visuals, dtype=torch.float32).unsqueeze(1).to(device) / 255.0
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_all), 128):
            all_preds.append(model(X_all[i:i+128]).argmax(1).cpu().numpy())
    pred_indices = np.concatenate(all_preds)

    if not args.quiet:
        # Redirect output if -o is provided
        out_f = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
        try:
            for r in range(args.lines):
                row = pred_indices[r*EXPECTED_COLS : (r+1)*EXPECTED_COLS]
                out_f.write("".join([ALPHABET[i] for i in row]) + "\n")
        finally:
            if args.output:
                out_f.close()

    inf_averages = calculate_bucket_averages(visuals, pred_indices)
    if args.debug:
        show_outliers(visuals, pred_indices, inf_averages, "Inference Bucket Deviations")

if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, OSError) as e:
        log(str(e))
        sys.exit(1)
