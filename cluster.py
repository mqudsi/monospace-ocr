#!/usr/bin/env -S uv.exe run

import cv2
import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as tf
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import argparse
import json


# --- DETERMINISM ---
def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=> -":;.'
CLUSTERS = len(ALPHABET)
EXPECTED_COLS = 78
TARGET_CELL_SIZE = (32, 32)
WEIGHTS_PATH = "./weights_file"
CONFIG_PATH = "./grid_config.json"
CHECKPOINT_PATH = "./checkpoint.pt"


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def save_checkpoint(model, optimizer, scheduler, epoch, loss):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, CHECKPOINT_PATH)
    log(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, scheduler):
    """Load training checkpoint if exists. Returns starting epoch."""
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        log(f"Resumed from checkpoint epoch {start_epoch}")
        return start_epoch
    return 0


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Input is 1 (gray) + 2 (coords) = 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            # We stop pooling here to keep an 8x8 grid for high-detail spatial features
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  # Generalizes better
            nn.Linear(512, num_classes),
        )

    def add_coords(self, x):
        # Generates X and Y coordinate maps from -1 to 1
        bs, _, h, w = x.size()
        xx = (
            torch.linspace(-1, 1, w, device=x.device)
            .view(1, 1, 1, w)
            .expand(bs, 1, h, w)
        )
        yy = (
            torch.linspace(-1, 1, h, device=x.device)
            .view(1, 1, h, 1)
            .expand(bs, 1, h, w)
        )
        return torch.cat([x, xx, yy], dim=1)

    def forward(self, x):
        x = self.add_coords(x)
        return self.fc(self.conv(x))


# Here the pixels have been inverted (white text on black background)
# This means that thresholding is already inverted! Lower values are
# WHITE not BLACK.
def normalize_character_soft(raw_cell, h_step):
    if raw_cell.size == 0:
        return np.zeros(TARGET_CELL_SIZE, dtype=np.uint8)

    # Wipe the left gutter to prevent gutter-line contamination.
    # Note: we can't do that to the right without clipping characters!
    base_cell = raw_cell.copy()
    base_cell[:, 0] = 0

    # Use a high threshold to find only the most reliable, solid ink.
    # This prevents faint speckles from shifting the character's baseline,
    # but we won't pass this heavily denoised input to the CNN.
    # The higher the second parameter the heavier the denoising.
    _, hard_mask = cv2.threshold(base_cell, 110, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(hard_mask)

    canvas = np.zeros(TARGET_CELL_SIZE, dtype=np.uint8)
    if coords is None:
        return canvas

    # Get the bounding box of the hard-denoised cell
    bx, by, bw, bh = cv2.boundingRect(coords)

    # Crop the original cell using the coordinates from the hard-denoised one.
    # Use a small 2px safety padding to ensure subpixel hints aren't clipped.
    pad = 2
    y1, y2 = max(0, by - pad), min(base_cell.shape[0], by + bh + pad)
    x1, x2 = max(0, bx - pad), min(base_cell.shape[1], bx + bw + pad)

    char_crop = base_cell[y1:y2, x1:x2]
    curr_h, curr_w = char_crop.shape

    # Scale based on the expected row height (h_step) to keep character sizes consistent
    scale = 28.0 / h_step
    nw, nh = max(1, int(curr_w * scale)), max(1, int(curr_h * scale))

    # Force fit into 32x32 if necessary
    if nw > 30 or nh > 30:
        f = 30.0 / max(nw, nh)
        nw, nh = int(nw * f), int(nh * f)

    # Resize the (cropped) character to fit our training cells
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(char_crop, (nw, nh), interpolation=interp)

    # Improve CNN performance by bottom-aligning cells.
    # Every character is anchored to a fixed baseline (2px from the bottom).
    target_y = 32 - nh - 2
    target_x = (32 - nw) // 2

    if target_y < 0:
        target_y = 0
    if target_y + nh > 32:
        nh = 32 - target_y

    # Place the cropped character onto the black canvas
    canvas[target_y : target_y + nh, target_x : target_x + nw] = resized[:nh, :nw]

    return canvas


def solve_grid_2d(img, gx, gy, gw, gh, num_lines):
    def score_axis(projection, n_segments, start_guess, dim_guess):
        best_cost = float("inf")
        best_params = (start_guess, dim_guess)

        for s_try in range(start_guess - 4, start_guess + 5):
            for d_try in range(dim_guess - 8, dim_guess + 9):
                step = d_try / n_segments
                gutter_ink = 0
                # Vectorized sum for speed and cleaner logic equivalent to original
                # Grid lines at: s, s+step, s+2*step...
                positions = s_try + np.arange(n_segments + 1) * step
                indices = np.round(positions).astype(int)

                # Boundary checks
                valid = (indices >= 0) & (indices < len(projection))
                gutter_ink = np.sum(projection[indices[valid]])

                if gutter_ink < best_cost:
                    best_cost, best_params = gutter_ink, (s_try, d_try)

        # Linear (mostly vertical) drift cancellation
        coarse_s, coarse_d = best_params
        refined_cost = best_cost
        refined_params = (float(coarse_s), float(coarse_d))

        # Scan +/- 2.0 pixels in total width (adjusts step size by fractions of a pixel)
        # Resolution 0.05px total width -> ~0.0006px per cell
        dim_range = np.linspace(coarse_d - 2.0, coarse_d + 2.0, 41)

        # Scan +/- 1.0 pixel in start position (recenters the grid)
        start_range = np.linspace(coarse_s - 1.0, coarse_s + 1.0, 21)

        for d_val in dim_range:
            step = d_val / n_segments
            rel_pos = np.arange(n_segments + 1) * step

            for s_val in start_range:
                positions = s_val + rel_pos
                indices = np.round(positions).astype(int)

                valid = (indices >= 0) & (indices < len(projection))
                cost = np.sum(projection[indices[valid]])

                if cost < refined_cost:
                    refined_cost = cost
                    refined_params = (s_val, d_val)

        return refined_params

    # Y-Axis (Rows)
    y_proj = np.sum(img[:, gx : gx + gw], axis=1)
    y_s, h_t = score_axis(y_proj, num_lines, gy, gh)

    # X-Axis (Cols)
    # We use the detected Y-range to filter the X-projection.
    y_start_clean = max(0, int(y_s))
    y_end_clean = min(img.shape[0], int(y_s + h_t))
    x_proj = np.sum(img[y_start_clean:y_end_clean, :], axis=0)
    x_s, w_t = score_axis(x_proj, EXPECTED_COLS, gx, gw)

    return x_s, y_s, w_t, h_t


def extract_cells(image_path, num_lines, grid_params=None, debug=False):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    # Upscale 2x (nearest neighbor)
    img = cv2.resize(img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    bg = np.median(img)
    inv = cv2.absdiff(img, int(bg))

    if grid_params:
        xs, ys, wt, ht = grid_params
        log(f"Using cached grid: x={xs:.1f}, y={ys:.1f}, w={wt:.1f}, h={ht:.1f}")
    else:
        _, mask = cv2.threshold(inv, 10, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask)
        if coords is None:
            return None, None
        gx, gy, gw, gh = cv2.boundingRect(coords)
        xs, ys, wt, ht = solve_grid_2d(inv, gx, gy, gw, gh, num_lines)
        log(f"Detected grid: x={xs}, y={ys}, w={wt}, h={ht}")

    w_step, h_step = wt / EXPECTED_COLS, ht / num_lines

    if debug:
        # Pad image slightly so labels are always visible
        pad = 40
        dbg_base = cv2.copyMakeBorder(
            inv, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
        )
        dbg = cv2.cvtColor(dbg_base, cv2.COLOR_GRAY2BGR)

        # Adjust grid coordinates for padding
        pxs, pys = xs + pad, ys + pad
        font = cv2.FONT_HERSHEY_SIMPLEX
        f_scale = 0.35
        color_line = (0, 255, 0)
        color_txt = (0, 255, 255)

        # 1-indexed Row Labels (Left and Right)
        for r in range(num_lines + 1):
            y = int(pys + r * h_step)
            cv2.line(dbg, (int(pxs), y), (int(pxs + wt), y), color_line, 1)
            if r < num_lines:
                y_mid = int(pys + (r + 0.6) * h_step)
                # Left
                cv2.putText(
                    dbg, str(r + 1), (int(pxs) - 30, y_mid), font, f_scale, color_txt, 1
                )
                # Right
                cv2.putText(
                    dbg,
                    str(r + 1),
                    (int(pxs + wt) + 5, y_mid),
                    font,
                    f_scale,
                    color_txt,
                    1,
                )

        # 1-indexed Column Labels (Top and Bottom)
        for c in range(EXPECTED_COLS + 1):
            x = int(pxs + c * w_step)
            cv2.line(dbg, (x, int(pys)), (x, int(pys + ht)), color_line, 1)
            if c < EXPECTED_COLS and (
                (c + 1) % 5 == 0 or c == 0 or c == EXPECTED_COLS - 1
            ):
                x_mid = int(pxs + c * w_step + 2)
                # Top
                cv2.putText(
                    dbg, str(c + 1), (x_mid, int(pys) - 10), font, f_scale, color_txt, 1
                )
                # Bottom
                cv2.putText(
                    dbg,
                    str(c + 1),
                    (x_mid, int(pys + ht) + 20),
                    font,
                    f_scale,
                    color_txt,
                    1,
                )

        fig = plt.figure(figsize=(14, 10))
        # This allows the image to expand when the window is maximized
        ax = fig.add_subplot(111)
        ax.imshow(dbg)
        ax.set_title("Grid Debug View (1,1-indexed Labels on All Sides)")
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    cells = []
    for r in range(num_lines):
        for c in range(EXPECTED_COLS):
            # Fix clipping at extreme right/left by expanding margins.
            y1, y2 = int(ys + r * h_step), int(ys + (r + 1) * h_step)
            x1, x2 = int(xs + c * w_step - 0.5), int(xs + (c + 1) * w_step + 0.5)
            cell_raw = inv[max(0, y1) : y2, max(0, x1) : x2]
            cells.append(normalize_character_soft(cell_raw, h_step))

    return np.array(cells), (xs, ys, wt, ht)


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
        grid_row = r_idx + 1 + line_offset
        line = line.ljust(EXPECTED_COLS)
        if len(line) != EXPECTED_COLS:
            raise ValueError(
                f"Format Error in {path}: Row {grid_row} length is {len(line)}, expected {EXPECTED_COLS}."
            )
        for c_idx, char in enumerate(line):
            # Use `#` (in training file) and `-` (internally) as placeholders for
            # characters we're not sure about and want to skip training against.
            if char == "#":
                labels.append(-1)
                continue
            if char not in char_to_idx:
                raise ValueError(
                    f"Alphabet Error: '{char}' at ({grid_row}, {c_idx + 1}) not in ALPHABET."
                )
            labels.append(char_to_idx[char])
    return np.array(labels), len(raw_lines)


def calculate_bucket_averages(visuals, labels):
    averages = np.zeros((CLUSTERS, 32, 32), dtype=np.uint8)
    for i in range(CLUSTERS):
        mask = labels == i
        if np.any(mask):
            averages[i] = np.mean(visuals[mask], axis=0).astype(np.uint8)
    return averages


def show_outliers(
    visuals,
    labels,
    ref_averages,
    title,
    model_preds=None,
    save_path=None,
    indices_map=None,
):
    n_cols = 11
    n_rows = (CLUSTERS + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 1.83))
    ax_f = axes.flatten()
    # Pre-disable all subplots to handle empty slots in the grid
    for a in ax_f:
        a.axis("off")

    overall_max_mse = 0.0
    for i, char in enumerate(ALPHABET):
        mask = labels == i
        indices = np.where(mask)[0]
        if len(indices) == 0:
            if i < len(ax_f):
                ax_f[i].axis("off")
            continue

        avg_img = ref_averages[i]
        subset_cells = visuals[indices].astype(float)
        ref_float = avg_img.astype(float)
        mse = np.mean((subset_cells - ref_float) ** 2, axis=(1, 2))
        worst_idx_in_group = np.argmax(mse)
        max_mse_val = mse[worst_idx_in_group]
        overall_max_mse = max(overall_max_mse, max_mse_val)
        global_idx = indices[worst_idx_in_group]
        outlier_img = visuals[global_idx]

        actual_idx = indices_map[global_idx] if indices_map is not None else global_idx
        line, col = (actual_idx // EXPECTED_COLS) + 1, (actual_idx % EXPECTED_COLS) + 1
        meta = ""
        if model_preds is not None:
            pred_char = ALPHABET[model_preds[global_idx]]
            meta = f" P='{pred_char}'"

        combined = np.zeros((32, 65), dtype=np.uint8)
        combined[:, :32] = avg_img
        combined[:, 33:65] = outlier_img
        ax_f[i].imshow(combined, cmap="magma")
        ax_f[i].set_title(f"'{char}' n={len(indices)} ({line},{col}){meta}", fontsize=7)
        ax_f[i].axis("off")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
    return overall_max_mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "train_path", nargs="?", default=None, help="Top N lines of ground truth"
    )
    parser.add_argument(
        "bottom_train_path",
        nargs="?",
        default=None,
        help="Optional: Bottom N lines of ground truth",
    )
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("-o", "--output", help="Path to write output instead of stdout")
    parser.add_argument(
        "--lines", type=int, default=65, help="Total grid lines in image"
    )
    args = parser.parse_args()

    log(f"Input: {args.image}")
    if args.output:
        log(f"Output: {args.output}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(CLUSTERS).to(device)

    # Determine if we should load a memorized grid
    grid_params = None
    if not args.train_path and os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
                grid_params = (cfg["xs"], cfg["ys"], cfg["wt"], cfg["ht"])
                log("Loaded grid configuration from file.")
        except Exception as e:
            log(f"Warning: Could not load config: {e}. Re-detecting grid.")

    # Suppress grid debug view if in inference mode with an output path
    grid_debug = args.debug and not (not args.train_path and args.output)
    visuals, detected_params = extract_cells(
        args.image, args.lines, grid_params=grid_params, debug=grid_debug
    )
    if visuals is None:
        return

    if args.train_path:
        # --- TRAINING MODE ---
        gt_labels_list = []
        gt_visuals_list = []
        gt_indices_list = []

        labels_top, n_lines_top = parse_training_file(args.train_path, 0)
        gt_labels_list.append(labels_top)
        gt_visuals_list.append(visuals[: n_lines_top * EXPECTED_COLS])
        gt_indices_list.append(np.arange(0, n_lines_top * EXPECTED_COLS))
        log(f"Loaded {n_lines_top} lines from top training file.")

        if args.bottom_train_path:
            with open(args.bottom_train_path, "r", encoding="utf-8") as f:
                n_lines_bot = len([l for l in f.read().splitlines() if l.strip()])

            offset = args.lines - n_lines_bot
            labels_bot, _ = parse_training_file(args.bottom_train_path, offset)
            gt_labels_list.append(labels_bot)
            gt_visuals_list.append(
                visuals[offset * EXPECTED_COLS : (offset + n_lines_bot) * EXPECTED_COLS]
            )
            gt_indices_list.append(
                np.arange(
                    offset * EXPECTED_COLS, (offset + n_lines_bot) * EXPECTED_COLS
                )
            )
            log(f"Loaded {n_lines_bot} lines from bottom training file.")

        all_gt_labels = np.concatenate(gt_labels_list)
        all_gt_visuals = np.concatenate(gt_visuals_list)
        all_gt_indices = np.concatenate(gt_indices_list)

        # Filter out cells marked for skipping via #/-1
        valid_mask = all_gt_labels != -1
        all_gt_labels = all_gt_labels[valid_mask]
        all_gt_visuals = all_gt_visuals[valid_mask]
        all_gt_indices = all_gt_indices[valid_mask]

        ground_truth_averages = calculate_bucket_averages(all_gt_visuals, all_gt_labels)
        if args.debug:
            max_dev = show_outliers(
                all_gt_visuals,
                all_gt_labels,
                ground_truth_averages,
                "TRAINING TYPO CHECK: Avg vs Max Outlier",
                indices_map=all_gt_indices,
            )
            log(f"Max bucket deviation: {max_dev:.2f}")

        X = torch.tensor(all_gt_visuals, dtype=torch.float32).unsqueeze(1) / 255.0
        Y = torch.tensor(all_gt_labels, dtype=torch.long)
        train_idx, _ = train_test_split(
            np.arange(len(Y)), test_size=0.1, random_state=42
        )
        train_loader = DataLoader(
            TensorDataset(X[train_idx], Y[train_idx]), batch_size=64, shuffle=True
        )

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)
        criterion = nn.CrossEntropyLoss()
        
        # Load checkpoint if exists
        start_epoch = load_checkpoint(model, optimizer, scheduler)
        
        for epoch in range(start_epoch, 53):
            # Save checkpoint at start of each epoch (except first if just loaded)
            if epoch > start_epoch:
                save_checkpoint(model, optimizer, scheduler, epoch, l_sum)
            
            model.train()
            l_sum = 0
            # Loop twice per epoch: once for clean ground truth, once for augmented shifts
            for augmented_pass in [False, True]:
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    if augmented_pass:
                        shift_x = random.uniform(-2.0, 2.0)
                        shift_y = random.uniform(-2.0, 2.0)
                        N, C, H, W = xb.size()
                        theta = torch.tensor(
                            [[[1, 0, -2 * shift_x / W], [0, 1, -2 * shift_y / H]]],
                            device=xb.device,
                        ).repeat(N, 1, 1)

                        grid = tf.affine_grid(theta, xb.size(), align_corners=False)
                        xb = tf.grid_sample(xb, grid, align_corners=False)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    l_sum += loss.item()
            scheduler.step()
            curr_lr = optimizer.param_groups[0]["lr"]
            log(
                f"Epoch {epoch + 1:02d} | Loss: {l_sum / (2 * len(train_loader)):.4f} | LR: {curr_lr:.5f}"
            )

        # Save Weights
        torch.save(model.state_dict(), WEIGHTS_PATH)
        # Save Grid Config (Memorization)
        with open(CONFIG_PATH, "w") as f:
            json.dump(
                {
                    "xs": float(detected_params[0]),
                    "ys": float(detected_params[1]),
                    "wt": float(detected_params[2]),
                    "ht": float(detected_params[3]),
                },
                f,
            )
        log(f"Memorized grid and saved weights.")
    else:
        # --- INFERENCE MODE ---
        if not os.path.exists(WEIGHTS_PATH):
            log("Error: Weights not found. Provide a training file to generate them.")
            return
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))

    model.eval()
    X_all = torch.tensor(visuals, dtype=torch.float32).unsqueeze(1).to(device) / 255.0
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_all), 128):
            all_preds.append(model(X_all[i : i + 128]).argmax(1).cpu().numpy())
    pred_indices = np.concatenate(all_preds)

    if not args.quiet:
        out_f = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
        try:
            for r in range(args.lines):
                row = pred_indices[r * EXPECTED_COLS : (r + 1) * EXPECTED_COLS]
                out_f.write("".join([ALPHABET[i] for i in row]) + "\n")
        finally:
            if args.output:
                out_f.close()

    if args.debug:
        inf_averages = calculate_bucket_averages(visuals, pred_indices)
        pf_path = (
            (os.path.splitext(args.output)[0] + "-proof.png")
            if (not args.train_path and args.output)
            else None
        )
        max_dev = show_outliers(
            visuals,
            pred_indices,
            inf_averages,
            "Inference Results: Max Outliers",
            save_path=pf_path,
        )
        log(f"Max bucket deviation: {max_dev:.2f}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, OSError) as e:
        log(str(e))
        sys.exit(1)
