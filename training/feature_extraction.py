"""
training/feature_extraction.py
================================
Extracts ResNet-18 visual features from real SumMe / TVSum videos
downloaded via training/download_dataset.py (kagglehub).

Usage (from project root):
    python -m training.feature_extraction

What it does:
  1. Locates the Kaggle dataset using kagglehub.
  2. Finds all .mp4 / .avi / .mkv / .mov video files.
  3. For each video, samples frames at FRAME_SAMPLE_FPS (default 2 fps).
  4. Extracts a 512-dim ResNet-18 feature vector per frame.
  5. Saves per-video features to datasets/video_features/<video_id>.npy
     (skip already-processed videos — safe to re-run).
  6. Loads SumMe / TVSum ground-truth annotations.
  7. Aligns per-frame features with per-frame importance scores.
  8. Saves the aligned dataset to datasets/processed_dataset.json.

SumMe annotations  : GT/summe_gt.h5  (keys: video_name -> gt_score  [N_frames])
TVSum  annotations : annotations/ydata-tvsum50-anno.tsv
                     (video_name, category, importance_scores CSV per row)
"""

import os
import sys
import glob
import json
import re
import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("OpenCV not found. Run: pip install opencv-python")

try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
except ImportError:
    sys.exit("PyTorch / torchvision / Pillow not found.")

try:
    import kagglehub
except ImportError:
    sys.exit("kagglehub not found. Run: pip install kagglehub")

import training.config as cfg


# ---------------------------------------------------------------------------
# CNN Feature Extractor (ResNet-18, no FC layer)
# ---------------------------------------------------------------------------
class ResNet18Extractor:
    def __init__(self, device="cpu"):
        self.device = device
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = torch.nn.Identity()
        resnet.to(device)
        resnet.eval()
        self.model = resnet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, pil_image):
        img = pil_image.convert("RGB")
        t   = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(t)
        return feat.squeeze(0).cpu().numpy()   # (512,)


# ---------------------------------------------------------------------------
# Frame sampler
# ---------------------------------------------------------------------------
def sample_frames(video_path, sample_fps=2):
    """
    Yields (timestamp_sec, PIL.Image) pairs sampled at `sample_fps` fps.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {video_path}")
        return

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(1, int(round(native_fps / sample_fps)))
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            timestamp = frame_idx / native_fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield timestamp, Image.fromarray(rgb)
        frame_idx += 1

    cap.release()


# ---------------------------------------------------------------------------
# Per-video feature extraction (with caching)
# ---------------------------------------------------------------------------
def extract_video_features(video_path, extractor, cache_dir, sample_fps=2):
    video_id  = os.path.splitext(os.path.basename(video_path))[0]
    cache_npy = os.path.join(cache_dir, f"{video_id}.npy")
    ts_cache  = os.path.join(cache_dir, f"{video_id}_timestamps.npy")

    if os.path.exists(cache_npy) and os.path.exists(ts_cache):
        feats  = np.load(cache_npy)
        tstamps = np.load(ts_cache)
        return feats, tstamps, video_id

    feats_list, ts_list = [], []
    for ts, pil in sample_frames(video_path, sample_fps):
        feat = extractor.extract(pil)
        feats_list.append(feat)
        ts_list.append(ts)

    if not feats_list:
        return None, None, video_id

    feats   = np.stack(feats_list)    # (N, 512)
    tstamps = np.array(ts_list)

    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_npy, feats)
    np.save(ts_cache, tstamps)
    print(f"  Cached {len(feats)} frames -> {cache_npy}")
    return feats, tstamps, video_id


# ---------------------------------------------------------------------------
# SumMe annotation loader
# ---------------------------------------------------------------------------
def load_summe_annotations(dataset_root):
    """
    Returns dict: {video_name: np.array of importance scores [0,1], N_frames}
    SumMe stores ground-truth in GT/summe_gt.h5 (HDF5) or per-video .mat files.
    """
    annotations = {}

    # Try HDF5 first
    h5_path = os.path.join(dataset_root, "GT", "summe_gt.h5")
    if not os.path.exists(h5_path):
        # Search recursively
        found = glob.glob(os.path.join(dataset_root, "**", "summe_gt.h5"), recursive=True)
        h5_path = found[0] if found else None

    if h5_path and os.path.exists(h5_path):
        try:
            import h5py
            with h5py.File(h5_path, "r") as hf:
                for key in hf.keys():
                    scores = hf[key]["gt_score"][:]   # (N_frames,)
                    # Normalize to [0,1]
                    mn, mx = scores.min(), scores.max()
                    if mx > mn:
                        scores = (scores - mn) / (mx - mn)
                    annotations[key] = scores.astype(np.float32)
            print(f"[SumMe] Loaded {len(annotations)} videos from HDF5.")
            return annotations
        except Exception as e:
            print(f"[SumMe] HDF5 load failed ({e}), trying .mat files...")

    # Fallback: per-video .mat files in GT/
    import glob as _glob
    mat_files = _glob.glob(os.path.join(dataset_root, "**", "*.mat"), recursive=True)
    for mat_path in mat_files:
        try:
            import scipy.io as sio
            data = sio.loadmat(mat_path)
            # SumMe .mat contains 'gt_score' key
            if "gt_score" in data:
                scores = data["gt_score"].flatten().astype(np.float32)
                mn, mx = scores.min(), scores.max()
                if mx > mn:
                    scores = (scores - mn) / (mx - mn)
                vid_name = os.path.splitext(os.path.basename(mat_path))[0]
                annotations[vid_name] = scores
        except Exception:
            continue

    print(f"[SumMe] Loaded {len(annotations)} videos from .mat files.")
    return annotations


# ---------------------------------------------------------------------------
# TVSum annotation loader
# ---------------------------------------------------------------------------
def load_tvsum_annotations(dataset_root):
    """
    Returns dict: {video_name: np.array of mean importance scores [0,1]}
    TVSum stores annotations in ydata-tvsum50-anno.tsv:
      video_name, category, scores (comma-separated, one row per annotator).
    """
    annotations = {}

    tsv_path = None
    for root, _, files in os.walk(dataset_root):
        for fn in files:
            if "anno" in fn.lower() and fn.endswith(".tsv"):
                tsv_path = os.path.join(root, fn)
                break
        if tsv_path:
            break

    if not tsv_path:
        print("[TVSum] WARNING: annotation tsv not found.")
        return annotations

    raw = {}   # video_name -> list of score arrays
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            vid_id   = parts[0].strip()
            score_str = parts[2].strip() if len(parts) > 2 else parts[-1].strip()
            try:
                scores = np.array([float(x) for x in score_str.split(",")], dtype=np.float32)
                raw.setdefault(vid_id, []).append(scores)
            except ValueError:
                continue

    for vid_id, annotator_scores in raw.items():
        # Average across annotators, then normalize to [0,1]
        mean_scores = np.mean(annotator_scores, axis=0)
        mn, mx = mean_scores.min(), mean_scores.max()
        if mx > mn:
            mean_scores = (mean_scores - mn) / (mx - mn)
        annotations[vid_id] = mean_scores.astype(np.float32)

    print(f"[TVSum] Loaded {len(annotations)} videos from {tsv_path}")
    return annotations


# ---------------------------------------------------------------------------
# Feature-to-annotation alignment
# ---------------------------------------------------------------------------
def align_features_to_scores(features, timestamps, gt_scores, gt_fps):
    """
    Given:
      features   : (N_sampled, 512) ResNet features at `sample_fps`
      timestamps : (N_sampled,)     seconds
      gt_scores  : (M,)             ground-truth at `gt_fps`
      gt_fps     : float            annotation framerate

    Returns:
      aligned_features : (N_sampled, 512)
      aligned_scores   : (N_sampled,)
    """
    aligned_scores = np.zeros(len(timestamps), dtype=np.float32)
    for i, ts in enumerate(timestamps):
        gt_idx = min(int(round(ts * gt_fps)), len(gt_scores) - 1)
        aligned_scores[i] = gt_scores[gt_idx]
    return features, aligned_scores


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_dataset(
    dataset_root=None,
    output_path=None,
    cache_dir=None,
    sample_fps=None,
    device="cpu",
):
    dataset_root = dataset_root or _find_dataset_root()
    output_path  = output_path  or cfg.PROCESSED_DATASET_PATH
    cache_dir    = cache_dir    or cfg.FEATURES_CACHE_DIR
    sample_fps   = sample_fps   or cfg.FRAME_SAMPLE_FPS

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[FeatureExtraction] Dataset root : {dataset_root}")
    print(f"[FeatureExtraction] Feature cache: {cache_dir}")
    print(f"[FeatureExtraction] Output       : {output_path}")

    # Init CNN extractor
    extractor = ResNet18Extractor(device=device)

    # Load annotations
    summe_annot = load_summe_annotations(dataset_root)
    tvsum_annot = load_tvsum_annotations(dataset_root)
    print(f"[Annotations] SumMe: {len(summe_annot)} videos, TVSum: {len(tvsum_annot)} videos")

    # Find all video files
    video_exts = {".mp4", ".avi", ".mkv", ".mov", ".MP4", ".AVI"}
    all_videos = []
    for root, dirs, files in os.walk(dataset_root):
        for fn in files:
            if os.path.splitext(fn)[1] in video_exts:
                all_videos.append(os.path.join(root, fn))

    print(f"[Videos] Found {len(all_videos)} video files")

    dataset = []

    for vpath in sorted(all_videos):
        video_id = os.path.splitext(os.path.basename(vpath))[0]

        print(f"\n  Processing: {video_id}")
        features, timestamps, vid_id = extract_video_features(
            vpath, extractor, cache_dir, sample_fps
        )

        if features is None or len(features) == 0:
            print(f"  [SKIP] No frames extracted.")
            continue

        # Match to ground-truth annotation
        gt_scores = None
        dataset_type = "unknown"

        # Try SumMe first (exact then fuzzy)
        for key in summe_annot:
            if key.lower() == video_id.lower() or video_id.lower() in key.lower():
                gt_scores    = summe_annot[key]
                dataset_type = "SumMe"
                print(f"  Matched SumMe annotation: '{key}' ({len(gt_scores)} frames)")
                break

        # Try TVSum
        if gt_scores is None:
            for key in tvsum_annot:
                if key.lower() == video_id.lower() or video_id.lower() in key.lower():
                    gt_scores    = tvsum_annot[key]
                    dataset_type = "TVSum"
                    print(f"  Matched TVSum annotation: '{key}' ({len(gt_scores)} frames)")
                    break

        if gt_scores is None:
            print(f"  [SKIP] No matching annotation for '{video_id}'")
            continue

        # Determine annotation fps
        gt_fps = cfg.SUMME_FPS if dataset_type == "SumMe" else cfg.TVSUM_FPS
        _, aligned_scores = align_features_to_scores(
            features, timestamps, gt_scores, gt_fps
        )

        dataset.append({
            "video_id"        : video_id,
            "dataset_type"    : dataset_type,
            "num_segments"    : len(features),
            "visual_features" : features.tolist(),
            "text_features"   : [[0.0] * 384] * len(features),  # placeholder; filled during training
            "importance_scores": aligned_scores.tolist(),
        })

    print(f"\n[Done] {len(dataset)} videos processed.")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f)

    print(f"[Saved] {output_path}")
    return output_path


def _find_dataset_root():
    """Use kagglehub to locate the downloaded dataset, or raise."""
    try:
        import kagglehub
        path = kagglehub.dataset_download(
            "python16/summe-videos-and-tvsum-videos/versions/1"
        )
        print(f"[kagglehub] Dataset path: {path}")
        return path
    except Exception as e:
        raise RuntimeError(
            f"Could not locate dataset: {e}\n"
            "Run: python training/download_dataset.py  first."
        )


if __name__ == "__main__":
    build_dataset()
