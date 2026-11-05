#!/usr/bin/env python3
"""
Compare YOLO26 NMS-free (one-to-one head) vs traditional one-to-many + NMS.

Main metric:
    Number of unique bbox pairs with IoU > overlap_iou.

Example:
    python compare_nms.py \
        --model runs/detect/train/weights/best.pt \
        --images dataset/images/test \
        --conf 0.25 \
        --overlap-iou 0.7 \
        --nms-iou 0.7 \
        --imgsz 640 \
        --device 0

Output:
    - Per-image statistics
    - Summary statistics
    - CSV file
"""

from pathlib import Path
import argparse
import csv
import time

import torch
from ultralytics import YOLO


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU.

    boxes1: [N, 4] xyxy
    boxes2: [M, 4] xyxy

    Returns:
        IoU matrix [N, M]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (boxes1.shape[0], boxes2.shape[0]),
            device=boxes1.device
        )

    # Intersection
    lt = torch.maximum(
        boxes1[:, None, :2],
        boxes2[None, :, :2]
    )

    rb = torch.minimum(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:]
    )

    wh = (rb - lt).clamp(min=0)

    intersection = wh[:, :, 0] * wh[:, :, 1]

    # Area
    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
        * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    )

    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
        * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    )

    union = area1[:, None] + area2[None, :] - intersection

    return intersection / (union + 1e-7)


def count_overlapping_pairs(
    boxes: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float = 0.6,
    same_class_only: bool = False,
):
    """
    Count UNIQUE bbox pairs having IoU > threshold.

    Example:
        bbox 0 overlaps bbox 1 -> counted once
        not twice as (0,1) and (1,0).

    Returns:
        pair_count
        max_iou
        mean_overlap_iou
    """

    n = len(boxes)

    if n < 2:
        return 0, 0.0, 0.0

    ious = box_iou(boxes, boxes)

    # Only upper triangle:
    # removes diagonal and duplicated pairs.
    mask = torch.triu(
        torch.ones_like(ious, dtype=torch.bool),
        diagonal=1
    )

    # Optionally compare only boxes belonging to the same class.
    if same_class_only:
        same_class = classes[:, None] == classes[None, :]
        mask &= same_class

    selected_ious = ious[mask]

    if selected_ious.numel() == 0:
        return 0, 0.0, 0.0

    overlapping = selected_ious > iou_threshold

    count = int(overlapping.sum().item())

    max_iou = float(selected_ious.max().item())

    if count > 0:
        mean_overlap_iou = float(
            selected_ious[overlapping].mean().item()
        )
    else:
        mean_overlap_iou = 0.0

    return count, max_iou, mean_overlap_iou


def extract_result(result):
    """
    Extract xyxy boxes, confidence and classes from Ultralytics Results.
    """

    if result.boxes is None or len(result.boxes) == 0:
        device = (
            result.boxes.xyxy.device
            if result.boxes is not None
            else torch.device("cpu")
        )

        return (
            torch.empty((0, 4), device=device),
            torch.empty((0,), device=device),
            torch.empty((0,), device=device),
        )

    boxes = result.boxes.xyxy.detach()
    confs = result.boxes.conf.detach()
    classes = result.boxes.cls.detach().long()

    return boxes, confs, classes


def evaluate_result(result, overlap_iou):
    """
    Calculate overlap metrics for one Result object.
    """

    boxes, confs, classes = extract_result(result)

    # All classes
    overlap_all, max_iou_all, mean_iou_all = count_overlapping_pairs(
        boxes=boxes,
        classes=classes,
        iou_threshold=overlap_iou,
        same_class_only=False,
    )

    # Same class only
    overlap_same, max_iou_same, mean_iou_same = count_overlapping_pairs(
        boxes=boxes,
        classes=classes,
        iou_threshold=overlap_iou,
        same_class_only=True,
    )

    return {
        "num_boxes": len(boxes),

        "overlap_pairs_all": overlap_all,
        "overlap_pairs_same_class": overlap_same,

        "max_iou_all": max_iou_all,
        "max_iou_same_class": max_iou_same,

        "mean_overlap_iou_all": mean_iou_all,
        "mean_overlap_iou_same_class": mean_iou_same,

        "mean_conf": (
            float(confs.mean().item())
            if len(confs) > 0
            else 0.0
        ),
    }


def get_image_paths(image_dir):
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: {image_dir}"
        )

    paths = [
        p
        for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    paths.sort()

    if not paths:
        raise RuntimeError(
            f"No images found in {image_dir}"
        )

    return paths


@torch.inference_mode()
def run_prediction(
    model,
    image_path,
    mode,
    conf,
    nms_iou,
    imgsz,
    device,
    max_det,
):
    """
    Run YOLO26 inference.

    mode="nms_free":
        nms=False
        -> one-to-one head
        -> no NMS

    mode="nms":
        nms=None
        -> one-to-many head
        -> traditional Ultralytics NMS
    """

    common_args = dict(
        source=str(image_path),
        conf=conf,
        imgsz=imgsz,
        device=device,
        max_det=max_det,
        verbose=False,
    )

    start = time.perf_counter()

    if mode == "nms_free":
        results = model.predict(
            **common_args,
            nms=False,
        )

    elif mode == "nms":
        results = model.predict(
            **common_args,
            nms=None,
            iou=nms_iou,
        )

    else:
        raise ValueError(
            f"Unknown inference mode: {mode}"
        )

    elapsed = time.perf_counter() - start

    return results[0], elapsed


def print_summary(title, stats, total_images):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(f"Images                     : {total_images}")

    print(
        f"Total detections           : "
        f"{stats['num_boxes']}"
    )

    print(
        f"Overlap pairs IoU > threshold (all classes): "
        f"{stats['overlap_all']}"
    )

    print(
        f"Overlap pairs IoU > threshold (same class) : "
        f"{stats['overlap_same']}"
    )

    print(
        f"Images with overlap (all)  : "
        f"{stats['images_overlap_all']}/{total_images}"
    )

    print(
        f"Images with overlap (same) : "
        f"{stats['images_overlap_same']}/{total_images}"
    )

    if stats["num_boxes"] > 0:
        print(
            f"Overlap pairs / 100 boxes  : "
            f"{100 * stats['overlap_same'] / stats['num_boxes']:.3f}"
        )

    print(
        f"Total inference time       : "
        f"{stats['time']:.4f} s"
    )

    print(
        f"Average time / image       : "
        f"{1000 * stats['time'] / total_images:.3f} ms"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare YOLO26 NMS-free vs traditional NMS"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO26 .pt checkpoint"
    )

    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Directory containing test images"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold. Default: 0.25"
    )

    parser.add_argument(
        "--overlap-iou",
        type=float,
        default=0.7,
        help=(
            "IoU threshold used to define overlapping duplicate boxes. "
            "Default: 0.6"
        )
    )

    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.7,
        help=(
            "IoU threshold used by traditional NMS. "
            "Default: 0.7"
        )
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help='Device, e.g. "0", "cpu"'
    )

    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum detections per image"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="nms_comparison.csv",
        help="Output CSV"
    )

    args = parser.parse_args()

    # ----------------------------------------------------------
    # Load model
    # ----------------------------------------------------------

    print(f"Loading model: {args.model}")

    model = YOLO(args.model)

    image_paths = get_image_paths(args.images)

    print(f"Found {len(image_paths)} images")
    print(f"Confidence threshold : {args.conf}")
    print(f"Overlap IoU threshold: {args.overlap_iou}")
    print(f"NMS IoU threshold    : {args.nms_iou}")
    print(f"Image size           : {args.imgsz}")
    print(f"Device               : {args.device}")
    print()

    rows = []

    summary = {
        "nms_free": {
            "num_boxes": 0,
            "overlap_all": 0,
            "overlap_same": 0,
            "images_overlap_all": 0,
            "images_overlap_same": 0,
            "time": 0.0,
        },

        "nms": {
            "num_boxes": 0,
            "overlap_all": 0,
            "overlap_same": 0,
            "images_overlap_all": 0,
            "images_overlap_same": 0,
            "time": 0.0,
        },
    }

    # ----------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------

    for idx, image_path in enumerate(image_paths, start=1):

        # ======================================================
        # NMS-free
        # ======================================================

        result_free, time_free = run_prediction(
            model=model,
            image_path=image_path,
            mode="nms_free",
            conf=args.conf,
            nms_iou=args.nms_iou,
            imgsz=args.imgsz,
            device=args.device,
            max_det=args.max_det,
        )

        metrics_free = evaluate_result(
            result_free,
            args.overlap_iou
        )

        # ======================================================
        # Traditional one-to-many + NMS
        # ======================================================

        result_nms, time_nms = run_prediction(
            model=model,
            image_path=image_path,
            mode="nms",
            conf=args.conf,
            nms_iou=args.nms_iou,
            imgsz=args.imgsz,
            device=args.device,
            max_det=args.max_det,
        )

        metrics_nms = evaluate_result(
            result_nms,
            args.overlap_iou
        )

        # ------------------------------------------------------
        # Update statistics
        # ------------------------------------------------------

        for key, metrics, elapsed in [
            ("nms_free", metrics_free, time_free),
            ("nms", metrics_nms, time_nms),
        ]:
            summary[key]["num_boxes"] += metrics["num_boxes"]

            summary[key]["overlap_all"] += (
                metrics["overlap_pairs_all"]
            )

            summary[key]["overlap_same"] += (
                metrics["overlap_pairs_same_class"]
            )

            summary[key]["time"] += elapsed

            if metrics["overlap_pairs_all"] > 0:
                summary[key]["images_overlap_all"] += 1

            if metrics["overlap_pairs_same_class"] > 0:
                summary[key]["images_overlap_same"] += 1

        # ------------------------------------------------------
        # CSV row
        # ------------------------------------------------------

        row = {
            "image": str(image_path),

            # NMS-free
            "nms_free_boxes":
                metrics_free["num_boxes"],

            "nms_free_overlap_all":
                metrics_free["overlap_pairs_all"],

            "nms_free_overlap_same_class":
                metrics_free["overlap_pairs_same_class"],

            "nms_free_max_iou":
                metrics_free["max_iou_same_class"],

            "nms_free_mean_overlap_iou":
                metrics_free["mean_overlap_iou_same_class"],

            "nms_free_mean_conf":
                metrics_free["mean_conf"],

            "nms_free_time_ms":
                time_free * 1000,

            # NMS
            "nms_boxes":
                metrics_nms["num_boxes"],

            "nms_overlap_all":
                metrics_nms["overlap_pairs_all"],

            "nms_overlap_same_class":
                metrics_nms["overlap_pairs_same_class"],

            "nms_max_iou":
                metrics_nms["max_iou_same_class"],

            "nms_mean_overlap_iou":
                metrics_nms["mean_overlap_iou_same_class"],

            "nms_mean_conf":
                metrics_nms["mean_conf"],

            "nms_time_ms":
                time_nms * 1000,
        }

        rows.append(row)

        print(
            f"[{idx:4d}/{len(image_paths)}] "
            f"{image_path.name:<30} | "
            f"NMS-free: boxes={metrics_free['num_boxes']:3d}, "
            f"overlap={metrics_free['overlap_pairs_same_class']:3d} | "
            f"NMS: boxes={metrics_nms['num_boxes']:3d}, "
            f"overlap={metrics_nms['overlap_pairs_same_class']:3d}"
        )

    # ----------------------------------------------------------
    # Save CSV
    # ----------------------------------------------------------

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    print_summary(
        "NMS-FREE / ONE-TO-ONE",
        summary["nms_free"],
        len(image_paths)
    )

    print_summary(
        "TRADITIONAL ONE-TO-MANY + NMS",
        summary["nms"],
        len(image_paths)
    )

    # ----------------------------------------------------------
    # Direct comparison
    # ----------------------------------------------------------

    free_overlap = summary["nms_free"]["overlap_same"]
    nms_overlap = summary["nms"]["overlap_same"]

    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(
        f"NMS-free overlapping pairs : {free_overlap}"
    )

    print(
        f"NMS overlapping pairs      : {nms_overlap}"
    )

    if nms_overlap > 0:
        ratio = free_overlap / nms_overlap

        print(
            f"NMS-free / NMS ratio       : {ratio:.4f}"
        )

    free_boxes = summary["nms_free"]["num_boxes"]
    nms_boxes = summary["nms"]["num_boxes"]

    if free_boxes > 0:
        free_rate = (
            free_overlap / free_boxes * 100
        )
    else:
        free_rate = 0

    if nms_boxes > 0:
        nms_rate = (
            nms_overlap / nms_boxes * 100
        )
    else:
        nms_rate = 0

    print()
    print(
        f"NMS-free overlap / 100 boxes: "
        f"{free_rate:.4f}"
    )

    print(
        f"NMS overlap / 100 boxes     : "
        f"{nms_rate:.4f}"
    )

    print()
    print(f"CSV saved to: {output_path}")


if __name__ == "__main__":
    main()