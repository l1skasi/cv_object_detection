import os
import shutil
from pathlib import Path
from typing import Optional, List, Tuple

import pandas as pd
from PIL import Image

IMAGE_ROOT = Path("../data/100k/train")


OUT_DIR = Path("../data/my_yolo_data")

CSV_PATHS = {
    "train": Path("../data/labels/train_labels.csv"),
    "val":   Path("../data/labels/val_labels.csv"),
    "test":  Path("../data/labels/test_labels.csv"),
}


KEEP_CLASSES = ["car", "traffic sign", "traffic light", "person"]


USE_SYMLINKS = False

EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


CLASS2ID = {c: i for i, c in enumerate(KEEP_CLASSES)}


def ensure_dirs():
    for split in CSV_PATHS.keys():
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def normalize_category(cat: str) -> str:
    return " ".join(str(cat).strip().split())


def resolve_image_path(root_dirs: List[Path], image_value: str) -> Optional[Path]:
    image_value = str(image_value).strip()
    p_value = Path(image_value)
    stem = p_value.stem


    if p_value.suffix:
        for d in root_dirs:
            p = d / p_value.name
            if p.exists():
                return p


    for d in root_dirs:
        for ext in EXTS:
            p = d / (stem + ext)
            if p.exists():
                return p

    for d in root_dirs:
        hits = list(d.rglob(stem + ".*"))
        if hits:
            hits_sorted = sorted(hits, key=lambda x: (x.suffix.lower() != ".jpg", str(x)))
            return hits_sorted[0]

    return None


def yolo_line_from_xyxy(x1, y1, x2, y2, w_img, h_img, cls_id) -> Optional[str]:

    x1 = max(0.0, min(float(x1), w_img))
    x2 = max(0.0, min(float(x2), w_img))
    y1 = max(0.0, min(float(y1), h_img))
    y2 = max(0.0, min(float(y2), h_img))

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 0 or bh <= 0:
        return None

    xc = (x1 + x2) / 2.0 / w_img
    yc = (y1 + y2) / 2.0 / h_img
    bw = bw / w_img
    bh = bh / h_img

    return f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if USE_SYMLINKS:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def convert_split(split: str, csv_path: Path) -> Tuple[int, int, int, int]:
   
    df = pd.read_csv(csv_path)

    required = {"image", "category", "x1", "y1", "x2", "y2"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"{csv_path} is missing columns: {missing_cols}")

    df["image"] = df["image"].astype(str).str.strip()
    df["category"] = df["category"].apply(normalize_category)


    df = df[df["category"].isin(KEEP_CLASSES)].copy()

    images_written = 0
    images_missing = 0
    label_files_written = 0
    boxes_written = 0


    for image_id, g in df.groupby("image"):
        src_img = resolve_image_path([IMAGE_ROOT], image_id)
        if src_img is None:
            images_missing += 1
            print(f"[WARN] Missing image for id '{image_id}' (split={split})")
            continue


        try:
            with Image.open(src_img) as im:
                w_img, h_img = im.size
        except Exception as e:
            images_missing += 1
            print(f"[WARN] Could not open image '{src_img}': {e}")
            continue


        lines: List[str] = []
        for _, row in g.iterrows():
            cls_id = CLASS2ID[row["category"]]
            line = yolo_line_from_xyxy(
                row["x1"], row["y1"], row["x2"], row["y2"],
                w_img, h_img, cls_id
            )
            if line:
                lines.append(line)

        label_path = OUT_DIR / "labels" / split / (Path(image_id).stem + ".txt")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        label_files_written += 1
        boxes_written += len(lines)

        dst_img = OUT_DIR / "images" / split / (Path(image_id).stem + src_img.suffix)
        link_or_copy(src_img, dst_img)
        images_written += 1

    return images_written, images_missing, label_files_written, boxes_written


def write_data_yaml():
    yaml_text = f"""# Ultralytics YOLO dataset config
path: {OUT_DIR.resolve()}
train: images/train
val: images/val
test: images/test

names:
"""
    for i, name in enumerate(KEEP_CLASSES):
        yaml_text += f"  {i}: {name}\n"

    (OUT_DIR / "data.yaml").write_text(yaml_text, encoding="utf-8")


def main():
    # Quick sanity check: does IMAGE_ROOT contain images?
    any_img = next(IMAGE_ROOT.glob("*.jpg"), None)
    if any_img is None:
        print(f"[WARN] No .jpg found directly in {IMAGE_ROOT}. "
              f"If images are in subfolders, that's okay (rglob will handle it), "
              f"but double-check IMAGE_ROOT is correct.")

    ensure_dirs()

    total_written = 0
    total_missing = 0
    total_label_files = 0
    total_boxes = 0

    for split, csv_path in CSV_PATHS.items():
        w, m, lf, b = convert_split(split, csv_path)
        total_written += w
        total_missing += m
        total_label_files += lf
        total_boxes += b
        print(f"[{split}] images written: {w}, missing: {m}, label files: {lf}, boxes: {b}")

    write_data_yaml()



if __name__ == "__main__":
    main()
