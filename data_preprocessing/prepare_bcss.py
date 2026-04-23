#data preparation script to create 4-class patch dataset from BCSS
import os
import cv2
import numpy as np
import shutil
import random
from tqdm.notebook import tqdm


INPUT_ROOT = "/kaggle/input/bcss-dataset"
OUTPUT_DIR = "/kaggle/working/dataset_patches"

PATCH_SIZE = 224
STRIDE = 224
DESIRED_MPP = 0.50
SOURCE_MPP  = 0.25
RESIZE_FACTOR = SOURCE_MPP / DESIRED_MPP  # 0.5

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED = 42

#  Final 4-class mapping after merging critically rare classes
# GT code → final class label
GT_TO_LABEL = {
    1:  "tumor",                   
    20: "tumor",                   
    2:  "stroma",                  
    19: "stroma",                  
    3:  "lymphocytic_infiltrate",  
    10: "lymphocytic_infiltrate",  
    11: "lymphocytic_infiltrate",  
    4:  "necrosis_or_debris",      
}

TARGET_LABELS = [
    "tumor",
    "stroma",
    "lymphocytic_infiltrate",
    "necrosis_or_debris",
]

# Priority: necrosis > immune > tumor > stroma
# Rarest meaningful classes assigned first
LABEL_PRIORITY = [
    "necrosis_or_debris",
    "lymphocytic_infiltrate",
    "tumor",
    "stroma",
]

DOMINANT_THRESHOLD   = 0.35
OUTSIDE_ROI_THRESHOLD = 0.50
# =================================================

def find_bcss_paths():
    print("🔍 Searching for BCSS dataset...")
    for root, dirs, files in os.walk(INPUT_ROOT):
        if "images" in dirs and "masks" in dirs:
            img_dir = os.path.join(root, "images")
            if len(os.listdir(img_dir)) > 0:
                return img_dir, os.path.join(root, "masks")
    for root, dirs, files in os.walk(INPUT_ROOT):
        if "images" in dirs:
            potential_img = os.path.join(root, "images")
            potential_mask = os.path.join(os.path.dirname(potential_img), "masks")
            if os.path.exists(potential_mask):
                return potential_img, potential_mask
    return None, None

def setup_directories():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    for split in ["train", "val", "test"]:
        for label in TARGET_LABELS:
            os.makedirs(os.path.join(OUTPUT_DIR, split, label), exist_ok=True)
    print(f" Created 4-class directory structure at: {OUTPUT_DIR}")

def get_split_assignments(slide_files):
    patient_map = {}
    for filename in slide_files:
        patient_id = filename[:12]
        if patient_id not in patient_map:
            patient_map[patient_id] = []
        patient_map[patient_id].append(filename)

    unique_patients = list(patient_map.keys())
    print(f"🔍 Found {len(unique_patients)} unique patients across {len(slide_files)} images.")

    random.seed(SEED)
    random.shuffle(unique_patients)

    n_total = len(unique_patients)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)

    train_patients = unique_patients[:n_train]
    val_patients   = unique_patients[n_train: n_train + n_val]
    test_patients  = unique_patients[n_train + n_val:]

    train_slides, val_slides, test_slides = set(), set(), set()
    for pid in train_patients: train_slides.update(patient_map[pid])
    for pid in val_patients:   val_slides.update(patient_map[pid])
    for pid in test_patients:  test_slides.update(patient_map[pid])

    print(f" Split (Files): Train={len(train_slides)}, "
          f"Val={len(val_slides)}, Test={len(test_slides)}")
    return train_slides, val_slides, test_slides

def get_patch_label(patch_mask):
    """
    Assigns one of 4 survival-relevant labels using merged GT codes.
    Priority: necrosis > immune > tumor > stroma
    Returns label string or None if patch should be skipped.
    """
    total = patch_mask.size
    if total == 0:
        return None

    unique, counts = np.unique(patch_mask, return_counts=True)
    pixel_counts = dict(zip(unique.tolist(), counts.tolist()))

    # Skip if outside_roi dominates
    if pixel_counts.get(0, 0) / total > OUTSIDE_ROI_THRESHOLD:
        return None

    foreground_total = total - pixel_counts.get(0, 0)
    if foreground_total == 0:
        return None

    # Aggregate pixel counts per final label (handles merged GT codes)
    label_counts = {label: 0 for label in TARGET_LABELS}
    for gt_code, label in GT_TO_LABEL.items():
        label_counts[label] += pixel_counts.get(gt_code, 0)

    # Compute ratios over foreground
    label_ratios = {
        label: count / foreground_total
        for label, count in label_counts.items()
    }

    # Assign by priority
    for label in LABEL_PRIORITY:
        if label_ratios[label] >= DOMINANT_THRESHOLD:
            return label

    return None  # Mixed patch — skip

def process_and_save_patches():
    images_dir, masks_dir = find_bcss_paths()
    if not images_dir:
        print(" Error: Could not find dataset.")
        return

    setup_directories()

    valid_exts = ('.png', '.tif', '.svs', '.jpg')
    slide_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(valid_exts)]

    train_slides, val_slides, test_slides = get_split_assignments(slide_files)
    print(f" Processing {len(slide_files)} slides with 4-class labeling...")

    stats = {
        split: {label: 0 for label in TARGET_LABELS}
        for split in ["train", "val", "test"]
    }
    skipped = 0

    for slide_name in tqdm(slide_files):
        if slide_name in train_slides:   split = "train"
        elif slide_name in val_slides:   split = "val"
        else:                            split = "test"

        base_name = os.path.splitext(slide_name)[0]
        mask_path = os.path.join(masks_dir, slide_name)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(masks_dir, base_name + ".png")
            if not os.path.exists(mask_path):
                continue

        try:
            image = cv2.imread(os.path.join(images_dir, slide_name))
            if image is None: continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path, 0)
            if mask is None: continue

            if RESIZE_FACTOR != 1.0:
                new_w = int(image.shape[1] * RESIZE_FACTOR)
                new_h = int(image.shape[0] * RESIZE_FACTOR)
                image = cv2.resize(image, (new_w, new_h),
                                   interpolation=cv2.INTER_LINEAR)
                mask  = cv2.resize(mask,  (new_w, new_h),
                                   interpolation=cv2.INTER_NEAREST)

            h, w, _ = image.shape

            for y in range(0, h - PATCH_SIZE, STRIDE):
                for x in range(0, w - PATCH_SIZE, STRIDE):
                    patch_mask  = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                    patch_image = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

                    label = get_patch_label(patch_mask)
                    if label is None:
                        skipped += 1
                        continue

                    save_name = f"{base_name}_{x}_{y}.png"
                    save_path = os.path.join(OUTPUT_DIR, split, label, save_name)
                    cv2.imwrite(save_path,
                                cv2.cvtColor(patch_image, cv2.COLOR_RGB2BGR))
                    stats[split][label] += 1

        except Exception as e:
            print(f" Error on {slide_name}: {e}")
            continue

    print("\nProcessing Complete!")
    print(f" Skipped: {skipped}")
    print("\n Final 4-Class Distribution:")
    for split in ["train", "val", "test"]:
        total = sum(stats[split].values())
        print(f"\n  {split.upper()} ({total} patches):")
        for label in TARGET_LABELS:
            count = stats[split][label]
            pct = 100 * count / total if total > 0 else 0
            print(f"    {label:30s}: {count:6d}  ({pct:.1f}%)")

if __name__ == "__main__":
    process_and_save_patches()