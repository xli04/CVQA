import json
import os
from collections import defaultdict

# Paths (adjust to your environment)
qtype_dir = r"Original"  # where the *q*.json files live
img_type_map_path = r"/datasets/ImgId_cate_map.json"

# Question-type → filename suffix mapping
# Only the part AFTER karpathy_<split>_
qtype_suffixes = {
    0: "q_recognition.json",
    1: "q_location.json",
    2: "q_judge.json",
    3: "q_commonsense.json",
    4: "q_count.json",
    5: "q_action.json",
    6: "q_color.json",
    7: "q_type.json",
    8: "q_subcategory.json",
    9: "q_causal.json",
}

# Task definitions (unchanged)
task_definitions = {
    1: {8, 63, 51, 62, 87, 54, 55, 49, 67, 64},
    2: {88, 78, 41, 44, 58, 84, 53, 82, 43, 79},
    3: {73, 1, 3, 52, 36, 65, 28, 42, 4, 81},
    4: {72, 18, 39, 47, 14, 20, 50, 61, 56, 70},
    5: {38, 48, 6, 7, 77, 59, 76, 37, 21, 25},
    6: {40, 23, 46, 10, 19, 74, 13, 24, 31, 35},
    7: {11, 80, 17, 15, 75, 85, 89, 16, 33, 90},
    8: {57, 27, 32, 22, 34, 2, 60, 9, 5, 86},
}

# Load the global ImgID → CategoryID mapping once
with open(img_type_map_path, "r") as f:
    img_type_map = json.load(f)

# Helper: build tasks_data for one split
def collect_entries_for_split(split: str):
    """
    Returns a dict task_id -> list[entry] for the given split.
    """
    tasks_data = defaultdict(list)
    for qtype, suffix in qtype_suffixes.items():
        fname = f"karpathy_{split}_{suffix}"
        file_path = os.path.join(qtype_dir, fname)
        if not os.path.exists(file_path):
            print(f"[{split}] File missing: {file_path}")
            continue
        with open(file_path, "r") as f:
            data = json.load(f)
        for entry in data:
            img_id = str(entry["img_id"])
            cate_id = img_type_map.get(img_id)
            if cate_id is None:
                # image not in the category map
                continue
            # assign to all tasks that include this category
            for t_id, cate_set in task_definitions.items():
                if cate_id in cate_set:
                    tasks_data[t_id].append(entry)
    return tasks_data

# Main: iterate over splits
splits = ["train", "val", "test"]
for split in splits:
    tasks_data = collect_entries_for_split(split)
    if not tasks_data:
        print(f"[{split}] No data collected — nothing written.")
        continue
    out_dir = f"reorganized_tasks_{split}"
    os.makedirs(out_dir, exist_ok=True)
    for t_id, entries in tasks_data.items():
        out_path = os.path.join(out_dir, f"karpathy_{split}_task_{t_id}.json")
        with open(out_path, "w") as out_f:
            json.dump(entries, out_f, indent=2)
    print(f"[{split}] Completed: {len(tasks_data)} tasks saved to '{out_dir}'")

print("All splits processed.")
