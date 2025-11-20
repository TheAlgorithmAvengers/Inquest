import os

import shutil

from tqdm import tqdm

resources1 = os.path.join("csv","resources1")
resources2 = os.path.join("csv","resources2")
output_root = os.path.join("csv","resources")

os.makedirs(output_root, exist_ok=True)

def copy_with_unique_name(src_file, dest_folder):
    filename = os.path.basename(src_file)
    name, ext = os.path.splitext(filename)
    new_name = filename
    counter = 2
    while os.path.exists(os.path.join(dest_folder, new_name)):
        new_name = f"{name}_{counter}{ext}"
        counter += 1
    shutil.copy(src_file, os.path.join(dest_folder, new_name))
    #print("Copied:", new_name)

for subfolder in tqdm(os.listdir(resources1), desc="Merging folders"):
    folder1 = os.path.join(resources1, subfolder)
    folder2 = os.path.join(resources2, subfolder)
    if os.path.isdir(folder1) and os.path.isdir(folder2):
        print(f"Merging: {subfolder}")
        dest_folder = os.path.join(output_root, subfolder)
        os.makedirs(dest_folder, exist_ok=True)
        for f in os.listdir(folder1):
            src = os.path.join(folder1, f)
            if os.path.isfile(src):
                copy_with_unique_name(src, dest_folder)
        for f in os.listdir(folder2):
            src = os.path.join(folder2, f)
            if os.path.isfile(src):
                copy_with_unique_name(src, dest_folder)

print("Merge complete.")