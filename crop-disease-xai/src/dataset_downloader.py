import os
import sys
import json
import urllib.request
import random
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import RAW_DATA_DIR, TARGET_CLASSES, EXAMPLES_DIR, ensure_directories

GITHUB_API_BASE = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color"

def download_plant_disease_dataset(samples_per_class=200):
    """
    Downloads real leaf images from official PlantVillage repository on GitHub.
    
    Args:
        samples_per_class (int): Number of real images to download per class.
    """
    ensure_directories()
    print(f"==================================================")
    print(f"Downloading PlantVillage Dataset ({samples_per_class} images/class)...")
    print(f"==================================================")
    
    total_downloaded = 0
    
    for gh_folder, clean_name in TARGET_CLASSES.items():
        class_dir = RAW_DATA_DIR / clean_name.replace(" ", "_")
        os.makedirs(class_dir, exist_ok=True)
        
        # Check existing images count
        existing_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(existing_files) >= samples_per_class:
            print(f"[SKIP] Class '{clean_name}' already has {len(existing_files)} images.")
            continue
            
        print(f"\nFetching image metadata for: '{clean_name}' ({gh_folder})...")
        api_url = f"{GITHUB_API_BASE}/{gh_folder}"
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        
        try:
            with urllib.request.urlopen(req) as resp:
                files_info = json.loads(resp.read().decode('utf-8'))
                image_files = [f for f in files_info if f['name'].lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # Limit to target samples count
                selected_files = image_files[:samples_per_class]
                print(f"Found {len(image_files)} images on remote. Downloading {len(selected_files)}...")
                
                class_downloaded = 0
                for idx, file_info in enumerate(selected_files):
                    img_name = file_info['name']
                    save_path = class_dir / img_name
                    
                    if save_path.exists():
                        class_downloaded += 1
                        continue
                        
                    download_url = file_info['download_url']
                    img_req = urllib.request.Request(download_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(img_req) as img_resp:
                        with open(save_path, 'wb') as out_f:
                            out_f.write(img_resp.read())
                    
                    class_downloaded += 1
                    if (idx + 1) % 50 == 0 or (idx + 1) == len(selected_files):
                        print(f"  [{idx + 1}/{len(selected_files)}] Downloaded: {img_name}")
                
                print(f"[SUCCESS] Downloaded {class_downloaded} images for class '{clean_name}'.")
                total_downloaded += class_downloaded
                
        except Exception as e:
            print(f"[ERROR] Failed to fetch data for {clean_name}: {e}")
            
    # Prepare sample leaf examples for testing UI
    copy_sample_leaves()
    print(f"\n==================================================")
    print(f"Dataset Acquisition Complete! Total Images in Raw: {get_total_raw_count()}")
    print(f"==================================================")

def copy_sample_leaves():
    """Copy 1 sample image from each class to examples/sample_leaves for easy testing."""
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    for clean_name in TARGET_CLASSES.values():
        class_dir = RAW_DATA_DIR / clean_name.replace(" ", "_")
        if class_dir.exists():
            files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                src_file = class_dir / files[0]
                dest_file = EXAMPLES_DIR / f"sample_{clean_name.lower().replace(' ', '_')}.jpg"
                with open(src_file, 'rb') as f_in, open(dest_file, 'wb') as f_out:
                    f_out.write(f_in.read())
                print(f"Copied sample leaf for UI testing: {dest_file.name}")

def get_total_raw_count():
    total = 0
    for root, _, files in os.walk(RAW_DATA_DIR):
        total += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    return total

if __name__ == "__main__":
    download_plant_disease_dataset(samples_per_class=200)
