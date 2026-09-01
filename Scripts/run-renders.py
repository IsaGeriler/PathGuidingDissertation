import os
import sys
import subprocess
import time
import csv
from datetime import datetime

# --- Dynamic Path Resolution ---
# 1. Get the directory where this python script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Define the Base Directory of your repository
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# --- Configuration ---
# Path to Renderer built dynamically
RENDERER_BIN = os.path.join(BASE_DIR, "x64", "Release", "PathGuidingDissertation.exe")

# Directory to save output HDR/PNG images
OUTPUT_DIR = os.path.join(BASE_DIR, "Images", "Experiments")

# Summary log file
LOG_CSV = os.path.join(OUTPUT_DIR, "render_benchmark_results.csv")

# Test Scenes (Cleaner way to define them)
SCENE_NAMES = [
    "kitchen",
    "cornell-box",
    "staircase",
    "classroom",
    "dining-room"
]
# Build the absolute paths dynamically for whatever machine this runs on
SCENES = [os.path.join(BASE_DIR, "Scenes", name) for name in SCENE_NAMES]

# Methods to compare
METHODS = ["path_trace", "photon_map", "path_guide_pss", "path_guide_photon"]

# Sample counts for convergence testing (Equal-Sample test)
SPP_LIST = [32, 128, 512]

# --- Pre-run Checks ---
if not os.path.exists(RENDERER_BIN):
    print(f"[ERROR]: Renderer executable not found at:\n{RENDERER_BIN}")
    print("Did you forget to build the project in Visual Studio (Release mode)?")
    sys.exit(1)

for scene in SCENES:
    if not os.path.exists(scene):
        print(f"[WARNING]: Scene folder not found: {scene}")

# --- Batch Execution ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prepare CSV Logging
csv_fields = ["timestamp", "scene", "method", "spp", "render_time_seconds", "output_file"]
file_exists = os.path.isfile(LOG_CSV)

with open(LOG_CSV, mode="a", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if not file_exists:
        writer.writeheader()

    print("=" * 60)
    print(f"STARTING BATCH RENDERING AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 60)

    total_renders = len(SCENES) * len(METHODS) * len(SPP_LIST)
    current_count = 0

    for scene_path in SCENES:
        scene_name = os.path.basename(scene_path)

        for method in METHODS:
            for spp in SPP_LIST:
                current_count += 1

                out_filename = f"{scene_name}_{method}_{spp}spp.hdr"
                out_filepath = os.path.join(OUTPUT_DIR, out_filename)

                cmd = [
                    RENDERER_BIN,
                    "-scene", scene_path,
                    "-method", method,
                    "-SPP", str(spp),
                    "-outputFilename", out_filepath
                ]

                print(f"\n[{current_count}/{total_renders}] Rendering: {scene_name} | Method: {method} | SPP: {spp} ")
                
                start_time = time.time()

                try:
                    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    elapsed_time = time.time() - start_time

                    print(f"--> COMPLETED in {elapsed_time:.2f} seconds.")

                    writer.writerow({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "scene": scene_name,
                        "method": method,
                        "spp": spp,
                        "render_time_seconds": round(elapsed_time, 3),
                        "output_file": out_filepath
                    })
                    csv_file.flush()

                except subprocess.CalledProcessError as e:
                    print(f"[ERROR]: Render failed for {scene_name} ({method}, {spp} spp)!")
                    print(f"Error output:\n{e.stderr}")
                    continue

print("\n" + "=" * 60)
print(f"ALL BATCH RENDERS COMPLETE! Check 'render_benchmark_results.csv' in:\n{OUTPUT_DIR}")
print("=" * 60)