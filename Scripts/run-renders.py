import os
import subprocess
import time
import csv
from datetime import datetime

# --- Configuration ---
# Path to Renderer
RENDERER_BIN = "C:/Users/u5749205/source/repos/IsaGeriler/PathGuidingDissertation/x64/Release/PathGuidingDissertation.exe"

# Directory to save output HDR/PNG images
OUTPUT_DIR = "C:/Users/u5749205/source/repos/IsaGeriler/PathGuidingDissertation/Images/PathTraceVsPathGuide"

# Summary log file
LOG_CSV = os.path.join(OUTPUT_DIR, "render_benchmark_results.csv")

# Test Scenes
SCENES = [
    #"C:/Users/u5749205/source/repos/IsaGeriler/PathGuidingDissertation/Scenes/cornell-box",
    #"C:/Users/u5749205/source/repos/IsaGeriler/PathGuidingDissertation/Scenes/kitchen",
    "C:/Users/u5749205/source/repos/IsaGeriler/PathGuidingDissertation/Scenes/staircase"
]

# Methods to compare
METHODS = ["path_trace", "photon_map", "path_guide"]

# Sample counts for convergence testing (Equal-Sample test)
# TO:DO - Add 512, 1024 (and 2048 if enough time!) [Path Guide is slow as a turtle!]
SPP_LIST = [64, 128, 256]
# --- Configuration End ---

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
    print("=" * 60)

    total_renders = len(SCENES) * len(METHODS) * len(SPP_LIST)
    current_count = 0

    for scene_path in SCENES:
        scene_name = os.path.splitext(os.path.basename(scene_path))[0]

        for method in METHODS:
            for spp in SPP_LIST:
                current_count += 1

                # Construct clean output filename
                out_filename = f"{scene_name}_{method}_{spp}spp.hdr"
                out_filepath = os.path.join(OUTPUT_DIR, out_filename)

                # Construct CLI command for your custom engine
                # Adjust flags (--scene, --method, etc.) to match your engine's argument parser!
                cmd = [
                    RENDERER_BIN,
                    "-scene", scene_path,
                    "-method", method,
                    "-SPP", str(spp),
                    "-outputFilename", out_filepath
                ]

                print(f"\n[{current_count}/{total_renders}] Rendering: {scene_name} | Method: {method} | SPP: {spp}")
                print(f"Running command: {' '.join(cmd)}")

                # Measure exact execution time
                start_time = time.time()

                try:
                    # Run the engine as a subprocess
                    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    elapsed_time = time.time() - start_time

                    print(f"--> COMPLETED in {elapsed_time:.2f} seconds.")

                    # Log to CSV
                    writer.writerow({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "scene": scene_name,
                        "method": method,
                        "spp": spp,
                        "render_time_seconds": round(elapsed_time, 3),
                        "output_file": out_filepath
                    })
                    csv_file.flush() # Flush to disk immediately

                except subprocess.CalledProcessError as e:
                    print(f"❌ ERROR: Render failed for {scene_name} ({method}, {spp} spp)!")
                    print(f"Error output:\n{e.stderr}")
                    # Continue script so one crash doesn't stop the whole night!
                    continue

print("\n" + "=" * 60)
print("ALL BATCH RENDERS COMPLETE! Check 'render_benchmark_results.csv' for data.")
print("=" * 60)
# --- Batch Execution End ---