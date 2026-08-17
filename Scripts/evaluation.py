# --- Import the necessary libraries for the evaluation script ---
import os
import re
# By default, OpenCV library has this flag disabled, and FLIP does not support .hdr files
# We need to convert .hdr files to .exr when evaluating FLIP, thus enabling this flag manually
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import imageio.v3 as iio
import pandas as pd
from flip_evaluator.flip_python_api import evaluate as flip_evaluate
from skimage.metrics import structural_similarity as ssim

# --- Functions for metric evaluations ---
def calculate_mse(test_hdr, gt_hdr):
    return np.mean((test_hdr - gt_hdr) ** 2)

def calculate_rmse(test_hdr, gt_hdr):
    return np.sqrt(calculate_mse(test_hdr, gt_hdr))

def calculate_mrse(test_hdr, gt_hdr, epsilon=1e-2):
    numerator = (test_hdr - gt_hdr) ** 2
    denominator = gt_hdr ** 2 + epsilon
    return np.mean(numerator / denominator)

def calculate_rrmse(test_hdr, gt_hdr, epsilon=1e-2):
    return np.sqrt(calculate_mrse(test_hdr, gt_hdr, epsilon))

def calculate_mae(test_hdr, gt_hdr):
    return np.mean(np.abs(test_hdr - gt_hdr))

def calculate_psnr(test_hdr, gt_hdr):
    mse = calculate_mse(test_hdr, gt_hdr)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(gt_hdr)
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim(test_png_filepath, gt_png_filepath):
    test_ldr = cv2.imread(test_png_filepath, cv2.IMREAD_GRAYSCALE)
    gt_ldr = cv2.imread(gt_png_filepath, cv2.IMREAD_GRAYSCALE)

    if test_ldr is None or gt_ldr is None:
        print(f"WARNING: Image not found for {test_png_filepath} or {gt_png_filepath}")
        return -1.0
    return ssim(gt_ldr, test_ldr, data_range=255)

# --- FLIP and EXR Conversion Helpers ---
def create_temp_exr(hdr_pixels, output_path):
    bgr_pixels = cv2.cvtColor(hdr_pixels, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr_pixels)

def run_flip(test_hdr_pixels, gt_hdr_pixels, test_base_path, gt_base_path, flip_output_dir):
    temp_test_exr = f"{test_base_path}_TEMP.exr"
    temp_gt_exr = f"{gt_base_path}_TEMP.exr"
    
    create_temp_exr(test_hdr_pixels, temp_test_exr)
    if not os.path.exists(temp_gt_exr):
        create_temp_exr(gt_hdr_pixels, temp_gt_exr)

    basename = os.path.basename(test_base_path)
    error_map_path = os.path.join(flip_output_dir, f"{basename}_flip_map.png")
    os.makedirs(flip_output_dir, exist_ok=True)

    try:
        metrics = flip_evaluate(temp_gt_exr, temp_test_exr, "HDR")
        error_map_array = metrics[0]
        flip_val = float(metrics[1])
        
        error_map_8bit = np.clip(error_map_array * 255.0, 0, 255).astype(np.uint8)
        iio.imwrite(error_map_path, error_map_8bit)       
    except Exception as e:
        print(f"FLIP API Failed for {basename}: {e}")
        flip_val = -1.0
        
    if os.path.exists(temp_test_exr):
        os.remove(temp_test_exr)
    return flip_val

# --- Main function ---
def evaluate_scene(gt_base_path, test_base_path, results_dir):
    gt_hdr = iio.imread(f"{gt_base_path}.hdr").astype(np.float32)
    test_hdr = iio.imread(f"{test_base_path}.hdr").astype(np.float32)

    mse_val = calculate_mse(test_hdr, gt_hdr)
    rmse_val = calculate_rmse(test_hdr, gt_hdr)
    mrse_val = calculate_mrse(test_hdr, gt_hdr)
    rrmse_val = calculate_rrmse(test_hdr, gt_hdr)
    mae_val = calculate_mae(test_hdr, gt_hdr)
    psnr_val = calculate_psnr(test_hdr, gt_hdr)
    ssim_val = calculate_ssim(f"{test_base_path}.png", f"{gt_base_path}.png")
    
    # Save FLIP maps in the new Results structure
    flip_dir = os.path.join(results_dir, "FLIP")
    flip_val = run_flip(test_hdr, gt_hdr, test_base_path, gt_base_path, flip_dir)

    # --- Robust Naming Parsing ---
    # Expected format: scene_method1_method2_128spp
    filename = os.path.basename(test_base_path)
    parts = filename.split('_')
    
    scene_name = parts[0]
    
    # Extract only the numbers from the last part (handles "128", "128spp", "SPP128", etc.)
    spp_match = re.search(r'\d+', parts[-1])
    spp = int(spp_match.group()) if spp_match else 0
    
    # Method is everything between scene name and SPP
    method = "_".join(parts[1:-1])

    row = {
        "Scene": scene_name,
        "Method": method,
        "SPP": spp,
        "MSE": mse_val,
        "RMSE": rmse_val,
        "MRSE": mrse_val,
        "RRMSE": rrmse_val,
        "MAE": mae_val,
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "FLIP": flip_val 
    }
    return row

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    
    # Base directory where all evaluation outputs will go
    master_results_dir = os.path.join(script_dir, "EvaluationResults")

    # --- Configuration for Multiple Scenes ---
    # Add Dining Room and Classroom here after implementing Environment Map for Photon Map
    SCENES_CONFIG = {
        "CornellBox": {
            "gt_base": os.path.join(repo_root, "Images", "GroundTruths", "CornellBoxPathTrace8192SPP"),
            "tests": [
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "CornellBoxResults", "cornell-box_path_guide_64spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "CornellBoxResults", "cornell-box_path_trace_64spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "CornellBoxResults", "cornell-box_photon_map_64spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "CornellBoxResults", "cornell-box_path_guide_128spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "CornellBoxResults", "cornell-box_path_trace_128spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "CornellBoxResults", "cornell-box_photon_map_128spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "CornellBoxResults", "cornell-box_path_guide_256spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "CornellBoxResults", "cornell-box_path_trace_256spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "CornellBoxResults", "cornell-box_photon_map_256spp")
            ]
        },
        "Kitchen": {
            "gt_base": os.path.join(repo_root, "Images", "GroundTruths", "KitchenPathTrace8192SPP"),
            "tests": [
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "KitchenResults", "kitchen_path_guide_64spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "KitchenResults", "kitchen_path_trace_64spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "KitchenResults", "kitchen_photon_map_64spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "KitchenResults", "kitchen_path_guide_128spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "KitchenResults", "kitchen_path_trace_128spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "KitchenResults", "kitchen_photon_map_128spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "KitchenResults", "kitchen_path_guide_256spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "KitchenResults", "kitchen_path_trace_256spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "KitchenResults", "kitchen_photon_map_256spp")
            ]
        },
        "Staircase": {
            "gt_base": os.path.join(repo_root, "Images", "GroundTruths", "StaircasePathTrace8192SPP"),
            "tests": [
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "StaircaseResults", "staircase_path_guide_64spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "StaircaseResults", "staircase_path_trace_64spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "StaircaseResults", "staircase_photon_map_64spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "StaircaseResults", "staircase_path_guide_128spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "StaircaseResults", "staircase_path_trace_128spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "StaircaseResults", "staircase_photon_map_128spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "StaircaseResults", "staircase_path_guide_256spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "StaircaseResults", "staircase_path_trace_256spp"),
                os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "StaircaseResults", "staircase_photon_map_256spp")
            ]
        }
    }

    # Iterate through every scene in the config
    for scene_id, config in SCENES_CONFIG.items():
        gt_base = config["gt_base"]
        test_bases = config["tests"]
        
        # Skip if no tests defined
        if not test_bases:
            continue

        print(f"--- Evaluating Scene: {scene_id} ---")
        
        # Create a dedicated directory for this scene's results
        scene_results_dir = os.path.join(master_results_dir, scene_id)
        os.makedirs(scene_results_dir, exist_ok=True)

        scene_results = []
        for test in test_bases:
            if not os.path.exists(f"{test}.hdr"):
                print(f"File not found, skipping: {test}.hdr")
                continue
            
            print(f"  Processing: {os.path.basename(test)}")
            row_data = evaluate_scene(gt_base, test, scene_results_dir)
            scene_results.append(row_data)

        # Save CSV for this specific scene
        if scene_results:
            df = pd.DataFrame(scene_results)
            csv_path = os.path.join(scene_results_dir, f"{scene_id}_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved {scene_id} results to: {csv_path}\n")
            print(df)
            print("-" * 50)
        
        # Cleanup the Ground Truth temp EXR file for this scene
        temp_gt_exr = f"{gt_base}_TEMP.exr"
        if os.path.exists(temp_gt_exr):
            os.remove(temp_gt_exr)
            
    print("Evaluation Complete!")