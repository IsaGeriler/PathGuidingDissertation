# --- Import the necessary libraries for the evaluation script ---
import os
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
    # Mean Square Error
    return np.mean((test_hdr - gt_hdr) ** 2)

def calculate_rmse(test_hdr, gt_hdr):
    # Relative Mean Square Error (just square root of MSE)
    return np.sqrt(calculate_mse(test_hdr, gt_hdr))

def calculate_mrse(test_hdr, gt_hdr, epsilon=1e-2):
    # Standart Relative MSE for Rendering
    numerator = (test_hdr - gt_hdr) ** 2
    denominator = gt_hdr ** 2 + epsilon
    return np.mean(numerator / denominator)

def calculate_rrmse(test_hdr, gt_hdr, epsilon=1e-2):
    # Square Root of MRSE
    return np.sqrt(calculate_mrse(test_hdr, gt_hdr, epsilon))

def calculate_mae(test_hdr, gt_hdr):
    # Mean Absolute Error (L1 Loss)
    # Using this as MC Rendering may yield fireflies
    # MAE is less sensitive to this, compared to MSE
    return np.mean(np.abs(test_hdr - gt_hdr))

def calculate_psnr(test_hdr, gt_hdr):
    # Peak Signal to Noise Ratio
    mse = calculate_mse(test_hdr, gt_hdr)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(gt_hdr)
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim(test_png_filepath, gt_png_filepath):
    # Structural Similarity (Human Perception)
    test_ldr = cv2.imread(test_png_filepath, cv2.IMREAD_GRAYSCALE)
    gt_ldr = cv2.imread(gt_png_filepath, cv2.IMREAD_GRAYSCALE)

    if test_ldr is None or gt_ldr is None:
        print(f"WARNING: Image not found for {test_png_filepath} or {gt_png_filepath}")
        return -1.0
    return ssim(gt_ldr, test_ldr, data_range=255)

# --- FLIP and EXR Conversion Helpers ---
def create_temp_exr(hdr_pixels, output_path):
    # imageio loads the image as RGB, OpenCV saves as BGR.
    # We convert it to ensure colors are correct.
    bgr_pixels = cv2.cvtColor(hdr_pixels, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr_pixels)

def run_flip(test_hdr_pixels, gt_hdr_pixels, test_base_path, gt_base_path):
    # Setup paths for temporary EXR files
    temp_test_exr = f"{test_base_path}_TEMP.exr"
    temp_gt_exr = f"{gt_base_path}_TEMP.exr"
    
    # Save the EXR files
    create_temp_exr(test_hdr_pixels, temp_test_exr)
    if not os.path.exists(temp_gt_exr):
        create_temp_exr(gt_hdr_pixels, temp_gt_exr)

    # Figure out where to save the FLIP Error Map
    output_dir = os.path.dirname(test_base_path)
    basename = os.path.basename(test_base_path)
    flip_dir = os.path.join(output_dir, "FLIP")
    error_map_path = os.path.join(flip_dir, f"{basename}_flip_map.png")

    os.makedirs(flip_dir, exist_ok=True)

    try:
        # Call NVIDIA's FLIP API
        metrics = flip_evaluate(temp_gt_exr, temp_test_exr, "HDR")
        
        # metrics[0] -> The FLIP Error Map (RGB Array)
        # metrics[1] -> The Mean FLIP Score
        error_map_array = metrics[0]
        flip_val = float(metrics[1])
        
        # Convert the float array (0.0 to 1.0) into a standard 8-bit PNG (0 to 255)
        error_map_8bit = np.clip(error_map_array * 255.0, 0, 255).astype(np.uint8)
        
        # We use imageio (iio) because it natively understands RGB arrays
        iio.imwrite(error_map_path, error_map_8bit)       
    except Exception as e:
        print(f"FLIP API Failed: {e}")
        flip_val = -1.0
        
    # Clean up the temporary EXR files
    if os.path.exists(temp_test_exr):
        os.remove(temp_test_exr)
    return flip_val

# --- Main function ---
def evaluate_scene(test_name, gt_base_path, test_base_path):
    # Load HDR images
    gt_hdr = iio.imread(f"{gt_base_path}.hdr").astype(np.float32)
    test_hdr = iio.imread(f"{test_base_path}.hdr").astype(np.float32)

    # Compute metrics
    mse_val = calculate_mse(test_hdr, gt_hdr)
    rmse_val = calculate_rmse(test_hdr, gt_hdr)
    mrse_val = calculate_mrse(test_hdr, gt_hdr)
    rrmse_val = calculate_rrmse(test_hdr, gt_hdr)
    mae_val = calculate_mae(test_hdr, gt_hdr)
    psnr_val = calculate_psnr(test_hdr, gt_hdr)
    ssim_val = calculate_ssim(f"{test_base_path}.png", f"{gt_base_path}.png")
    
    # Pass the actual pixel data and paths to our updated FLIP function
    flip_val = run_flip(test_hdr, gt_hdr, test_base_path, gt_base_path)

    folder_name = os.path.basename(test_base_path)
    parts = folder_name.split('_')

    row = {
        "Scene": test_name,
        "Method": parts[1],
        "SPP": int(parts[2]),
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
    # Absolute path logic ensuring the script never gets lost
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    # Define the Ground Truth
    gt_base = os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "Kitchen_GT")

    # Define the test case directories
    test_bases = [
        os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "Kitchen_PathGuide_128"),
        os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "Kitchen_PathTrace_128"),
        os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "Kitchen_PathGuide_256"),
        os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "Kitchen_PathTrace_256"),
        os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "Kitchen_PathGuide_512"),
        os.path.join(repo_root, "Images", "PathTraceVsPathGuide", "Kitchen_PathTrace_512")
    ]

    # Populate the calculated results in a list
    results = []
    for test in test_bases:
        row_data = evaluate_scene("Kitchen", gt_base, test)
        results.append(row_data)

    # Save CSV using absolute path
    df = pd.DataFrame(results)
    csv_path = os.path.join(script_dir, "experiment_results.csv")
    df.to_csv(csv_path, index=False)
    
    # Cleanup the GT temp file
    temp_gt_exr = f"{gt_base}_TEMP.exr"
    if os.path.exists(temp_gt_exr):
        os.remove(temp_gt_exr)

    # Print the results in the terminal  
    print(f"\nSaved results to: {csv_path}\n")
    print(df)