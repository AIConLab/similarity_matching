import sys
import os

# Get the path to the package directory
package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, package_dir)

# Add the path to the accelerated_features directory
accelerated_features_dir = os.path.join(package_dir, 'libs', 'accelerated_features')
sys.path.insert(0, accelerated_features_dir)

# Add the modules directory to the path
modules_dir = os.path.join(accelerated_features_dir, 'modules')
sys.path.insert(0, modules_dir)

# Now you can import XFeat
from modules.xfeat import XFeat
import numpy as np
import torch
import cv2
import argparse

xfeat = XFeat()

def compare_images(target_image, scene_image):
    # Convert numpy arrays to torch tensors
    target_tensor = torch.from_numpy(target_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    scene_tensor = torch.from_numpy(scene_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # Detect and compute features
    target_output = xfeat.detectAndCompute(target_tensor, top_k=4096)[0]
    scene_output = xfeat.detectAndCompute(scene_tensor, top_k=4096)[0]
    
    # Match the two images
    mkpts_0, mkpts_1 = xfeat.match_xfeat(target_tensor, scene_tensor)
    
    # Convert keypoints to numpy arrays for OpenCV operations
    kpts_0 = target_output['keypoints'].cpu().numpy()
    kpts_1 = scene_output['keypoints'].cpu().numpy()
    
    # Check if mkpts_0 and mkpts_1 are already numpy arrays
    if not isinstance(mkpts_0, np.ndarray):
        mkpts_0_np = mkpts_0.cpu().numpy()
        mkpts_1_np = mkpts_1.cpu().numpy()
    else:
        mkpts_0_np = mkpts_0
        mkpts_1_np = mkpts_1
    
    # Ensure we have matching points before trying to estimate homography
    if len(mkpts_0_np) < 4 or len(mkpts_1_np) < 4:
        print("Not enough matching points to estimate homography")
        return 0.0  # Return lowest similarity score
    
    # Estimate homography
    H, mask = cv2.findHomography(mkpts_0_np, mkpts_1_np, cv2.RANSAC, 5.0)
    
    # Calculate metrics
    """
        - Number of matches: This is the number of feature points that were successfully matched between the two images.
        - Number of inliers: These are the matched points that are consistent with the estimated homography (transformation between the images). For identical images, this should be equal to the number of matches.
        - Inlier ratio: This is the ratio of inliers to total matches. A value of 1.0 means all matches are consistent with the homography, which is expected for identical images.
        - Feature ratio: This is the ratio of the number of features in the image with fewer features to the image with more features. For identical images, this should be 1.0.
        - Number of keypoints in target image: This is the number of feature points detected in the target image.
    """
    # Calculate metrics
    num_matches = len(mkpts_0)

    num_inliers = np.sum(mask) if mask is not None else 0
    inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0
    feature_ratio = min(len(kpts_0), len(kpts_1)) / max(len(kpts_0), len(kpts_1))
    
    max_features = 4096  # This should match the top_k value in detectAndCompute
    match_ratio = num_matches / max_features

    # Debug output
    print(f"Number of matches: {num_matches}")
    print(f"Number of inliers: {num_inliers}")
    print(f"Inlier ratio: {inlier_ratio}")
    print(f"Feature ratio: {feature_ratio}")
    print(f"Match ratio: {match_ratio}")
    print(f"Number of keypoints in target image: {len(kpts_0)}")
    print(f"Number of keypoints in scene image: {len(kpts_1)}")

    # Check for identical images
    if (inlier_ratio == 1.0 and 
        feature_ratio == 1.0 and 
        len(kpts_0) == len(kpts_1) == num_matches):
        similarity = 1.0
    else:
        # Calculate similarity for non-identical images
        similarity = min(1.0, (0.4 * inlier_ratio + 0.3 * feature_ratio + 0.3 * match_ratio) * (1 + match_ratio))

    return similarity

def run_standard_mode():
    while True:
        # Read image dimensions
        line = sys.stdin.buffer.readline().strip()
        if not line:
            break
        h1, w1, c1, h2, w2, c2 = map(int, line.split())
        
        # Read image data
        target_data = sys.stdin.buffer.read(h1 * w1 * c1)
        scene_data = sys.stdin.buffer.read(h2 * w2 * c2)
        
        if not target_data or not scene_data:
            break
        
        # Convert to numpy arrays and reshape
        target_image = np.frombuffer(target_data, dtype=np.uint8).reshape((h1, w1, c1))
        scene_image = np.frombuffer(scene_data, dtype=np.uint8).reshape((h2, w2, c2))
        
        # Compare images and get similarity
        similarity = compare_images(target_image, scene_image)
        
        # Write result to stdout
        sys.stdout.buffer.write(f"{similarity:.6f}\n".encode('utf-8'))
        sys.stdout.buffer.flush()

def run_debug_mode(target_img_path, scene_img_path):
    target_image = cv2.imread(target_img_path)
    scene_image = cv2.imread(scene_img_path)

    if target_image is None:
        print(f"ERROR: Could not read target image from {target_img_path}")
        return
    if scene_image is None:
        print(f"ERROR: Could not read scene image from {scene_img_path}")
        return

    similarity = compare_images(target_image, scene_image)
    print(f"Similarity score: {similarity:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XFeat Image Comparison')
    parser.add_argument('--target_img_path', type=str, help='Path to the target image')
    parser.add_argument('--scene_img_path', type=str, help='Path to the scene image')
    args = parser.parse_args()

    if args.target_img_path and args.scene_img_path:
        run_debug_mode(args.target_img_path, args.scene_img_path)
    else:
        run_standard_mode()