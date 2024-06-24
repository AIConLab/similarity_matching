from ..libs.accelerated_features.modules.xfeat import XFeat
import sys
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
    mkpts_0_np = mkpts_0.cpu().numpy()
    mkpts_1_np = mkpts_1.cpu().numpy()

    # Estimate homography
    H, mask = cv2.findHomography(mkpts_0_np, mkpts_1_np, cv2.RANSAC, 5.0)
    
    # Calculate metrics
    num_matches = len(mkpts_0)
    num_inliers = np.sum(mask)
    inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0
    
    feature_ratio = min(len(kpts_0), len(kpts_1)) / max(len(kpts_0), len(kpts_1))
    
    # Combine metrics (you can adjust the weights)
    similarity = 0.5 * inlier_ratio + 0.3 * feature_ratio + 0.2 * (num_matches / 4096)
    
    return similarity

def run_standard_mode():
    while True:
        # Read image dimensions
        line = sys.stdin.readline().strip()
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
        print(f"{similarity:.6f}")
        sys.stdout.flush()

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