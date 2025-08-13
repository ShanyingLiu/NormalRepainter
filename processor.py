from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import cv2
import sys
import os

def normalize_vectors_batch(vectors):
    """Vectorized normalization for 3*N arrays."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

def merge_similar_colors(image, tolerance=1000):
    """
    Merge similar colors by rounding to nearest bin
    """
    img = image.astype(np.uint8)
    bins = np.floor_divide(img, tolerance) * tolerance
    bins = np.clip(bins, 0, 255).astype(np.uint8)
    flat_bins = bins.reshape(-1, bins.shape[-1])
    unique_colors, inverse_indices = np.unique(flat_bins, axis=0, return_inverse=True)
    return unique_colors, inverse_indices

def process_images(base_color_path, normal_map_path, output_path, max_clusters_per_color=5, tolerance=5):
    """
    Seam removal + smoothing
    """
    print(f"Loading images: {base_color_path}, {normal_map_path}")
    base_img = Image.open(base_color_path).convert("RGB")
    normal_img = Image.open(normal_map_path).convert("RGB")
    
    base_array = np.array(base_img)
    normal_array = np.array(normal_img).astype(np.float32)
    
    height, width = base_array.shape[:2]
    normals_xyz = (normal_array / 255.0 * 2.0) - 1.0
    
    base_flat = base_array.reshape(-1, 3)
    normals_flat = normals_xyz.reshape(-1, 3)
    normals_flat = normalize_vectors_batch(normals_flat)
    
    print(f"Merging colors within tolerance: {tolerance}")
    unique_colors, inverse_indices = merge_similar_colors(base_flat, tolerance)
    print(f"Found {len(unique_colors)} merged color groups.")
    
    output_normals = normals_flat.copy()
    
    for color_idx in range(len(unique_colors)):
        mask = inverse_indices == color_idx
        color_pixels = np.where(mask)[0]
        if len(color_pixels) < 2:
            continue
        
        color_normals = normals_flat[color_pixels]
        
        if len(color_pixels) < 10:
            avg_normal = np.mean(color_normals, axis=0)
            avg_normal /= np.linalg.norm(avg_normal)
            output_normals[color_pixels] = avg_normal
        else:
            n_clusters = min(max_clusters_per_color, len(color_pixels) // 5)
            if n_clusters < 2:
                avg_normal = np.mean(color_normals, axis=0)
                avg_normal /= np.linalg.norm(avg_normal)
                output_normals[color_pixels] = avg_normal
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
                cluster_labels = kmeans.fit_predict(color_normals)
                for cluster_id in range(n_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    if np.any(cluster_mask):
                        cluster_pixels = color_pixels[cluster_mask]
                        cluster_normals = color_normals[cluster_mask]
                        avg_normal = np.mean(cluster_normals, axis=0)
                        avg_normal /= np.linalg.norm(avg_normal)
                        output_normals[cluster_pixels] = avg_normal
    
    output_normals = output_normals.reshape(height, width, 3)
    output_rgb = ((output_normals + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    
    # Post-processing: smoothing
    normals_f = (output_rgb.astype(np.float32) / 255.0) * 2.0 - 1.0
    smoothed = cv2.GaussianBlur(normals_f, (5, 5), sigmaX=5)
    
    lengths = np.linalg.norm(smoothed, axis=2, keepdims=True)
    lengths[lengths == 0] = 1.0
    smoothed /= lengths
    
    output_rgb_smoothed = ((smoothed + 1.0) / 2.0 * 255.0).astype(np.uint8)
    
    Image.fromarray(output_rgb_smoothed, mode="RGB").save(output_path)
    print(f"Processed normal map saved to {output_path}")

def process_images_simple(base_color_path, normal_map_path, output_path):
    """
    Fallback without sklearn.
    """
    print("Using simple version (no clustering).")
    base_img = Image.open(base_color_path).convert("RGB")
    normal_img = Image.open(normal_map_path).convert("RGB")
    
    base_array = np.array(base_img)
    normal_array = np.array(normal_img).astype(np.float32)
    
    height, width = base_array.shape[:2]
    normals_xyz = (normal_array / 255.0 * 2.0) - 1.0
    
    base_flat = base_array.reshape(-1, 3)
    normals_flat = normals_xyz.reshape(-1, 3)
    normals_flat = normalize_vectors_batch(normals_flat)
    
    base_view = base_flat.view(dtype=[('r', np.uint8), ('g', np.uint8), ('b', np.uint8)])
    base_1d = base_view.ravel()
    unique_colors, inverse_indices = np.unique(base_1d, return_inverse=True)
    
    output_normals = normals_flat.copy()
    for color_idx in range(len(unique_colors)):
        mask = inverse_indices == color_idx
        color_pixels = np.where(mask)[0]
        if len(color_pixels) > 1:
            avg_normal = np.mean(normals_flat[color_pixels], axis=0)
            avg_normal /= np.linalg.norm(avg_normal)
            output_normals[color_pixels] = avg_normal
    
    output_normals = output_normals.reshape(height, width, 3)
    output_rgb = ((output_normals + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    
    Image.fromarray(output_rgb, mode="RGB").save(output_path)
    print(f"Processed normal map saved to {output_path}")

if __name__ == "__main__":
    tolerance = 25
    if len(sys.argv) < 4:
        print("Usage: python processor.py base_color.png normal_map.png output.png [tolerance]")
        sys.exit(1)
    
    base_color_path = sys.argv[1]
    normal_map_path = sys.argv[2]
    output_path = sys.argv[3]
    
    if len(sys.argv) > 4:
        tolerance = int(sys.argv[4])
    
    try:
        process_images(base_color_path, normal_map_path, output_path, tolerance=tolerance)
    except ImportError:
        process_images_simple(base_color_path, normal_map_path, output_path)
