from PIL import Image
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import cv2

def normalize_vectors_batch(vectors):
    """Vectorized normalization"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

def merge_similar_colors(image, tolerance=1000):
    """
    Merge similar colors quickly by rounding to nearest bin
    """
    img = image.astype(np.uint8)

    # Quantize colors by binning
    bins = np.floor_divide(img, tolerance) * tolerance
    bins = np.clip(bins, 0, 255).astype(np.uint8)

    # Flatten for unique lookup
    flat_bins = bins.reshape(-1, bins.shape[-1])

    # Get unique colors & mapping
    unique_colors, inverse_indices = np.unique(flat_bins, axis=0, return_inverse=True)

    return unique_colors, inverse_indices


def process_images(base_color_path, normal_map_path, output_path, max_clusters_per_color=5, tolerance=5):
    """
    Seam removal using vectorized operations
    """
    print("Loading images...")
    base_img = Image.open(base_color_path).convert("RGB")
    normal_img = Image.open(normal_map_path).convert("RGB")
    
    base_array = np.array(base_img)
    normal_array = np.array(normal_img).astype(np.float32)
    
    height, width = base_array.shape[:2]
    normals_xyz = (normal_array / 255.0 * 2.0) - 1.0
    
    # Reshape for flat arrays
    base_flat = base_array.reshape(-1, 3)
    normals_flat = normals_xyz.reshape(-1, 3)
    
    # Normalize all normals at once
    normals_flat = normalize_vectors_batch(normals_flat)
    
    print("Finding unique colors (with tolerance)...")
    # Merge colors that are within the tolerance
    unique_colors, mapping = merge_similar_colors(base_flat, tolerance)
    inverse_indices = mapping  # Now maps each pixel to its merged color index
    
    print(f"Found {len(unique_colors)} merged color groups...")
    
    output_normals = normals_flat.copy()
    
    # itr through each merged color
    for color_idx, unique_color in enumerate(unique_colors):
        # Find all pixels with this merged color
        mask = inverse_indices == color_idx
        color_pixels = np.where(mask)[0]
        
        if len(color_pixels) < 2:
            continue
            
        # Get normals for this color
        color_normals = normals_flat[color_pixels]
        
        if len(color_pixels) < 10: # disregard small groups
            avg_normal = np.mean(color_normals, axis=0)
            avg_normal = avg_normal / np.linalg.norm(avg_normal)
            output_normals[color_pixels] = avg_normal
        else:
            # K-means clustering
            n_clusters = min(max_clusters_per_color, len(color_pixels) // 5)
            if n_clusters < 2:
                # Just average if too few pixels
                avg_normal = np.mean(color_normals, axis=0)
                avg_normal = avg_normal / np.linalg.norm(avg_normal)
                output_normals[color_pixels] = avg_normal
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
                cluster_labels = kmeans.fit_predict(color_normals)
                
                for cluster_id in range(n_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    if np.sum(cluster_mask) > 0:
                        cluster_pixels = color_pixels[cluster_mask]
                        cluster_normals = color_normals[cluster_mask]
                        avg_normal = np.mean(cluster_normals, axis=0)
                        avg_normal = avg_normal / np.linalg.norm(avg_normal)
                        output_normals[cluster_pixels] = avg_normal
    
    print("Converting back to image format...")
    output_normals = output_normals.reshape(height, width, 3)
    
    output_rgb = ((output_normals + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    


    #post processing
    normals_f = (output_rgb.astype(np.float32) / 255.0) * 2.0 - 1.0

    #smoothed = cv2.bilateralFilter(normals_f, d=3, sigmaColor=0.1, sigmaSpace=2)

    smoothed = cv2.GaussianBlur(normals_f, (5, 5), sigmaX=50)

    # Renormalize normals to unit length
    lengths = np.linalg.norm(smoothed, axis=2, keepdims=True)
    lengths[lengths == 0] = 1.0
    smoothed /= lengths

    # Convert back to RGB in [0,255]
    output_rgb_smoothed = ((smoothed + 1.0) / 2.0 * 255.0).astype(np.uint8)

    Image.fromarray(output_rgb, mode="RGB").save(output_path)
    print(f"Processed normal map saved to {output_path}")

# Fallback if sklearn is not available
def process_images_simple(base_color_path, normal_map_path, output_path):
    """
    Simple version without sklearn - just averages all same-colored pixels
    """
    print("Loading images...")
    base_img = Image.open(base_color_path).convert("RGB")
    normal_img = Image.open(normal_map_path).convert("RGB")
    
    base_array = np.array(base_img)
    normal_array = np.array(normal_img).astype(np.float32)
    
    height, width = base_array.shape[:2]
    normals_xyz = (normal_array / 255.0 * 2.0) - 1.0
    
    print("Processing...")
    # Reshape for vectorized operations
    base_flat = base_array.reshape(-1, 3)
    normals_flat = normals_xyz.reshape(-1, 3)
    normals_flat = normalize_vectors_batch(normals_flat)
    
    # Create color index for grouping
    base_view = base_flat.view(dtype=[('r', np.uint8), ('g', np.uint8), ('b', np.uint8)])
    base_1d = base_view.ravel()
    unique_colors, inverse_indices = np.unique(base_1d, return_inverse=True)
    
    output_normals = normals_flat.copy()
    
    # Average normals for each color group
    for color_idx in range(len(unique_colors)):
        mask = inverse_indices == color_idx
        color_pixels = np.where(mask)[0]
        
        if len(color_pixels) > 1:
            avg_normal = np.mean(normals_flat[color_pixels], axis=0)
            avg_normal = avg_normal / np.linalg.norm(avg_normal)
            output_normals[color_pixels] = avg_normal
    
    # Reshape and save
    output_normals = output_normals.reshape(height, width, 3)
    output_rgb = ((output_normals + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(output_rgb, mode="RGB").save(output_path)
    print(f"Processed normal map saved to {output_path}")

if __name__ == "__main__":
    try:
        print("Begin repainting normal map with tolerance {tolerance}")
        process_images("base_color.png", "normal_map.png", "output_normal_map.png", tolerance=25)
    except ImportError:
        print("sklearn not available, using simple version...")
        process_images_simple("base_color.png", "normal_map.png", "output_normal_map.png")
