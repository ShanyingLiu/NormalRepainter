from PIL import Image
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

def normalize_vectors_batch(vectors):
    """Vectorized normalization"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

def process_images(base_color_path, normal_map_path, output_path, max_clusters_per_color=5):
    """
    Ultra-fast normal map seam removal using vectorized operations
    """
    print("Loading images...")
    base_img = Image.open(base_color_path).convert("RGB")
    normal_img = Image.open(normal_map_path).convert("RGB")
    
    base_array = np.array(base_img)
    normal_array = np.array(normal_img).astype(np.float32)
    
    height, width = base_array.shape[:2]
    normals_xyz = (normal_array / 255.0 * 2.0) - 1.0
    
    print("Reshaping data...")
    # Reshape to work with flat arrays (much faster)
    base_flat = base_array.reshape(-1, 3)
    normals_flat = normals_xyz.reshape(-1, 3)
    
    # Normalize all normals at once (vectorized)
    normals_flat = normalize_vectors_batch(normals_flat)
    
    print("Finding unique colors...")
    # Find unique colors much faster using numpy
    base_view = base_flat.view(dtype=[('r', np.uint8), ('g', np.uint8), ('b', np.uint8)])
    base_1d = base_view.ravel()
    unique_colors, inverse_indices = np.unique(base_1d, return_inverse=True)
    
    print(f"Found {len(unique_colors)} unique colors, processing...")
    
    output_normals = normals_flat.copy()
    
    # Process each unique color
    for color_idx, unique_color in enumerate(unique_colors):
        # Find all pixels with this color (vectorized)
        mask = inverse_indices == color_idx
        color_pixels = np.where(mask)[0]
        
        if len(color_pixels) < 2:
            continue
            
        # Get normals for this color
        color_normals = normals_flat[color_pixels]
        
        if len(color_pixels) < 10:
            # For small groups, just average everything
            avg_normal = np.mean(color_normals, axis=0)
            avg_normal = avg_normal / np.linalg.norm(avg_normal)
            output_normals[color_pixels] = avg_normal
        else:
            # Use K-means clustering for larger groups (much faster than manual grouping)
            n_clusters = min(max_clusters_per_color, len(color_pixels) // 5)
            if n_clusters < 2:
                # Just average if too few pixels
                avg_normal = np.mean(color_normals, axis=0)
                avg_normal = avg_normal / np.linalg.norm(avg_normal)
                output_normals[color_pixels] = avg_normal
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
                cluster_labels = kmeans.fit_predict(color_normals)
                
                # Average normals within each cluster
                for cluster_id in range(n_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    if np.sum(cluster_mask) > 0:
                        cluster_pixels = color_pixels[cluster_mask]
                        cluster_normals = color_normals[cluster_mask]
                        avg_normal = np.mean(cluster_normals, axis=0)
                        avg_normal = avg_normal / np.linalg.norm(avg_normal)
                        output_normals[cluster_pixels] = avg_normal
    
    print("Converting back to image format...")
    # Reshape back to image dimensions
    output_normals = output_normals.reshape(height, width, 3)
    
    # Convert back to RGB [0,255]
    output_rgb = ((output_normals + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    
    Image.fromarray(output_rgb, mode="RGB").save(output_path)
    print(f"Processed normal map saved to {output_path}")

# Fallback version if sklearn is not available
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
        # Try the advanced version first
        process_images("base_color.png", "normal_map.png", "output_normal_map.png")
    except ImportError:
        print("sklearn not available, using simple version...")
        process_images_simple("base_color.png", "normal_map.png", "output_normal_map.png")