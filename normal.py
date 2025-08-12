from PIL import Image
import numpy as np
from collections import deque

def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

def average_direction(vectors):
    avg = np.mean(vectors, axis=0)
    norm = np.linalg.norm(avg)
    if norm == 0:
        return np.array([0, 0, 1], dtype=np.float32)
    return avg / norm

def get_neighbors(y, x, height, width):
    neighbors = []
    for ny, nx in [(y-1,x),(y+1,x),(y,x-1),(y,x+1)]:
        if 0 <= ny < height and 0 <= nx < width:
            neighbors.append((ny, nx))
    return neighbors

def process_images(base_color_path, normal_map_path, output_path):
    base_img = Image.open(base_color_path).convert("RGB")
    normal_img = Image.open(normal_map_path).convert("RGB")

    base_array = np.array(base_img)
    normal_array = np.array(normal_img).astype(np.float32) / 255.0

    height, width, _ = base_array.shape
    normals_xyz = (normal_array * 2.0) - 1.0
    new_normals = normals_xyz.copy()

    visited = np.zeros((height, width), dtype=bool)

    for y in range(height):
        for x in range(width):
            if visited[y, x]:
                continue

            color = tuple(base_array[y, x])
            # BFS to find all connected pixels with the same color
            queue = deque()
            queue.append((y, x))
            connected_pixels = []

            while queue:
                cy, cx = queue.popleft()
                if visited[cy, cx]:
                    continue
                if tuple(base_array[cy, cx]) != color:
                    continue

                visited[cy, cx] = True
                connected_pixels.append((cy, cx))

                for ny, nx in get_neighbors(cy, cx, height, width):
                    if not visited[ny, nx] and tuple(base_array[ny, nx]) == color:
                        queue.append((ny, nx))

            # Average normals for this connected region
            region_normals = np.array([normals_xyz[py, px] for py, px in connected_pixels])
            region_normals = normalize_vectors(region_normals)
            avg_normal = average_direction(region_normals)

            # Apply averaged normal to all pixels in this region
            for py, px in connected_pixels:
                new_normals[py, px] = avg_normal

    # Convert back to [0,255] RGB space and save
    output_rgb = ((new_normals + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(output_rgb, mode="RGB").save(output_path)
    print(f"Processed normal map saved to {output_path}")

# Example usage:
process_images("base_color.png", "normal_map.png", "output_normal_map.png")
