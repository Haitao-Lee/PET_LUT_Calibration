import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# # Generate random centers
# num_clusters = 256
# centers = np.random.randint(0, 256, size=(num_clusters, 2))

# # Create a blank clustering result image
# clustered_image = np.zeros((256, 256))

# # Assign each pixel to a cluster
# for i in range(256):
#     for j in range(256):
#         pixel = np.array([i, j])
#         distances = np.linalg.norm(centers - pixel, axis=1)
#         cluster_index = np.argmin(distances)
#         clustered_image[i, j] = cluster_index

# # Compute cluster boundaries
# def compute_cluster_boundaries(clustered_image):
#     boundaries = np.zeros_like(clustered_image, dtype=bool)
#     for i in range(1, clustered_image.shape[0]):
#         for j in range(1, clustered_image.shape[1]):
#             if clustered_image[i, j] != clustered_image[i-1, j] or clustered_image[i, j] != clustered_image[i, j-1]:
#                 boundaries[i, j] = True
#     return boundaries

# # Get cluster boundaries
# boundaries = compute_cluster_boundaries(clustered_image)

# # Display the clustering result and boundaries
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(clustered_image, cmap='jet')
# plt.axis('off')
# plt.title('Clustered Image')

# plt.subplot(1, 2, 2)
# plt.imshow(boundaries, cmap='gray')
# plt.axis('off')
# plt.title('Cluster Boundaries')

# plt.tight_layout()
# plt.show()


import numpy as np
import random
from scipy.spatial import cKDTree, distance

# Retry without voronoi_labels (not available in current skimage)
# We'll skip the Voronoi-based implementation for now

import numpy as np
import random
from scipy.spatial import KDTree
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt


def replace_nearest_point(points, x, y):
    """
    Find the point in the 256x2 array closest to the specified (x, y) and replace it with (x, y).
    
    Parameters:
    points -- 2D numpy array of shape (256, 2), coordinates of the points (x, y)
    x -- x coordinate of the specified point
    y -- y coordinate of the specified point
    
    Returns:
    new_points -- the new array after replacement
    """
    if points.shape[1] != 2:
        raise ValueError("points 必须是形状为 (N, 2) 的二维数组")
    
    # Compute the Euclidean distance from each point to (x, y)
    distances = np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2)
    
    # Find the index of the closest point
    nearest_index = np.argmin(distances)
    
    # Create a new array and replace the nearest point
    new_points = points.copy()
    new_points[nearest_index] = [x, y]
    
    return new_points


def generate_cluster_col(n_clusters):
    return np.array(random.sample(range(256), n_clusters))


def get_boundaries(label_img):
    boundaries = np.zeros_like(label_img, dtype=bool)
    boundaries[1:, :] |= (label_img[1:, :] != label_img[:-1, :])
    boundaries[:, 1:] |= (label_img[:, 1:] != label_img[:, :-1])
    return boundaries


def method_kdtree(img, pts):
    H, W = img.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pixels = np.stack((yy.ravel(), xx.ravel()), axis=-1)
    tree = KDTree(pts)
    _, cluster_indices = tree.query(pixels)
    cluster_col = generate_cluster_col(pts.shape[0])
    clustered_flat = cluster_col[cluster_indices]
    label_img = clustered_flat.reshape(H, W)
    boundaries = get_boundaries(label_img)
    return label_img, boundaries


def method_distance_transform(img, pts):
    H, W = img.shape
    mask = np.zeros((H, W, pts.shape[0]), dtype=bool)
    for i, (y, x) in enumerate(pts):
        if 0 <= y < H and 0 <= x < W:
            mask[int(y), int(x), i] = True

    dist_map = np.stack([distance_transform_edt(~m) for m in np.rollaxis(mask, 2)], axis=-1)
    cluster_indices = np.argmin(dist_map, axis=-1)
    cluster_col = generate_cluster_col(pts.shape[0])
    label_img = cluster_col[cluster_indices]
    boundaries = get_boundaries(label_img)
    return label_img, boundaries


def add_random_noise_to_points(points, n, image_shape=(256, 256)):
    """
    Add random ±n noise to a set of 2D coordinates.

    Parameters:
    points (ndarray): Original coordinates of shape (256, 2)
    n (int): Maximum absolute noise to be added in each direction
    image_shape (tuple): Size of the image (height, width) to ensure points remain valid

    Returns:
    noisy_points (ndarray): Coordinates after adding random noise, shape (256, 2)
    """
    # Generate random noise in range [-n, n] for each point
    noise = np.random.normal(loc=0.0, scale=n / 2.0, size=points.shape).astype(int)

    # Add noise to original points
    noisy_points = points + noise

    # Clip the points to stay within image boundaries
    noisy_points[:, 0] = np.clip(noisy_points[:, 0], 0, image_shape[0] - 1)
    noisy_points[:, 1] = np.clip(noisy_points[:, 1], 0, image_shape[1] - 1)

    return noisy_points


def clusterAndBoundary(img, pts, color=None):
    H, W = img.shape
    if color is None:
        color = list(range(1, 257)) 
        random.shuffle(color)   
    markers = np.zeros((H, W), dtype=int)
    for i, (y, x) in enumerate(pts):
        if 0 <= y < H and 0 <= x < W:
            markers[int(y), int(x)] = color[i]

    elevation_map = np.ones_like(img)
    label_img = watershed(elevation_map, markers)
    boundaries = get_boundaries(label_img)
    return label_img, boundaries


def method_potential_field(img, pts, sigma=20.0):
    H, W = img.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pot = np.zeros((H, W, pts.shape[0]))

    for i, (y, x) in enumerate(pts):
        pot[:, :, i] = np.exp(-((yy - y)**2 + (xx - x)**2) / (2 * sigma**2))

    cluster_indices = np.argmax(pot, axis=-1)
    cluster_col = generate_cluster_col(pts.shape[0])
    label_img = cluster_col[cluster_indices]
    boundaries = get_boundaries(label_img)
    return label_img, boundaries


# # Dummy input
# img_dummy = np.zeros((256, 256))
# np.random.seed(0)
# pts_dummy = np.random.randint(0, 256, size=(256, 2))

# # Apply available methods
# results = {
#     'kdtree': method_kdtree(img_dummy, pts_dummy),
#     'distance_transform': method_distance_transform(img_dummy, pts_dummy),
#     'watershed': method_watershed(img_dummy, pts_dummy),
#     'potential_field': method_potential_field(img_dummy, pts_dummy)
# }

# results.keys()


# def clusterAndBoundary(img, pts):
#     """
#     Perform clustering based on given centers and identify boundary pixels.

#     Parameters:
#     img (ndarray): Input image array (used only for shape reference).
#     pts (ndarray): Array of cluster centers, shape (n_centers, 2).

#     Returns:
#     clustered_image (ndarray): Image where each pixel is assigned a cluster value.
#     boundaries (ndarray): Boolean array indicating boundary pixels.
#     """
#     # Initialize cluster centers and create a blank clustered image
#     centers = pts
#     clustered_image = np.zeros(img.shape)

#     # Generate random cluster labels (up to 256 unique colors)
#     cluster_col = random.sample(range(centers.shape[0]), centers.shape[0])

#     # Assign each pixel to the nearest cluster center
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             pixel = np.array([i, j])
#             # Compute distances to all cluster centers
#             distances = np.linalg.norm(centers - pixel, axis=1)
#             # Assign the pixel to the nearest cluster
#             cluster_index = np.argmin(distances)
#             clustered_image[i, j] = cluster_col[cluster_index]

#     # Identify boundary pixels
#     boundaries = np.zeros_like(clustered_image, dtype=bool)
#     for i in range(1, clustered_image.shape[0]):
#         for j in range(1, clustered_image.shape[1]):
#             # Check if the current pixel's cluster differs from its neighbors
#             if (clustered_image[i, j] != clustered_image[i-1, j] or 
#                 clustered_image[i, j] != clustered_image[i, j-1]):
#                 boundaries[i, j] = True
#     return clustered_image, boundaries


# def clusterAndBoundary(img, pts):
#     H, W = img.shape
#     n_centers = pts.shape[0]

#     # Generate pixel coordinates grid: shape (H*W, 2)
#     yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
#     pixels = np.stack((yy.ravel(), xx.ravel()), axis=-1)  # shape: (H*W, 2)

#     # Compute distances from each pixel to all centers: shape (H*W, n_centers)
#     dists = np.linalg.norm(pixels[:, None, :] - pts[None, :, :], axis=2)

#     # Find closest center index for each pixel
#     cluster_indices = np.argmin(dists, axis=1)

#     # Map cluster indices to cluster_col (randomized labels)
#     cluster_col = np.array(random.sample(range(256), n_centers))  # optional random label
#     clustered_flat = cluster_col[cluster_indices]

#     # Reshape to original image size
#     clustered_image = clustered_flat.reshape(H, W)

#     # Compute boundaries using convolution-style difference (fast)
#     boundaries = np.zeros_like(clustered_image, dtype=bool)
#     boundaries[1:, :] |= (clustered_image[1:, :] != clustered_image[:-1, :])
#     boundaries[:, 1:] |= (clustered_image[:, 1:] != clustered_image[:, :-1])

#     return clustered_image, boundaries




def fine_tune_position_by_maximization(original_img, clustered_img, peaks):
    """
    Fine-tune the positions of peaks by maximizing their corresponding values 
    in the original image based on cluster classification.
    
    Parameters:
        original_img (numpy.ndarray): The original image (e.g., intensity map) 
                                      from which values are extracted.
        clustered_img (numpy.ndarray): An image where each pixel is labeled with a cluster ID.
        peaks (numpy.ndarray): An array of shape (n, 2) representing the (row, col) coordinates of peaks.
    
    Returns:
        numpy.ndarray: An updated array of peaks with fine-tuned positions.
    
    Process:
        - For each peak, find all pixels in the same cluster.
        - Identify the pixel in the cluster with the maximum value in the original image.
        - Update the peak's position to this maximum-value pixel.
    """
    for i in range(peaks.shape[0]):
        # Get the cluster label at the current peak's position
        cluster_label = clustered_img[peaks[i][0], peaks[i][1]]
        
        # Find all indices belonging to the same cluster
        cluster_indices = np.argwhere(clustered_img == cluster_label)
        
        # Extract corresponding values from the original image
        cluster_values = original_img[cluster_indices[:, 0], cluster_indices[:, 1]]
        
        # Find the index of the maximum value in the cluster
        max_index = np.argmax(cluster_values)
        
        # Update the peak to the position of the maximum value
        peaks[i, :] = cluster_indices[max_index]
    
    return peaks


from scipy.ndimage import distance_transform_edt

def fine_tune_position_by_average_center(original_img, clustered_img, peaks):
    """
    Refine peak positions by identifying the most significant point within each cluster 
    using intensity values and distance transformation.

    Parameters:
        original_img (numpy.ndarray): 2D array representing the original image intensities.
        clustered_img (numpy.ndarray): 2D array representing clustered regions (e.g., labeled clusters).
        peaks (numpy.ndarray): An (n, 2) array representing the initial peak coordinates.

    Returns:
        numpy.ndarray: Updated peak positions refined based on intensity and distance transformation.
    """
    for i in range(peaks.shape[0]):
        # Step 1: Get the cluster label for the current peak position
        cluster_label = clustered_img[peaks[i][0], peaks[i][1]]
        
        # Step 2: Find all indices belonging to the same cluster
        cluster_indices = np.argwhere(clustered_img == cluster_label)
        
        if len(cluster_indices) == 0:
            continue  # Skip if no valid cluster indices are found
        
        # Step 3: Extract intensity values for the cluster from the original image
        cluster_values = original_img[cluster_indices[:, 0], cluster_indices[:, 1]]
        
        # Step 4: Calculate the average intensity value of the cluster
        average_value = np.mean(cluster_values)
        
        # Step 5: Create a binary mask for pixels exceeding the average intensity
        mask = np.zeros_like(original_img, dtype=bool)
        mask[cluster_indices[:, 0], cluster_indices[:, 1]] = original_img[cluster_indices[:, 0], cluster_indices[:, 1]]
        mask[mask >= average_value] = 1
        # Step 6: Perform distance transformation on the mask
        distance_map = distance_transform_edt(mask)
        
        # Step 7: Find the maximum distance point (most central point in the cluster)
        max_idx = np.unravel_index(np.argmax(distance_map), distance_map.shape)
        
        # Step 8: Update the peak position with the new refined coordinates
        peaks[i, :] = max_idx
    
    return peaks

def largest_connected_island(coords, connectivity=4):
    """
    Find the largest connected group (island) of 2D coordinates.

    Parameters:
        coords (numpy.ndarray): An (n, 2) array of 2D coordinates.
        connectivity (int): Connectivity type, 4 or 8. 
                            - 4: Only horizontal and vertical neighbors are connected.
                            - 8: Horizontal, vertical, and diagonal neighbors are connected.

    Returns:
        numpy.ndarray: An array of coordinates belonging to the largest connected island.
    """
    if len(coords) == 0:
        return np.array([])

    # Convert coordinates to a set for faster lookup
    coord_set = set(map(tuple, coords))
    visited = set()
    largest_island = []

    # Define neighbor directions based on connectivity
    if connectivity == 4:
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    elif connectivity == 8:
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]
    else:
        raise ValueError("Connectivity must be 4 or 8")

    def dfs(coord):
        """Depth-First Search to explore an island."""
        stack = [coord]
        island = []
        while stack:
            current = stack.pop()
            if current not in visited and current in coord_set:
                visited.add(current)
                island.append(current)
                for dx, dy in directions:
                    neighbor = (current[0] + dx, current[1] + dy)
                    if neighbor in coord_set and neighbor not in visited:
                        stack.append(neighbor)
        return island

    # Explore each coordinate to find connected components
    for coord in coord_set:
        if coord not in visited:
            island = dfs(coord)
            if len(island) > len(largest_island):
                largest_island = island

    return np.array(largest_island)


class GravityGuidedRegressionModule(nn.Module):
    def __init__(self, radius=1, max_iters=20, tol=1e-5):
        """
        A PyTorch module that simulates the iterative movement of points on depth maps
        towards local maxima within a defined neighborhood radius, with support for batch processing.

        Args:
            radius (int): Radius of the local neighborhood to search for the maximum.
            max_iters (int): Maximum number of iterations for the movement process.
            tol (float): Convergence tolerance. Movement stops when maximum displacement is below this value.
        """
        super(GravityGuidedRegressionModule, self).__init__()
        self.radius = radius
        self.max_iters = max_iters
        self.tol = tol


    @staticmethod
    def smooth_tensor(tensor, kernel_size=3):
        """
        Smooth the input tensor while keeping the same dimensions.
        :param tensor: Input tensor of shape (batch_size, channels, height, width).
        :param kernel_size: Size of the smoothing kernel (default: 3).
        :return: Smoothed tensor with the same dimensions as the input.
        """
        assert kernel_size % 2 == 1, "Kernel size must be odd to ensure symmetry"
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=tensor.device) / (kernel_size ** 2)
        smoothed_tensor = F.conv2d(tensor, kernel, padding=kernel_size // 2, groups=1)
        return smoothed_tensor

    @staticmethod
    def find_adjacent_col_groups(nums):
        """
        Find groups of integers in the list where elements are adjacent (difference is 1),
        and filter out groups with fewer than 2 elements.
        :param nums: List of integers.
        :return: List of lists containing grouped adjacent integers.
        """
        if not nums:
            return []

        # Sort the numbers to ensure sequential order
        nums = sorted(nums)

        groups = []
        current_group = [nums[0]]

        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1] + 1:  # Check if current number is adjacent to the previous
                current_group.append(nums[i])
            else:
                if len(current_group) > 1:  # Add group only if it has more than 1 element
                    groups.append(current_group)
                current_group = [nums[i]]  # Start a new group

        # Add the last group if it has more than 1 element
        if len(current_group) > 1:
            groups.append(current_group)

        return groups

    @staticmethod
    def find_adjacent_row_groups(nums):
        """
        Find groups of integers in the list where elements are separated by 16,
        and filter out groups with fewer than 2 elements.
        :param nums: List of integers.
        :return: List of lists containing grouped integers separated by 16.
        """
        if not nums:
            return []

        # Sort the numbers to ensure sequential order
        nums = sorted(nums)

        groups = []
        current_group = [nums[0]]

        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1] + 16:  # Check if current number is separated by 16 from the previous
                current_group.append(nums[i])
            else:
                if len(current_group) > 1:  # Add group only if it has more than 1 element
                    groups.append(current_group)
                current_group = [nums[i]]  # Start a new group

        # Add the last group if it has more than 1 element
        if len(current_group) > 1:
            groups.append(current_group)

        return groups


    @staticmethod
    def find_overlapping_indices(points, threshold=5):
        """
        Identify groups of overlapping points based on a distance threshold for each batch.

        Args:
            points (torch.Tensor): Tensor of shape (batch_size, num_points, 2) representing the coordinates.
            threshold (float): Distance threshold below which points are considered overlapping.

        Returns:
            list: A list of lists, where each inner list contains indices of overlapping points for each batch.
        """
        batch_size, num_points, _ = points.shape
        overlapping_groups = []

        for b in range(batch_size):
            # Compute pairwise distances
            distances = torch.cdist(points[b].float().unsqueeze(0), points[b].float().unsqueeze(0), p=2).squeeze(0)  # Shape: (num_points, num_points)

            # Identify overlapping points
            visited = set()
            batch_groups = []
            for i in range(num_points):
                if i in visited:
                    continue

                # Find all points within the threshold distance
                overlapping = (distances[i] < threshold).nonzero(as_tuple=True)[0].tolist()

                # Only consider groups with more than one point
                if len(overlapping) > 1:
                    batch_groups.append(overlapping)

                # Mark all points in the group as visited
                visited.update(overlapping)

            overlapping_groups.append(batch_groups)

        return overlapping_groups

    def process_overlap_points(self, start_points, end_points):
        """
        Resolve overlapping points by retaining only the nearest point to the maximum
        and resetting others to their original positions for each batch.

        Args:
            start_points (torch.Tensor): Original positions of the points, shape (batch_size, num_points, 2).
            end_points (torch.Tensor): Updated positions of the points after movement, shape (batch_size, num_points, 2).

        Returns:
            torch.Tensor: Adjusted positions of the points, shape (batch_size, num_points, 2).
        """
        batch_size = start_points.size(0)
        overlap_groups = self.find_overlapping_indices(end_points)
        for b in range(batch_size):
            for overlap_group in overlap_groups[b]:
                pairs = self.find_adjacent_col_groups(overlap_group)
                for pair in pairs:
                    if end_points[b, pair[0], 1] > start_points[b, pair[-1], 1]:
                        end_points[b, pair[0], :] = start_points[b, pair[0], :]
                        overlap_group.remove(pair[0])
                    elif end_points[b, pair[0], 1] < start_points[b, pair[0], 1]:
                        end_points[b, pair[-1], :] = start_points[b, pair[-1], :]
                        overlap_group.remove(pair[-1])    
                    elif pair[0] > 15:
                        diff1 = end_points[b, pair[0]] - end_points[b, pair[0] - 16]
                        diff2 = end_points[b, pair[0]] - end_points[b, pair[-1] - 16]
                        if torch.abs(diff1[1])/(torch.abs(diff1[0]) + 1e-4) < torch.abs(diff2[1])/(torch.abs(diff2[0]) + 1e-4):
                            end_points[b, pair[-1], :] = start_points[b, pair[-1], :]
                            overlap_group.remove(pair[-1])
                        else:
                            end_points[b, pair[0], :] = start_points[b, pair[0], :]
                            overlap_group.remove(pair[0])
                    elif pair[-1] < 240:
                        diff1 = end_points[b, pair[0]] - end_points[b, pair[0] + 16]
                        diff2 = end_points[b, pair[0]] - end_points[b, pair[-1] + 16]
                        if torch.abs(diff1[1])/(torch.abs(diff1[0]) + 1e-4) < torch.abs(diff2[1])/(torch.abs(diff2[0]) + 1e-4):
                            end_points[b, pair[-1], :] = start_points[b, pair[-1], :]
                            overlap_group.remove(pair[-1])
                        else:
                            end_points[b, pair[0], :] = start_points[b, pair[0], :]
                            overlap_group.remove(pair[0])                
                    # elif pair[0] > 0 and pair[-1] < 255 and torch.norm(end_points[b, pair[0]] - start_points[b, pair[-1] + 1]) < torch.norm(end_points[b, pair[0]] - start_points[b, pair[0] - 1]):
                    #     end_points[b, pair[0], :] = start_points[b, pair[0], :]
                    #     overlap_group.remove(pair[0])
                    # elif pair[0] > 0 and pair[-1] < 255 and torch.norm(end_points[b, pair[0]] - start_points[b, pair[-1] + 1]) > torch.norm(end_points[b, pair[0]] - start_points[b, pair[0] - 1]):
                    #     end_points[b, pair[-1], :] = start_points[b, pair[-1], :]
                    #     overlap_group.remove(pair[-1])
                        
                pairs = self.find_adjacent_row_groups(overlap_group)
                for pair in pairs:
                    if end_points[b, pair[0], 0] > start_points[b, pair[-1], 0]:
                        end_points[b, pair[0], :] = start_points[b, pair[0], :]
                        overlap_group.remove(pair[0])
                    elif end_points[b, pair[0], 0] < start_points[b, pair[0], 0]:
                        end_points[b, pair[-1], :] = start_points[b, pair[-1], :]
                        overlap_group.remove(pair[-1])
                    elif pair[0] % 16 > 0:
                        diff1 = end_points[b, pair[0]] - end_points[b, pair[0] - 1]
                        diff2 = end_points[b, pair[0]] - end_points[b, pair[-1] - 1]
                        if torch.abs(diff1[0])/(torch.abs(diff1[1]) + 1e-4) < torch.abs(diff2[0])/(torch.abs(diff2[1]) + 1e-4):
                            end_points[b, pair[-1], :] = start_points[b, pair[-1], :]
                            overlap_group.remove(pair[-1])
                        else:
                            end_points[b, pair[0], :] = start_points[b, pair[0], :]
                            overlap_group.remove(pair[0])
                    elif pair[-1] % 16 < 15:
                        diff1 = end_points[b, pair[0]] - end_points[b, pair[0] + 1]
                        diff2 = end_points[b, pair[0]] - end_points[b, pair[-1] + 1]
                        if torch.abs(diff1[0])/(torch.abs(diff1[1]) + 1e-4) < torch.abs(diff2[0])/(torch.abs(diff2[1]) + 1e-4):
                            end_points[b, pair[-1], :] = start_points[b, pair[-1], :]
                            overlap_group.remove(pair[-1])
                        else:
                            end_points[b, pair[0], :] = start_points[b, pair[0], :]
                            overlap_group.remove(pair[0])
                    # elif pair[0] > 15 and pair[-1] < 240 and torch.norm(end_points[b, pair[0]] - start_points[b, pair[-1] + 16]) < torch.norm(end_points[b, pair[0]] - start_points[b, pair[0] - 16]):
                    #     end_points[b, pair[0], :] = start_points[b, pair[0], :]
                    #     overlap_group.remove(pair[0])
                    # elif pair[0] > 15 and pair[-1] < 240 and torch.norm(end_points[b, pair[0]] - start_points[b, pair[-1] + 16]) > torch.norm(end_points[b, pair[0]] - start_points[b, pair[0] - 16]):
                    #     end_points[b, pair[-1], :] = start_points[b, pair[-1], :]
                    #     overlap_group.remove(pair[-1])
                                         
                if len(overlap_group) > 1:
                    target_point = end_points[b, overlap_group[0]].clone()
                    min_dist = torch.inf
                    final_idx = None
                    for idx in overlap_group:
                        end_points[b, idx, :] = start_points[b, idx, :]  # Reset position
                        dist = torch.norm(start_points[b, idx] - target_point)
                        if dist < min_dist:
                            min_dist = dist
                            final_idx = idx
                    end_points[b, final_idx] = target_point
               
        return end_points

    def forward(self, depth_maps, start_points):
        """
        Perform iterative movement of points towards local maxima on depth maps for a batch of data.

        Args:
            depth_maps (torch.Tensor): Depth maps tensor of shape (batch_size, 1, height, width),
                                       where each pixel contains depth information.
            start_points (torch.Tensor): Tensor of initial point coordinates, shape (batch_size, num_points, 2).

        Returns:
            torch.Tensor: Final positions of points after convergence, shape (batch_size, num_points, 2).
        """
        batch_size, _, height, width = depth_maps.shape
        num_points = start_points.shape[1]
        device = depth_maps.device

        # Remove channel dimension from depth_maps
        depth_maps = depth_maps.squeeze(1)  # Shape: (batch_size, height, width)

        # Initialize positions of the points
        points = start_points.clone().float()  # Shape: (batch_size, num_points, 2)
        depth_maps = self.smooth_tensor(depth_maps)
        # Create relative offsets for the local neighborhood
        offsets = torch.stack(torch.meshgrid(
            torch.arange(-self.radius, self.radius + 1, device=device),
            torch.arange(-self.radius, self.radius + 1, device=device)
        ), dim=-1).view(-1, 2)  # Shape: (K, 2), where K is the number of offsets

        for _ in range(self.max_iters):
            # Clamp points to ensure they remain within valid map boundaries
            points[..., 0] = points[..., 0].clamp(self.radius, height - self.radius - 1)
            points[..., 1] = points[..., 1].clamp(self.radius, width - self.radius - 1)

            # Compute neighborhood coordinates for each point
            local_coords = points.unsqueeze(2) + offsets.unsqueeze(0).unsqueeze(0)  # Shape: (batch_size, num_points, K, 2)
            local_coords[..., 0] = local_coords[..., 0].clamp(0, height - 1)  # Clamp x-coordinates
            local_coords[..., 1] = local_coords[..., 1].clamp(0, width - 1)   # Clamp y-coordinates

            # Extract depth values for all neighborhood points
            local_x = local_coords[..., 0].long()  # Shape: (batch_size, num_points, K)
            local_y = local_coords[..., 1].long()  # Shape: (batch_size, num_points, K)
            batch_indices = torch.arange(batch_size, device=device).view(-1, 1, 1).expand_as(local_x)
            local_values = depth_maps[batch_indices, local_x, local_y]  # Shape: (batch_size, num_points, K)

            # Find the offset corresponding to the maximum depth value
            max_idx = torch.argmax(local_values, dim=-1)  # Shape: (batch_size, num_points)
            next_points = local_coords[torch.arange(batch_size, device=device).view(-1, 1), 
                                       torch.arange(num_points, device=device).view(1, -1), 
                                       max_idx]  # Shape: (batch_size, num_points, 2)

            # Check for convergence (if the maximum displacement is below the tolerance)
            if torch.norm(next_points - points, dim=-1).max() < self.tol:
                break

            # Update positions
            points = torch.round(next_points)

        all_indices = torch.arange(256)
        mod_16 = all_indices % 16
        div_16 = all_indices // 16
        corner_mask = ((div_16 < 5) & (mod_16 < 5)) | ((div_16 < 5) & (mod_16 > 10)) | ((div_16 > 10) & (mod_16 < 5)) | ((div_16 > 10) & (mod_16 > 10)) 
        valid_indices = torch.round(all_indices[corner_mask])
        points[:, valid_indices, :] = torch.round(start_points[:, valid_indices, :]).to(points.dtype) 
        # Resolve overlapping points
        points = self.process_overlap_points(start_points, points)

        return points
    
    
def find_peaks(depth_map, start_points, radius=2, max_iters=100, tol=1e-3):
    """
    Simulate "ball rolling to peak" for multiple points on a single depth map.

    Args:
        depth_map (torch.Tensor): Single depth map with shape (height, width).
        start_points (torch.Tensor): Tensor of starting points with shape (num_points, 2).
        radius (int): Neighborhood radius to explore local maximum.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance for stopping condition.

    Returns:
        torch.Tensor: Final coordinates for each point, shape (num_points, 2).
    """
    height, width = depth_map.shape
    num_points = start_points.shape[0]
    device = depth_map.device

    # Initialize ball positions
    points = start_points.clone().float()  # (num_points, 2)

    # Generate grid offsets
    offsets = torch.stack(torch.meshgrid(
        torch.arange(-radius, radius + 1, device=device),
        torch.arange(-radius, radius + 1, device=device)
    ), dim=-1).view(-1, 2)  # (K, 2)

    for _ in range(max_iters):
        # Clamp points to valid range
        points[:, 0] = points[:, 0].clamp(radius, height - radius - 1)
        points[:, 1] = points[:, 1].clamp(radius, width - radius - 1)

        # Expand points to neighborhoods
        local_coords = points.unsqueeze(1) + offsets.unsqueeze(0)  # (num_points, K, 2)
        local_coords[..., 0] = local_coords[..., 0].clamp(0, height - 1)
        local_coords[..., 1] = local_coords[..., 1].clamp(0, width - 1)

        # Extract depth values for neighborhoods
        local_x = local_coords[:, :, 0].long()  # (num_points, K)
        local_y = local_coords[:, :, 1].long()  # (num_points, K)
        local_values = depth_map[local_x, local_y]  # (num_points, K)

        # Find the direction of the maximum value
        max_idx = torch.argmax(local_values, dim=-1)  # (num_points,)
        next_points = local_coords[torch.arange(num_points, device=device), max_idx]  # (num_points, 2)

        # Check for convergence
        if torch.norm(next_points - points, dim=-1).max() < tol:
            break

        # Update points
        points = next_points

    return points


def generate_heatmap(image_shape, points, radius=3, intensity=1.0):
    """
    Generate heatmaps for a batch of points using efficient tensor operations.
    
    Args:
        image_shape (tuple): Shape of the output image (batch_size, 1, H, W).
        points (torch.Tensor): Tensor of shape (batch_size, 1, num_points, 2), where each point is (y, x).
        radius (int): Radius of the heatmap (standard deviation of Gaussian kernel).
        intensity (float): Peak intensity of the heatmap.
    
    Returns:
        torch.Tensor: Heatmap image of shape (batch_size, 1, H, W).
    """
    batch_size, _, height, width = image_shape
    num_points = points.shape[2]

    # Generate coordinate grid for the entire image
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=points.device),
        torch.arange(width, device=points.device),
        indexing="ij"
    )
    y_grid = y_grid.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
    x_grid = x_grid.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]

    # Expand points to match the grid
    points_y = points[..., 0].unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, 1, num_points, 1, 1]
    points_x = points[..., 1].unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, 1, num_points, 1, 1]

    # Compute squared distance from each pixel to each point
    dist_squared = ((y_grid - points_y) ** 2 + (x_grid - points_x) ** 2) / (radius ** 2)

    # Compute Gaussian weights
    gaussian_weights = torch.exp(-0.5 * dist_squared) * intensity  # Shape: [batch_size, 1, num_points, H, W]

    # Sum up contributions from all points
    heatmap = gaussian_weights.sum(dim=2)  # Combine all points: [batch_size, 1, H, W]

    # Clamp the heatmap to ensure valid values
    heatmap = torch.clamp(heatmap, 0, 1)
    return heatmap



class PointAdjustmentModule(nn.Module):
    """
    Module to adjust input points based on their proximity to reference points.
    This module applies a mirror transformation followed by a regression adjustment 
    (via the Gravity Guided Regression Module, GGRM) to refine points that are within 
    a specified distance threshold.
    """
    def __init__(self, threshold, smooth_sign=True, process_sign=False):
        """
        Initialize the point adjustment module.

        Args:
            threshold (float): Distance threshold to determine if a point should be adjusted.
            smooth_sign (bool): Whether to use smooth sign in GGRM (default: True).
            process_sign (bool): Flag for additional processing in GGRM (default: False).
        """
        super(PointAdjustmentModule, self).__init__()
        self.threshold = threshold
        self.GGRM = GravityGuidedRegressionModule()

    def forward(self, x, tensor1, tensor2):
        """
        Adjust the points in tensor1 based on their proximity to reference points in tensor2.

        Args:
            x: Additional input required by GGRM (the meaning depends on its implementation).
            tensor1 (torch.Tensor): Points to be adjusted, shape [batch_size, 1, 256, 2].
            tensor2 (torch.Tensor): Reference points, shape [batch_size, 1, 256, 2].

        Returns:
            torch.Tensor: Adjusted tensor1 with the same shape as the input.
        """
        assert tensor1.shape == tensor2.shape, f"Input tensor shapes mismatch: {tensor1.shape} vs {tensor2.shape}"

        # Check for NaN or Inf values
        if torch.isnan(tensor1).any() or torch.isnan(tensor2).any():
            raise ValueError("Input tensors contain NaN values.")
        if torch.isinf(tensor1).any() or torch.isinf(tensor2).any():
            raise ValueError("Input tensors contain Inf values.")

        # First mirror transformation
        inverse_tensor = torch.round(tensor2 + 1.5 * (tensor1 - tensor2))
        valid_tensor = self.GGRM(x, inverse_tensor.reshape(-1, 256, 2))
        valid_tensor = valid_tensor.reshape_as(tensor1)
        distances = torch.norm(valid_tensor - tensor2, dim=-1)
        indices1 = torch.where(distances < self.threshold)

        # Second mirror transformation
        inverse_tensor = torch.round(tensor2 - (tensor1 - tensor2))
        valid_tensor = self.GGRM(x, inverse_tensor.reshape(-1, 256, 2))
        valid_tensor = valid_tensor.reshape_as(tensor1)
        distances = torch.norm(valid_tensor - tensor2, dim=-1)
        indices2 = torch.where(distances < self.threshold)

        # Get intersection of valid indices based on last dimension (point index)
        common_idx = torch.isin(indices1[1], indices2[1])
        filtered_i0 = indices1[0][common_idx]
        filtered_i1 = indices1[1][common_idx]

        # Additional filtering: only keep indices where mod/div of 16 ∈ [5,10]
        mod_16 = filtered_i1 % 16
        div_16 = filtered_i1 // 16
        corner_mask = ((div_16 < 5) & (mod_16 < 5)) | ((div_16 < 5) & (mod_16 > 10)) | ((div_16 > 10) & (mod_16 < 5)) | ((div_16 > 10) & (mod_16 > 10)) 
        # Final indices after all filtering
        update_idx = (filtered_i0[~corner_mask], filtered_i1[~corner_mask])
        update_idx = (filtered_i0, filtered_i1)

        # Apply update
        tensor1[update_idx[0], update_idx[1], :] = \
            torch.round(valid_tensor[update_idx[0], update_idx[1], :]).to(tensor1.dtype)

        return tensor1
    
    
    
    
class AverageBasedFilterModule(nn.Module):
    def __init__(self, filter_rate, corner_location=20, crop=40):
        """
        Initialize the module with a given filter rate, corner location, and crop size.
        
        :param filter_rate: The rate to scale the mean for filtering.
        :param corner_location: The starting location of the corner regions.
        :param crop: The size of the crop from the image borders.
        """
        super(AverageBasedFilterModule, self).__init__()
        self.filter_rate = filter_rate
        self.center_regions = [crop, 256 - crop]
        self.corner_regions = [
            [(corner_location, corner_location), (corner_location + crop, corner_location + crop)],
            [(corner_location, 256 - corner_location - crop), (corner_location + crop, 256 - corner_location)],
            [(256 - corner_location - crop, corner_location), (256 - corner_location, corner_location + crop)],
            [(256 - corner_location - crop, 256 - corner_location - crop), (256 - corner_location, 256 - corner_location)]
        ]

    def forward(self, x):
        """
        Apply the filtering operation on the input tensor.
        
        :param x: Input tensor with shape [batch_size, 1, img_size, img_size].
        :return: Filtered tensor with the same shape as input.
        """
        B, C, H, W = x.shape
        for i in range(B):
            # Process the center region
            middle_x = x[i, :, self.center_regions[0]:self.center_regions[1], self.center_regions[0]:self.center_regions[1]]
            
            # Compute the mean for the cropped region of each image in the batch
            mean_values = torch.mean(middle_x, dim=(1, 2), keepdim=True)  # Shape: [1, 1, 1]
            
            # Compute the threshold based on the mean values
            threshold = mean_values * self.filter_rate  # Shape: [1, 1, 1]
            
            # Apply the threshold to the cropped region
            filtered_middle_x = torch.where(middle_x >= threshold, middle_x, torch.zeros_like(middle_x))  # Set values below threshold to 0
            
            # Replace the cropped region in the original tensor with the filtered values
            x[i, :, self.center_regions[0]:self.center_regions[1], self.center_regions[0]:self.center_regions[1]] = filtered_middle_x
            
        return x
    


def high_pass_by_region(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Divide a 2D image into 9 regions, compute the mean in each region,
    threshold each region by its mean (high‐pass filter), and return both
    the filtered image and a per-pixel threshold map.

    Regions are defined on an H×W image as follows (H=W=256 assumed, but
    works for any H,W divisible by 4):

        R1  R2  R3
        R4  R5  R6
        R7  R8  R9

    with boundaries at H/4 and 3H/4 (and likewise W).

    Parameters
    ----------
    img : np.ndarray, shape (H, W)
        Grayscale image.

    Returns
    -------
    filtered : np.ndarray, shape (H, W)
        The same image, but pixels below their region’s mean set to zero.
    thresh_map : np.ndarray, shape (H, W)
        A map where each pixel’s value is the mean of its region.
    """
    H, W = img.shape
    h1, h2 = H//4, 3*H//4
    w1, w2 = W//4, 3*W//4

    # Define slices for the 9 regions
    regions = [
        (slice(0,   h1), slice(0,   w1)),  # R1
        (slice(0,   h1), slice(w1,  w2)),  # R2
        (slice(0,   h1), slice(w2,  W)),   # R3
        (slice(h1,  h2), slice(0,   w1)),  # R4
        (slice(h1,  h2), slice(w1,  w2)),  # R5
        (slice(h1,  h2), slice(w2,  W)),   # R6
        (slice(h2,  H), slice(0,   w1)),   # R7
        (slice(h2,  H), slice(w1,  w2)),   # R8
        (slice(h2,  H), slice(w2,  W)),    # R9
    ]

    filtered = img.copy()
    thresh_map = np.zeros_like(img, dtype=float)

    for sl_h, sl_w in regions:
        region = img[sl_h, sl_w]
        mean_val = region.mean()
        # store the threshold
        thresh_map[sl_h, sl_w] = mean_val
        # zero out low‐value pixels
        mask = region < mean_val
        filtered[sl_h, sl_w][mask] = 0

    return filtered, thresh_map
