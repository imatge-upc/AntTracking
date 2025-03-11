
import numpy as np
import warnings


def batch_obbox_vertices(obbox):
    # obbox as cx, cy, w, h, angle
    # Returns a N, 4, 2 matrix where the corners rotate counterclockwise starting by the unrotated bottom-left corner

    centers = obbox[:, :2]
    half_widths = obbox[:, 2] /  2
    half_heights = obbox[:, 3] / 2
    thetas = obbox[:, 4]

    corners_local = np.array([
        [-half_widths, -half_heights],  # First corner (x, y)
        [ half_widths, -half_heights],  # Second corner (x, y)
        [ half_widths,  half_heights],  # Third corner (x, y)
        [-half_widths,  half_heights]   # Fourth corner (x, y)
    ]) # (4, 2, -1)

    corners_local = np.transpose(corners_local, (2, 0, 1))  # Shape becomes (N, 4, 2)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    rotation_matrices = np.stack([
        cos_t, -sin_t, sin_t, cos_t
    ], axis=-1).reshape(-1, 2, 2)

    obbox_vertices = np.einsum('nij,nkj->nik', corners_local, rotation_matrices)
    obbox_vertices += centers[:, np.newaxis, :]

    return obbox_vertices

def batch_obbox_edges(obbox_vertices):
    
    # Extract edges for each rectangle (shape: (N, 4, 2, 2))
    # np.roll: copy of obboxes with a shift on the vertices
    # Each edge is [start, end]
    edges = np.stack([obbox_vertices, np.roll(obbox_vertices, shift=-1, axis=1)], axis=2)
    return edges

def batch_obbox_edges_intesection(obbox_edges_test, obbox_edges_gt):

    n_test = obbox_edges_test.shape[0]
    n_gt = obbox_edges_gt.shape[0]

    # Prepare pairwise edges (good to have in mind the (N, M, 4, 4, 2, 2) shape)
    edges_test = obbox_edges_test[:, None, :, None, :, :] # (N, 1, 4, 1, 2, 2)
    edges_gt = obbox_edges_gt[None, :, None, :, :, :]     # (1, M, 1, 4, 2, 2)
    
    # Extract points for the parametric equations (shape: ({N, 1}, {1, M}, {4, 1}, {1, 4}, 2))
    P1 = edges_test[..., 0, :]  # Start of each edge of the first rectangle
    P2 = edges_test[..., 1, :]  # End of each edge of the first rectangle
    Q1 = edges_gt[..., 0, :]  # Start of each edge of the second rectangle
    Q2 = edges_gt[..., 1, :]  # End of each edge of the second rectangle
    
    # Direction vectors
    dP = P2 - P1  # Direction vector of edges in the first rectangle (N, 1, 4, 1, 2)
    dQ = Q2 - Q1  # Direction vector of edges in the second rectangle (1, M, 1, 4, 2)

    # Extract components ({N, 1}, {1, M}, {4, 1}, {1, 4}) and broadcast to the shape (N, M, 4, 4)
    dP = np.broadcast_to(dP, (n_test, n_gt, 4, 4, 2)) # (N, M, 4, 4)
    dQ = np.broadcast_to(dQ, (n_test, n_gt, 4, 4, 2)) # (N, M, 4, 4)

    # Stack the components to create the matrix A
    A = np.stack([
        np.stack([dP[..., 0], -dQ[..., 0]], axis=-1),  # First row
        np.stack([dP[..., 1], -dQ[..., 1]], axis=-1)   # Second row
    ], axis=-2).reshape(n_test, n_gt, 4, 4, 2, 2) # (N, M, 4, 4, 2, 2)

    B = np.stack([
        Q1[..., 0] - P1[..., 0],
        Q1[..., 1] - P1[..., 1]
    ], axis=-1).reshape(n_test, n_gt, 4, 4, 2)
    
    # Solve for t and u
    det = np.linalg.det(A)  # Determinant of A (shape: (N, M, 4, 4))
    valid = det != 0  # Valid intersections only where det != 0 
    # NOTE: If overlapped edges, the perpendicular edges will cross at the desired point
    t_u = np.full_like(B, -1) # (N, M, 4, 4, 2)
    t_u[valid] = np.linalg.solve(A[valid], B[valid])  # Solve for t and u
    t, u = t_u[..., 0], t_u[..., 1] # (N, M, 4, 4)
    
    # Filter valid t, u (0 <= t, u <= 1)
    valid_intersections = (0 <= t) & (t <= 1) & (0 <= u) & (u <= 1) # (N, M, 4, 4)
    intersection_points = (P1 + t[..., None] * dP) * valid_intersections[..., None] # (N, M, 4, 4, 2)

    return intersection_points, valid_intersections

def batch_obbox_vertices_inside_obbox_edges(obbox_vertices, obbox_edges):

    # Unpack edges (M, 4, 2, 2)
    edge_start = obbox_edges[..., 0, :]  # (M, 4, 2)
    edge_end = obbox_edges[..., 1, :]    # (M, 4, 2)

    # Direction vectors for edges and points
    edge_dir = edge_end - edge_start  # (M, 4, 2)
    vertex_vector = obbox_vertices[:, None, :, None, :] - edge_start[None, :, None, :, :]  # (N, M, 4, 4, 2)

    # Cross product to determine which side of the edge the vertex is on
    cross_product = np.cross(edge_dir[None, :, None, ...], vertex_vector) # (N, M, 4, 4)

    # Check sign of cross products across all edges (edge points are considered outside)
    is_left_side = cross_product > 0  # Positive cross product indicates left side
    is_right_side = cross_product < 0  # Negative cross product indicates right side

    # For a point to be inside the OBB, it must be consistently on one side (left or right) for all edges
    inside_all_edges = np.all(is_left_side, axis=-1) | np.all(is_right_side, axis=-1)  # (N, M, 4)

    return inside_all_edges

def batch_vertices_to_polygons(all_points, valid_mask, max_points=8):

    max_points = min(max_points, all_points.shape[2])

    # Sorting the valid points at the beggining and taking the first max_points
    indices = np.argsort(valid_mask, axis=-1)[..., ::-1]  # Sort by validity
    sorted_points = np.take_along_axis(all_points, indices[..., None], axis=2)  # (N, M, 24, 2)
    polygons = sorted_points[..., :max_points, :]  # (N, M, max_points, 2)
    
    # Get a valid point if there is at least one
    valid_counts = np.sum(valid_mask, axis=-1)  # (N, M)
    last_valid_idx = np.clip(valid_counts - 1, 0, max_points - 1) # (N, M)
    last_valid_vertex = np.take_along_axis(sorted_points, last_valid_idx[..., None, None], axis=2)  # (N, M, 1, 2)

    # Repeat the valid vertex to fill polygons with fewer than 8 points
    repeat_mask = np.arange(max_points)[None, None, :] >= valid_counts[..., None]  # (N, M, max_points)
    polygons = np.where(repeat_mask[..., None], last_valid_vertex, polygons)

    # Replace invalid polygons (fewer than 3 vertices) with NaN
    invalid_polygons = valid_counts < 3  # Polygons with fewer than 3 vertices
    polygons[invalid_polygons] = np.nan

    return polygons

def batch_sort_polygons(polygons):
    # polygons are (N, M, 8, 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning) # A lot of invalid polygons will raise this

        centroid = np.nanmean(polygons, axis=-2, keepdims=True) # (N, M, 1, 2)
    
    relative_positions = polygons - centroid  # (N, M, 8, 2)
    angles = np.arctan2(relative_positions[..., 1], relative_positions[..., 0])  # (N, M, 8)
    
    sort_indices = np.argsort(angles, axis=-1)  # (N, M, 8)
    sorted_polygons = np.take_along_axis(polygons, sort_indices[..., None], axis=-2)  # (N, M, 8, 2)

    return sorted_polygons

def batch_process_obbox_intersection_vertices(obbox_test_vertices, obbox_gt_vertices, intersection_points, valid_intersections, obbox_test_mask, obbox_gt_mask):

    # Flatten the intersection points and mask
    intersection_points_flat = intersection_points.reshape(intersection_points.shape[0], intersection_points.shape[1], -1, 2)  # (N, M, 16, 2)
    valid_intersections_flat = valid_intersections.reshape(valid_intersections.shape[0], valid_intersections.shape[1], -1)  # (N, M, 16)

    # Broadcast the vertices to the output, careful: invalid points will become 0,0
    inside_test_vertices = obbox_test_vertices[:, None, :, :] * obbox_test_mask[..., None]  # (N, M, 4, 2)
    inside_gt_vertices = obbox_gt_vertices[None, :, :, :] * obbox_gt_mask[..., None]  # (N, M, 4, 2)
    
    # Joining points and masks into one
    all_points = np.concatenate([
        intersection_points_flat,  # (N, M, 16, 2)
        inside_test_vertices,  # (N, M, 4, 2)
        inside_gt_vertices  # (N, M, 4, 2)
    ], axis=2)  # (N, M, 24, 2)

    valid_mask = np.concatenate([
        valid_intersections_flat,  # (N, M, 16)
        obbox_test_mask,  # (N, M, 4)
        obbox_gt_mask  # (N, M, 4)
    ], axis=2)  # (N, M, 24)

    return all_points, valid_mask

def batch_intersection_obbox(obbox_test, obbox_gt):

    # Extract counterclockwise vertices for each obboxes starting with the bottom-left corner
    obbox_test_vertices = batch_obbox_vertices(obbox_test) # (N, 4, 2)
    obbox_gt_vertices = batch_obbox_vertices(obbox_gt) # (M, 4, 2)

    if obbox_test_vertices.size == 0 or obbox_gt_vertices.size == 0:
        return np.empty([obbox_test_vertices.shape[0], obbox_gt_vertices.shape[0], 8, 2])

    # Extract edges for each rectangle
    edges_test = batch_obbox_edges(obbox_test_vertices) # (N, 4, 2, 2)
    edges_gt = batch_obbox_edges(obbox_gt_vertices) # (M, 4, 2, 2)

    # Compute intersection points between edges (N, M, 4, 4, 2) and validity mask (N, M, 4, 4)
    intersections, valid_intersections = batch_obbox_edges_intesection(edges_test, edges_gt)

    # Check vertices of one OBB inside the other
    obbox_test_inside_gt_mask = batch_obbox_vertices_inside_obbox_edges(obbox_test_vertices, edges_gt) # (N, M, 4)
    obbox_gt_inside_test_mask = batch_obbox_vertices_inside_obbox_edges(obbox_gt_vertices, edges_test) # (M, N, 4)
    obbox_gt_inside_test_mask = obbox_gt_inside_test_mask.transpose((1, 0, 2)) # (N, M, 4)

    all_points, valid_mask = batch_process_obbox_intersection_vertices(obbox_test_vertices, obbox_gt_vertices, intersections, valid_intersections, obbox_test_inside_gt_mask, obbox_gt_inside_test_mask)
    polygons = batch_vertices_to_polygons(all_points, valid_mask, max_points=8) # maximum possible polygon with 2 obboxes has 8 points
    sorted_polygons = batch_sort_polygons(polygons) # Counterclock-wise vertices sorting

    return sorted_polygons

def batch_process_obbox_enclosure_vertices(obbox_test_vertices, obbox_gt_vertices, obbox_test_mask, obbox_gt_mask):

    n_test = obbox_test_vertices.shape[0]
    n_gt = obbox_gt_vertices.shape[0]

    outside_test_vertices = obbox_test_vertices[:, None, :, :] * obbox_test_mask[..., None]  # (N, M, 4, 2)
    outside_gt_vertices = obbox_gt_vertices[None, :, :, :] * obbox_gt_mask[..., None]  # (N, M, 4, 2)
    
    # Joining points and masks into one
    all_points = np.concatenate([
        outside_test_vertices,  # (N, M, 4, 2)
        outside_gt_vertices  # (N, M, 4, 2)
    ], axis=2)  # (N, M, 8, 2)

    valid_mask = np.concatenate([
        obbox_test_mask,  # (N, M, 4)
        obbox_gt_mask  # (N, M, 4)
    ], axis=2)  # (N, M, 8)

    distances = np.linalg.norm(
        obbox_test_vertices[:, None, :, None, :] - obbox_gt_vertices[None, :, None, :, :],
        axis=-1
    ) # (N, M, 4, 4)
    fully_coincident = np.all(np.any(distances == 0, axis=2), axis=2)  # Shape: (N, M)
    flat_argmin_indices = np.argmin(distances.reshape(n_test, n_gt, -1), axis=-1)  # Shape: (N, M)
    min_dist_test_indices, min_dist_gt_indices = np.unravel_index(flat_argmin_indices, (4, 4))

    test_mask = np.ones((n_test, n_gt, 4), dtype=bool)
    gt_mask = np.ones((n_test, n_gt, 4), dtype=bool)

    batch_indices = np.arange(n_test)[:, None]
    pair_indices = np.arange(n_gt)[None, :]

    # The minimum distance does not belong to the enclosure if the obboxes are not coincident
    test_mask[batch_indices, pair_indices, min_dist_test_indices] = False
    gt_mask[batch_indices, pair_indices, min_dist_gt_indices] = False

    # If the obboxes are coincident, either of them is enough
    fully_coincident_expanded = np.broadcast_to(fully_coincident[..., None], (n_test, n_gt, 4))  # Expand to (N, M, 1)
    test_mask[fully_coincident_expanded] = True
    gt_mask[fully_coincident_expanded] = False

    pair_mask_extended = np.concatenate([test_mask, gt_mask], axis=-1)  # Shape: (N, M, 8)

    all_valid = np.all(valid_mask, axis=-1, keepdims=True)  # Shape: (N, M, 1)
    final_valid_mask = np.where(all_valid, valid_mask & pair_mask_extended, valid_mask)

    return all_points, final_valid_mask

def batch_enclosure_obbox(obbox_test, obbox_gt):

    # Extract counterclockwise vertices for each obboxes starting with the bottom-left corner
    obbox_test_vertices = batch_obbox_vertices(obbox_test) # (N, 4, 2)
    obbox_gt_vertices = batch_obbox_vertices(obbox_gt) # (M, 4, 2)

    # Extract edges for each rectangle
    edges_test = batch_obbox_edges(obbox_test_vertices) # (N, 4, 2, 2)
    edges_gt = batch_obbox_edges(obbox_gt_vertices) # (M, 4, 2, 2)

    # Check vertices of one OBB inside the other 
    obbox_test_outside_gt_mask = ~batch_obbox_vertices_inside_obbox_edges(obbox_test_vertices, edges_gt) # (N, M, 4)
    obbox_gt_outside_test_mask = ~batch_obbox_vertices_inside_obbox_edges(obbox_gt_vertices, edges_test) # (M, N, 4)
    obbox_gt_outside_test_mask = obbox_gt_outside_test_mask.transpose((1, 0, 2)) # (N, M, 4) 

    all_points, valid_mask = batch_process_obbox_enclosure_vertices(obbox_test_vertices, obbox_gt_vertices, obbox_test_outside_gt_mask, obbox_gt_outside_test_mask)

    # maximum possible polygon with 2 obboxes has 6 points (If there is no intersection, the nearest points of the 2 bboxes should be excluded)
    polygons = batch_vertices_to_polygons(all_points, valid_mask, max_points=6)
    sorted_polygons = batch_sort_polygons(polygons) # Counterclock-wise vertices sorting

    return sorted_polygons

def batch_intersection_and_enclosure_obbox(obbox_test, obbox_gt):
    # Extract counterclockwise vertices for each obboxes starting with the bottom-left corner
    obbox_test_vertices = batch_obbox_vertices(obbox_test) # (N, 4, 2)
    obbox_gt_vertices = batch_obbox_vertices(obbox_gt) # (M, 4, 2)

    # Extract edges for each rectangle
    edges_test = batch_obbox_edges(obbox_test_vertices) # (N, 4, 2, 2)
    edges_gt = batch_obbox_edges(obbox_gt_vertices) # (M, 4, 2, 2)

    # Compute intersection points between edges (N, M, 4, 4, 2) and validity mask (N, M, 4, 4)
    intersections, valid_intersections = batch_obbox_edges_intesection(edges_test, edges_gt)

    # Check vertices of one OBB inside the other
    obbox_test_inside_gt_mask = batch_obbox_vertices_inside_obbox_edges(obbox_test_vertices, edges_gt) # (N, M, 4)
    obbox_gt_inside_test_mask = batch_obbox_vertices_inside_obbox_edges(obbox_gt_vertices, edges_test) # (M, N, 4)
    obbox_gt_inside_test_mask = obbox_gt_inside_test_mask.transpose((1, 0, 2)) # (N, M, 4)
    obbox_test_outside_gt_mask = ~obbox_test_inside_gt_mask
    obbox_gt_outside_test_mask = ~obbox_gt_inside_test_mask

    all_points_intersection, valid_mask_intersection = batch_process_obbox_intersection_vertices(obbox_test_vertices, obbox_gt_vertices, intersections, valid_intersections, obbox_test_inside_gt_mask, obbox_gt_inside_test_mask)
    intersection_polygons = batch_vertices_to_polygons(all_points_intersection, valid_mask_intersection, max_points=8) # maximum possible polygon with 2 obboxes has 8 points
    sorted_intersection_polygons = batch_sort_polygons(intersection_polygons) # Counterclock-wise vertices sorting

    all_points_enclosure, valid_mask_enclosure = batch_process_obbox_enclosure_vertices(obbox_test_vertices, obbox_gt_vertices, obbox_test_outside_gt_mask, obbox_gt_outside_test_mask)
    enclosure_polygons = batch_vertices_to_polygons(all_points_enclosure, valid_mask_enclosure, max_points=6) # maximum possible polygon with 2 obboxes has 6 points
    sorted_enclosure_polygons = batch_sort_polygons(enclosure_polygons) # Counterclock-wise vertices sorting

    return sorted_intersection_polygons, sorted_enclosure_polygons
