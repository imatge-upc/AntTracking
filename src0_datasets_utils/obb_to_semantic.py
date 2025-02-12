
import cv2
import numpy as np
import os
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.validation import make_valid
from tqdm import tqdm
from ultralytics import SAM

from ceab_ants.io.mot_loader import PrecomputedOMOTDetector
from ceab_ants.io.video_contextmanager import VideoCapture


#gt_path = "~/data/videos_learning/20241018_1324_0000_00_gt.txt"
gt_path = "~/videos_ini/20241018_1324_0000_00_gt.txt"
#video_path = "~/data/videos_learning/20241018/color_20241018_1324_0000_00.mkv"
video_path = "~/videos_ini/color_20241018_1324_0000_00.mkv"
dataset_name = "yolo_seg_UNKNOWN_colors_20241018_1324"
output_dir = "yolo_seg_UNKNOWN_colors_20241018_1324"
train_freq = 24
p_val = 0.3
p_train = 1 - p_val
tolerance = 0.01
max_polygon_points = 20

gt_path = os.path.expanduser(gt_path)
video_path = os.path.expanduser(video_path)


def obb2bbox(obboxes, h, w, sam_size):
    x, y, w, h, a = obboxes[:5]
    a = np.radians(a)
    w_half, h_half = w / 2, h / 2
    cos_a, sin_a = np.cos(a), np.sin(a)

    x_offsets = np.array([w_half, -w_half, -w_half, w_half]).T
    y_offsets = np.array([h_half, h_half, -h_half, -h_half]).T
    
    x_corners = x[:, None] + x_offsets * cos_a[:, None] - y_offsets * sin_a[:, None]
    y_corners = y[:, None] + x_offsets * sin_a[:, None] + y_offsets * cos_a[:, None]
    
    x_min, x_max = x_corners.min(axis=1), x_corners.max(axis=1)
    y_min, y_max = y_corners.min(axis=1), y_corners.max(axis=1)
    
    return np.stack([x_min, y_min, x_max, y_max], axis=1)
    #return np.stack([x_min / h * sam_size, y_min / w * sam_size, x_max / h * sam_size, y_max / w * sam_size], axis=1)
    #return np.stack([x_min / h, y_min / w, x_max / h, y_max / w], axis=1)

def obb_to_points(obb):
    x, y, w, h, angle = obb[:5]
    angle = np.radians(angle)
    w_half, h_half = w / 2, h / 2

    # Compute the four corners of the OBB
    corners = np.array([
        [x + w_half * np.cos(angle) - h_half * np.sin(angle),
         y + w_half * np.sin(angle) + h_half * np.cos(angle)],
        [x - w_half * np.cos(angle) - h_half * np.sin(angle),
         y - w_half * np.sin(angle) + h_half * np.cos(angle)],
        [x - w_half * np.cos(angle) + h_half * np.sin(angle),
         y - w_half * np.sin(angle) - h_half * np.cos(angle)],
        [x + w_half * np.cos(angle) + h_half * np.sin(angle),
         y + w_half * np.sin(angle) - h_half * np.cos(angle)]
    ])

    center = [x, y]

    return center, corners.tolist()

def obb2polygon(obb):
    _, corners = obb_to_points(obb)
    return Polygon(corners)

def get_crops(image, crop_size=1024, overlap=256):
    h, w = image.shape[:2]
    step = crop_size - overlap

    crops = []
    positions = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            y_end = min(y + crop_size, h)
            x_end = min(x + crop_size, w)

            # Pad the crop if it goes beyond the image boundaries
            crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            crop[: y_end - y, : x_end - x] = image[y:y_end, x:x_end]

            crops.append(crop)
            positions.append((x, y))

    return crops, positions

def crop_bboxes(bboxes, x_offset, y_offset, crop_size=1024):
    
    cropped_bboxes = []    
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        
        x_min_new, x_max_new = x_min - x_offset, x_max - x_offset
        y_min_new, y_max_new = y_min - y_offset, y_max - y_offset
        
        if x_max_new < 0 or y_max_new < 0 or x_min_new > crop_size or y_min_new > crop_size:
            continue

        x_min_new, x_max_new = np.clip([x_min_new, x_max_new], 0, crop_size)
        y_min_new, y_max_new = np.clip([y_min_new, y_max_new], 0, crop_size)
        
        cropped_bboxes.append([x_min_new, y_min_new, x_max_new, y_max_new])

    return np.array(cropped_bboxes)

def crop_points(center, corners, x_offset, y_offset, crop_size=1024):
    center = [center[0] - x_offset, center[1] - y_offset]
    corners = [[x - x_offset, y - y_offset] for x, y in corners]

    # Filter out points that are completely outside the crop
    corners = [p for p in corners if 0 <= p[0] < crop_size and 0 <= p[1] < crop_size]
    if not (0 <= center[0] < crop_size and 0 <= center[1] < crop_size):
        return None, []

    return center, corners

def reconstruct_masks(mask_polygons, positions, crop_size=1024, overlap=200):
    
    adjusted_polygons = []
    for (x_offset, y_offset), crop_masks in zip(positions, mask_polygons):
        for mask in crop_masks:
            translated_coords = [(x + x_offset, y + y_offset) for x, y in mask.exterior.coords]
            adjusted_polygons.append(Polygon(translated_coords))

    return adjusted_polygons

def process_crops(model, crops, bboxes, positions):
    mask_polygons_per_crop = []
    for (x_offset, y_offset), crop in zip(positions, crops):
        local_bboxes = crop_bboxes(bboxes, x_offset, y_offset, 1024)
        
        if len(local_bboxes) == 0:
            mask_polygons_per_crop.append([])
            continue

        results = model(crop, bboxes=local_bboxes, verbose=False) # Pretty slow if not in cuda, not allowed batch processing. So just in this case, it can be applied multiple instances of a model in 1 GPU (in a future because I don't know how to do so yet).

        mask_polygons = [
            Polygon(np.array(mask.xyn[0]) * [1024, 1024])
            for mask in results[0].masks if len(mask.xyn[0]) > 0
        ]
        mask_polygons_per_crop.append(mask_polygons)    

    full_image_masks = reconstruct_masks(mask_polygons_per_crop, positions, 1024, 200)  
    return full_image_masks

def process_crops_with_points(model, crops, obbs, positions):
    mask_polygons_per_crop = []
    
    for (x_offset, y_offset), crop in zip(positions, crops):

        points = []
        labels = []

        for obb in obbs:
            center, corners = obb_to_points(obb)
            center, corners = crop_points(center, corners, x_offset, y_offset, 1024)

            if center is None:
                continue 

            while len(corners) < 4:
                corners.append(corners[-1])
            
            points.append([center] + corners)
            labels.append([1] + [0] * len(corners))

        if len(points) == 0:
            mask_polygons_per_crop.append([])
            continue

        results = model(crop, points=points, labels=labels, verbose=False)

        mask_polygons = [
            Polygon(np.array(mask.xyn[0]) * [1024, 1024])
            for mask in results[0].masks if len(mask.xyn[0]) > 0
        ] # I'm using mask.xyn[0]. maybe I should keep all parts...
        mask_polygons_per_crop.append(mask_polygons)

    return reconstruct_masks(mask_polygons_per_crop, positions, 1024, 200)

def debug_plot(frame, obb_polygons, mask_polygons):
    from matplotlib import pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    ax1.imshow(frame)
    ax2.imshow(frame)

    for obb in obb_polygons:
        x, y = obb.exterior.xy
        ax1.plot(x, y, 'r-', linewidth=2, label="OBB" if "OBB" not in ax1.get_legend_handles_labels()[1] else "")

    for mask in mask_polygons:
        x, y = mask.exterior.xy
        ax2.plot(x, y, 'g-', linewidth=2, label="Mask" if "Mask" not in ax2.get_legend_handles_labels()[1] else "")

    ax1.legend()
    ax1.set_title("Visualization of OBBs")
    ax2.legend()
    ax2.set_title("Visualization of SAM Masks")
    plt.show()

def debug_plot_with_mask_points(frame, obb_polygons, mask_points):
    from matplotlib import pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    ax1.imshow(frame)
    ax2.imshow(frame)

    for obb in obb_polygons:
        x, y = obb.exterior.xy
        ax1.plot(x, y, 'r-', linewidth=2, label="OBB" if "OBB" not in ax1.get_legend_handles_labels()[1] else "")

    for points in mask_points:
        points = np.array(points)
        points = np.vstack([points, points[0]])
        ax2.plot(points[:, 0], points[:, 1], color='green', linewidth=2, label='Mask' if 'Mask' not in plt.gca().get_legend_handles_labels()[1] else "")

    ax1.legend()
    ax1.set_title("Visualization of OBBs")
    ax2.legend()
    ax2.set_title("Visualization of SAM Masks")
    plt.show()

if __name__ == "__main__":
    
    model = SAM("sam2.1_b.pt")
    omot = PrecomputedOMOTDetector(gt_path)
    last_frame = omot.last_frame

    for folder in ["train", "val", "test"]:
        os.makedirs(f"{output_dir}/images/{folder}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{folder}", exist_ok=True)

    with VideoCapture(video_path) as capture:    
        for fr in tqdm(range(1, last_frame + 1)):

            ret, frame = capture.read()
            if not ret:
                break
            
            img_h, img_w = frame.shape[:2]

            obbs = omot(fr, override=True) # [x, y, w, h, a, score]
            bboxes = obb2bbox(obbs, img_h, img_w, 1024) # [x_min, y_min, x_max, y_max] 1024 is not used now

            crops, positions = get_crops(frame, crop_size=1024, overlap=200)
            #mask_polygons = process_crops(model, crops, bboxes, positions)
            mask_polygons = process_crops_with_points(model, crops, obbs, positions)
            
            obb_polygons = [obb2polygon(obb) for obb in obbs]

            mode = "test" if fr % train_freq != 1 else ("val" if fr < last_frame * p_val else "train")
            image_filename = f"{output_dir}/images/{mode}/frame_{fr:06d}.png"
            label_filename = f"{output_dir}/labels/{mode}/frame_{fr:06d}.txt"
            output_file = f"{output_dir}/{dataset_name}.yaml"

            cv2.imwrite(image_filename, frame) # NOTE: CALCULA space cuota may be too small, if needed, run with this line commented

            #debug_plot(frame, obb_polygons, mask_polygons)

            with open(label_filename, "w") as f:
                
                mask_points_to_plot = []
                used_obbs = set()
                for obb_idx, obb_polygon in enumerate(obb_polygons):

                    best_mask = None
                    best_intersection_area = 0

                    for mask_polygon in mask_polygons:
                        
                        # Stable and usually better but sometimes misses
                        mask_polygon_valid = make_valid(mask_polygon)
                        if isinstance(mask_polygon_valid, MultiPolygon):
                            mask_polygon_valid = max(mask_polygon_valid.geoms, key=lambda p: obb_polygon.intersection(p).area)

                        intersection = obb_polygon.intersection(mask_polygon_valid).simplify(tolerance, preserve_topology=True)
                        if isinstance(intersection, MultiPolygon):
                            intersection = max(intersection.geoms, key=lambda p: obb_polygon.intersection(p).area)
                        elif isinstance(intersection, GeometryCollection):
                            intersection = [geom for geom in intersection.geoms if isinstance(geom, Polygon) and not geom.is_empty]
                            if intersection:
                                intersection = max(intersection, key=lambda p: obb_polygon.intersection(p).area)
                            else:
                                intersection = None

                        if intersection:

                            intersection_area = intersection.area
                            if intersection_area > best_intersection_area:
                                best_intersection_area = intersection_area
                                best_mask = intersection
                        
                        # Unstable but completes previous
                        try:
                            mask_polygon = mask_polygon.buffer(0)

                            intersection = obb_polygon.intersection(mask_polygon).simplify(tolerance, preserve_topology=True)
                            if isinstance(intersection, MultiPolygon):
                                intersection = max(intersection.geoms, key=lambda p: p.area)

                            if intersection:

                                intersection_area = intersection.area
                                if intersection_area > best_intersection_area:
                                    best_intersection_area = intersection_area
                                    best_mask = intersection
                        except:
                            continue
                    
                    if isinstance(best_mask, Polygon) and not best_mask.is_empty:
    
                        mask_points = np.array(best_mask.exterior.coords[:-1])
                    
                        if len(mask_points) > max_polygon_points:
                            step = len(mask_points) // max_polygon_points
                            mask_points = mask_points[::step][:max_polygon_points]
                        
                        mask_points_to_plot.append(mask_points)

                        mask_str = " ".join([f"{x / img_w} {y / img_h}" for x, y in mask_points])
                        print(f"0 {mask_str}", end="\n", file=f)

            #debug_plot_with_mask_points(frame.copy(), obb_polygons, mask_points_to_plot)

        config_text = f"""
        # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
        path: ./{output_dir}  # dataset root dir
        train: images/train  # train images (relative to 'path')
        val: images/val  # val images (relative to 'path')
        test:  images/test # test images (optional)

        # Classes
        names:
            0: ant

        """

        with open(output_file, 'w') as f:
            f.write(config_text)
