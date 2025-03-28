
from collections import defaultdict
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import random


def apply_homography(points, H):
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    dst = points_h @ H.T
    dst /= dst[:, 2:3]
    return dst[:, :2]

def clip_polygon(polygon, width, height):
    from shapely.geometry import Polygon, box

    poly = Polygon([tuple(pt[0]) for pt in polygon])
    valid_area = box(0, 0, width - 1, height - 1)
    clipped = poly.intersection(valid_area)

    if clipped.is_empty:
        return None

    if clipped.geom_type == 'Polygon':
        return np.array(clipped.exterior.coords, dtype=np.int32).reshape(-1, 1, 2)
    elif clipped.geom_type == 'MultiPolygon':
        max_poly = max(clipped.geoms, key=lambda p: p.area) # Shouldn't happen but it's safer to contemplate it
        return np.array(max_poly.exterior.coords, dtype=np.int32).reshape(-1, 1, 2)

def get_obb(mot_line, trk=True):
    mot_line = mot_line.flatten()

    x, y, width, height = mot_line[2:6]
    #if trk : x, y = x + width / 2, y + height / 2
    angle = np.deg2rad(mot_line[10])

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    half_w, half_h = width / 2, height / 2

    obb = np.array([
        [x + half_w * cos_a - half_h * sin_a, y + half_w * sin_a + half_h * cos_a],  # Top-right
        [x - half_w * cos_a - half_h * sin_a, y - half_w * sin_a + half_h * cos_a],  # Top-left
        [x - half_w * cos_a + half_h * sin_a, y - half_w * sin_a - half_h * cos_a],  # Bottom-left
        [x + half_w * cos_a + half_h * sin_a, y + half_w * sin_a - half_h * cos_a]   # Bottom-right
    ], dtype=np.int32)

    return obb

class TrackMemory():
    # TODO: At some point make a python interface for https://medialab.github.io/iwanthue/ to get a colormap improved for colorblind.

    def __init__(self, memory_size, num_ids, colormap=None):

        self.memory_size = memory_size
        self.tracks = defaultdict(list)
        self.age = defaultdict(lambda : 0)
        self.colors = dict()

        self.num_ids = num_ids
        colormap = colormap or cm.get_cmap("tab10")
        if isinstance(colormap, mcolors.Colormap):
            self.colormap = [tuple(int(c * 255) for c in colormap(i / num_ids)[:3]) for i in range(num_ids)]
            if colormap == cm.get_cmap("tab10") and num_ids > 10:
                random.shuffle(self.colormap)
        elif isinstance(colormap, dict):
            self.colormap = colormap
 
    def update_tracks(self, mot_frame):

        for track in mot_frame:
            track_id = int(track[1])
            center = (int(track[2]), int(track[3]))
            center = (int(track[2] - track[4] / 2), int(track[3] - track[5] / 2))

            self.tracks[track_id].append(center)

            if track_id not in self.colors.keys():
                self.colors[track_id] = self.colormap[len(self.colors) % min(self.num_ids, len(self.colormap))]
        
        if self.memory_size > 0:
            for track_id in self.tracks.copy().keys():
                self.age[track_id] = min((self.age[track_id] + 1), self.memory_size + 1)
            
                if self.age[track_id] > self.memory_size:
                    self.tracks[track_id].pop(0)
                    if len(self.tracks[track_id]) == 0:
                        del self.tracks[track_id]
                        del self.age[track_id]
                    else:
                        self.age[track_id] -= 1

    def generate_tracks(self, mot_frame):
        self.update_tracks(mot_frame)
        
        for track_id, positions in self.tracks.items():

            current_detection = mot_frame[mot_frame[:, 1] == track_id].flatten()
            if current_detection.shape[0] == 0:
                obb = None
            else:
                obb = get_obb(current_detection)

            x, y = positions[-1]
            memory_line = positions.copy()
            color = self.colors[track_id]

            yield obb, (int(x), int(y)), memory_line, color

class FrameDrawer():

    def __init__(self, cam_homographies, real_world_homography, motfile=None, mot_real_world=True, video_resolution=None, camera_resolution=None, output_resolution=None, memory_size=1, colormap=None, obb_thickness=1, point_thickness=8, line_thickness=2, max_ids=None):
        """
        At main, cam_homographies was loaded as a matrix of 12x12x3x3, the main know which cameras are useful and which camera is the base for real_world_homography, so the main filter them into a list of 3x3 homographies
        mot_real_world=True means the motfile already has applied real_world_homography, if False real_world_homography will be applied.
        """

        camera_resolution = camera_resolution or (4000., 3000.)
        video_resolution = video_resolution or camera_resolution
        self.output_resolution = output_resolution or (4000., 3000.)
        self.motfile = motfile

        self.mot_homography, self.cam_homographies, self.coverage = self.prepare_homographies(cam_homographies, real_world_homography, mot_real_world, video_resolution, camera_resolution, self.output_resolution)
        self.processed_mot = self.process_mot(motfile) if motfile else None

        self.tracks = None
        if self.processed_mot is not None:
            if self.processed_mot[0, 1] != -1: 
                if max_ids is not None:
                    unique, counts = np.unique(self.processed_mot[:, 1], return_counts=True)
                    unique = unique[counts >= np.partition(counts, -max_ids)[-max_ids]]
                    self.processed_mot = self.processed_mot[np.isin(self.processed_mot[:, 1], unique), ...]
                
                self.tracks = TrackMemory(memory_size, len(np.unique(self.processed_mot[:, 1])), colormap=colormap)
                self.colormap = colormap
            else:
                colormap = colormap or cm.get_cmap("tab10")
                if isinstance(colormap, mcolors.Colormap):
                    # maximum number of id per frame
                    max_colors = max([np.sum(self.processed_mot[:, 0] == i) for i in np.unique(self.processed_mot[:, 0])])
                    self.colormap = [tuple(int(c * 255) for c in colormap(i / max_colors)[:3]) for i in range(max_colors)]
                    if colormap == cm.get_cmap("tab10") and max_colors > 10:
                        random.shuffle(self.colormap)

                elif isinstance(colormap, dict):
                    self.colormap = colormap
                
                self.max_radius = np.max(np.sqrt(self.processed_mot[:, 4] ** 2 + self.processed_mot[:, 5] ** 2))

        self.obb_thickness = obb_thickness
        self.point_thickness = point_thickness / len(self.cam_homographies)
        if self.point_thickness < 1 : self.point_thickness = 3 # NOTE: When drawing multiple cams, big points in pixels are larger than the ants
        self.line_thickness = line_thickness

    def prepare_homographies(self, cam_homographies, real_world_homography, mot_real_world, video_resolution, camera_resolution, output_resolution):
        mot_homography = np.eye(3) if mot_real_world else real_world_homography.copy()
        cam_homographies = cam_homographies.copy()

        scale_0 = (camera_resolution[0] / video_resolution[0], camera_resolution[1] / video_resolution[1])
        resize_matrix_0 = np.array([[scale_0[0], 0., 0.], [0., scale_0[1], 0.], [0., 0., 1.]])

        for i in range(len(cam_homographies)):
            cam_homographies[i] = real_world_homography @ cam_homographies[i] @ resize_matrix_0

        image_corners = np.array([
            [0., 0., 1.], [video_resolution[0], 0., 1.], [video_resolution[0], video_resolution[1], 1.], [0., video_resolution[1], 1.]]
        ).T

        corners = []
        for H in cam_homographies:
            transformed_corners = np.dot(H, image_corners)
            transformed_corners /= transformed_corners[2]
            corners.extend(transformed_corners[:2].T)
        corners = np.asarray(corners)
        
        min_x, min_y = np.min(corners, axis=0)
        max_x, max_y = np.max(corners, axis=0)
        w, h = max_x - min_x, max_y - min_y
        scale = min(output_resolution[0] / w, output_resolution[1] / h)

        center_x, center_y = scale * w / 2, scale * h / 2
        center_x_out, center_y_out = output_resolution[0] / 2, output_resolution[1] / 2
        
        translation_matrix_1 = np.array([[1., 0., -min_x], [0., 1., -min_y], [0., 0., 1.]])
        resize_matrix = np.array([[scale, 0., 0.], [0., scale, 0.], [0., 0., 1.]])
        translation_matrix_2 = np.array([[1., 0., center_x_out - center_x], [0., 1., center_y_out - center_y], [0., 0., 1.]])

        mot_homography = translation_matrix_2 @ resize_matrix @ translation_matrix_1 @ mot_homography
        for i in range(len(cam_homographies)):
            cam_homographies[i] = translation_matrix_2 @ resize_matrix @ translation_matrix_1 @ cam_homographies[i]

        width, height = self.output_resolution
        coverage = np.zeros((height, width), dtype=np.int32)

        corners = np.array([
            [0, 0],
            [video_resolution[0] - 1, 0],
            [video_resolution[0] - 1, video_resolution[1] - 1],
            [0, video_resolution[1] - 1]
        ])

        for H in cam_homographies:
            transformed_corners = apply_homography(corners, H)

            polygon = transformed_corners.astype(np.int32)
            polygon = polygon.reshape((-1, 1, 2))

            clipped_polygon = clip_polygon(polygon, width, height)
            if clipped_polygon is not None:
                temp_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(temp_mask, [clipped_polygon], 1)
                coverage += temp_mask
        
        coverage[coverage == 0] = 1
        coverage = np.repeat(coverage[..., np.newaxis], 3, axis=2)
        
        return mot_homography, cam_homographies, coverage

    def process_mot(self, motfile):
        detections = np.loadtxt(motfile, delimiter=',')

        src_p = np.hstack((detections[:, 2:4], np.ones((detections.shape[0], 1))))
        src_s = np.hstack((detections[:, 4:6], np.zeros((detections.shape[0], 1))))
        
        dst_p = np.matmul(self.mot_homography, src_p.T).T
        dst_s = np.matmul(self.mot_homography, src_s.T).T
        
        detections[:, 2:4] = dst_p[:, :2] / dst_p[:, 2, np.newaxis]
        detections[:, 4:6] = np.abs(dst_s[:, :2])
        
        # Update angle
        angle_rad = np.deg2rad(detections[:, 10])
        major_axis_x = (detections[:, 4] / 2) * np.cos(angle_rad)
        major_axis_y = (detections[:, 4] / 2) * np.sin(angle_rad)
        
        point1 = np.vstack([detections[:, 2] + major_axis_x, detections[:, 3] + major_axis_y, np.ones_like(detections[:, 2])]).T
        point2 = np.vstack([detections[:, 2] - major_axis_x, detections[:, 3] - major_axis_y, np.ones_like(detections[:, 2])]).T
        
        transformed_point1 = np.matmul(self.mot_homography, point1.T).T
        transformed_point2 = np.matmul(self.mot_homography, point2.T).T
        
        transformed_point1 = transformed_point1[:, :2] / transformed_point1[:, 2, np.newaxis]
        transformed_point2 = transformed_point2[:, :2] / transformed_point2[:, 2, np.newaxis]
        
        detections[:, 10] = np.rad2deg(np.arctan2(
            transformed_point2[:, 1] - transformed_point1[:, 1],
            transformed_point2[:, 0] - transformed_point1[:, 0]
        ))
        
        return detections

    def process_cam(self, frame, cam_id):
        # This happens before "draw" so draw already has all the data in good conditions for blending
        H = self.cam_homographies[cam_id].reshape(3, 3)
        transformed_frame = cv2.warpPerspective(frame, H, (int(self.output_resolution[0]), int(self.output_resolution[1])))
        transformed_frame = (transformed_frame / self.coverage).astype(np.uint8)
        return transformed_frame
    
    def join_frames(self, frame_list):
        stitched_frame = frame_list[0].copy()

        for frame in frame_list[1:]:
            stitched_frame = cv2.addWeighted(stitched_frame, 1, frame, 1, 0)
            #mask = np.all(stitched_frame == 0, axis=2)
            #stitched_frame[mask] = frame[mask]

        return stitched_frame

    def draw_detections(self, frame, mot_frame):

        overlap_index = 1
        detection_positions = []
        for detection in mot_frame:
            center = tuple(map(int, detection[2:4])) # + detection[4:6] / 2))
            obb = get_obb(detection, trk=False)

            if len(detection_positions) == 0:
                distances = np.inf
            else:
                distances = np.linalg.norm(np.asarray(detection_positions).reshape(-1, 2) - np.asarray(center).reshape(-1, 2), axis=1)
            is_close = np.min(distances) < self.max_radius
            if is_close:
                color_index = overlap_index
                overlap_index += 1
            else:
                color_index = 0
            
            detection_positions.append(center)
            detection_color = tuple(int(c) for c in self.colormap[color_index % len(self.colormap)]) # In case user colormap and color_index > len(colormap)
            cv2.polylines(frame, [obb], isClosed=True, color=detection_color, thickness=self.obb_thickness)
            cv2.circle(frame, center, radius=self.point_thickness, color=detection_color, thickness=-1)
        
        return frame

    def draw_tracks(self, frame, mot_frame):
        for obb, center, memory_line, color in self.tracks.generate_tracks(mot_frame):

            color = tuple(int(c) for c in color)

            if len(memory_line) > 1:
                cv2.polylines(frame, [np.array(memory_line, dtype=np.int32)], isClosed=False, color=color, thickness=self.line_thickness)
            
            if obb is not None:
                cv2.polylines(frame, [obb], isClosed=True, color=color, thickness=self.obb_thickness)
                cv2.circle(frame, center, radius=self.point_thickness, color=color, thickness=-1)
            else:
                if len(memory_line) > 1:
                    start_point, end_point = memory_line[-2], memory_line[-1]
                    cv2.arrowedLine(frame, start_point, end_point, color, thickness=self.point_thickness, line_type=cv2.FILLED, tipLength=0.3)
                else:
                    cv2.circle(frame, memory_line[-1], radius=self.point_thickness, color=color, thickness=-1)
                    
        return frame

    def draw(self, frame_list, fr):
        #frame_list = [frame for _, frame in frame_list]
        frame = self.join_frames(frame_list)

        if self.processed_mot is not None:
            mot_frame = self.processed_mot[self.processed_mot[:, 0] == fr, :]
            if self.tracks:
                frame = self.draw_tracks(frame, mot_frame)
            elif len(mot_frame) > 0:
                frame = self.draw_detections(frame, mot_frame)
        
        return frame

    __call__ = draw
