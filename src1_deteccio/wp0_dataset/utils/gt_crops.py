
import numpy as np
from scipy import optimize
from shapely import Polygon, affinity

from ceab_ants.detection.dataset_prep.bbox_random_centered_crop import propose_random_crop, find_intersections, improve_crop, desired_movments


def random_crop(tracks, seen, crop_width, crop_height, width, height):

    current = np.argwhere(~seen).flatten()[0]
    trk = tracks[current]
    
    initial, final, low, high = propose_random_crop(trk, crop_width, crop_height, width, height)
    intersect = find_intersections(tracks, initial, final)

    unseen_intersect = intersect & (~seen)
    unseen_tracks = tracks[unseen_intersect, :]

    initial, final = improve_crop(unseen_tracks, initial, final, low, high, width, height)

    return initial, final

def filter_annotations(tracks, seen, initial, final):

    delta_w, delta_h = desired_movments(tracks, initial, final)
    seen = seen | ((delta_w == 0) & (delta_h == 0))
    within = ((np.abs(delta_w) < tracks[:, 2]) & (np.abs(delta_h) < tracks[:, 3]))

    return within, seen

def adjust_bbox_annotations(tracks, within, initial, crop_width, crop_height):
    tracks_save = tracks[within, :].copy() # x, y, w, h, s (transformation from x1y1x2y2_to_bbox on main script)
    new_left = tracks_save[:, 0] - initial[0]
    tracks_save[:, 0] = np.clip(new_left, 0, None)
    tracks_save[:, 2] = np.minimum(tracks_save[:, 2], tracks_save[:, 2] + new_left)
    tracks_save[:, 2] = np.minimum(tracks_save[:, 2], crop_width - tracks_save[:, 0])

    new_up = tracks_save[:, 1] - initial[1]
    tracks_save[:, 1] = np.clip(new_up, 0, None)
    tracks_save[:, 3] = np.minimum(tracks_save[:, 3], tracks_save[:, 3] + new_up)
    tracks_save[:, 3] = np.minimum(tracks_save[:, 3], crop_height - tracks_save[:, 1])

    tracks_save[:, 0] = (tracks_save[:, 0] + (tracks_save[:, 2] / 2)) / crop_width
    tracks_save[:, 1] = (tracks_save[:, 1] + (tracks_save[:, 3] / 2)) / crop_height
    tracks_save[:, 2] = tracks_save[:, 2] / crop_width
    tracks_save[:, 3] = tracks_save[:, 3] / crop_height

    return tracks_save # Normalized (cx, cy, w, h)

def sort_clockwise(vertices):
    centroid = np.mean(vertices, axis=0)
    
    def angle(vertex):
        dx, dy = vertex[0] - centroid[0], vertex[1] - centroid[1]
        return np.arctan2(dy, dx)
    
    vertices = sorted(vertices, key=angle) #, reverse=True)
    
    distances = np.linalg.norm(vertices, axis=1)
    min_index = np.argmin(distances)
    vertices = np.roll(vertices, -min_index, axis=0)
    return np.array(vertices)

def adjust_obbox_annotations(tracks, within, initial, crop_width, crop_height, min_area=120): # Min area anotació 360, min w 12 i min h 21
    tracks_save = tracks[within, :].copy() # x, y, w, h, a, s
    
    tracks_save[:, 0] = tracks_save[:, 0] - initial[0] # x', y, w, h, a, s
    tracks_save[:, 1] = tracks_save[:, 1] - initial[1] # x', y', w, h, a, s

    build_obb = lambda x, y, w, h, a, _ : affinity.translate(affinity.rotate(Polygon([(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]), a, use_radians=False), x, y)
    boundary = build_obb(crop_width / 2, crop_height / 2, crop_width, crop_height, 0, 0)

    def process_outside(obb):
        sh_obb = build_obb(*obb)

        if np.isclose(boundary.intersection(sh_obb).area, sh_obb.area):
            return sh_obb
        
        max_area = 0
        best_rect = None
        for angle in np.linspace(0, 180 - 0.5, 180, endpoint=True): # Brute-force the angle but if not it doesn't work
            rotated = affinity.rotate(sh_obb, angle, use_radians=False)
            minx, miny, maxx, maxy = rotated.bounds
            diag = np.linalg.norm(((maxx-minx), (maxy-miny)))

            def objective(params):
                new_obb = build_obb(*params, angle, 0) 
                return new_obb.union(sh_obb.intersection(boundary)).area - 0.3 * new_obb.intersection(sh_obb).intersection(boundary).area - 0.7 * new_obb.intersection(sh_obb).area
            
            result = optimize.minimize(
                objective, # minimize union and maximize intersection with original obbox
                [rotated.centroid.x, rotated.centroid.y, diag * np.sin(obb[-2]), diag * np.cos(obb[-2])], # Start from original obbox
                bounds=[
                    (minx, maxx),
                    (miny, maxy), 
                    (1, diag),
                    (1, diag), 
                ],
                constraints={'type' : 'eq', 'fun' : lambda params : build_obb(*params, angle, 0).intersection(boundary).area - build_obb(*params, angle, 0).area} # Be inside the crop
            )

            rect = build_obb(*result.x, angle, 0)
            if rect.intersection(sh_obb).intersection(boundary).area > max_area:
                max_area = rect.intersection(sh_obb).intersection(boundary).area
                best_rect = rect
                
        if best_rect is None:
            try:
                minx, miny, maxx, maxy = sh_obb.intersection(boundary).bounds
                best_rect = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)])
            except:
                return None
        
        if best_rect.area > min_area:
            return best_rect
        return None
    
    new_poly_gen = (process_outside(obb) for obb in tracks_save)
    tracks_save = np.asarray([sort_clockwise(poly.exterior.coords[:-1]) for poly in new_poly_gen if poly is not None]).reshape(-1, 8)
    tracks_save[:, ::2] = tracks_save[:, ::2] / crop_width
    tracks_save[:, 1::2] = tracks_save[:, 1::2] / crop_height

    return tracks_save # Normalized (x1, y1, x2, y2, x3, y3, x4, y4)
