
from copy import copy
from hmac import new
from turtle import update
import numpy as np
from shapely import Point, unary_union
from shapely.geometry import Polygon, box


def order_corners_by_distance_to_edge(polygon, fov_width, fov_height, composite_img_width, composite_img_height):
    corners = list(polygon.exterior.coords)[:-1]
    ordered_corners = []
    for x, y in corners:
        adjusted_corner = adjust_corner_for_fov_limits((x, y), fov_width, fov_height, composite_img_width, composite_img_height)
        dist_left = adjusted_corner[0]
        dist_right = composite_img_width - adjusted_corner[0]
        dist_top = adjusted_corner[1]
        dist_bottom = composite_img_height - adjusted_corner[1]
        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
        ordered_corners.append(((adjusted_corner[0], adjusted_corner[1]), min_dist))
        
    ordered_corners.sort(key=lambda tup: tup[1], reverse=True)
    return [corner for corner, _ in ordered_corners]


def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))


def adjust_corner_for_fov_limits(corner, fov_width, fov_height, composite_img_width, composite_img_height):
    x, y = corner
    center_x = clamp(x, fov_width // 2, composite_img_width - fov_width // 2)
    center_y = clamp(y, fov_height // 2,
                     composite_img_height - fov_height // 2)
    return (center_x, center_y)


def create_fov_box(position, fov_width, fov_height):
    x, y = position
    return box(x - fov_width // 2, y - fov_height // 2,
               x + fov_width // 2, y + fov_height // 2)


def verify_complete_coverage(polygon, captured_regions):
    if not captured_regions:
        return 0.0
    covered_area = unary_union([polygon.intersection(fov_box)
                               for fov_box in captured_regions])
    if polygon.area == 0:
        return 1.0
    return covered_area.area / polygon.area


def create_initial_grid(polygon_geom, fov_width, fov_height, composite_img_width, composite_img_height, step):
    grid = {}
    fov_half_width = fov_width // 2
    fov_half_height = fov_height // 2

    x_positions = np.arange(0, composite_img_width, step)
    y_positions = np.arange(0, composite_img_height, step)
    for i, x in enumerate(x_positions):
        for j, y in enumerate(y_positions):
            if (0 <= x - fov_half_width and x + fov_half_width <= composite_img_width and
                    0 <= y - fov_half_height and y + fov_half_height <= composite_img_height):
                fov_box = box(x - fov_half_width, y - fov_half_height,
                              x + fov_half_width, y + fov_half_height)
                polygon_intersection = fov_box.intersection(polygon_geom)
                polygon_area = polygon_intersection.area if not polygon_intersection.is_empty else 0

                grid_key = (i, j)
                grid[grid_key] = {
                    'position': (x, y),
                    'fov_box': fov_box,
                    'polygon_area': polygon_area,
                    'available': True,
                    'distance_to_covered': float('inf')
                }
    return grid, x_positions, y_positions


def mark_nearby_positions_unavailable(grid, x_positions, y_positions, new_position, radio_steps=3):
    pos_x, pos_y = new_position
    x_idx, y_idx = np.argmin(np.abs(x_positions - pos_x)
                             ), np.argmin(np.abs(y_positions - pos_y))

    for i in range(max(0, x_idx - radio_steps), min(len(x_positions), x_idx + radio_steps + 1)):
        for j in range(max(0, y_idx - radio_steps), min(len(y_positions), y_idx + radio_steps + 1)):
            grid_key = (i, j)
            if grid_key in grid:
                grid[grid_key]['available'] = False


def get_exclusion_range(x_positions, y_positions, new_position, radio_steps=3):
    pos_x, pos_y = new_position
    x_idx = np.argmin(np.abs(x_positions - pos_x))
    y_idx = np.argmin(np.abs(y_positions - pos_y))

    x_start = max(0, x_idx - radio_steps)
    x_end = min(len(x_positions), x_idx + radio_steps + 1)
    y_start = max(0, y_idx - radio_steps)
    y_end = min(len(y_positions), y_idx + radio_steps + 1)

    return (x_start, x_end), (y_start, y_end)


def update_grid_distances_and_availability(grid, covered_regions, step, exclusion_ranges=None):
    if exclusion_ranges:
        for x_range, y_range in exclusion_ranges:
            for i in range(x_range[0], x_range[1]):
                for j in range(y_range[0], y_range[1]):
                    grid_key = (i, j)
                    if grid_key in grid:
                        grid[grid_key]['available'] = False
    if not covered_regions:
        return

    covered_area = unary_union(covered_regions)
    buffer_offset = -step * 25
    covered_area_buffered = covered_area.buffer(buffer_offset)

    for grid_key, grid_info in grid.items():
        if not grid_info['available'] or not grid_info['distance_to_covered'] or grid_info['distance_to_covered'] == 0.0:
            continue
        position = grid_info['position']
        point = Point(position)
        distance_to_covered = point.distance(covered_area)
        if covered_area_buffered.contains(point):
            grid_info['available'] = False
            grid_info['distance_to_covered'] = 0.0
        elif covered_area.contains(point) or not grid_info['distance_to_covered'] or grid_info['distance_to_covered'] == 0.0:
            grid_info['distance_to_covered'] = 0.0
            continue
        grid_info['distance_to_covered'] = distance_to_covered


def get_candidates_from_grid(grid, polygon_geom, covered_regions, min_overlap, max_candidates):
    if not covered_regions:
      return []
    
    covered_area = unary_union(covered_regions)
    candidates = []
    for _, grid_info in grid.items():
        if not grid_info['available']:
            continue
        position = grid_info['position']
        fov_box = grid_info['fov_box']
        covered_background = covered_area.difference(polygon_geom)
        fov_background = fov_box.difference(polygon_geom)
        background_overlap = covered_background.intersection(fov_background)
        overlap_ratio = background_overlap.area / fov_box.area if fov_box.area > 0 else 0
        
        new_polygon_area = fov_box.intersection(polygon_geom.difference(covered_area)).area
        new_polygon_area_norm = new_polygon_area / polygon_geom.area if polygon_geom.area > 0 else 0
        
        if overlap_ratio >= min_overlap:
            area_weight = new_polygon_area_norm
            overlap_weight = overlap_ratio
            combined_weight = 0.9 * area_weight + 0.1 * overlap_weight
            candidates.append((position, combined_weight, overlap_ratio, new_polygon_area_norm))
            
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:max_candidates]

def calculate_photo_positions_with_tree(polygon, fov_width, fov_height, composite_img_height, composite_img_width, min_overlap, max_candidates, max_depth):
    step = int(fov_width / 25)
    polygon_coords = [(float(point[0][0]), float(point[0][1])) for point in polygon]
    polygon_geom = Polygon(polygon_coords).buffer(0)
    
    grid, x_positions, y_positions = create_initial_grid(polygon_geom, fov_width, fov_height, composite_img_width, composite_img_height, step)
    
    corners_ordered = order_corners_by_distance_to_edge(polygon_geom, composite_img_width, composite_img_height)
    best_initial = corners_ordered[0]
    
    initial_fov = create_fov_box(best_initial, fov_width, fov_height)
    
    exclusion_range = get_exclusion_range(x_positions, y_positions, best_initial)
    
    update_grid_distances_and_availability(grid, [initial_fov], step, exclusion_ranges=[exclusion_range])
    
    tree = [
      {
        'positions': [best_initial],
        'regions': [initial_fov],
        'depth': 1,
        'grid_state': grid.copy()
      }
    ]
    
    all_branches = []
    while tree:
        current_branch = tree.pop(0)
        positions = current_branch['positions']
        regions = current_branch['regions']
        depth = current_branch['depth']
        grid_state = current_branch['grid_state']
        
        if depth >= max_depth:
            return all_branches, max_depth
                
        candidates = get_candidates_from_grid(grid_state, polygon_geom, regions, min_overlap, max_candidates)
        
        for candidate_pos, weight, overlap, new_pol in candidates:
            new_grid = copy.deepcopy(grid_state)
            new_fov = create_fov_box(candidate_pos, (fov_width, fov_height))
            new_regions = regions + [new_fov]
            new_positions = positions + [candidate_pos]
            exclusion_range = get_exclusion_range(x_positions, y_positions, candidate_pos)
            update_grid_distances_and_availability(new_grid, new_regions, step, exclusion_ranges=[exclusion_range])
            
            coverage = verify_complete_coverage(polygon_geom, new_regions)
            if coverage >= 0.998:
                max_depth_reached = depth + 1
                print(f"Coverage achieved at depth {max_depth_reached} with positions: {new_positions}")
                return [new_positions]
            
            tree.append({
                'positions': new_positions,
                'regions': new_regions,
                'depth': depth + 1,
                'grid_state': new_grid
            })
            