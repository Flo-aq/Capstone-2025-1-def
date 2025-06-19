import numpy as np

from Functions.SecondModule.SecondModuleFunctions import (
    extend_line, get_extreme_points, get_horizontal_lines_from_sorted,
    calculate_distances_to_points, determine_target_length,
    find_polygon_from_intersections, extend_line_opposite
)

def calculate_corner_distances(unique_corners, points_dict):
    """
    Calculates distances from corners to points.
    
    Args:
        unique_corners (list): List of corner points
        points_dict (dict): Dictionary with left and right points
        
    Returns:
        list: List of tuples (corner, min_distance, closest_direction)
    """
    distances = []
    for corner in unique_corners:
        min_dist, direction = calculate_distances_to_points(corner, points_dict)
        distances.append((corner, min_dist, direction))
    return sorted(distances, key=lambda x: x[1])

def find_polygon_from_two_unclean_intersections(grouped_lines_top, grouped_lines_bottom, 
                                              unique_corners, width_px, height_px):
    """
    Creates a polygon from two unclean intersections by extending lines to target length.

    Args:
        grouped_lines_top (dict): Dictionary of top lines grouped by angle {angle: [(x1,y1,x2,y2)]}
        grouped_lines_bottom (dict): Dictionary of bottom lines grouped by angle {angle: [(x1,y1,x2,y2)]}
        unique_corners (list): List of unique corner points [(x,y)]
        width_px (int): Image width in pixels
        height_px (int): Image height in pixels

    Returns:
        numpy.ndarray: Array of polygon vertices ordered clockwise
    """
    top_angles = sorted(grouped_lines_top.keys())
    bottom_angles = sorted(grouped_lines_bottom.keys())
    
    horizontal_top_lines = get_horizontal_lines_from_sorted(
        grouped_lines_top, 
        top_angles if len(top_angles) == 1 else top_angles
    )
    horizontal_bottom_lines = get_horizontal_lines_from_sorted(
        grouped_lines_bottom,
        bottom_angles if len(bottom_angles) == 1 else bottom_angles
    )
    
    top_points = get_extreme_points(horizontal_top_lines)
    bottom_points = get_extreme_points(horizontal_bottom_lines)
    
    target_length = determine_target_length(unique_corners, width_px, height_px)
    
    distances_top = calculate_corner_distances(unique_corners, top_points)
    distances_bottom = calculate_corner_distances(unique_corners, bottom_points)
    
    top_start = distances_top[0][0]
    bottom_start = distances_bottom[0][0]
    is_right_corner = distances_top[0][2] == "right"
    
    if is_right_corner:
        top_end = extend_line(top_start, top_points["left"], target_length)
        bottom_end = extend_line(bottom_start, bottom_points["left"], target_length)
    else:
        current_length = np.sqrt(np.sum((top_start - top_points["left"])**2))
        top_end = extend_line_opposite(top_points["left"], top_start, 
                                     abs(target_length - current_length))
        bottom_end = extend_line_opposite(bottom_points["left"], bottom_start,
                                        abs(target_length - current_length))
    
    all_corners = np.vstack((unique_corners, [top_end, bottom_end]))
    return find_polygon_from_intersections(all_corners)