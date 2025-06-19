import numpy as np

from Functions.SecondModule.SecondModuleFunctions import determine_target_length, extend_line, extend_line_opposite, find_polygon_from_intersections, get_extreme_points



def extend_from_direction(points_dict, start_direction, target_length, current_length=None):
    """
    Extends lines based on the starting direction and target length.
    
    Args:
        points_dict (dict): Dictionary with left and right points
        start_direction (str): Direction to start from ('left' or 'right')
        target_length (float): Target length for extension
        current_length (float, optional): Current length if extending opposite
        
    Returns:
        tuple: Start and end points
    """
    if start_direction == "left":
        start = points_dict["left"]
        end = extend_line(start, points_dict["right"], target_length)
    else:
        start = points_dict["right"]
        if current_length is not None:
            end = extend_line_opposite(points_dict["left"], start, 
                                     abs(target_length - current_length))
    return start, end

def find_polygon_from_two_clean_intersections(unique_corners, sorted_top_lines, 
                                            sorted_bottom_lines, width_px, height_px):
    """
    Creates a polygon from two clean intersections.
    """
    # Get horizontal lines
    top_angles = sorted(sorted_top_lines.keys())
    bottom_angles = sorted(sorted_bottom_lines.keys())
    
    horizontal_top_lines = sorted_top_lines[top_angles[1]]
    horizontal_bottom_lines = sorted_bottom_lines[bottom_angles[1]]
    
    # Get extreme points
    top_points = get_extreme_points(horizontal_top_lines)
    bottom_points = get_extreme_points(horizontal_bottom_lines)
    
    # Determine target length and closest corner
    target_length = determine_target_length(unique_corners, width_px, height_px)
    
    unique_corners_array = np.array(unique_corners)
    min_corner = unique_corners_array[
        np.argmin(unique_corners_array[:, 0] + unique_corners_array[:, 1])
    ]
    
    # Determine extension direction
    dist_top_left = np.sqrt(np.sum((top_points["left"] - min_corner)**2))
    dist_top_right = np.sqrt(np.sum((top_points["right"] - min_corner)**2))
    start_direction = "left" if dist_top_left < dist_top_right else "right"
    
    # Extend lines
    current_length = None
    if start_direction == "right":
        current_length = np.sqrt(np.sum(
            (top_points["right"] - top_points["left"])**2
        ))
    
    top_start, top_end = extend_from_direction(
        top_points, start_direction, target_length, current_length
    )
    bottom_start, bottom_end = extend_from_direction(
        bottom_points, start_direction, target_length, current_length
    )
    
    # Create final polygon
    all_corners = np.vstack((unique_corners, [top_end, bottom_end]))
    return find_polygon_from_intersections(all_corners)