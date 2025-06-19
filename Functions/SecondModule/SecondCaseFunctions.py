import numpy as np
from Functions.SecondModule.SecondModuleFunctions import extend_line_opposite, get_extreme_points, order_polygon_points


def process_line_points(lines, remaining_height):
    """
    Process lines to find extreme points and create extensions.

    Args:
        lines (list): List of line segments [(x1,y1), (x2,y2)]
        remaining_height (float): Available height for extending lines

    Returns:
        tuple: (dict of extreme points, list of extension points)
            - points: Dictionary with 'left', 'right', 'top', 'bottom' extreme points
            - extensions: List containing two extended points from right to left and vice versa
    """
    points = get_extreme_points(lines)
    extensions = [
        extend_line_opposite(points["right"], points["left"], remaining_height),
        extend_line_opposite(points["left"], points["right"], remaining_height)
    ]
    return points, extensions

def get_intersection_points(ext_points_top, ext_points_bottom):
    """
    Calculate intersection points between top and bottom perpendicular lines.

    Args:
        ext_points_top (dict): Dictionary with 'left' and 'right' points for top lines
        ext_points_bottom (dict): Dictionary with 'left' and 'right' points for bottom lines

    Returns:
        numpy.ndarray: Array of intersection points forming polygon vertices
    """
    perp_top = create_perpendicular_lines(ext_points_top, True)
    perp_bottom = create_perpendicular_lines(ext_points_bottom, False)

    center = np.mean([
        ext_points_top["left"], ext_points_top["right"],
        ext_points_bottom["left"], ext_points_bottom["right"]
    ], axis=0)

    lines = []
    for name, line in {**perp_top, **perp_bottom}.items():
        dist = np.linalg.norm(line[0] - center)
        lines.append((name, line, dist))
    
    closest_lines = sorted(lines, key=lambda x: x[2])[:2]
    
    points = []
    for _, line, _ in closest_lines:
        points.extend([line[0], line[1]])
    
    return np.array(points)

def reconstruct_polygon_from_paralell_lines(lines_top, lines_bottom, height_px):
    """
    Reconstruct a polygon from two sets of parallel lines.

    Args:
        lines_top (list): List of top line segments [(x1,y1), (x2,y2)]
        lines_bottom (list): List of bottom line segments [(x1,y1), (x2,y2)]
        height_px (int): Height of the image in pixels

    Returns:
        numpy.ndarray: Array of polygon vertices ordered clockwise
    """
    top_points = get_extreme_points(lines_top)
    bottom_points = get_extreme_points(lines_bottom)
    height_top = height_px - np.linalg.norm(np.array(top_points["right"]) - np.array(top_points["left"]))
    height_bottom = height_px - np.linalg.norm(np.array(bottom_points["right"]) - np.array(bottom_points["left"]))
    top_points, top_extensions = process_line_points(lines_top, height_top)
    bottom_points, bottom_extensions = process_line_points(lines_bottom, height_bottom)
    polygon_points = get_intersection_points(
        dict(zip(["left", "right"], [top_extensions[0], top_extensions[1]])),
        dict(zip(["left", "right"], [bottom_extensions[0], bottom_extensions[1]]))
    )
    return polygon_points

def create_perpendicular_lines(points, is_top=True, length=3000):
    """
    Creates perpendicular lines from given points in specified direction.

    Args:
        points (dict): Dictionary with 'left' and 'right' points defining the base line
        is_top (bool): Direction of perpendicular lines - True for downward, False for upward
        length (int): Length of perpendicular lines in pixels

    Returns:
        dict: Dictionary containing two perpendicular lines:
            - 'left': (start_point, end_point) for left perpendicular line
            - 'right': (start_point, end_point) for right perpendicular line
    """
    dx = points["right"][0] - points["left"][0]
    dy = points["right"][1] - points["left"][1]
    
    if is_top:
        perp_dx, perp_dy = -dy, dx
    else:
        perp_dx, perp_dy = dy, -dx
        
    mag = np.sqrt(perp_dx**2 + perp_dy**2)
    perp_dx, perp_dy = perp_dx/mag, perp_dy/mag
    
    left_end = (
        int(points["left"][0] + perp_dx * length),
        int(points["left"][1] + perp_dy * length)
    )
    right_end = (
        int(points["right"][0] + perp_dx * length),
        int(points["right"][1] + perp_dy * length)
    )
    
    return {
        "left": (points["left"], left_end),
        "right": (points["right"], right_end)
    }

