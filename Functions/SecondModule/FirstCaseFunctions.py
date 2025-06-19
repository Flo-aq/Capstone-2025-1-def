import numpy as np

from Functions.SecondModule.SecondModuleFunctions import extend_line, extend_line_opposite, get_extreme_points


def get_vertical_and_horizontal_lines(grouped_lines):
    """
    Separates lines into vertical and horizontal groups based on their angles.
    
    Lines with angles between 45° and 135° are considered vertical,
    while the rest are considered horizontal.
    
    Args:
        lines (list): List of line segments where each line is [(x1,y1), (x2,y2)]
    
    Returns:
        tuple: Two lists containing:
            - vertical_lines: Lines with angles between 45° and 135°
            - horizontal_lines: Lines with angles < 45° or > 135°
    """
    angles = sorted(grouped_lines.keys())
    vertical_lines = []
    horizontal_lines = []
    
    for angle in angles:
        if 45 <= angle <= 135:
            vertical_lines.extend(grouped_lines[angle])
        else:
            horizontal_lines.extend(grouped_lines[angle])
            
    return vertical_lines, horizontal_lines

def create_lines_from_extremes(vertical_lines, horizontal_lines):
    """
    Creates representative lines from the extreme points of vertical and horizontal line groups.
    
    For vertical lines, creates a line from leftmost to rightmost points.
    For horizontal lines, creates a line from topmost to bottommost points.
    
    Args:
        vertical_lines (list): List of vertical line segments
        horizontal_lines (list): List of horizontal line segments
    
    Returns:
        tuple: Two single-element lists containing:
            - vertical_line: [(leftmost_point, rightmost_point)]
            - horizontal_line: [(topmost_point, bottommost_point)]
    """
    vertical_points = get_extreme_points(vertical_lines)
    vertical_line = [(vertical_points['left'], vertical_points['right'])]
    
    horizontal_points = get_extreme_points(horizontal_lines)
    horizontal_line = [(horizontal_points['top'], horizontal_points['bottom'])]
    
    return vertical_line, horizontal_line


    
def reconstruct_top_polygon(width_px, height_px, vertical_line, horizontal_line):
    """
    Reconstructs a polygon representing a paper sheet when its only visible in the top image.
    
    Extends the detected lines to match the expected paper dimensions and creates
    a parallel vertical line to complete the polygon.
    
    Args:
        width_px (float): Expected paper width in pixels
        height_px (float): Expected paper height in pixels
        vertical_line (list): Single-element list containing the vertical line [(x1,y1), (x2,y2)]
        horizontal_line (list): Single-element list containing the horizontal line [(x1,y1), (x2,y2)]
    
    Returns:
        numpy.ndarray: Array of 4 points defining the reconstructed polygon,
                      ordered clockwise from top-left
    """
    vertical_start = vertical_line[0][0]
    vertical_end = extend_line(vertical_start, vertical_line[0][1], width_px)
    
    horizontal_start = horizontal_line[0][0]
    horizontal_current_end = horizontal_line[0][1]
    current_height = np.sqrt(
        (horizontal_current_end[0] - horizontal_start[0])**2 + 
        (horizontal_current_end[1] - horizontal_start[1])**2
    )
    remaining_height = height_px - current_height
    horizontal_end = extend_line_opposite(horizontal_start, horizontal_current_end, remaining_height)
    parallel_vertical = create_parallel_line(vertical_start, vertical_end, horizontal_end, width_px)
    
    return np.array([
        vertical_start,
        vertical_end,
        parallel_vertical[1],
        parallel_vertical[0]
    ], dtype=np.int32)

def reconstruct_bottom_polygon(width_px, height_px, vertical_line, horizontal_line):
    """
    Reconstructs a polygon representing a paper sheet when its only visible in the bottom image.
    
    Similar to reconstruct_top_polygon but handles the case where the visible part
    is at the bottom image.
    
    Args:
        width_px (float): Expected paper width in pixels
        height_px (float): Expected paper height in pixels
        vertical_line (list): Single-element list containing the vertical line [(x1,y1), (x2,y2)]
        horizontal_line (list): Single-element list containing the horizontal line [(x1,y1), (x2,y2)]
    
    Returns:
        numpy.ndarray: Array of 4 points defining the reconstructed polygon,
                      ordered clockwise from top-left
    """
    vertical_start = vertical_line[0][1]
    vertical_end = extend_line(vertical_start, vertical_line[0][0], width_px)
    
    horizontal_start = horizontal_line[0][1]
    horizontal_current_end = horizontal_line[0][0]
    current_height = np.sqrt(
        (-horizontal_current_end[0] + horizontal_start[0])**2 + 
        (-horizontal_current_end[1] + horizontal_start[1])**2
    )
    remaining_height = height_px - current_height
    horizontal_end = extend_line_opposite(horizontal_start, horizontal_current_end, remaining_height)
    parallel_vertical = create_parallel_line(vertical_start, vertical_end, horizontal_end, width_px)
    
    return np.array([
        vertical_start,
        vertical_end,
        parallel_vertical[1],
        parallel_vertical[0]
    ], dtype=np.int32)
    
def create_parallel_line(p1, p2, offset_point, length):
    """
    Creates a parallel line segment from an offset point using the direction of a reference line.
    
    Args:
        p1 (tuple): Starting point (x, y) of the reference line
        p2 (tuple): Ending point (x, y) of the reference line 
        offset_point (tuple): Starting point (x, y) for the parallel line
        length (float): Desired length of the parallel line segment
        
    Returns:
        tuple: A tuple containing two points ((x1,y1), (x2,y2)) representing the parallel line segment,
        where the first point is the offset_point and the second is the calculated end point
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # Normalizar el vector direccional
    mag = np.sqrt(dx*dx + dy*dy)
    dx, dy = dx/mag, dy/mag
    
    # Calcular el punto final
    end_point = (
        int(offset_point[0] + dx * length),
        int(offset_point[1] + dy * length)
    )
    return (offset_point, end_point)