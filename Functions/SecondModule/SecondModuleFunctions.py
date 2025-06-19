import numpy as np
import cv2


def group_lines_by_angle(lines, angle_tolerance=5):
    """
    Group lines by their angles within a tolerance.

    Args:
        lines (list): List of line segments
        angle_tolerance (float): Maximum angle difference for grouping

    Returns:
        dict: Groups of lines keyed by their base angle
    """
    angles = [(calculate_angle(line[0], line[1]), line) for line in lines]
    angles.sort(key=lambda x: x[0])

    line_groups = {}

    for angle, line in angles:
        matched = False
        for base_angle in line_groups:
            if abs(angle - base_angle) < angle_tolerance:
                line_groups[base_angle].append(line)
                matched = True
                break

        if not matched:
            line_groups[angle] = [line]

    return line_groups


def calculate_angle(p1, p2):
    """
    Calculate the angle between two points relative to horizontal.

    Args:
        p1 (tuple): First point coordinates (x,y)
        p2 (tuple): Second point coordinates (x,y)

    Returns:
        float: Angle in degrees (0-180)
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = np.degrees(np.arctan2(dy, dx)) % 180
    return angle


def extend_line(p1, p2, length):
    """
    Extends a line from p1 to p2 by a specified length.

    Args:
        p1 (tuple): The first point (x, y) of the line
        p2 (tuple): The second point (x, y) of the line 
        length (float): The length to extend the line by

    Returns:
        tuple: The coordinates of the end point of the extended line
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    mag = np.sqrt(dx*dx + dy*dy)
    dx, dy = dx/mag, dy/mag

    end_point = (
        int(p1[0] + dx * length),
        int(p1[1] + dy * length)
    )
    return end_point


def extend_line_opposite(p1, p2, length):
    """
    Extends a line in the opposite direction from p1 to p2 by a specified length.

    Args:
        p1 (tuple): The first point (x, y) of the line
        p2 (tuple): The second point (x, y) of the line 
        length (float): The length to extend the line by

    Returns:
        tuple: The coordinates of the end point of the extended line
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # Normalizar el vector direccional
    mag = np.sqrt(dx*dx + dy*dy)
    dx, dy = dx/mag, dy/mag
    # Calcular el punto final
    end_point = (
        int(p1[0] - dx * length),
        int(p1[1] - dy * length)
    )
    return end_point


def get_extreme_points(lines):
    """
    Extracts the extreme points (leftmost, rightmost, topmost, bottommost) from a set of lines.
    Args:
        lines (list): A list of lines where each line is represented by two points [(x1,y1), (x2,y2)]
    Returns:
        dict: A dictionary containing the extreme points with keys:
            - 'left': Point with minimum x-coordinate (leftmost)
            - 'right': Point with maximum x-coordinate (rightmost) 
            - 'top': Point with minimum y-coordinate (topmost)
            - 'bottom': Point with maximum y-coordinate (bottommost)
            Each point is represented as a numpy array [x, y]
    """
    all_points = []
    for line in lines:
        all_points.extend([line[0], line[1]])

    points = np.array(all_points)
    leftmost_idx = np.argmin(points[:, 0])
    rightmost_idx = np.argmax(points[:, 0])
    topmost_idx = np.argmin(points[:, 1])
    bottommost_idx = np.argmax(points[:, 1])

    return {
        'left': points[leftmost_idx],
        'right': points[rightmost_idx],
        'top': points[topmost_idx],
        'bottom': points[bottommost_idx]
    }


def are_points_similar(p1, p2, tolerance=100):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) < tolerance


def find_intersections_between_groups(group1_lines, group2_lines, canvas_shape, tolerance=5):
    """
    Find intersection points between two groups of lines.

    Args:
        group1_lines (list): First group of lines
        group2_lines (list): Second group of lines
        canvas_shape (tuple): Image dimensions (height, width)
        tolerance (int): Pixel tolerance for unique intersections

    Returns:
        numpy.ndarray: Array of unique intersection points
    """
    unique_intersections = set()

    for line1 in group1_lines:
        for line2 in group2_lines:
            x1, y1 = line1[0]
            x2, y2 = line1[1]
            x3, y3 = line2[0]
            x4, y4 = line2[1]

            denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

            if denominator == 0:
                continue

            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator

            px = int(x1 + t * (x2 - x1))
            py = int(y1 + t * (y2 - y1))

            if 0 <= px < canvas_shape[1] and 0 <= py < canvas_shape[0]:
                is_unique = True

                for existing_x, existing_y in unique_intersections:
                    if np.sqrt((px - existing_x)**2 + (py - existing_y)**2) < tolerance:
                        is_unique = False
                        break

                if is_unique:
                    unique_intersections.add((px, py))
    return np.array(list(unique_intersections))


def extend_all_lines_and_find_corners(img, all_lines, extension_length=1000, tolerance=100):
    """
    Extends all lines and finds their intersection points to determine corners.

    Args:
        img (numpy.ndarray): Input image
        all_lines (list): List of line segments [(x1,y1), (x2,y2)]
        extension_length (int): Length to extend lines by
        tolerance (int): Pixel tolerance for considering points as similar

    Returns:
        list: List of unique corner points found from line intersections
    """
    extended_lines = []
    for line in all_lines:
        p1, p2 = line
        ext_p1 = extend_line_opposite(p2, p1, extension_length)
        ext_p2 = extend_line(p1, p2, extension_length)
        extended_lines.append((ext_p1, ext_p2))

    grouped_lines = group_lines_by_angle(extended_lines)
    angles = sorted(grouped_lines.keys())

    all_intersections = []
    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):
            intersections = find_intersections_between_groups(
                grouped_lines[angles[i]],
                grouped_lines[angles[j]],
                (img.shape[1] * 3, img.shape[0] * 3)
            )
            if len(intersections) > 0:
                all_intersections.extend(intersections)

    unique_corners = []
    for intersection in all_intersections:
        is_unique = True
        for existing_corner in unique_corners:
            if are_points_similar(intersection, existing_corner, tolerance):
                is_unique = False
                break
        if is_unique:
            unique_corners.append(intersection)
    return unique_corners


def find_polygon_from_intersections(intersections):
    """
    Find the convex hull polygon from intersection points.

    Args:
        intersections (numpy.ndarray): Array of intersection points

    Returns:
        numpy.ndarray: Convex hull of points or None if insufficient points
    """
    if not isinstance(intersections, np.ndarray):
        intersections = np.array(intersections)
    if len(intersections.shape) == 2:
        intersections = intersections.reshape(-1, 1, 2)
    intersections = intersections.astype(np.float32)
    hull = cv2.convexHull(intersections)
    return hull


def order_polygon_points(points, centroid=None):
    """
    Orders polygon points clockwise starting from the topmost point.

    Args:
        points (list/ndarray): List of points defining the polygon vertices
        centroid (ndarray, optional): Centroid point of the polygon. If None, 
                                    calculated from points. Defaults to None.

    Returns:
        list: Points ordered clockwise around the centroid starting from topmost point
    """
    if centroid is None:
        centroid = np.mean(points, axis=0)

    return sorted(points, key=lambda p: (np.arctan2(p[1] - centroid[1],
                                                    p[0] - centroid[0]) + np.pi) % (2*np.pi))


def standardize_polygon(polygon):
    """
    Standardizes polygon representation to ordered points in clockwise direction.

    Args:
        polygon (list/ndarray): Input polygon points, can be in various formats

    Returns:
        ndarray: Standardized polygon points as int32 array ordered clockwise,
                empty list if input is invalid
    """
    if polygon is None or len(polygon) == 0:
        return []
    if not isinstance(polygon, np.ndarray):
        polygon = np.array(polygon)

    polygon = polygon.reshape(-1, 2)

    if len(polygon) > 2:
        centroid = np.mean(polygon, axis=0)
        polygon = np.array(order_polygon_points(
            polygon, centroid), dtype=np.int32)

    return polygon


def determine_target_length(unique_corners, width_px, height_px):
    """
    Determines target length based on distance between corners.

    Args:
        unique_corners (list): List of corner points
        width_px (int): Width in pixels
        height_px (int): Height in pixels

    Returns:
        float: Target length for extension
    """
    distance = np.sqrt((unique_corners[0][0] - unique_corners[1][0])**2 +
                       (unique_corners[0][1] - unique_corners[1][1])**2)
    return height_px if width_px - 50 <= distance <= width_px + 50 else width_px


def get_horizontal_lines_from_sorted(sorted_lines, angles):
    """
    Gets horizontal lines from sorted lines dictionary.

    Args:
        sorted_lines (dict): Dictionary of lines sorted by angle
        angles (list): List of sorted angles

    Returns:
        list: Horizontal lines
    """
    if len(angles) == 1:
        return sorted_lines[angles[0]]
    return sorted_lines[angles[1]]


def calculate_distances_to_points(corner, points_dict):
    """
    Calculates distances from a corner to left and right points.

    Args:
        corner (numpy.ndarray): Corner point
        points_dict (dict): Dictionary with 'left' and 'right' points

    Returns:
        tuple: (min_distance, 'left' or 'right' indicating closest point)
    """
    dist_left = np.sqrt(np.sum((corner - points_dict["left"])**2))
    dist_right = np.sqrt(np.sum((corner - points_dict["right"])**2))
    return min(dist_left, dist_right), "left" if dist_left < dist_right else "right"


def process_uncovered_area(uncovered_coords, bounds, fov_width, fov_height, margin_px):
    """
    Calculate new camera positions for uncovered areas.

    Args:
        uncovered_coords (tuple): Y and X coordinates of uncovered pixels
        bounds (dict): Dictionary with area boundaries
        fov_width (float): Camera field of view width
        fov_height (float): Camera field of view height
        margin_px (int): Margin in pixels for position calculation

    Returns:
        list: New camera positions to cover uncovered areas
    """
    y_coords, x_coords = uncovered_coords
    if len(x_coords) == 0:
        return []

    x_max, x_min = np.max(x_coords), np.min(x_coords)
    y_max, y_min = np.max(y_coords), np.min(y_coords)

    if (x_max - x_min) > (fov_width + margin_px) or (y_max - y_min) > (fov_height + margin_px):
        return [
            (bounds['min_x'] + x_min + fov_width/2 - margin_px,
             bounds['min_y'] + y_min + fov_height/2 - margin_px),  # Top-left
            (bounds['min_x'] + x_max - fov_width/2 + margin_px,
             bounds['min_y'] + y_max - fov_height/2 + margin_px)   # Bottom-right
        ]
    else:
        return [(bounds['min_x'] + (x_max + x_min)/2,
                bounds['min_y'] + (y_max + y_min)/2)]


def create_coverage_masks(polygon_points, positions, bounds, fov_width, fov_height):
    """
    Create masks for polygon area and covered areas.

    Args:
        polygon_points (numpy.ndarray): Points defining the polygon
        positions (list): List of camera positions
        bounds (dict): Dictionary with area boundaries
        fov_width (float): Camera field of view width
        fov_height (float): Camera field of view height

    Returns:
        tuple: (polygon mask, covered areas mask)
    """
    mask = np.zeros(
        (int(bounds['height'])+1, int(bounds['width'])+1), dtype=np.uint8)
    polygon_for_mask = polygon_points - [bounds['min_x'], bounds['min_y']]
    cv2.fillPoly(mask, [polygon_for_mask.astype(np.int32)], 1)

    covered_mask = np.zeros_like(mask)
    for x, y in positions:
        x_norm = x - bounds['min_x']
        y_norm = y - bounds['min_y']
        x1 = max(0, int(x_norm - fov_width/2))
        y1 = max(0, int(y_norm - fov_height/2))
        x2 = min(mask.shape[1], int(x_norm + fov_width/2))
        y2 = min(mask.shape[0], int(y_norm + fov_height/2))
        covered_mask[y1:y2, x1:x2] = 1

    return mask, covered_mask


def calculate_photo_positions_diagonal(polygon, fov_width, fov_height, margin_px=54, corners=None):
    """
    Calculate optimal camera positions for complete polygon coverage.

    Args:
        polygon (numpy.ndarray): Points defining the polygon
        fov_width (float): Camera field of view width
        fov_height (float): Camera field of view height
        margin_px (int): Margin in pixels for position calculation
        corners (str): Starting corner strategy ("topleft-bottomright" or "topright-bottomleft")

    Returns:
        tuple: (list of positions, coverage percentage)
    """
    polygon_points = polygon.reshape(-1, 2)
    bounds = {
        'min_x': np.min(polygon_points[:, 0]),
        'max_x': np.max(polygon_points[:, 0]),
        'min_y': np.min(polygon_points[:, 1]),
        'max_y': np.max(polygon_points[:, 1]),
        'width': np.max(polygon_points[:, 0]) - np.min(polygon_points[:, 0]),
        'height': np.max(polygon_points[:, 1]) - np.min(polygon_points[:, 1])
    }

    if bounds['width'] <= fov_width and bounds['height'] <= fov_height:
        return [(bounds['min_x'] + bounds['width']/2, bounds['min_y'] + bounds['height']/2)], 100.0

    positions = []
    if corners == "topleft-bottomright":
        positions.extend([
            (bounds['min_x'] + fov_width/2 - margin_px,
             bounds['min_y'] + fov_height/2 - margin_px),
            (bounds['max_x'] - fov_width/2 + margin_px,
             bounds['max_y'] - fov_height/2 + margin_px)
        ])
    else:
        positions.extend([
            (bounds['max_x'] - fov_width/2 + margin_px,
             bounds['min_y'] + fov_height/2 - margin_px),
            (bounds['min_x'] + fov_width/2 - margin_px,
             bounds['max_y'] - fov_height/2 + margin_px)
        ])

    max_iterations = 3
    iteration = 0

    while iteration < max_iterations:
        mask, covered_mask = create_coverage_masks(
            polygon_points, positions, bounds, fov_width, fov_height)
        uncovered = np.logical_and(mask, np.logical_not(covered_mask))

        if not np.any(uncovered):
            coverage_percentage = 100 * (1 - np.sum(uncovered) / np.sum(mask))
            return positions, coverage_percentage

        uncovered_coords = np.where(uncovered)
        new_positions = process_uncovered_area(
            uncovered_coords, bounds, fov_width, fov_height, margin_px)

        if not new_positions:
            coverage_percentage = 100 * (1 - np.sum(uncovered) / np.sum(mask))
            return positions, coverage_percentage

        positions.extend(new_positions)
        iteration += 1

    mask, covered_mask = create_coverage_masks(
        polygon_points, positions, bounds, fov_width, fov_height)
    uncovered = np.logical_and(mask, np.logical_not(covered_mask))
    coverage_percentage = 100 * (1 - np.sum(uncovered) / np.sum(mask))

    return positions, coverage_percentage
