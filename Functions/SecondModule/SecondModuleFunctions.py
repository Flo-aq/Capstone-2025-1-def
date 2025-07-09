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


def extend_all_lines_and_find_corners(img, all_lines, extension_length=1000, tolerance=150):
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
    best_pair = None
    best_diff = float('inf')
    for i in range(len(angles)):
        for j in range(i+1, len(angles)):
            diff = abs((angles[i] - angles[j]) % 180)
            diff = min(diff, 180 - diff)  # Asegura que la diferencia esté en [0, 90]
            if abs(diff - 90) < best_diff:
                best_diff = abs(diff - 90)
                best_pair = (angles[i], angles[j])

    # Filtrar grouped_lines para dejar solo los dos grupos más perpendiculares
    if best_pair:
        grouped_lines = {angle: grouped_lines[angle] for angle in best_pair}
        angles = list(best_pair)

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
    img_center = np.array([img.shape[1] / 2, img.shape[0] / 2])
    for intersection in all_intersections:
        is_unique = True
        replace_idx = None
        for idx, existing_corner in enumerate(unique_corners):
            if are_points_similar(intersection, existing_corner, tolerance):
                # Comparar distancias al centro
                dist_new = np.linalg.norm(np.array(intersection) - img_center)
                dist_existing = np.linalg.norm(np.array(existing_corner) - img_center)
                if dist_new > dist_existing:
                    replace_idx = idx
                is_unique = False
                break
        if is_unique:
            unique_corners.append(intersection)
        elif replace_idx is not None:
            unique_corners[replace_idx] = intersection
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


def calculate_photo_positions_diagonal(polygon, fov_width, fov_height, margin_px=10, corners=None):
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


###################################

def calculate_photo_positions_diagonal_with_overlap(polygon, fov_width, fov_height, mm_to_px_h, mm_to_px_v, height, width, corners=None, border_mm=30, min_overlap=0.3):
    """
    Calcula posiciones de cámara: primero en esquinas opuestas, luego cubre áreas no cubiertas
    alineando el borde del FOV con el borde de la bounding box del área no cubierta.
    Si no se cumple el overlap, prueba dos posiciones desplazadas en Y.
    """
    polygon_points = polygon.reshape(-1, 2)
    border_px_h = border_mm * mm_to_px_h
    border_px_v = border_mm * mm_to_px_v

    bounds = {
        'min_x': np.min(polygon_points[:, 0]) + border_px_h,
        'max_x': np.max(polygon_points[:, 0]) - border_px_h,
        'min_y': np.min(polygon_points[:, 1]) + border_px_v,
        'max_y': np.max(polygon_points[:, 1]) - border_px_v,
        'width': np.max(polygon_points[:, 0]) - np.min(polygon_points[:, 0]) - 2 * border_px_h,
        'height': np.max(polygon_points[:, 1]) - np.min(polygon_points[:, 1]) - 2 * border_px_v
    }

    positions = []
    # 1. Esquinas opuestas
    if corners == "topleft-bottomright":
        positions.append((bounds['min_x'] + fov_width/2, bounds['min_y'] + fov_height/2))
        positions.append((bounds['max_x'] - fov_width/2, bounds['max_y'] - fov_height/2))
    else:  # "topright-bottomleft"
        positions.append((bounds['max_x'] - fov_width/2, bounds['min_y'] + fov_height/2))
        positions.append((bounds['min_x'] + fov_width/2, bounds['max_y'] - fov_height/2))

    max_iterations = 10
    iteration = 0

    while iteration < max_iterations:
        # Crear máscaras
        mask, covered_mask = create_coverage_masks(
            polygon_points, positions, bounds, fov_width, fov_height)
        uncovered = np.logical_and(mask, np.logical_not(covered_mask))
        
        if uncovered:
          uncovered_uint8 = (uncovered.astype(np.uint8)) * 255
          contours, _ = cv2.findContours(uncovered_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
          uncovered_areas_and_contours = [[cv2.contourArea(contour), contour] for contour in contours if cv2.contourArea(contour) > 0]
          img_center_x, img_center_y = width / 2, height / 2
          if uncovered_areas_and_contours:
            for area, contour in uncovered_areas_and_contours:
                x, y, w, h = cv2.boundingRect(contour)
                bbox_center_x = x + w / 2
                #Area a la izq
                if bbox_center_x < img_center_x:
                    new_x = x - fov_width/2
                    new_y = y
                else:
                    new_x = x + fov_width/2
                    new_y = y
                  

def get_starting_position(bounds, corners, fov_width, fov_height):
    """Obtener la posición inicial según la esquina especificada"""
    if corners == "topleft-bottomright":
        return (bounds['min_x'] + fov_width/2, bounds['min_y'] + fov_height/2)
    else:  # "topright-bottomleft"
        return (bounds['max_x'] - fov_width/2, bounds['min_y'] + fov_height/2)

def find_next_position_sequential(positions, bounds, fov_width, fov_height, step_x, step_y, 
                                direction, row, corners, polygon_points, min_overlap):
    """Encontrar la siguiente posición en la secuencia boustrophedon"""
    
    if not positions:
        return None
    
    last_position = positions[-1]
    
    # Calcular posición candidata basada en la dirección actual
    if row == 0:  # Primera fila
        if direction == 'right':
            candidate_x = last_position[0] + step_x
            candidate_y = last_position[1]
        else:  # direction == 'left'
            candidate_x = last_position[0] - step_x
            candidate_y = last_position[1]
    else:  # Filas siguientes
        if direction == 'right':
            # Buscar el inicio de la nueva fila (lado izquierdo)
            candidate_x = bounds['min_x'] + fov_width/2
            candidate_y = bounds['min_y'] + fov_height/2 + row * step_y
        else:  # direction == 'left'
            # Buscar el inicio de la nueva fila (lado derecho)
            candidate_x = bounds['max_x'] - fov_width/2
            candidate_y = bounds['min_y'] + fov_height/2 + row * step_y
        
        # Si ya estamos en la fila, avanzar horizontalmente
        if len(positions) > 1 and abs(positions[-1][1] - candidate_y) < step_y/2:
            if direction == 'right':
                candidate_x = last_position[0] + step_x
                candidate_y = last_position[1]
            else:
                candidate_x = last_position[0] - step_x
                candidate_y = last_position[1]
    
    candidate_position = (candidate_x, candidate_y)
    
    
    # Verificar si la posición está dentro de los límites
    if not is_position_within_bounds(candidate_position, bounds, fov_width, fov_height):
        return None
    
    # Verificar si está dentro del polígono
    if not is_position_in_polygon(candidate_position, polygon_points, fov_width, fov_height):
        return None
    
    # Verificar que tenga overlap suficiente con las imágenes anteriores
    if not has_sufficient_overlap_with_previous(candidate_position, positions, fov_width, fov_height, min_overlap):
        return None
    
    return candidate_position

def is_position_within_bounds(position, bounds, fov_width, fov_height):
    """Verificar si una posición está dentro de los límites del área"""
    return (bounds['min_x'] + fov_width/2 <= position[0] <= bounds['max_x'] - fov_width/2 and
            bounds['min_y'] + fov_height/2 <= position[1] <= bounds['max_y'] - fov_height/2)

def is_position_in_polygon(position, polygon_points, fov_width, fov_height):
    """Verificar si una posición de cámara cubre área significativa dentro del polígono"""
    # Verificar el centro
    if cv2.pointPolygonTest(polygon_points.astype(np.int32), position, False) >= 0:
        return True
    
    # Verificar múltiples puntos dentro del FOV
    test_points = [
        (position[0], position[1]),  # centro
        (position[0] - fov_width/4, position[1] - fov_height/4),
        (position[0] + fov_width/4, position[1] - fov_height/4),
        (position[0] + fov_width/4, position[1] + fov_height/4),
        (position[0] - fov_width/4, position[1] + fov_height/4)
    ]
    
    points_inside = sum(1 for point in test_points 
                       if cv2.pointPolygonTest(polygon_points.astype(np.int32), point, False) >= 0)
    
    # Aceptar si al menos 2 puntos están dentro del polígono
    return points_inside >= 2

def has_sufficient_overlap_with_previous(candidate_position, previous_positions, fov_width, fov_height, min_overlap):
    """Verificar si una posición candidata tiene overlap suficiente con las posiciones anteriores"""
    if not previous_positions:
        return True
    
    # Debe tener overlap mínimo con al menos una imagen anterior
    for prev_pos in previous_positions:
        overlap_ratio = calculate_overlap_area(candidate_position, prev_pos, fov_width, fov_height)
        if overlap_ratio >= min_overlap:
            return True
    
    return False

def calculate_overlap_area(pos1, pos2, fov_width, fov_height):
    """Calcular el área de overlap entre dos posiciones"""
    rect1 = [pos1[0] - fov_width/2, pos1[1] - fov_height/2, 
             pos1[0] + fov_width/2, pos1[1] + fov_height/2]
    rect2 = [pos2[0] - fov_width/2, pos2[1] - fov_height/2, 
             pos2[0] + fov_width/2, pos2[1] + fov_height/2]
    
    x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]))
    y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]))
    
    overlap_area = x_overlap * y_overlap
    image_area = fov_width * fov_height
    
    return overlap_area / image_area if image_area > 0 else 0

def is_coverage_complete(positions, polygon_points, bounds, fov_width, fov_height):
    """Verificar si las posiciones actuales proporcionan cobertura completa"""
    if not positions:
        return False
    
    # Crear máscara de cobertura
    coverage_mask = create_coverage_mask_from_positions(positions, bounds, fov_width, fov_height)
    polygon_mask = create_polygon_mask(polygon_points, bounds)
    
    # Verificar si hay área no cubierta
    uncovered_area = np.sum(polygon_mask & ~coverage_mask)
    total_polygon_area = np.sum(polygon_mask)
    
    # Considerar cobertura completa si menos del 1% queda sin cubrir
    print(f"Uncovered area: {uncovered_area}, Total polygon area: {total_polygon_area}")
    coverage_ratio = 1 - (uncovered_area / total_polygon_area) if total_polygon_area > 0 else 1
    return coverage_ratio >= 1

def ensure_complete_coverage_sequential(positions, polygon_points, bounds, fov_width, fov_height, min_overlap):
    """Asegurar cobertura completa añadiendo posiciones adicionales si es necesario"""
    max_additional = 20  # Máximo número de posiciones adicionales
    added = 0
    
    while added < max_additional:
        if is_coverage_complete(positions, polygon_points, bounds, fov_width, fov_height):
            break
        
        # Encontrar área más grande no cubierta
        coverage_mask = create_coverage_mask_from_positions(positions, bounds, fov_width, fov_height)
        polygon_mask = create_polygon_mask(polygon_points, bounds)
        uncovered_mask = polygon_mask & ~coverage_mask
        
        if np.sum(uncovered_mask) == 0:
            break
        
        # Encontrar centro del área no cubierta más grande
        best_position = find_best_additional_position(
            uncovered_mask, positions, bounds, fov_width, fov_height, polygon_points, min_overlap
        )
        
        if best_position is None:
            break
        
        positions.append(best_position)
        added += 1
    
    return positions

def find_best_additional_position(uncovered_mask, existing_positions, bounds, fov_width, fov_height, 
                                polygon_points, min_overlap):
    """Encontrar la mejor posición adicional para cubrir área no cubierta"""
    # Encontrar contornos de áreas no cubiertas
    contours, _ = cv2.findContours(uncovered_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_position = None
    best_score = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < (fov_width * fov_height * 0.05):  # Ignorar áreas muy pequeñas
            continue
        
        # Encontrar centro del área
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"]) + (bounds['min_x'] - bounds['width']/2)
            cy = int(M["m01"] / M["m00"]) + (bounds['min_y'] - bounds['height']/2)
            
            candidate_pos = (cx, cy)
            
            # Verificar validez de la posición
            if (is_position_within_bounds(candidate_pos, bounds, fov_width, fov_height) and
                is_position_in_polygon(candidate_pos, polygon_points, fov_width, fov_height) and
                has_sufficient_overlap_with_previous(candidate_pos, existing_positions, fov_width, fov_height, min_overlap)):
                
                if area > best_score:
                    best_score = area
                    best_position = candidate_pos
    
    return best_position

def create_coverage_mask_from_positions(positions, bounds, fov_width, fov_height):
    """Crear máscara de cobertura desde lista de posiciones"""
    mask = np.zeros((int(bounds['height']) + 1, int(bounds['width']) + 1), dtype=bool)
    
    for pos in positions:
        x1 = int(pos[0] - fov_width/2 - (bounds['min_x'] - bounds['width']/2))
        y1 = int(pos[1] - fov_height/2 - (bounds['min_y'] - bounds['height']/2))
        x2 = int(pos[0] + fov_width/2 - (bounds['min_x'] - bounds['width']/2))
        y2 = int(pos[1] + fov_height/2 - (bounds['min_y'] - bounds['height']/2))
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(mask.shape[1], x2)
        y2 = min(mask.shape[0], y2)
        
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    
    return mask

def create_polygon_mask(polygon_points, bounds):
    """Crear máscara del polígono"""
    mask = np.zeros((int(bounds['height']) + 1, int(bounds['width']) + 1), dtype=np.uint8)
    
    adjusted_polygon = polygon_points.astype(np.float64).copy()
    adjusted_polygon[:, 0] -= (bounds['min_x'] - bounds['width']/2)
    adjusted_polygon[:, 1] -= (bounds['min_y'] - bounds['height']/2)
    
    cv2.fillPoly(mask, [adjusted_polygon.astype(np.int32)], True)
    return mask

def validate_stitching_sequence(positions, fov_width, fov_height, min_overlap):
    """Validar que la secuencia de posiciones es válida para stitching"""
    if len(positions) < 2:
        return True, "Secuencia válida (una sola imagen o vacía)"
    
    validation_results = []
    total_overlaps = 0
    
    for i in range(len(positions) - 1):
        current_pos = positions[i]
        next_pos = positions[i + 1]
        
        overlap = calculate_overlap_area(current_pos, next_pos, fov_width, fov_height)
        total_overlaps += overlap
        
        validation_results.append({
            'from_image': i + 1,
            'to_image': i + 2,
            'overlap_ratio': overlap,
            'valid': overlap >= min_overlap,
            'distance': np.sqrt((current_pos[0] - next_pos[0])**2 + (current_pos[1] - next_pos[1])**2)
        })
    
    all_valid = all(result['valid'] for result in validation_results)
    avg_overlap = total_overlaps / len(validation_results) if validation_results else 0
    
    summary = f"Secuencia {'VÁLIDA' if all_valid else 'INVÁLIDA'} - Overlap promedio: {avg_overlap:.1%}"
    
    return all_valid, summary, validation_results

def print_stitching_pattern(positions, fov_width, fov_height):
    """Imprimir el patrón de stitching para visualización"""
    if not positions:
        print("No hay posiciones para mostrar")
        return
    
    print("\n=== PATRÓN DE STITCHING SECUENCIAL ===")
    print("Secuencia de captura:")
    
    for i, pos in enumerate(positions, 1):
        print(f"  {i:2d}. X:{pos[0]:7.1f}, Y:{pos[1]:7.1f}")
        
        if i < len(positions):
            next_pos = positions[i]
            overlap = calculate_overlap_area(pos, next_pos, fov_width, fov_height)
            distance = np.sqrt((pos[0] - next_pos[0])**2 + (pos[1] - next_pos[1])**2)
            print(f"      ↓ Overlap: {overlap:.1%}, Distancia: {distance:.1f}px")
    
    print(f"\nTotal de imágenes: {len(positions)}")

# Función auxiliar para mantener compatibilidad
def create_coverage_masks(polygon_points, positions, bounds, fov_width, fov_height):
    """Función de compatibilidad con el código existente"""
    polygon_mask = create_polygon_mask(polygon_points, bounds)
    coverage_mask = create_coverage_mask_from_positions(positions, bounds, fov_width, fov_height)
    return polygon_mask, coverage_mask