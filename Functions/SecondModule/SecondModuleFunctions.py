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

def calculate_photo_positions_diagonal_with_overlap(polygon, fov_width, fov_height, mm_to_px_h, mm_to_px_v, 
                                                   corners=None, border_mm=30, min_overlap=0.3):
    """
    Calculate optimal camera positions ensuring 100% coverage with ordered sequence for stitching.
    
    Args:
        polygon (numpy.ndarray): Points defining the polygon
        fov_width (float): Camera field of view width
        fov_height (float): Camera field of view height
        mm_to_px_h (float): Millimeters to pixels horizontal conversion
        mm_to_px_v (float): Millimeters to pixels vertical conversion
        corners (str): Starting corner strategy ("topleft-bottomright" or "topright-bottomleft")
        border_mm (float): Border margin in millimeters
        min_overlap (float): Minimum overlap ratio required (0.3 = 30%)
        
    Returns:
        tuple: (ordered list of positions, coverage percentage - always 100%)
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
    
    # Si el área completa cabe en una sola imagen
    if bounds['width'] <= fov_width and bounds['height'] <= fov_height:
        return [(bounds['min_x'] + bounds['width']/2, bounds['min_y'] + bounds['height']/2)], 100.0
    
    # Generar grid de posiciones con overlap garantizado
    step_x = fov_width * (1 - min_overlap)
    step_y = fov_height * (1 - min_overlap)
    
    # Calcular número de imágenes necesarias en cada dirección
    n_cols = int(np.ceil(bounds['width'] / step_x))
    n_rows = int(np.ceil(bounds['height'] / step_y))
    
    # Ajustar el step para distribución uniforme
    if n_cols > 1:
        step_x = (bounds['width'] - fov_width) / (n_cols - 1)
    if n_rows > 1:
        step_y = (bounds['height'] - fov_height) / (n_rows - 1)
    
    # Generar todas las posiciones posibles
    all_positions = []
    for row in range(n_rows):
        for col in range(n_cols):
            x = bounds['min_x'] + fov_width/2 + col * step_x
            y = bounds['min_y'] + fov_height/2 + row * step_y
            all_positions.append((x, y, row, col))
    
    # Filtrar posiciones que están dentro del polígono
    valid_positions = []
    for pos in all_positions:
        if is_position_in_polygon(pos[:2], polygon_points, fov_width, fov_height):
            valid_positions.append(pos)
    
    if not valid_positions:
        return [], 0.0
    
    # Ordenar posiciones según el patrón de barrido
    ordered_positions = order_positions_for_stitching(valid_positions, corners)
    
    # Verificar y ajustar para garantizar 100% cobertura
    ordered_positions = ensure_complete_coverage(
        ordered_positions, polygon_points, bounds, fov_width, fov_height, min_overlap
    )
    
    return [(pos[0], pos[1]) for pos in ordered_positions], 100.0

def is_position_in_polygon(position, polygon_points, fov_width, fov_height):
    """Verificar si una posición de cámara cubre área dentro del polígono"""
    # Verificar el centro
    if cv2.pointPolygonTest(polygon_points.astype(np.int32), position, False) >= 0:
        return True
    
    # Verificar las esquinas del FOV
    corners = [
        (position[0] - fov_width/2, position[1] - fov_height/2),
        (position[0] + fov_width/2, position[1] - fov_height/2),
        (position[0] + fov_width/2, position[1] + fov_height/2),
        (position[0] - fov_width/2, position[1] + fov_height/2)
    ]
    
    for corner in corners:
        if cv2.pointPolygonTest(polygon_points.astype(np.int32), corner, False) >= 0:
            return True
    
    return False

def order_positions_for_stitching(positions, corners):
    """Ordenar posiciones en patrón boustrophedon (ida y vuelta por filas)"""
    if not positions:
        return []
    
    # Convertir a array para facilitar manipulación
    pos_array = np.array(positions)
    
    # Ordenar por filas primero
    rows = sorted(set(pos_array[:, 2]))
    sorted_positions = []
    
    if corners == "topleft-bottomright":
        # Patrón: →→→→→ (izq a der)
        #         ↓
        #         ←←←←← (der a izq)  
        #         ↓
        #         →→→→→ (izq a der)
        for i, row in enumerate(rows):
            row_positions = pos_array[pos_array[:, 2] == row]
            if i % 2 == 0:  # Filas pares: izquierda a derecha
                row_positions = row_positions[row_positions[:, 3].argsort()]
            else:  # Filas impares: derecha a izquierda
                row_positions = row_positions[row_positions[:, 3].argsort()[::-1]]
            sorted_positions.extend(row_positions.tolist())
            
    else:  # "topright-bottomleft"
        # Patrón: ←←←←← (der a izq)
        #         ↓
        #         →→→→→ (izq a der)
        #         ↓
        #         ←←←←← (der a izq)
        for i, row in enumerate(rows):
            row_positions = pos_array[pos_array[:, 2] == row]
            if i % 2 == 0:  # Filas pares: derecha a izquierda
                row_positions = row_positions[row_positions[:, 3].argsort()[::-1]]
            else:  # Filas impares: izquierda a derecha
                row_positions = row_positions[row_positions[:, 3].argsort()]
            sorted_positions.extend(row_positions.tolist())
    
    return sorted_positions

def ensure_complete_coverage(ordered_positions, polygon_points, bounds, fov_width, fov_height, min_overlap):
    """Garantizar cobertura completa del polígono"""
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        # Crear máscara de cobertura actual
        coverage_mask = create_coverage_mask_from_positions(
            ordered_positions, bounds, fov_width, fov_height
        )
        polygon_mask = create_polygon_mask(polygon_points, bounds)
        
        # Encontrar área no cubierta dentro del polígono
        uncovered_mask = polygon_mask & ~coverage_mask
        
        if np.sum(uncovered_mask) == 0:
            break  # Cobertura completa
        
        # Encontrar posiciones adicionales para cubrir áreas faltantes
        additional_positions = find_additional_positions(
            uncovered_mask, ordered_positions, bounds, fov_width, fov_height, 
            polygon_points, min_overlap
        )
        
        if not additional_positions:
            # Si no se pueden encontrar más posiciones, expandir el grid
            ordered_positions = expand_coverage_grid(
                ordered_positions, bounds, fov_width, fov_height, min_overlap
            )
            break
        
        # Insertar nuevas posiciones en la secuencia óptima
        ordered_positions = insert_positions_optimally(
            ordered_positions, additional_positions, fov_width, fov_height
        )
        
        iteration += 1
    
    return ordered_positions

def create_coverage_mask_from_positions(positions, bounds, fov_width, fov_height):
    """Crear máscara de cobertura desde lista de posiciones"""
    mask = np.zeros((int(bounds['height']) + 1, int(bounds['width']) + 1), dtype=bool)
    
    for pos in positions:
        # Calcular rectángulo de cobertura
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
    mask = np.zeros((int(bounds['height']) + 1, int(bounds['width']) + 1), dtype=bool)
    
    # Ajustar coordenadas del polígono
    adjusted_polygon = polygon_points.copy()
    adjusted_polygon[:, 0] -= (bounds['min_x'] - bounds['width']/2)
    adjusted_polygon[:, 1] -= (bounds['min_y'] - bounds['height']/2)
    
    cv2.fillPoly(mask, [adjusted_polygon.astype(np.int32)], True)
    return mask

def find_additional_positions(uncovered_mask, existing_positions, bounds, fov_width, fov_height, 
                            polygon_points, min_overlap):
    """Encontrar posiciones adicionales para cubrir áreas no cubiertas"""
    additional_positions = []
    
    # Encontrar centros de áreas no cubiertas
    contours, _ = cv2.findContours(uncovered_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) < (fov_width * fov_height * 0.1):  # Ignorar áreas muy pequeñas
            continue
            
        # Encontrar centro del área no cubierta
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"]) + (bounds['min_x'] - bounds['width']/2)
            cy = int(M["m01"] / M["m00"]) + (bounds['min_y'] - bounds['height']/2)
            
            candidate_pos = (cx, cy, -1, -1)  # -1 indica posición adicional
            
            # Verificar si está en polígono y tiene overlap suficiente
            if (is_position_in_polygon((cx, cy), polygon_points, fov_width, fov_height) and
                has_sufficient_overlap_with_sequence(candidate_pos, existing_positions, fov_width, fov_height, min_overlap)):
                additional_positions.append(candidate_pos)
    
    return additional_positions

def has_sufficient_overlap_with_sequence(candidate_pos, existing_positions, fov_width, fov_height, min_overlap):
    """Verificar si una posición tiene overlap suficiente con la secuencia existente"""
    for existing_pos in existing_positions:
        overlap_ratio = calculate_overlap_area(candidate_pos[:2], existing_pos[:2], fov_width, fov_height)
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

def insert_positions_optimally(ordered_positions, additional_positions, fov_width, fov_height):
    """Insertar posiciones adicionales en la secuencia óptima"""
    if not additional_positions:
        return ordered_positions
    
    result = ordered_positions.copy()
    
    for add_pos in additional_positions:
        best_insert_idx = 0
        min_distance = float('inf')
        
        # Encontrar la mejor posición para insertar
        for i in range(len(result) + 1):
            total_distance = 0
            
            if i > 0:
                total_distance += np.sqrt((add_pos[0] - result[i-1][0])**2 + (add_pos[1] - result[i-1][1])**2)
            if i < len(result):
                total_distance += np.sqrt((add_pos[0] - result[i][0])**2 + (add_pos[1] - result[i][1])**2)
            
            if total_distance < min_distance:
                min_distance = total_distance
                best_insert_idx = i
        
        result.insert(best_insert_idx, add_pos)
    
    return result

def expand_coverage_grid(positions, bounds, fov_width, fov_height, min_overlap):
    """Expandir el grid de cobertura si es necesario"""
    # Como último recurso, añadir posiciones en los bordes
    expanded_positions = positions.copy()
    
    # Añadir posiciones adicionales en los bordes
    border_positions = [
        (bounds['min_x'] + fov_width/4, bounds['min_y'] + fov_height/2, -1, -1),
        (bounds['max_x'] - fov_width/4, bounds['min_y'] + fov_height/2, -1, -1),
        (bounds['min_x'] + fov_width/2, bounds['min_y'] + fov_height/4, -1, -1),
        (bounds['min_x'] + fov_width/2, bounds['max_y'] - fov_height/4, -1, -1),
    ]
    
    for border_pos in border_positions:
        if border_pos not in expanded_positions:
            expanded_positions.append(border_pos)
    
    return expanded_positions

def validate_stitching_sequence(positions, fov_width, fov_height, min_overlap):
    """Validar que la secuencia de posiciones es válida para stitching"""
    if len(positions) < 2:
        return True, "Secuencia válida (una sola imagen o vacía)"
    
    validation_results = []
    total_overlaps = 0
    
    for i in range(len(positions) - 1):
        current_pos = positions[i][:2] if len(positions[i]) > 2 else positions[i]
        next_pos = positions[i + 1][:2] if len(positions[i + 1]) > 2 else positions[i + 1]
        
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


# Función auxiliar para mantener compatibilidad
def create_coverage_masks(polygon_points, positions, bounds, fov_width, fov_height):
    """Función de compatibilidad con el código existente"""
    polygon_mask = create_polygon_mask(polygon_points, bounds)
    
    # Convertir posiciones si es necesario
    position_coords = [(pos[0], pos[1]) if len(pos) > 2 else pos for pos in positions]
    coverage_mask = create_coverage_mask_from_positions(position_coords, bounds, fov_width, fov_height)
    
    return polygon_mask, coverage_mask