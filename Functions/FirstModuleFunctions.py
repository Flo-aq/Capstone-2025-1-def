import cv2
import numpy as np

def first_module_process_image(image, kernel_size, sigma):
    """
    Process an image through a series of transformations for edge detection.
    
    Args:
        image (numpy.ndarray): Input BGR image
        kernel_size (int): Size of Gaussian blur kernel
        sigma (float): Standard deviation for Gaussian blur
    
    Returns:
        numpy.ndarray: Binary image ready for edge detection
    """
    rotated = cv2.rotate(image, cv2.ROTATE_180)
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
    binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]
    
    return binary

def add_border(img, border_size, color=[0,0,0]):
    """
    Add a border of specified size and color to an image.
    
    Args:
        img (numpy.ndarray): Input image
        border_size (int): Width of border in pixels
        color (list): RGB color values for border, default black
    
    Returns:
        numpy.ndarray: Image with added border
    """
    image = img.copy()
    return cv2.copyMakeBorder(
        image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=color
    )

def remove_border(img, border_size):
    """
    Remove border of specified size from an image.
    
    Args:
        img (numpy.ndarray): Input image with border
        border_size (int): Width of border to remove
    
    Returns:
        numpy.ndarray: Image with border removed
    """
    return img[border_size:-border_size, border_size:-border_size]

def detect_edges(img, canny_1, canny_2):
    """
    Detect edges in an image using Canny edge detection.
    
    Args:
        img (numpy.ndarray): Input image
        canny_1 (int): First threshold for Canny detector
        canny_2 (int): Second threshold for Canny detector
    
    Returns:
        numpy.ndarray: Binary edge image
    """
    edges = cv2.Canny(img, canny_1, canny_2)
    return edges

def process_edges_and_remove_border(img_with_border, edges, border_size):
    """
    Remove border from both original image and edge detection result.
    
    Args:
        img_with_border (numpy.ndarray): Original image with border
        edges (numpy.ndarray): Edge detection result with border
        border_size (int): Width of border to remove
    
    Returns:
        tuple: (image without border, edges without border)
    """
    img_no_border = remove_border(img_with_border, border_size)
    edges_no_border = remove_border(edges, border_size)
    return img_no_border, edges_no_border

def get_largest_edge(edges):
    """
    Find the largest contour in an edge image.
    
    Args:
        edges (numpy.ndarray): Binary edge image
    
    Returns:
        numpy.ndarray: Largest contour or None if no contours found
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour
    return None

def approximate_polygon(contour, epsilon_factor=0.02):
    """
    Approximate a contour to a polygon.
    
    Args:
        contour (numpy.ndarray): Input contour
        epsilon_factor (float): Approximation accuracy factor
    
    Returns:
        numpy.ndarray: Approximated polygon vertices
    """
    perimeter = cv2.arcLength(contour, True)
    epsilon = epsilon_factor * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx

def is_border_line(p1, p2, shape, tolerance=10):
    """
    Check if a line segment lies on the border of an image.
    
    Args:
        p1 (tuple): First point coordinates (x,y)
        p2 (tuple): Second point coordinates (x,y)
        shape (tuple): Image dimensions (height, width)
        tolerance (int): Pixel tolerance for border detection
    
    Returns:
        bool: True if line is on border, False otherwise
    """
    height, width = shape[:2]
    x1, y1 = p1
    x2, y2 = p2

    y_aligned = abs(y1 - y2) < tolerance
    x_aligned = abs(x1 - x2) < tolerance

    if y_aligned:
        y_avg = (y1 + y2) / 2
        return abs(y_avg) < tolerance or abs(y_avg - height) < tolerance
        
    if x_aligned:
        x_avg = (x1 + x2) / 2
        return abs(x_avg) < tolerance or abs(x_avg - width) < tolerance
        
    return False