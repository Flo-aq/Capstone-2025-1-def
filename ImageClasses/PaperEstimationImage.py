import numpy as np
from Functions.SecondModule.FirstCaseFunctions import create_lines_from_extremes, get_vertical_and_horizontal_lines, reconstruct_bottom_polygon, reconstruct_top_polygon
from Functions.SecondModule.FourthCaseFunctions import find_polygon_from_two_unclean_intersections
from Functions.SecondModule.SecondCaseFunctions import reconstruct_polygon_from_paralell_lines
from Functions.SecondModule.SecondModuleFunctions import calculate_photo_positions_diagonal, extend_all_lines_and_find_corners, find_polygon_from_intersections, group_lines_by_angle, standardize_polygon
from Functions.SecondModule.ThirdCaseFunctions import find_polygon_from_two_clean_intersections
from ImageClasses.Image import Image
import time

class PaperEstimationImage(Image):
    """
    A specialized Image class that handles the composition and processing of two images
    to detect polygons and calculate optimal camera positions for capture.
    Inherits from the base Image class.
    """
    def __init__(self, camera, top_image, bottom_image, parameters=None):
        """
        Initialize ImageFunction2 with required images and parameters.

        Args:
            camera (Camera): Camera object for image acquisition
            top_image (Image): First image to process (top position)
            bottom_image (Image): Second image to process (bottom position)
            parameters (dict): Processing parameters for the second module

        Raises:
            ValueError: If parameters or any image is None
        """
        super().__init__(function=2, image=None, camera=camera)
        if parameters is None:
            raise ValueError("Parameters cannot be None")
        if top_image is None:
            raise ValueError("Top image cannot be None")
        if bottom_image is None:
            raise ValueError("Bottom image cannot be None")
        self.parameters = parameters["second_module"]
        self.polygon = None
        self.top_image = top_image
        self.bottom_image = bottom_image
        self.case = 0
        self.grouped_lines = None
        self.unique_corners = None
    

    def create_image(self):
        """
        Create a composite image by combining top and bottom images.
        Places images at opposite corners of a white canvas.
        """
        print("Creating composite image...")
        composite = np.ones((self.height_px, self.width_px, 3), dtype=np.uint8) * 255
        
        h_top, w_top = self.top_image.image.shape[:2]
        h_bottom, w_bottom = self.bottom_image.image.shape[:2]
        
        composite[0:h_top, 0:w_top] = self.top_image.image
    
        bottom_y = self.height_px - h_bottom
        bottom_x = self.width_px - w_bottom
        
        composite[bottom_y:self.height_px, bottom_x:self.width_px] = self.bottom_image.image

        self.image = composite
    
    def get_lines(self):
        """
        Detect and process lines from both top and bottom images.
        Adjusts bottom image line positions based on offset and determines
        processing case based on line detection results.
        """
        h_bottom = self.bottom_image.height_px
        w_bottom = self.bottom_image.width_px
        bottom_offset = [
            self.width_px - w_bottom,
            self.width_px - h_bottom
        ]

        self.top_image.get_lines()
        self.bottom_image.get_lines()
        
        original_bottom_lines = self.bottom_image.lines.copy()

        self.bottom_image.lines = [
            (
                [p1[0] + bottom_offset[0], p1[1] + bottom_offset[1]],
                [p2[0] + bottom_offset[0], p2[1] + bottom_offset[1]]
            )
            for p1, p2 in original_bottom_lines
        ]
        
        if len(self.top_image.lines) == 0 and (len(original_bottom_lines) + len(self.top_image.lines) != 0):
            self.lines = self.bottom_image.lines
            grouped_lines = group_lines_by_angle(self.lines)
            self.grouped_lines = grouped_lines
            self.case = 1
            
        elif len(original_bottom_lines) == 0 and (len(original_bottom_lines) + len(self.top_image.lines) != 0):
            self.lines = self.top_image.lines
            grouped_lines = group_lines_by_angle(self.lines)
            self.grouped_lines = grouped_lines
            self.case = 1
            
        elif len(self.top_image.lines) != 0 and len(original_bottom_lines) != 0:
            self.lines = self.top_image.lines + self.bottom_image.lines 
            grouped_lines = group_lines_by_angle(self.lines)
            self.grouped_lines = grouped_lines
            if len(grouped_lines.keys()) == 1:
                self.case = 2
            else:
                self.bottom_image.lines_by_angles = group_lines_by_angle(self.bottom_image.lines)
                self.top_image.lines_by_angles = group_lines_by_angle(self.top_image.lines)
                self.unique_corners = extend_all_lines_and_find_corners(self.image, self.lines)
                if len(self.unique_corners) == 2:
                    top_angles = sorted(self.top_image.lines_by_angles.keys())
                    bottom_angles = sorted(self.bottom_image.lines_by_angles.keys())
                    if len(top_angles) == 1 or len(bottom_angles) == 1:
                        self.case = 4
                    else:
                        self.case = 3
        
    def get_polygon(self, width_mm, height_mm):
        """
        Generate polygon representation based on detected lines and case.
        
        Args:
            width_mm (float): Paper width in millimeters
            height_mm (float): Paper height in millimeters
        
        Returns:
            numpy.ndarray: Standardized polygon vertices ordered clockwise
        """
        self.get_lines()
        if self.case == 0:
            polygon = find_polygon_from_intersections(self.unique_corners) if len(self.unique_corners) >= 3 else []
            return standardize_polygon(polygon)
        
        width_px = width_mm / self.camera.mm_per_px_h
        height_px = height_mm / self.camera.mm_per_px_v
        
        if self.case == 1:
            vertical_lines, horizontal_lines = get_vertical_and_horizontal_lines(self.grouped_lines)
            if len(vertical_lines) == 0 or len(horizontal_lines) == 0:
                raise ValueError("No vertical nor horizontal lines found")

            vertical_line, horizontal_line = create_lines_from_extremes(
            vertical_lines, horizontal_lines)
            
            if self.top_image.has_paper:
                polygon =  reconstruct_top_polygon(
                    width_px, height_px,
                    vertical_line, horizontal_line)
            else:
                polygon = reconstruct_bottom_polygon(
                width_px, height_px,
                vertical_line, horizontal_line)
                
            return standardize_polygon(polygon)
        elif self.case == 2:
            polygon = reconstruct_polygon_from_paralell_lines(self.top_image.lines, self.bottom_image.lines, height_px)
            return standardize_polygon(polygon)
        
        elif self.case == 3:
            polygon = find_polygon_from_two_clean_intersections(self.unique_corners, self.top_image.lines_by_angles, self.bottom_image.lines_by_angles, width_px, height_px)
            return standardize_polygon(polygon)

        elif self.case == 4:
            polygon = find_polygon_from_two_unclean_intersections(self.top_image.lines_by_angles, self.bottom_image.lines_by_angles, self.unique_corners, width_px, height_px)
            return standardize_polygon(polygon)

        return []
    
    def process(self, width_mm, height_mm):
        """
        Process images to create composite and detect polygon.
        Measures and reports processing time for each step.

        Args:
            width_mm (float): Paper width in millimeters
            height_mm (float): Paper height in millimeters

        Returns:
            None
        """
        start_time = time.time()
        self.create_image()
        create_time = time.time() - start_time
        start_time = time.time()
        self.polygon = self.get_polygon(width_mm, height_mm)
        polygon_time = time.time() - start_time
        print(f"Tiempo de creación de imagen compuesta: {create_time:.3f} segundos")
        print(f"Tiempo de detección de polígono: {polygon_time:.3f} segundos")
        print(f"Tiempo total de procesamiento: {create_time + polygon_time:.3f} segundos")
        print(f"Caso detectado: {self.case}")
        print("--------------------\n")
        print("Polygon of paper detected")
    
    def calculate_capture_photos_positions(self):
        """
        Calculate optimal camera positions for capturing the detected polygon.
        Tests two diagonal strategies and returns the better one based on
        coverage and number of positions needed.

        Args:
            None

        Returns:
            list: List of (x,y) tuples representing optimal camera positions,
                empty list if no polygon was detected
        """
        if len(self.polygon) == 0:
            return []
            
        fov_width = self.camera.fov_h_px
        fov_height = self.camera.fov_v_px
        margin_px = int(self.parameters["camera_positioning_margin"] / self.camera.mm_per_px_h)  # 5mm margin
        print("Trying first diagonal strategy to get camera positions")
        positions1, coverage1 = calculate_photo_positions_diagonal(
            self.polygon, fov_width, fov_height, margin_px, "topleft-bottomright")
        print("Trying second diagonal strategy to get camera positions")
        positions2, coverage2 = calculate_photo_positions_diagonal(
            self.polygon, fov_width, fov_height, margin_px, "topright-bottomleft")
        print("Deciding which diagonal strategy to use")
        if len(positions1) < len(positions2) or (len(positions1) == len(positions2) and coverage1 > coverage2):
            print("Using first diagonal strategy")
            return positions1
        else:
            print("Using second diagonal strategy")
            return positions2
    