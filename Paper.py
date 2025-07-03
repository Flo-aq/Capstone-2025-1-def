import numpy as np
from ImageClasses.PaperEstimationImage import PaperEstimationImage
from ImageClasses.PaperImage import PaperImage
import cv2
import os

class Paper:
    """
    A class representing a paper document in the system.
    Handles paper detection, corner calculation, and capture position planning.
    """
    def __init__(self, config, camera_box, translator):
        """
        Initialize Paper object with configuration and camera settings.
        
        Args:
            config (dict): Configuration dictionary containing paper and image settings
            camera (Camera): Camera object for capturing images
            
        Raises:
            ValueError: If config or camera is not provided
        """
        if not config or not camera_box:
            raise ValueError(
                "Config and camera must be provided")
        if not translator:
            raise ValueError(
                "Translator must be provided")
        self.translator = translator
        self.camera_box = camera_box
        self.camera = camera_box.camera
        paper_config = config["paper"]
        self.image_config = config
        self.corners_mm = {
            'top_left': None,
            'top_right': None,
            'bottom_left': None,
            'bottom_right': None
        }
        self.corners_px = {
            'top_left': None,
            'top_right': None,
            'bottom_left': None,
            'bottom_right': None
        }
        self.polygon = None
        self.width_mm = paper_config["dimensions"]["width_mm"]
        self.height_mm = paper_config["dimensions"]["height_mm"]
        self.image = None
        self.paper_image = None
        self.text = ""
        self.translated_text = {}
    
    def set_position(self, top_image, bottom_image):
        """
        Set paper position using two images (top and bottom views).
        Creates composite image and detects paper polygon.
        
        Args:
            top_image (ImageFunction1): Processed top view image
            bottom_image (ImageFunction1): Processed bottom view image
            
        Raises:
            ValueError: If paper cannot be detected in either image
        """
        if top_image.polygon is [] and bottom_image.polygon is []:
            raise ValueError(
                "Could not detect paper in the provided images")
        self.composed_image = PaperEstimationImage(self.camera_box, top_image, bottom_image, self.image_config)
        self.composed_image.process(self.width_mm, self.height_mm)
        self.polygon = self.composed_image.polygon
        self.update_corners()
        
    def update_corners(self):
        """
        Update corner coordinates based on detected polygon.
        Converts pixel coordinates to millimeters and assigns to corners dictionary.
        """
        if self.polygon is None:
            return
        
        points = self.polygon.reshape(-1, 2)
        center = np.mean(points, axis=0)

        angles = np.arctan2(points[:, 1] - center[1],
                            points[:, 0] - center[0])

        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]

        corners = ['top_left', 'top_right', 'bottom_right', 'bottom_left']

        for i, corner in enumerate(corners):
            x_px, y_px = sorted_points[i]
            x_mm = x_px * self.camera.photo_mm_per_px_h
            y_mm = y_px * self.camera.photo_mm_per_px_v
            self.corners_mm[corner] = (x_mm, y_mm)
            self.corners_px[corner] = (x_px, y_px)
            
        print("Position of paper: ", self.corners_mm)
        print("Position of corners in pixels: ", self.corners_px)
        return

    def calculate_capture_positions(self):
        """
        Calculate optimal camera positions for capturing the entire paper.
        Uses the composite image to determine capture positions.
        
        Returns:
            list: List of camera positions for capturing the paper
        """
        self.capture_positions = self.composed_image.calculate_capture_photos_positions()
        
        print("Capture positions calculated: ", self.capture_positions)
    
    def get_text(self):
        print("Extracting and rotating image...")
        self.update_corners()
        image = self.image.extract_and_rotate(self.corners_px, self.width_mm, self.height_mm)
        self.paper_image = PaperImage(image, self.image_config, self.width_mm, self.height_mm)
        print("Image extracted and rotated.")
        print("Extracting text from image...")
        self.text = self.paper_image.get_text()
    
    def translate_text(self):
        if not self.text:
            raise ValueError("No text to translate. Please extract text first.")
        print("Translating text...")
        self.translated_text = self.translator.translate_full_text(self.text)
        print("Translation completed.")
        print("Original text:", self.text)
        print("Translated text:")
        os.system('chcp 65001')
        for line in self.translator.result["unicode"].split('\n'):
            print(line)
        
        