import numpy as np
from ImageClasses.Image import Image
import cv2

class PaperSectionImage(Image):
    """
    Specialized Image class for processing images with corner position tracking.
    Handles binary threshold processing of images.
    """
    def __init__(self, camera_box, image):
        """
        Initialize ImageFunction3 with camera and image.
        
        Args:
            camera (Camera): Camera object for image acquisition
            image (ndarray): Input image to process
        """
        super().__init__(function=3, image=image, camera_box=camera_box)
        self.corners_positions_mm = self.camera.get_fov_corners()
        self.corners_positions_px = self.calculate_corners_positions()
        self.mask = None
        self.red_polygons_contours = None
        self.original_img = None
        
    def calculate_corners_positions(self):
        """
        Calculate corner positions in pixel coordinates from millimeter positions.
        
        Returns:
            dict: Corner positions in pixels {corner_name: (x_px, y_px)}
        """
        pos = {}
        for corner, position in self.corners_positions_mm.items():
            x_px = position[0] / self.camera.mm_per_px_h
            y_px = position[1] / self.camera.mm_per_px_v
            pos[corner] = (int(x_px), int(y_px))
        return pos
    
    def process(self):
        """
        Process the image by converting to grayscale and applying binary threshold.
        Uses Otsu's method for optimal threshold selection.
        """
        if self.image is None:
            raise ValueError("No image to process")
        rotated = cv2.rotate(self.image, cv2.ROTATE_180)
        self.original_img = rotated.copy()
        self.create_mask(rotated)
        img_flat = rotated.reshape(-1, 3)
        darkest_idx = np.argmin(img_flat.sum(axis=1))
        darkest_color = img_flat[darkest_idx]
        result = rotated.copy()
        result[self.mask == 255] = darkest_color
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
        self.image = binary
        self.get_red_polygons_contours()
    
    def create_mask(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        self.mask = cv2.bitwise_or(mask1, mask2)
    
    def get_red_polygons_contours(self):
        self.red_polygons_contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
