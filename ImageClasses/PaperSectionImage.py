from ImageClasses.Image import Image
import cv2

class PaperSectionImage(Image):
    """
    Specialized Image class for processing images with corner position tracking.
    Handles binary threshold processing of images.
    """
    def __init__(self, camera, image):
        """
        Initialize ImageFunction3 with camera and image.
        
        Args:
            camera (Camera): Camera object for image acquisition
            image (ndarray): Input image to process
        """
        super().__init__(function=3, image=image, camera=camera)
        self.corners_positions_mm = self.camera.get_fov_corners()
        self.corners_positions_px = self.calculate_corners_positions()

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
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
        self.image = binary