from ImagesClasses.Image import Image
import numpy as np
import cv2

class ImageFunction4(Image):
    def __init__(self, camera, images, parameters):
        """
        Initialize ImageFunction4 with camera and list of images.
        
        Args:
            camera (Camera): Camera object for image acquisition
            images (list): List of ImageFunction3 instances to compose
        """
        super().__init__(function=4, image=None, camera=camera)
        self.images = images
        self.lines = None
        self.parameters = parameters["first_module"]
    
    def create_image(self):
        """
        Creates a composite panorama from multiple images using their corner positions.
        Places each image in its correct position in the reference system.
        
        Raises:
            ValueError: If no images are provided
        """
        if not self.images:
            raise ValueError("No images provided for composite image creation")
        panorama = np.zeros((self.height_px, self.width_px), dtype=np.uint8)
        current_panorama = panorama.copy()
        for idx, img in enumerate(self.images):
            if img.image is None:
                img.process()

            y_start = img.corners_positions_px['top_left'][1]
            x_start = img.corners_positions_px['top_left'][0]
            y_end = y_start + img.height_px
            x_end = x_start + img.width_px
            print(f"Placing image {idx + 1} at position: ({x_start}, {y_start}) to ({x_end}, {y_end})")
            current_panorama[y_start:y_end, x_start:x_end] = img.image

        print("Composite image created.")
        self.image = current_panorama
        
    def extract_and_rotate(self, corners_px, width, height):
        """
        Extracts and rotates paper region based on corner positions.

        Args:
            corners_positions_px (dict): Dictionary with corner positions in pixels
            width_mm (float): Target width in millimeters
            height_mm (float): Target height in millimeters
        
        Returns:
            ndarray: Transformed and rotated image
        """
        if self.image is None:
            raise ValueError("No image to process")
        if None in corners_px.values():
            raise ValueError("Corners positions cannot be None")

        corners = np.float32([
            corners_px['top_left'],
            corners_px['top_right'],
            corners_px['bottom_right'],
            corners_px['bottom_left']
        ])
        
        dst_width = int(width / self.camera.mm_per_px_h)
        dst_height = int(height / self.camera.mm_per_px_v)

        dst_points = np.float32([
            [0,0],
            [dst_width, 0],
            [dst_width, dst_height],
            [0, dst_height]
        ])
               
        matrix = cv2.getPerspectiveTransform(corners, dst_points)
        warped = cv2.warpPerspective(self.image, matrix, (dst_width, dst_height))
        
        return warped
    
