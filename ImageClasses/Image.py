class Image:
    """
    A class to handle image processing and coordinate transformations between
    pixel and millimeter spaces in both image and reference system contexts.
    """

    def __init__(self, function, image, camera_box):
        """
        Initialize the Image class with a function, image, and camera.

        Args:
            function (callable): Function to process the image.
            image (array): The image to be processed.
            camera (Camera): The camera object used for capturing images.
        """
        self.function = function
        self.image = image
        self.camera_box = camera_box
        self.camera = camera_box.camera

        if function not in [2, 4]:
            self.height_mm = self.camera.fov_h_mm
            self.width_mm = self.camera.fov_v_mm
            self.height_px = self.camera.fov_h_px
            self.width_px = self.camera.fov_v_px
        else:
            self.height_mm = self.camera_box.reference_system.range_of_motion_height_mm + self.camera.fov_h_mm
            self.width_mm = self.camera_box.reference_system.range_of_motion_width_mm+ self.camera.fov_v_mm
            self.height_px = int(self.height_mm / self.camera.mm_per_px_h)
            self.width_px = int(self.width_mm/ self.camera.mm_per_px_v)

    def get_mm_coordinates_in_img(self, x_px, y_px):
        """
        Convert pixel coordinates to millimeter coordinates within the image space.
        
        Args:
            x_px (int): X coordinate in pixels
            y_px (int): Y coordinate in pixels
            
        Returns:
            tuple: Millimeter coordinates (x_mm, y_mm) relative to image origin
        """
        x_mm = x_px * self.camera.mm_per_px_h
        y_mm = y_px * self.camera.mm_per_px_v

        return x_mm, y_mm

    def get_px_coordinates_in_rs(self, x_px, y_px):
        """
        Convert image pixel coordinates to reference system pixel coordinates.
        
        Args:
            x_px (int): X coordinate in image pixels
            y_px (int): Y coordinate in image pixels
            
        Returns:
            tuple: Pixel coordinates (x, y) in reference system space
        """
        x = (self.camera_box.x * self.camera.mm_per_px_h - self.width_px / 2) + x_px
        y = (self.camera_box.y * self.camera.mm_per_px_v - self.height_px / 2) + y_px

        return x, y

    def get_mm_coordinates_in_rs(self, x_px, y_px):
        """
        Convert image pixel coordinates to reference system millimeter coordinates.
        
        Args:
            x_px (int): X coordinate in image pixels
            y_px (int): Y coordinate in image pixels
            
        Returns:
            tuple: Millimeter coordinates (x, y) in reference system space
        """
        x = (self.camera_box.x - self.width_mm / 2) + \
            x_px * self.camera.mm_per_px_h
        y = (self.camera_box.y - self.height_mm / 2) + \
            y_px * self.camera.mm_per_px_v

        return x, y

    def capture_and_process(self):
        """
        Capture an image or video frame based on function mode and process if needed.
        
        """
        self.image = self.camera.capture_image()
        print("Processing image...")
        self.process()
        return

    def process(self):
        """
        Placeholder for image processing implementation.
        To be overridden by subclasses with specific processing logic.
        """
        pass
