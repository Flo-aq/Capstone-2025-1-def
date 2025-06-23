class ReferenceSystem:
    """
    A class that defines the reference system and coordinate boundaries for camera movement.
    Handles both physical dimensions in millimeters and pixel-based dimensions for photo/video modes.
    """

    def __init__(self, config):
        """
        Initialize the reference system with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary containing reference system parameters
                         including dimensions and coordinate boundaries
        """
        parameters = config["reference_system"]

        self.width_mm = parameters["dimensions"]["width_mm"]
        self.height_mm = parameters["dimensions"]["height_mm"]

        self.origin_x = 0
        self.origin_y = 0

        self.photo_width_px = 0
        self.photo_height_px = 0

        self.video_width_px = 0
        self.video_height_px = 0

    def set_photo_dimensions(self, mm_per_px_x, mm_per_px_y):
        """
        Calculate and set photo mode dimensions in pixels based on mm/pixel ratios.
        
        Args:
            mm_per_px_x (float): Millimeters per pixel in horizontal direction
            mm_per_px_y (float): Millimeters per pixel in vertical direction
        """
        self.photo_width_px = int(self.width_mm / mm_per_px_x)
        self.photo_height_px = int(self.height_mm / mm_per_px_y)

    def set_video_dimensions(self, mm_per_px_x, mm_per_px_y):
        """
        Calculate and set video mode dimensions in pixels based on mm/pixel ratios.
        
        Args:
            mm_per_px_x (float): Millimeters per pixel in horizontal direction
            mm_per_px_y (float): Millimeters per pixel in vertical direction
        """
        self.video_width_px = int(self.width_mm / mm_per_px_x)
        self.video_height_px = int(self.height_mm / mm_per_px_y)
    
    def set_limits(self, max_x, max_y):
        """
        Set the movement limits for the reference system.
        
        Args:
            max_x (float): Maximum X coordinate in mm
            min_x (float): Minimum X coordinate in mm
            max_y (float): Maximum Y coordinate in mm
            min_y (float): Minimum Y coordinate in mm
        """
        self.max_x = max_x
        self.min_x = 0
        self.max_y = max_y
        self.min_y = 0
    
