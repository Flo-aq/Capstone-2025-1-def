class CameraBox:
    """
    A class to manage the camera box movement and positioning.
    The camera box operates in a 2D space with position constraints.
    """
    def __init__(self, x_position, y_position, camera, reference_system):
        """
        Initialize a new CameraBox instance.

        Args:
            x_position (float): Initial X position in mm
            y_position (float): Initial Y position in mm
            camera (Camera): Camera instance for FOV calculations
            reference_system (object): Reference system defining movement boundaries
        
        Raises:
            ValueError: If reference system is None
        """
        if reference_system is None:
            raise ValueError("Reference system cannot be None")
            
        self.camera = camera
        self.reference_system = reference_system

        self.x = None
        self.y = None
        
        self.set_position(x_position, y_position)

    def set_position(self, x_position, y_position):
        """
        Set the position of the camera box within the defined boundaries.
        
        Args:
            x_position (float): X position in mm
            y_position (float): Y position in mm
        """
        self.x = x_position
        self.y = y_position

    def move_distance(self, axis, distance):
        if axis == 0:
            self.x += distance
        elif axis == 1:
            self.y += distance
    
    def get_specific_corner_coordinates(self, corner_name):
        corners = self.get_corner_coordinates()
        
        if corner_name not in corners:
            raise ValueError(f"Invalid corner. Must be one of {list(corners.keys())}")
        
        return corners[corner_name]
    
    def get_corner_coordinates(self):
      return {
            'top_left': (self.reference_system.min_x, self.reference_system.min_y),
            'top_right': (self.reference_system.max_x, self.reference_system.min_y),
            'bottom_left': (self.reference_system.min_x, self.reference_system.max_y),
            'bottom_right': (self.reference_system.max_x, self.reference_system.max_y)
        }

    