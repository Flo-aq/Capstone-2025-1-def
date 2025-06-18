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

        # Calculate movement boundaries based on camera FOV and reference system limits
        self._x_min_mm = reference_system.min_x + camera.fov_h_mm/2
        self._x_max_mm = reference_system.max_x - camera.fov_h_mm/2
        self._y_min_mm = reference_system.min_y + camera.fov_v_mm/2
        self._y_max_mm = reference_system.max_y - camera.fov_v_mm/2

        self._x = None
        self._y = None

        self.set_position(x_position, y_position)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = min(max(value, self._x_min_mm), self._x_max_mm)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = min(max(value, self._y_min_mm), self._y_max_mm)
    
    def set_position(self, x_position, y_position):
        """
        Set the position of the camera box within the defined boundaries.
        
        Args:
            x_position (float): X position in mm
            y_position (float): Y position in mm
        """
        self.x = x_position
        self.y = y_position

    def get_fov_corners(self):
        """
        Get the coordinates of the current field of view corners.
        Returns:
            dict: Corner coordinates in format {corner_name: (x, y)}
        """
        return {
            'top_left': (self.x - self.camera.fov_h_mm/2, self.y - self.camera.fov_v_mm/2),
            'top_right': (self.x + self.camera.fov_h_mm/2, self.y - self.camera.fov_v_mm/2),
            'bottom_left': (self.x - self.camera.fov_h_mm/2, self.y + self.camera.fov_v_mm/2),
            'bottom_right': (self.x + self.camera.fov_h_mm/2, self.y + self.camera.fov_v_mm/2)
        }

    def move_distance(self, axis, distance):
        if axis == 0:
            self.x += distance
        elif axis == 1:
            self.y += distance
    
    def get_specific_corner_coordinates(self, corner_name):
        corners = {
            'top-left': (self._x_min_mm, self._y_min_mm),
            'top-right': (self._x_max_mm, self._y_min_mm),
            'bottom-left': (self._x_min_mm, self._y_max_mm),
            'bottom-right': (self._x_max_mm, self._y_max_mm)
        }
        
        if corner_name not in corners:
            raise ValueError(f"Invalid corner. Must be one of {list(corners.keys())}")
        
        return corners[corner_name]
    
    def get_corner_coordinates(self):
      return {
          'top-left': (self._x_min_mm, self._y_min_mm),
          'top-right': (self._x_max_mm, self._y_min_mm),
          'bottom-left': (self._x_min_mm, self._y_max_mm),
          'bottom-right': (self._x_max_mm, self._y_max_mm)
      }

    