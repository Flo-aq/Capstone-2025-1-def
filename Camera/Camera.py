import numpy as np


class Camera:
    """ 
    A class to manage camera operations for photo capture.
    """
    def __init__(self, config):
        """
        Initialize a new Camera instance.

        Args:
            config (dict): Configuration dictionary containing settings for camera under 'camera_settings'
        """
        self.camera = None
        camera_settings = config['camera_settings']

        self.photo_h_res = camera_settings['photo_config']["size"][0]
        self.photo_v_res = camera_settings['photo_config']["size"][1]

        self.height_mm = camera_settings['height_mm']

        self.fov_h_deg = camera_settings['fov_h_deg']
        self.fov_v_deg = camera_settings['fov_v_deg']
        self.fov_h_rad = np.deg2rad(self.fov_h_deg)
        self.fov_v_rad = np.deg2rad(self.fov_v_deg)

        self.fov_h_mm, self.fov_v_mm = self.calculate_fov()
        self.photo_mm_per_px_h, self.photo_mm_per_px_v = self.calculate_mm_relation()
        
        self.fov_h_px, self.fov_v_px = self.photo_h_res, self.photo_v_res
        self.mm_per_px_h, self.mm_per_px_v = self.photo_mm_per_px_h, self.photo_mm_per_px_v

        self.captured_imgs = []
        
    def calculate_fov(self):
        """
        Calculate field of view dimensions in mm.
        Returns:
            tuple: (horizontal FOV in mm, vertical FOV in mm)
        """
        return 2 * self.height_mm * np.tan(self.fov_h_rad / 2), 2 * self.height_mm * np.tan(self.fov_v_rad / 2)

    def calculate_mm_relation(self):
        """
        Calculate mm per pixel ratios for photo mode.
        Returns:
            tuple: (photo_h_ratio, photo_v_ratio)
        """
        return self.fov_h_mm / self.photo_h_res, self.fov_v_mm / self.photo_v_res

    def initialize_camera(self):
        """Initialize camera hardware in photo mode."""
        if self.camera is None:
            # self.camera = Picamera2()
            camera_config = self.camera.create_preview_configuration(
                main={"size": (self.photo_h_res, self.photo_v_res)}
            )
            self.camera.configure(camera_config)
            self.camera.start()

    def release_camera(self):
        """Release camera hardware resources."""
        if self.camera is not None:
            self.camera.stop()
            self.camera.close()
            self.camera = None

    def capture_image(self):
        """
        Capture a single image.
        Returns:
            array: Numpy array containing the captured image
        """
        if self.camera is None:
            self.initialize_camera()
        img = self.camera.capture_array()
        self.captured_imgs.append(img)
        self.release_camera()
        return img
    
    