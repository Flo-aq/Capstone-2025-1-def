from Controllers.DeviceManager import DeviceManager
from Camera.Camera import Camera
from Camera.CameraBox import CameraBox
from ReferenceSystem import ReferenceSystem
from Functions.AuxFunctions import load_json
from os.path import join


class Main:
    def __init__(self):
        self.device_manager = DeviceManager()
        
        self.config = load_json(join("parameters.json"))
        
        self.camera = Camera(self.config)
        
        self.reference_system = ReferenceSystem(self.config)

        self.camera_box = CameraBox(0, 0, self.camera, self.reference_system)

        self.top_left_img = None
        self.bottom_right_img = None

        self.paper = None
        
        self.imgs = []
    
    def move_axis_arduino(self, axis, distance_mm):
        if not self.device_manager.arduino_mega_scanner:
            print("E: Arduino Mega Scanner not connected")
            return None
        if axis == 0:
            response = self.device_manager.arduino_mega_scanner.move_without_PID("X", distance_mm)
        else:
            response = self.device_manager.arduino_mega_scanner.move_without_PID("Y", distance_mm)
        
        if response.startswith("E:"):
            print(f"Error moving camera box: {response}")
            return None
        else:
            response_lines = response.split("\n")
            for line in response_lines:
              if line.startswith("E"):
                print(f"Error moving camera box: {line.split(' ')[1]}")
                return None
              elif line.startswith("OK"):
                distance_mm = float(line.split(' ')[1])
                if axis == 0:
                    self.camera_box.move_distance(0, distance_mm)
                else:
                    self.camera_box.move_distance(1, distance_mm)
    
    def move_to_position(self, x_mm, y_mm):
        distance_x = x_mm - self.camera_box.x
        distance_y = y_mm - self.camera_box.y
        
        self.move_specific_distance(distance_x, distance_y)
        
    def move_specific_distance(self, distance_x_mm, distance_y_mm):
        if distance_x_mm != 0:
            self.move_axis_arduino(0, distance_x_mm)
        
        if distance_y_mm != 0:
            self.move_axis_arduino(1, distance_y_mm)
    
    def move_to_corner(self, corner_name):
        corner_coords = self.camera_box.get_specific_corner_coordinates(corner_name)
        if corner_coords:
            self.move_to_position(corner_coords[0], corner_coords[1])
        else:
            print(f"E: Invalid corner name '{corner_name}'")
      
    def capture_image(self):
        image = self.camera.capture_image()
        return image
    
    def homing(self, set_custom_origin=False):
        if not self.device_manager.arduino_mega_scanner:
          print("Error: Arduino Mega Scanner not connected")
          return False
        
        if set_custom_origin:
            response = self.device_manager.arduino_mega_scanner.home_and_set_origin()
        else:
            response = self.device_manager.arduino_mega_scanner.full_homing()
        if response.startswith("E:"):
            print(f"Error during homing: {response}")
            return False
        
        # Actualizar la posición en el modelo de CameraBox
        self.camera_box.set_position(0, 0)
        
        return True

    def capture_corners_imgs(self):
        if not self.device_manager.arduino_mega_scanner:
            print("E: Arduino Mega Scanner not connected")
            return None
        
        corners = ["top-left", "top-right", "bottom-right", "bottom-left"]
        
        corner_imgs = {}
        
        for corner in corners:
          try:
            coords = self.camera_box.get_specific_corner_coordinates(corner)
            self.move_to_position(coords[0], coords[1]) 
            corner_imgs[corner] = self.capture_image()
          except Exception as e:
            print(f"E: Error capturing image for corner {corner}: {str(e)}")
            corner_imgs[corner] = None
        return corner_imgs

    def first_phase(self):
        if not self.homing(set_custom_origin=True):
            print("E: Initial homing failed. Exiting first phase.")
            return False

        img_1 = self.capture_image()

        if img_1 is None:
            print("E: Failed to capture image in first phase.")
            return False

        self.move_to_corner("bottom-right")

        img_2 = self.capture_image()

