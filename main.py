from Controllers.DeviceManager import DeviceManager
from Camera.Camera import Camera
from Camera.CameraBox import CameraBox

from ReferenceSystem import ReferenceSystem
from Functions.AuxFunctions import load_json
from ImageClasses.CornerImage import CornerImage
from os.path import join
import os
import cv2
from datetime import datetime


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
        print(distance_x)
        distance_y = y_mm - self.camera_box.y
        print(distance_y)
        
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
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"captured_corners_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)

        corners = ["top_left", "top_right", "bottom_right", "bottom_left"]
        
        corner_imgs = {}
        
        for corner in corners:
          try:
            coords = self.camera_box.get_specific_corner_coordinates(corner)
            self.move_to_position(coords[0], coords[1]) 
            img = self.capture_image()

            if img is not None:
                # Guardar imagen
                img_path = os.path.join(save_dir, f"{corner}.jpg")
                cv2.imwrite(img_path, img)
                corner_imgs[corner] = img
            else:
                corner_imgs[corner] = None

          except Exception as e:
            print(f"E: Error capturing image for corner {corner}: {str(e)}")
            corner_imgs[corner] = None

        return corner_imgs

    def first_phase(self):
          
        homing_success = self.homing(set_custom_origin=True)
        
        x_limit = float(self.device_manager.arduino_mega_scanner.max_limit_x())
        y_limit = float(self.device_manager.arduino_mega_scanner.max_limit_y())
        
        self.reference_system.set_limits(x_limit, y_limit)
    
        self.top_left_img = CornerImage(image=None, camera=self.camera, parameters=self.config)
        self.top_left_img.capture_and_process()

        if self.top_left_img.image is None:
            print("E: Failed to capture top left image.")
            return False
        
        save_dir = "ImagenesPrueba"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(save_dir, f"top_left_{timestamp}.jpg")
        cv2.imwrite(img_path, self.top_left_img.image)
        return True

        # self.bottom_right_img = ImageFunction1(image=None,camera=self.camera,parameters=self.config)
        # self.bottom_right_img.capture_and_process()

        # if self.bottom_right_img.image is None:
        #     print("E: Failed to capure bottom right image")
        
        # self.paper = Paper(self.config, self.camera, self.translator)
        # self.paper.set_position(self.top_left_img, self.bottom_right_img)

        # for i, pos in enumerate(self.paper.capture_positions):
        #     print(f"Moving camera to capture position {pos}...")
        #     print(f"Capturing image {i + 1}...")
        #     self.move_to_position(pos[0], pos[1])
        #     img = ImageFunction3(
        #         image=None,
        #         camera=self.camera
        #     )
        #     img.capture_and_process()
        #     self.imgs.append(img)

if __name__ == "__main__":
    main = Main()
    
    # Example usage
    if main.first_phase():
        print("First phase completed successfully.")
    else:
        print("First phase failed.")
    