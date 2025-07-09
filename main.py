# Imports de la biblioteca estándar
import os
import time
from datetime import datetime
from os.path import join

import cv2

from BrailleToCoordinates import BrailleToCoordinates
from Camera.Camera import Camera
from Camera.CameraBox import CameraBox
from Controllers.DeviceManager import DeviceManager
from Functions.AuxFunctions import load_json
from GCodeGenerator import GCodeGenerator
from ImageClasses.CornerImage import CornerImage
from ImageClasses.ImageStitcher import ImageStitcher
from ImageClasses.PaperSectionImage import PaperSectionImage
from ImageClasses.PaperRecompositionImage import PaperRecompositionImage
from Paper import Paper
from ReferenceSystem import ReferenceSystem
from Translator import Translator


class Main:
    def __init__(self):
        self.device_manager = DeviceManager()
        
        self.config = load_json(join("parameters.json"))
        
        self.camera = Camera(self.config)
        
        self.reference_system = ReferenceSystem(self.config)

        self.camera_box = CameraBox(0, 0, self.camera, self.reference_system)

        self.translator = Translator()
        
        self.braille_converter = BrailleToCoordinates(self.config)
        
        # self.g_code_generator = GCodeGenerator()
        
        self.image_stitcher = ImageStitcher()
        
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
            print(f"Moving to {corner_name} corner at coordinates: {corner_coords}")
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

    def flow(self):
        print("Starting first phase - homing...")
        homing_success = self.homing(set_custom_origin=True)
        if not homing_success:
            print("E: Homing failed")
            return False
        
        print("Getting X limit...")
        x_limit_response = self.device_manager.arduino_mega_scanner.max_limit_x()
        # No continuar hasta tener una respuesta válida
        if x_limit_response and not x_limit_response.startswith("E:"):
            try:
                x_limit = float(x_limit_response.split(" ")[1])
                print(f"X limit: {x_limit}")
            except ValueError:
                print(f"E: Invalid X limit value: {x_limit_response}")
                return False
        else:
            print("E: Failed to get X limit")
            return False
            
        
        print("Getting Y limit...")
        y_limit_response = self.device_manager.arduino_mega_scanner.max_limit_y()
        # No continuar hasta tener una respuesta válida
        if y_limit_response and not y_limit_response.startswith("E:"):
            try:
                y_limit = float(y_limit_response.split(" ")[1])
                print(f"Y limit: {y_limit}")
            except ValueError:
                print(f"E: Invalid Y limit value: {y_limit_response}")
                return False
        else:
            print("E: Failed to get Y limit")
            return False
        
        print(f"Setting reference system limits: X={x_limit}, Y={y_limit}")
        self.reference_system.set_limits(x_limit, y_limit)
        print(f"Reference system limits: X={self.reference_system.max_x}, Y={self.reference_system.max_y}")
        
    
        self.top_left_img = CornerImage(image=None, camera_box=self.camera_box, parameters=self.config)
        led_on_response = self.device_manager.arduino_mega_scanner.turn_on_leds()
        time.sleep(2)  # Esperar un segundo para que los LEDs se enciendan
        self.top_left_img.capture_and_process()
        led_off_response = self.device_manager.arduino_mega_scanner.turn_off_leds()

        if self.top_left_img.image is None:
            print("E: Failed to capture top left image.")
            return False         
        
        self.move_to_corner("bottom_right")
        self.bottom_right_img = CornerImage(image=None,camera_box=self.camera_box,parameters=self.config)
        led_on_response = self.device_manager.arduino_mega_scanner.turn_on_leds()
        time.sleep(2)  # Esperar un segundo para que los LEDs se enciendan
        self.bottom_right_img.capture_and_process()
        led_off_response = self.device_manager.arduino_mega_scanner.turn_off_leds()

        if self.bottom_right_img.image is None:
            print("E: Failed to capure bottom right image")
        
        
        self.paper = Paper(self.config, self.camera_box, self.translator)
        self.paper.set_position(self.top_left_img, self.bottom_right_img)
        self.paper.calculate_capture_positions()
        # # # if index == 0:
        # # #   start = "TL"
        # # # else:
        # # #   start = "TR"
        
        for i, pos in enumerate(self.paper.capture_positions):
            print(f"Moving camera to capture position {pos}...")
            print(f"Capturing image {i + 1}...")
            self.move_to_position(pos[0], pos[1])
            led_on_response = self.device_manager.arduino_mega_scanner.turn_on_leds()
            time.sleep(2)  # Esperar un segundo para que los LEDs se enciendan
            img = self.camera.capture_image()
            led_off_response = self.device_manager.arduino_mega_scanner.turn_off_leds()
            self.imgs.append(img)
        
        print("Creating paper image...")
            
        self.paper.image = PaperRecompositionImage(camera_box=self.camera_box, images=self.imgs, parameters=self.config, stitcher=self.image_stitcher)
        # # print("Creating paper image...")
        # # self.paper.image.create_image()
        # self.paper.get_text()
        # self.paper.translate_text()
        # print("Text captured and translated.")
        # print("Generating braille coordinates...")
        # self.braille_converter.binary_to_coordinates(self.paper.translated_text["binary"])
        # self.braille_converter.sort_coordinates()
        # print(self.braille_converter.sorted_coordinates[:10])  # Print first 10 coordinates for debugging
        # success = self.device_manager.arduino_mega_printer.print_braille_points(self.braille_converter.sorted_coordinates[:10])
        # if success:
        #     print("Braille printing completed successfully!")
        # else:
        #     print("E: Braille printing failed")
        
        # return success

if __name__ == "__main__":
    main = Main()
    
    success = main.flow()
    