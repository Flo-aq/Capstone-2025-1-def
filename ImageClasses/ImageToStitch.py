import cv2
import numpy as np

class ImageToStitch:
  def __init__(self, img):
    self.img = img
    self.mask = None
    self.red_polygons_contours = None
    self.process()
  
  def process(self):
    self.create_mask()
    self.extract_contours()
    
  def create_mask(self):
    """
    Crea una máscara que detecta áreas rojas y azules en la imagen.
    """
    hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
    
    # Definir rangos para rojo (en HSV el rojo está en los extremos del espectro)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    
    # Definir rango para azul
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Crear máscaras individuales
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Combinar máscaras rojas
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # Combinar máscaras rojas y azules
    self.mask = cv2.bitwise_or(mask_red, mask_blue)
    
    # Aplicar operaciones morfológicas para limpiar la máscara
    kernel = np.ones((5,5), np.uint8)
    self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)
    self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
  def extract_contours(self):
    self.red_polygons_contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
  