import cv2
import numpy as np

class ImageToStitch:
    def __init__(self, img):
        self.img = img
        self.mask = None
        self.red_polygons_contours = None
        self.binary = None
        self.gray_img = None
        self.text_mask = None
    
    def process(self):
        self.create_mask()
        self.extract_contours()
        self.create_paper_mask()
        self.create_text_mask()
        
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
    
    def create_paper_mask(self):
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Corregir: usar self.img en lugar de self.image
        img_without_blue = self.img.copy()
        img_without_blue[blue_mask > 0] = [0, 0, 0] 
        
        lab = cv2.cvtColor(img_without_blue, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        img_normalized = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR) 
        
        self.gray_img = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(self.gray_img, (151, 151), 0)
        
        _, self.binary = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)
    
    def create_text_mask(self):
        binary_text = cv2.adaptiveThreshold(self.gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, blockSize=7, C=4)
        
        kernel_small = np.ones((2,2), np.uint8)
        # Corregir: agregar operación morfológica faltante
        text_mask = cv2.morphologyEx(binary_text, cv2.MORPH_OPEN, kernel_small)
        kernel_medium = np.ones((3, 3), np.uint8)
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel_medium)
        
        height, width = self.img.shape[:2]
        min_size = max(3, int(width * height * 0.000001))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(text_mask, 8, cv2.CV_32S)
        filtered_mask = np.zeros_like(text_mask)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w_comp = stats[i, cv2.CC_STAT_WIDTH]
            h_comp = stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = h_comp / w_comp if w_comp > 0 else 0
            if (area >= min_size and 0.1 <= aspect_ratio <= 12 and
                max(w_comp, h_comp) <= min(width, height) / 4):
                filtered_mask[labels == i] = 255

        # Eliminar componentes en fondo azul
        background_mask = cv2.bitwise_not(self.binary)
        filtered_mask = cv2.bitwise_and(filtered_mask, background_mask)

        # Dilatación horizontal leve para unir letras
        kernel_h = np.ones((1, 2), np.uint8)
        kernel_v = np.ones((2, 1), np.uint8)
        final_mask = cv2.dilate(filtered_mask, kernel_h, iterations=1)
        final_mask = cv2.dilate(final_mask, kernel_v, iterations=1)
        self.text_mask = final_mask