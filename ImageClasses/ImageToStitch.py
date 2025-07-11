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
        img_bright = self.img.copy()
        hsv_bright = cv2.cvtColor(img_bright, cv2.COLOR_BGR2HSV)
        
        mask_white = hsv_bright[...,2] > 120
        hsv_bright[...,2][mask_white] = np.clip(hsv_bright[...,2][mask_white] * 1.7, 0, 255)
        img_bright = cv2.cvtColor(hsv_bright, cv2.COLOR_HSV2BGR)
        
        hsv = cv2.cvtColor(img_bright, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.3, 0, 255)
        hsv = hsv.astype(np.uint8)
        
        lower_blue = np.array([100, 95, 65])
        upper_blue = np.array([130, 255, 255])
        
        self.mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        kernel = np.ones((5,5), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
        
    def extract_contours(self):
        self.red_polygons_contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    def create_paper_mask(self):
        img_without_blue = self.img.copy()
        img_without_blue[self.mask > 0] = [0, 0, 0] 
        
        self.gray_img = cv2.cvtColor(img_without_blue, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(self.gray_img, (151, 151), 0)
        
        _, self.binary = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)
        self.binary = cv2.bitwise_not(self.binary) 
    
    def create_text_mask(self):
        height, width = self.gray_img.shape
        scale_factor = 1.0
        
        if width > 2000 or height > 2000:
            scale_factor = 0.5
            small_gray = cv2.resize(self.gray_img, None, fx=scale_factor, fy=scale_factor)
        else:
            small_gray = self.gray_img
        
        binary_text = cv2.adaptiveThreshold(small_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, blockSize=7, C=4)
        
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        text_mask = cv2.morphologyEx(binary_text, cv2.MORPH_OPEN, kernel_small)
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel_medium)
        
        contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_mask = np.zeros_like(text_mask)
        
        min_size = max(3, int(small_gray.shape[0] * small_gray.shape[1] * 0.000001))
        max_dimension = min(small_gray.shape[0], small_gray.shape[1]) / 4
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_size:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filtros de aspecto ratio y tamaño
            aspect_ratio = h / w if w > 0 else 0
            if (0.1 <= aspect_ratio <= 12 and max(w, h) <= max_dimension):
                cv2.fillPoly(filtered_mask, [contour], 255)
        
        if scale_factor != 1.0:
            filtered_mask = cv2.resize(filtered_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        
        filtered_mask = cv2.bitwise_and(filtered_mask, self.binary)
        
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        
        final_mask = cv2.dilate(filtered_mask, kernel_h, iterations=1)
        final_mask = cv2.dilate(final_mask, kernel_v, iterations=1)
        
        self.text_mask = final_mask