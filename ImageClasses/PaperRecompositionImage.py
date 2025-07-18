from math import e
import time
from tracemalloc import start
from ImageClasses.Image import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from datetime import datetime
from os.path import join

COLOR_PALETTE = ['#43005c', '#6e0060', '#95005d',
                '#b80056', '#d51e4b', '#eb443b', '#f96927', '#ff8d00']


class PaperRecompositionImage(Image):
    def __init__(self, camera_box, images, parameters, stitcher, path):
        """
        Initialize PaperRecompositionImage with camera and list of images.

        Args:
            camera_box (CameraBox): Camera object for image acquisition
            images (list): List of images to compose
            parameters (dict): Configuration parameters
            stitcher (ImageStitcher): Stitcher object for combining images
        """
        super().__init__(function=4, image=None, camera_box=camera_box)
        self.images = images
        self.lines = None
        self.parameters = parameters["first_module"]
        self.image_stitcher = stitcher
        self.masks = []
        self.paper_contour = None
        self.paper_mask = None
        self.corners_dict = None
        self.path = path
        # Realizar el stitching y obtener todos los valores devueltos
        directory = join("FlowImages", "single_images_to_stitch", "stitching")
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, img in enumerate(self.images):
            output_path = join(directory, f"image_{i}_{timestamp}.jpg")
            cv2.imwrite(output_path, img)
        start = time.time()
        result = self.image_stitcher.stitch_and_extract(self.images)
        end = time.time()
        with open(path, 'a') as f:
            f.write(f"Stitching time: {end - start:.2f} seconds\n")
    
        # Manejar los valores devueltos
        if isinstance(result, tuple) and len(result) >= 3:
            self.image, self.paper_contour, self.paper_mask, self.text_mask = result
            # Si hay contorno, calcular las esquinas ordenadas
            if self.paper_contour is not None:
                self.corners_dict = self.get_ordered_corners(self.paper_contour)
        else:
            # Si solo devuelve la imagen
            self.image = result
            # Detectar el polígono de la hoja
            self.paper_contour, self.paper_mask, img_without_background = self.get_paper_polygon(self.image)
            if self.paper_contour is not None:
                self.corners_dict = self.get_ordered_corners(self.paper_contour)
        panoramas_dir = os.path.join("FlowImages", "Panoramas")
        os.makedirs(panoramas_dir, exist_ok=True)
        panorama_path = os.path.join(panoramas_dir, f"panorama_{timestamp}.jpg")
        if self.image is not None:
            cv2.imwrite(panorama_path, self.image)
            print(f"Panorama guardada en: {panorama_path}")
    # Añadir las funciones transferidas desde ImageStitcher
    def get_paper_polygon(self, img):
        """Versión simplificada de get_paper_polygon_from_stitched_img para la clase PaperRecompositionImage"""
        # Verificar si tiene canal alpha
        has_alpha = len(img.shape) > 2 and img.shape[2] == 4
        
        # Convertir a HSV para segmentación por color
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV if not has_alpha else cv2.COLOR_BGRA2HSV)
        
        # Detectar áreas rojas y azules (fondo)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask_red1, mask_red2)
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Combinar máscaras de fondo
        background_mask = cv2.bitwise_or(red_mask, blue_mask)
        
        # Invertir para obtener la máscara de la hoja
        paper_mask = cv2.bitwise_not(background_mask)
        
        # Limpiar la máscara
        kernel = np.ones((15, 15), np.uint8)
        paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, paper_mask, img
        
        # Filtrar por área
        min_area = img.shape[0] * img.shape[1] * 0.05
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if not large_contours:
            return None, paper_mask, img
        
        # Obtener el contorno más grande
        paper_contour = max(large_contours, key=cv2.contourArea)
        
        # Aproximar el contorno
        epsilon = 0.02 * cv2.arcLength(paper_contour, True)
        approx_contour = cv2.approxPolyDP(paper_contour, epsilon, True)
        
        # Asegurar 4 vértices
        if len(approx_contour) != 4:
            rect = cv2.minAreaRect(paper_contour)
            approx_contour = np.int0(cv2.boxPoints(rect))
        
        # Crear máscara refinada
        refined_mask = np.zeros_like(paper_mask)
        cv2.drawContours(refined_mask, [approx_contour], 0, 255, -1)
        
        # Obtener las esquinas ordenadas
        
        # Eliminar fondo
        img_without_background = img.copy()
        background_color = [255, 255, 255, 0] if has_alpha else [255, 255, 255]
        background_pixels = np.where(refined_mask == 0)
        if len(background_pixels[0]) > 0:
            img_without_background[background_pixels] = background_color
        
        return approx_contour, refined_mask, img_without_background
            
    def get_ordered_corners(self, contour):
        """Ordena las esquinas del contorno en top-left, top-right, bottom-right, bottom-left"""
        # Asegurar que tenemos 4 puntos
        if len(contour) != 4:
            rect = cv2.minAreaRect(contour)
            contour = np.int0(cv2.boxPoints(rect))
        
        # Convertir a formato más manejable
        pts = contour.reshape(4, 2)
        
        # Ordenar por suma de coordenadas
        s = pts.sum(axis=1)
        rect = np.zeros((4, 2), dtype=np.int32)
        
        # Asignar esquinas
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        
        # Diferenciar entre top-right y bottom-left por la diferencia
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        
        return {
            'top_left': rect[0].tolist(),
            'top_right': rect[1].tolist(),
            'bottom_right': rect[2].tolist(),
            'bottom_left': rect[3].tolist()
        }
        
    def extract_and_rotate(self, corners_dict, img=None, width_mm=210, height_mm=297):
        """Extrae y rectifica la región del papel basada en las esquinas detectadas"""
        start = time.time()
        # Usar imagen proporcionada o la almacenada
        img_to_process = img if img is not None else self.image
                
        if img_to_process is None:
            raise ValueError("No image to process")
        if None in corners_dict.values():
            raise ValueError("Corner positions cannot be None")
                
        corners = np.float32([
            corners_dict['top_left'],
            corners_dict['top_right'],
            corners_dict['bottom_right'],
            corners_dict['bottom_left']
        ])
                
        # Calcular las distancias de los lados del polígono detectado
        top_side = np.linalg.norm(corners[1] - corners[0])  # top_right - top_left
        right_side = np.linalg.norm(corners[2] - corners[1])  # bottom_right - top_right
        bottom_side = np.linalg.norm(corners[3] - corners[2])  # bottom_left - bottom_right
        left_side = np.linalg.norm(corners[0] - corners[3])  # top_left - bottom_left
        
        # Calcular dimensiones promedio
        detected_width = (top_side + bottom_side) / 2
        detected_height = (left_side + right_side) / 2
        
        # Determinar si necesitamos rotar para que el lado más largo quede vertical
        if detected_width > detected_height:
            # El papel está en landscape, necesitamos rotarlo
            # Reordenar esquinas para rotar 90° (sentido horario)
            corners_rotated = np.float32([
                corners_dict['bottom_left'],   # nuevo top_left
                corners_dict['top_left'],      # nuevo top_right
                corners_dict['top_right'],     # nuevo bottom_right
                corners_dict['bottom_right']   # nuevo bottom_left
            ])
            corners = corners_rotated
            
            # Después de rotar: detected_width se convierte en height, detected_height en width
            dst_width = int(detected_height)   # lado corto
            dst_height = int(detected_width)   # lado largo
        else:
            # El papel ya está en portrait
            dst_width = int(detected_width)    # lado corto
            dst_height = int(detected_height)  # lado largo
                
        dst_points = np.float32([
            [0, 0],
            [dst_width, 0],
            [dst_width, dst_height],
            [0, dst_height]
        ])
                
        # Aplicar transformación de perspectiva
        matrix = cv2.getPerspectiveTransform(corners, dst_points)
        warped = cv2.warpPerspective(img_to_process, matrix, (dst_width, dst_height))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        background = cv2.GaussianBlur(gray, (21, 21), 0)
        normalized = cv2.divide(gray, background, scale=255)
        img_bin = cv2.adaptiveThreshold(
            normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 10
        )

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin, connectivity=8)
        mask = np.zeros_like(img_bin)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if area > 10:
                mask[labels == i] = 255
            else:
                x_start = max(0, x - 10)
                y_start = max(0, y - 10)
                x_end = min(img_bin.shape[1], x + w + 10)
                y_end = min(img_bin.shape[0], y + h + 10)
                if np.any(mask[y_start:y_end, x_start:x_end]):
                    mask[labels == i] = 255

        cleaned = mask
        
        
        # 4. remove_isolated_noise_components sobre el crop
        os.makedirs("FlowImages/warped_images", exist_ok=True)
        end = time.time()
        processing_time = end - start
        with open(self.path, 'a') as f:
            f.write(f"Paper extraction time: {processing_time:.2f} seconds\n")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("FlowImages/warped_images", f"warped_{timestamp}.jpg")
        cv2.imwrite(output_path, cleaned)
        print(f"Warped image saved to: {output_path}")
        return cleaned
        
    def save_debug_visualization(self, img, contour=None, corners_dict=None, output_dir="debug_imgs"):
        """Guarda imágenes de visualización para debugging"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar imagen original
        orig_path = os.path.join(output_dir, f"{timestamp}_original.jpg")
        cv2.imwrite(orig_path, img)
        
        # Si hay contorno, visualizarlo
        if contour is not None:
            vis_img = img.copy()
            cv2.drawContours(vis_img, [contour], 0, (0, 255, 0), 3)
            
            # Si hay esquinas, visualizarlas
            if corners_dict is not None:
                for name, pos in corners_dict.items():
                    pos_tuple = tuple(map(int, pos))
                    cv2.circle(vis_img, pos_tuple, 5, (0, 0, 255), -1)
                    cv2.putText(vis_img, name, (pos_tuple[0]+5, pos_tuple[1]+5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Guardar visualización
            vis_path = os.path.join(output_dir, f"{timestamp}_contour_vis.jpg")
            cv2.imwrite(vis_path, vis_img)
        
        print(f"Debug visualizations saved to {output_dir}")
        
    # Métodos de acceso y procesamiento
    def get_rectified_image(self):
        """Obtiene la imagen rectificada, rectificándola si es necesario"""
        if self.paper_contour is not None and self.corners_dict is not None:
            if not hasattr(self, 'rectified_image') or self.rectified_image is None:
                self.rectified_image = self.extract_and_rotate(self.corners_dict, self.image)
            return self.rectified_image
        return self.image
    
    def process_image(self):
        """Procesa la imagen stitched"""
        # Asegurar que tenemos la imagen rectificada
        rectified = self.get_rectified_image()
        
        # Código adicional de procesamiento según necesidades
        
        # Guardar debug
        self.save_debug_visualization(self.image, self.paper_contour, self.corners_dict)