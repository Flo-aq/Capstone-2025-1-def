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
    def __init__(self, camera_box, images, parameters, start):
        """
        Initialize ImageFunction4 with camera and list of images.

        Args:
            camera (Camera): Camera object for image acquisition
            images (list): List of ImageFunction3 instances to compose
        """
        super().__init__(function=4, image=None, camera_box=camera_box)
        self.images = images
        self.lines = None
        self.parameters = parameters["first_module"]
        self.masks = []
        self.start = start

    def create_image(self):
        """
        Creates a composite panorama from multiple images using their corner positions.
        Places each image in its correct position in the reference system.

        Raises:
            ValueError: If no images are provided
        """
        if not self.images:
            raise ValueError("No images provided for composite image creation")
        save_dir = "FlowImages/RecompositionImage/Images"
        os.makedirs(save_dir, exist_ok=True)
        
        for idx, img in enumerate(self.images):
          if img.image is None:
                img.process()
          if img.image is not None:
              filename = os.path.join(save_dir, f"img_{idx+1}.png")
              # Si la imagen es binaria o de un canal, guardar como está
              if len(img.originial_img.shape) == 2:
                  cv2.imwrite(filename, img.original_img)
              # Si es BGR, convertir a RGB para visualización estándar
              elif len(img.original_img.shape) == 3 and img.original_img.shape[2] == 3:
                  cv2.imwrite(filename, cv2.cvtColor(img.original_img, cv2.COLOR_BGR2RGB))
              print(f"Imagen {idx+1} guardada en: {filename}")
    
        panorama = self.images[0].original_img
        mask_panorama = self.images[0].mask
        contours_panorama = self.images[0].red_polygons_contours
        for idx, img in enumerate(self.images[1:]):
            
            result = self.stitch_imgs(img, panorama, mask_panorama, contours_panorama)
            if result is not None:
                panorama = result
                mask_panorama = self.create_mask(panorama)
                contours_panorama = self.get_red_polygons_contours(mask_panorama)
                self.save_panorama_step(panorama, step=idx+2)  # idx+2 porque la base es la 1

            else:
                print(f"Failed to stitch image {idx + 1} with the panorama.")
                    # panorama = np.zeros((self.height_px, self.width_px), dtype=np.uint8)
        # current_panorama = panorama.copy()
        # for idx, img in enumerate(self.images):
        #     if img.image is None:
        #         img.process()
        #     y_start = img.corners_positions_px['top_left'][1]
        #     x_start = img.corners_positions_px['top_left'][0]
        #     y_end = y_start + img.height_px
        #     x_end = x_start + img.width_px
        #     print(
        #         f"Placing image {idx + 1} at position: ({x_start}, {y_start}) to ({x_end}, {y_end})")
        #     current_panorama[y_start:y_end, x_start:x_end] = img.image[:img.height_px, :img.width_px]

        # print("Composite image created.")
        new_panorama = np.zeros((self.height_px, self.width_px, 3), dtype=np.uint8)
        y_start = self.images[0].corners_positions_px['top_left'][1]
        x_start = self.images[0].corners_positions_px['top_left'][0]
        y_end = y_start + panorama.shape[0]
        x_end = x_start + panorama.shape[1]
        new_panorama[y_start:y_end, x_start:x_end] = panorama[:panorama.shape[0], :panorama.shape[1]]
        img_flat = new_panorama.reshape(-1, 3)
        darkest_idx = np.argmin(img_flat.sum(axis=1))
        darkest_color = img_flat[darkest_idx]
        result = new_panorama.copy()
        result[mask_panorama == 255] = darkest_color
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
        self.image = binary
        self.save_final_binary(self.image)
        
        print("Composite image created.")

        
        # self.visualize_recomposition_process()
    def create_mask(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        return mask
    
    def get_red_polygons_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def stitch_imgs(self, img, fixed_img, mask_fixed_img, contours_fixed_img, area_threshold=0.3, perimeter_threshold=0.3):
        similar_pairs = self.find_similar_contours(img.red_polygons_contours, contours_fixed_img, area_threshold, perimeter_threshold)
        if len(similar_pairs) == 0:
            print("No similar contours found.")
            return None
        half = len(similar_pairs) // 2
        best_pairs = similar_pairs[:half] if half > 0 else similar_pairs
        selected_idxs1 = [pair[0] for pair in best_pairs]
        selected_idxs2 = [pair[1] for pair in best_pairs]
        
        result1 = self.extract_red_features_from_contours(img.original_img, img.mask, selected_idxs1)
        kp1, desc1, contour_masks1, all_contours_1 = result1
        
        result2 = self.extract_red_features_from_contours(fixed_img, mask_fixed_img, selected_idxs2)
        kp2, desc2, contour_masks2, all_contours_2 = result2
        
        matches = self.match_features(desc1, desc2)
        
        if len(matches) >= 4:
          src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
          dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
          homography, mask_homo = cv2.findHomography(src_pts, dst_pts, 
                                                   cv2.RANSAC, 5.0)
          h1, w1 = img.original_img.shape[:2]
          h2, w2 = fixed_img.shape[:2]
        
          corners_img1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
          corners_transformed = cv2.perspectiveTransform(corners_img1, homography)

          all_corners = np.concatenate([corners_transformed, 
                                     np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)])

          [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())
          [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())
          
          translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
          homography_adjusted = translation.dot(homography)
          panorama_width = x_max - x_min
          panorama_height = y_max - y_min
          
          panorama = cv2.warpPerspective(img.original_img, homography_adjusted, 
                                       (panorama_width, panorama_height))
          
          panorama[-y_min:h2-y_min, -x_min:w2-x_min] = fixed_img
          return panorama

    def find_similar_contours(self, contours1, contours2, area_threshold=0.3, perimeter_threshold=0.3):
        similar_pairs = []
        
        for i, cnt1 in enumerate(contours1):
            area1 = cv2.contourArea(cnt1)
            perimeter1 = cv2.arcLength(cnt1, True)
            
            if area1 < 50: 
                continue
                
            for j, cnt2 in enumerate(contours2):
                area2 = cv2.contourArea(cnt2)
                perimeter2 = cv2.arcLength(cnt2, True)
                
                if area2 < 50:
                    continue
                
                area_diff = abs(area1 - area2) / max(area1, area2)
                perimeter_diff = abs(perimeter1 - perimeter2) / max(perimeter1, perimeter2)
                
                if area_diff <= area_threshold and perimeter_diff <= perimeter_threshold:
                    similarity_score = (1 - area_diff) * (1 - perimeter_diff)
                    similar_pairs.append((i, j, similarity_score, area1, perimeter1, area2, perimeter2))
        
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return similar_pairs
    
    def extract_red_features_from_contours(self, image, mask, contour_indices):
        sift = cv2.SIFT_create()
    
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Crear máscara solo para los contornos seleccionados
        selected_mask = np.zeros_like(mask)
        contour_masks = []
        
        for idx in contour_indices:
            if idx < len(contours):
                single_contour_mask = np.zeros_like(mask)
                cv2.fillPoly(single_contour_mask, [contours[idx]], 255)
                selected_mask = cv2.bitwise_or(selected_mask, single_contour_mask)
                contour_masks.append(single_contour_mask)
        
        keypoints, descriptors = sift.detectAndCompute(image, selected_mask)
        
        return keypoints, descriptors, contour_masks, contours
    
    def match_features(self, desc1, desc2, ratio_threshold=0.75):
        if desc1 is None or desc2 is None:
            return []
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches
        
    def extract_and_rotate(self, corners_px, width, height):
        """
        Extracts and rotates paper region based on corner positions.

        Args:
            corners_positions_px (dict): Dictionary with corner positions in pixels
            width_mm (float): Target width in millimeters
            height_mm (float): Target height in millimeters

        Returns:
            ndarray: Transformed and rotated image
        """
        if self.image is None:
            raise ValueError("No image to process")
        if None in corners_px.values():
            raise ValueError("Corners positions cannot be None")
        print(corners_px)
        corners = np.float32([
            corners_px['top_left'],
            corners_px['top_right'],
            corners_px['bottom_right'],
            corners_px['bottom_left']
        ])

        dst_width = int(width / self.camera.mm_per_px_h)
        dst_height = int(height / self.camera.mm_per_px_v)

        dst_points = np.float32([
            [0, 0],
            [dst_width, 0],
            [dst_width, dst_height],
            [0, dst_height]
        ])

        matrix = cv2.getPerspectiveTransform(corners, dst_points)
        warped = cv2.warpPerspective(
            self.image, matrix, (dst_width, dst_height))

        original_img_copy = self.image.copy()
        self.visualize_perspective_transform(original_img_copy, warped, corners_px)

        return warped
    def save_panorama_step(self, panorama, step):
        """
        Guarda el estado actual del panorama en FlowImages/RecompositionImage con timestamp y número de paso.
        """
        save_dir = "FlowImages/RecompositionImage"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = join(save_dir, f"RecompositionProcess_step{step}_{timestamp}.png")
        # Si la imagen es BGR, conviértela a RGB para matplotlib
        if len(panorama.shape) == 3 and panorama.shape[2] == 3:
            img_to_save = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
        else:
            img_to_save = panorama
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(img_to_save, cmap='gray' if len(img_to_save.shape) == 2 else None)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Panorama paso {step} guardado en: {filename}")

    def save_final_binary(self, binary_img):
        """
        Guarda la imagen binaria final del panorama.
        """
        save_dir = "FlowImages/RecompositionImage"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = join(save_dir, f"Recomposition_final_{timestamp}.png")
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(binary_img, cmap='gray')
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Imagen binaria final guardada en: {filename}")
        
    def visualize_recomposition_process(self):
        """
        Visualiza y guarda el proceso de recomposición de la imagen.
        Muestra cómo se va formando la imagen compuesta a medida que se añaden imágenes individuales.
        """
        if not self.images:
            print("No hay imágenes para visualizar")
            return

        # Crear directorio si no existe
        save_dir = "FlowImages/RecompositionImage"
        os.makedirs(save_dir, exist_ok=True)

        # Crear timestamp para el nombre de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = join(save_dir, f"RecompositionProcess_{timestamp}.png")

        # Configurar estilo de visualización
        plt.style.use('dark_background')

        # Calcular número de filas y columnas para el grid
        n_images = len(self.images) + 1  # +1 para la imagen final
        n_cols = min(3, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols

        # Crear figura con subplots
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        fig.patch.set_facecolor('#121212')

        # Convertir axs a arreglo para acceder fácilmente si solo hay una fila
        if n_rows == 1:
            axs = np.array([axs])
        if n_cols == 1:
            axs = axs.reshape(-1, 1)

        # Base panorama (lienzo en blanco)
        panorama = np.zeros((self.height_px, self.width_px), dtype=np.uint8)
        current_panorama = panorama.copy()

        # Mostrar cada imagen siendo añadida progresivamente
        for idx, img in enumerate(self.images):
            # Añadir la imagen actual al panorama
            y_start = img.corners_positions_px['top_left'][1]
            x_start = img.corners_positions_px['top_left'][0]
            y_end = y_start + img.height_px
            x_end = x_start + img.width_px

            # Actualizar la imagen actual
            current_panorama = current_panorama.copy()
            current_panorama[y_start:y_end, x_start:x_end] = img.image

            # Mostrar en el subplot
            row, col = idx // n_cols, idx % n_cols
            axs[row, col].imshow(current_panorama, cmap='gray')
            axs[row, col].set_title(
                f"Imagen {idx+1} añadida")
            axs[row, col].axis('off')

            # Añadir un rectángulo para mostrar dónde se colocó la imagen
            rect = plt.Rectangle((x_start, y_start), (x_end-x_start), (y_end-y_start),
                                 linewidth=10, edgecolor=COLOR_PALETTE[idx % len(COLOR_PALETTE)],
                                 facecolor='none')
            axs[row, col].add_patch(rect)

        # Mostrar la imagen final
        row, col = (n_images - 1) // n_cols, (n_images - 1) % n_cols
        axs[row, col].imshow(current_panorama, cmap='gray')
        axs[row, col].set_title(
            "Imagen Recompuesta Final")
        axs[row, col].axis('off')

        # Ocultar los ejes vacíos
        for i in range(n_images, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axs[row, col].axis('off')


        # Guardar la figura
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(filename, facecolor=fig.get_facecolor(), dpi=150)
        plt.close()

        return 

    def visualize_perspective_transform(self, original_image, transformed_image, corners_px):
        """
        Visualiza y guarda el proceso de transformación de perspectiva.

        Args:
            original_image: Imagen original
            transformed_image: Imagen después de la transformación de perspectiva
            corners_px: Coordenadas de las esquinas usadas para la transformación
        """
        if original_image is None or transformed_image is None:
            print("No hay imágenes para visualizar")
            return

        # Crear directorio si no existe
        save_dir = "FlowImages/RecompositionImage"
        os.makedirs(save_dir, exist_ok=True)

        # Crear timestamp para el nombre de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = join(save_dir, f"PerspectiveTransform_{timestamp}.png")

        # Configurar estilo de visualización
        plt.style.use('dark_background')

        # Crear figura con subplots
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        fig.patch.set_facecolor('#121212')

        # Mostrar imagen original con las esquinas marcadas
        axs[0].imshow(original_image, cmap='gray')
        axs[0].set_title(
            "Imagen Original con Esquinas Detectadas")
        axs[0].axis('off')

        # Dibujar las esquinas en la imagen original
        corners = [
            corners_px['top_left'],
            corners_px['top_right'],
            corners_px['bottom_right'],
            corners_px['bottom_left'],
            corners_px['top_left']  # Cerrar el polígono
        ]

        # Dibujar el contorno
        corners_array = np.array(corners)
        axs[0].plot(corners_array[:, 0], corners_array[:, 1],
                    '-', color=COLOR_PALETTE[5], linewidth=10)

        # Marcar cada esquina
        corner_labels = ['TL', 'TR', 'BR', 'BL']
        for i, (corner, label) in enumerate(zip(corners[:4], corner_labels)):
            axs[0].scatter(corner[0], corner[1],
                           color=COLOR_PALETTE[i+1], s=100)

        # Mostrar imagen transformada
        axs[1].imshow(transformed_image, cmap='gray')
        axs[1].set_title("Imagen con Perspectiva Corregida")
        axs[1].axis('off')

        # Guardar la figura
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(filename, facecolor=fig.get_facecolor(), dpi=150)
        plt.close()
