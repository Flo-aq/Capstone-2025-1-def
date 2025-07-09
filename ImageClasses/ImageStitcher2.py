from unittest import result

from ImageClasses.ImageToStitch import ImageToStitch
import numpy as np
import cv2


class ImageStitcher2:
    def __init__(self, area_threshold=0.3, 
                 perimeter_threshold=0.3, 
                 min_matches=4,
                 processing_scale=0.5,
                 debug=False):
      
        self.area_threshold = area_threshold
        self.perimeter_threshold = perimeter_threshold
        self.min_matches = min_matches
        self.processing_scale = processing_scale
        self.debug = debug
    
    def stitch_and_extract(self, images):
      if not images:
          return None, None, None
        
      final_image, result_handler = self.stitch_images(images)
      
      if final_image is None or result_handler is None:
            return None, None, None
      
      contour_hoja = None
      if result_handler.binary is not None:
          # Invertir la máscara binaria para obtener el contorno de la hoja
          inverted_binary = cv2.bitwise_not(result_handler.binary)
          contours, _ = cv2.findContours(inverted_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
          
          if contours:
              # Encontrar el contorno con mayor área
              contour_hoja = max(contours, key=cv2.contourArea)
      
      # Extraer la paper mask
      paper_mask = result_handler.binary
      return final_image, contour_hoja, paper_mask

        
    
    def stitch_images(self, images):
        if not images:
            return None, None
        
        if len(images) == 1:
            image_handler = ImageToStitch(images[0])
            image_handler.process()
            return images[0], image_handler
        
        print(f"Stitching {len(images)} images...")
        handlers = []
        for i, img in enumerate(images):
            print(f"Processing image {i+1}/{len(images)}")
            image_handler = ImageToStitch(img)
            image_handler.process()
            handlers.append(image_handler)
        
        result_handler = handlers[0]
        for i in range(1, len(handlers)):
            print(f"Stitching image {i+1}/{len(handlers)}")
            result_handler = self.stitch_pair_with_masks(result_handler, handlers[i])
            
            if result_handler is None:
                print("Stitching failed, returning None")
                result_handler = handlers[i - 1]
        return result_handler.img, result_handler
      
    def stitch_pair_with_masks(self, handler1, handler2):
        img1 = handler1.img
        img2 = handler2.img
        
        text_area1 = np.count_nonzero(handler1.text_mask) if handler1.text_mask is not None else 0
        text_area2 = np.count_nonzero(handler2.text_mask) if handler2.text_mask is not None else 0
        paper_area1 = np.count_nonzero(handler1.binary) if handler1.binary is not None else 0
        paper_area2 = np.count_nonzero(handler2.binary) if handler2.binary is not None else 0

        min_area_img = min(img1.shape[0] * img1.shape[1], img2.shape[0] * img2.shape[1])
        
        ratio1 = 0.95 * text_area1 + 0.05 * paper_area1
        ratio2 = 0.95 * text_area2 + 0.05 * paper_area2
        
        if ratio2 > ratio1:
            handler1, handler2 = handler2, handler1
            img1, img2 = img2, img1
        
        panorama, H_adjusted, panorama_bounds = self.perform_stitching(img1, img2, handler1, handler2)
        if panorama is None:
            print("Stitching failed, returning None")
            return None
        
        result_handler = ImageToStitch(panorama)
        
        self.transform_and_combine_masks(result_handler, handler1, handler2, H_adjusted, panorama_bounds)
        
        return result_handler
      
    def perform_stitching(self, img1, img2, handler1, handler2):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        scale = self.processing_scale
        
        text_mask_1_full, paper_mask_1_full = handler1.text_mask, handler1.binary
        text_mask_2_full, paper_mask_2_full = handler2.text_mask, handler2.binary
        
        text_area1 = np.count_nonzero(text_mask_1_full)
        text_area2 = np.count_nonzero(text_mask_2_full)
        paper_area1 = np.count_nonzero(paper_mask_1_full)
        paper_area2 = np.count_nonzero(paper_mask_2_full)
        ratio1 = 0.95 * text_area1 + 0.05 * paper_area1
        ratio2 = 0.95 * text_area2 + 0.05 * paper_area2
        if ratio2 > ratio1:
            handler1, handler2 = handler2, handler1
            img1, img2 = img2, img1
        
        text_coverage_1 = np.count_nonzero(text_mask_1_full) / (h1 * w1)
        text_coverage_2 = np.count_nonzero(text_mask_2_full) / (h2 * w2)
        
        text_threshold = 0.001
        
        has_text1 = text_coverage_1 > text_threshold
        has_text2 = text_coverage_2 > text_threshold
        
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        img1_small = cv2.resize(img1, None, fx=scale, fy=scale)
        img2_small = cv2.resize(img2, None, fx=scale, fy=scale)
        
        mask1_small = cv2.resize(handler1.mask, None, fx=scale, fy=scale)
        mask2_small = cv2.resize(handler2.mask, None, fx=scale, fy=scale)
        
        text_mask1 = cv2.resize(handler1.text_mask, None, fx=scale, fy=scale) 
        text_mask2 = cv2.resize(handler2.text_mask, None, fx=scale, fy=scale) 
        
        # Combinar máscaras para detección de características
        
        contours1, _ = cv2.findContours(mask1_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(mask2_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if has_text1 and has_text2:
          similar_pairs = self.find_similar_contours(contours1, contours2, self.area_threshold, self.perimeter_threshold)
          top_contours_mask1 = np.zeros_like(mask1_small)
          top_contours_mask2 = np.zeros_like(mask2_small)
          
          top_pairs = similar_pairs[:min(15, len(similar_pairs))]
          
          for pair in top_pairs:
              idx1, idx2 = pair[0], pair[1]
              cv2.drawContours(top_contours_mask1, [contours1[idx1]], -1, 255, -1)
              cv2.drawContours(top_contours_mask2, [contours2[idx2]], -1, 255, -1)
          combined_mask1 = cv2.bitwise_or(text_mask1, top_contours_mask1)
          combined_mask2 = cv2.bitwise_or(text_mask2, top_contours_mask2)
        
        # Configurar ORB
          orb = cv2.ORB_create(
              nfeatures=7000,
              scaleFactor=1.05,
              nlevels=12,
              edgeThreshold=10,
              patchSize=21,
              fastThreshold=8,
              WTA_K=3,
              scoreType=cv2.ORB_HARRIS_SCORE,
              firstLevel=0
          )
        else:
          similar_pairs = self.find_similar_contours(
              contours1, contours2, 
              self.area_threshold, 
              self.perimeter_threshold
          )
          
          top_contours_mask1 = np.zeros_like(mask1_small)
          top_contours_mask2 = np.zeros_like(mask2_small)
          
          # Limitar a los 15 mejores pares o menos si no hay suficientes
          top_pairs = similar_pairs[:min(15, len(similar_pairs))]
          
          # Dibujar solo los contornos seleccionados en las nuevas máscaras
          for pair in top_pairs:
              idx1, idx2 = pair[0], pair[1]
              cv2.drawContours(top_contours_mask1, [contours1[idx1]], -1, 255, -1)
              cv2.drawContours(top_contours_mask2, [contours2[idx2]], -1, 255, -1)
              
          combined_mask1 = top_contours_mask1
          combined_mask2 = top_contours_mask2
          
          orb = cv2.ORB_create(
              nfeatures=4000,        # Menos características para contornos
              scaleFactor=1.2,       # Mayor escala para captar contornos completos
              nlevels=8,             # Menos niveles para formas más grandes
              edgeThreshold=31,      # Mayor umbral para formas más completas
              patchSize=51,          # Patches más grandes para contornos
              fastThreshold=20,      # Mayor umbral para detectar esquinas más definidas
              WTA_K=3,               # Aumentado para mayor discriminación
              scoreType=cv2.ORB_HARRIS_SCORE,
              firstLevel=0
          )
      
        # Detectar keypoints
        kp1, desc1 = orb.detectAndCompute(img1_small, combined_mask1)
        kp2, desc2 = orb.detectAndCompute(img2_small, combined_mask2)
        
        if desc1 is None or desc2 is None or len(kp1) < self.min_matches or len(kp2) < self.min_matches:
            print("No se pudieron extraer suficientes características")
            return None, None, None
        
        # Matcher para descriptores binarios
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        # Filtro de ratio de Lowe
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < self.min_matches:
            print(f"No hay suficientes coincidencias: {len(good_matches)}/{self.min_matches}")
            return None, None, None
        
        # Calcular homografía
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            print("No se pudo calcular la homografía")
            return None, None, None
        
        # Ajustar homografía para resolución original
        scale_matrix = np.array([
            [1/scale, 0, 0],
            [0, 1/scale, 0],
            [0, 0, 1]
        ])
        H_original = scale_matrix @ H @ np.linalg.inv(scale_matrix)
        
        # Calcular dimensiones del panorama
        corners = np.float32([
            [0, 0], [w1, 0], [w1, h1], [0, h1]
        ]).reshape(-1, 1, 2)
        
        transformed_corners = cv2.perspectiveTransform(corners, H_original)
        
        all_corners = np.concatenate([
            transformed_corners,
            np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        ])
        
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())
        
        # Ajustar homografía con traslación
        translation = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ])
        
        H_adjusted = translation @ H_original
        
        # Dimensiones del panorama
        width = x_max - x_min
        height = y_max - y_min
        
        # Crear el panorama
        panorama = cv2.warpPerspective(img1, H_adjusted, (width, height))
        
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[max(0, -y_min):max(0, -y_min) + h2, max(0, -x_min):max(0, -x_min) + w2] = 1
        
        # Mezclar con la segunda imagen
        y_start = max(0, -y_min)
        y_end = min(height, y_start + h2)
        x_start = max(0, -x_min)
        x_end = min(width, x_start + w2)
        
        img2_y_start = 0 if y_min <= 0 else -y_min
        img2_y_end = img2_y_start + (y_end - y_start)
        img2_x_start = 0 if x_min <= 0 else -x_min
        img2_x_end = img2_x_start + (x_end - x_start)
        
        # Verificar límites válidos
        if (img2_y_end > h2 or img2_x_end > w2 or
            img2_y_start < 0 or img2_x_start < 0 or
            y_start < 0 or x_start < 0 or
            y_end > height or x_end > width):
            print("Error en los límites de recorte")
            return None, None, None
            
        panorama_region = panorama[y_start:y_end, x_start:x_end]
        img2_region = img2[img2_y_start:img2_y_end, img2_x_start:img2_x_end]
        
        # Mezcla simple con máscara
        non_black_mask = np.any(panorama_region > 10, axis=2).astype(np.float32)
        alpha_panorama = non_black_mask
        alpha_img2 = 1 - alpha_panorama
        
        for c in range(3):
            panorama_region[:, :, c] = (
                alpha_panorama * panorama_region[:, :, c] +
                alpha_img2 * img2_region[:, :, c]
            )
        
        panorama_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
        #Calcular panorama bounds
        panorama_bounds = {
            'width': width,
            'height': height,
            'x_min': x_min,
            'y_min': y_min
        }
        return panorama, H_adjusted, panorama_bounds
    
    def transform_and_combine_masks(self, result_handler: ImageToStitch, 
                                   handler1: ImageToStitch, handler2: ImageToStitch,
                                   H_adjusted: np.ndarray, panorama_bounds: dict):
        """
        Transforma y combina todas las máscaras al espacio del panorama.
        """
        width = panorama_bounds['width']
        height = panorama_bounds['height']
        x_min = panorama_bounds['x_min']
        y_min = panorama_bounds['y_min']
        
        # Inicializar máscaras combinadas
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        combined_text_mask = np.zeros((height, width), dtype=np.uint8)
        combined_binary = np.zeros((height, width), dtype=np.uint8)
        
        # Transformar máscaras de la primera imagen
        if handler1.mask is not None:
            mask1_transformed = cv2.warpPerspective(handler1.mask, H_adjusted, (width, height))
            combined_mask = cv2.bitwise_or(combined_mask, mask1_transformed)
        
        if handler1.text_mask is not None:
            text_mask1_transformed = cv2.warpPerspective(handler1.text_mask, H_adjusted, (width, height))
            combined_text_mask = cv2.bitwise_or(combined_text_mask, text_mask1_transformed)
        
        if handler1.binary is not None:
            binary1_transformed = cv2.warpPerspective(handler1.binary, H_adjusted, (width, height))
            combined_binary = cv2.bitwise_or(combined_binary, binary1_transformed)
        
        # Añadir máscaras de la segunda imagen (sin transformar, solo trasladar)
        h2, w2 = handler2.img.shape[:2]
        y_start = max(0, -y_min)
        y_end = min(height, y_start + h2)
        x_start = max(0, -x_min)
        x_end = min(width, x_start + w2)
        
        img2_y_start = 0 if y_min <= 0 else -y_min
        img2_y_end = img2_y_start + (y_end - y_start)
        img2_x_start = 0 if x_min <= 0 else -x_min
        img2_x_end = img2_x_start + (x_end - x_start)
        
        # Verificar límites válidos para máscaras
        if (img2_y_end <= h2 and img2_x_end <= w2 and
            img2_y_start >= 0 and img2_x_start >= 0 and
            y_start >= 0 and x_start >= 0 and
            y_end <= height and x_end <= width):
            
            if handler2.mask is not None:
                mask2_region = handler2.mask[img2_y_start:img2_y_end, img2_x_start:img2_x_end]
                combined_mask[y_start:y_end, x_start:x_end] = cv2.bitwise_or(
                    combined_mask[y_start:y_end, x_start:x_end], mask2_region
                )
            
            if handler2.text_mask is not None:
                text_mask2_region = handler2.text_mask[img2_y_start:img2_y_end, img2_x_start:img2_x_end]
                combined_text_mask[y_start:y_end, x_start:x_end] = cv2.bitwise_or(
                    combined_text_mask[y_start:y_end, x_start:x_end], text_mask2_region
                )
            
            if handler2.binary is not None:
                binary2_region = handler2.binary[img2_y_start:img2_y_end, img2_x_start:img2_x_end]
                combined_binary[y_start:y_end, x_start:x_end] = cv2.bitwise_or(
                    combined_binary[y_start:y_end, x_start:x_end], binary2_region
                )
        
        # Asignar máscaras combinadas al resultado
        result_handler.mask = combined_mask
        result_handler.text_mask = combined_text_mask
        result_handler.binary = combined_binary
        
        # Extraer contornos de la máscara combinada
        result_handler.extract_contours()

    def analyze_similarity_criteria(self, contours1, contours2):
        similar_pairs = self.find_similar_contours()
        
        if not similar_pairs:
            print("No se encontraron pares similares para analizar")
            return
        
        # Crear listas para almacenar criterios
        pairs_idx = []
        scores = []
        area_diffs = []
        perim_diffs = []
        vertex_diffs = []
        aspect_diffs = []
        angle_diffs = []
        compactness_diffs = []
        vertex_dist_diffs = []
        
        # Para cada par, calcular todos los criterios
        for pair in similar_pairs:
            i, j, score = pair[0], pair[1], pair[2]
            pairs_idx.append((i, j))
            scores.append(score)
            
            # Calcular todas las métricas como en _find_similar_contours
            cnt1 = contours1[i]
            cnt2 = contours2[j]
            
            # Áreas
            area1 = cv2.contourArea(cnt1)
            area2 = cv2.contourArea(cnt2)
            area_diff = abs(area1 - area2) / max(area1, area2)
            area_diffs.append(area_diff)
            
            # Perímetros
            perim1 = cv2.arcLength(cnt1, True)
            perim2 = cv2.arcLength(cnt2, True)
            perim_diff = abs(perim1 - perim2) / max(perim1, perim2)
            perim_diffs.append(perim_diff)
            
            # Vértices
            epsilon1 = 0.02 * perim1
            epsilon2 = 0.02 * perim2
            approx1 = cv2.approxPolyDP(cnt1, epsilon1, True)
            approx2 = cv2.approxPolyDP(cnt2, epsilon2, True)
            vertex_diff = abs(len(approx1) - len(approx2)) / max(len(approx1), len(approx2))
            vertex_diffs.append(vertex_diff)
            
            # Proporciones
            x1, y1, w1, h1 = cv2.boundingRect(cnt1)
            x2, y2, w2, h2 = cv2.boundingRect(cnt2)
            aspect_ratio1 = max(w1, h1) / max(1, min(w1, h1))
            aspect_ratio2 = max(w2, h2) / max(1, min(w2, h2))
            aspect_diff = abs(aspect_ratio1 - aspect_ratio2) / max(aspect_ratio1, aspect_ratio2)
            aspect_diffs.append(aspect_diff)
            
            # Compacidad
            compactness1 = 4 * np.pi * area1 / (perim1 * perim1) if perim1 > 0 else 0
            compactness2 = 4 * np.pi * area2 / (perim2 * perim2) if perim2 > 0 else 0
            compactness_diff = abs(compactness1 - compactness2) / max(0.01, max(compactness1, compactness2))
            compactness_diffs.append(compactness_diff)
            
            # Calcular ángulos
            angles1 = self.calculate_vertex_angles(approx1)
            angles2 = self.calculate_vertex_angles(approx2)
            
            angle_diff = 1.0
            if angles1 and angles2:
                min_angles = min(len(angles1), len(angles2))
                if min_angles > 0:
                    angle_diffs_list = [abs(angles1[i % len(angles1)] - angles2[i % len(angles2)]) 
                                      for i in range(min_angles)]
                    angle_diff = sum(angle_diffs_list) / (min_angles * np.pi)
            angle_diffs.append(angle_diff)
            
            # Distancias relativas entre vértices
            vertex_distances1 = self.calculate_vertex_distances(approx1)
            vertex_distances2 = self.calculate_vertex_distances(approx2)
            
            vertex_dist_diff = 1.0
            if vertex_distances1 and vertex_distances2:
                min_vd = min(len(vertex_distances1), len(vertex_distances2))
                if min_vd > 0:
                    vd_diffs = [abs(vertex_distances1[i % len(vertex_distances1)] - 
                                  vertex_distances2[i % len(vertex_distances2)])
                              for i in range(min_vd)]
                    vertex_dist_diff = sum(vd_diffs) / min_vd
            vertex_dist_diffs.append(vertex_dist_diff)
        
    def calculate_vertex_angles(self, approx):
        """Calcula los ángulos entre vértices del contorno aproximado."""
        angles = []
        if len(approx) >= 3:
            for j in range(len(approx)):
                prev_pt = approx[(j-1) % len(approx)][0]
                curr_pt = approx[j][0]
                next_pt = approx[(j+1) % len(approx)][0]
                
                vec1 = prev_pt - curr_pt
                vec2 = next_pt - curr_pt
                
                dot = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 > 0 and norm2 > 0:
                    cos_angle = dot / (norm1 * norm2)
                    cos_angle = max(-1, min(1, cos_angle))  # Evitar errores numéricos
                    angle = np.arccos(cos_angle)
                    angles.append(angle)
        angles.sort()
        return angles

    def calculate_vertex_distances(self, approx):
        """Calcula las distancias normalizadas de los vértices al centroide."""
        distances = []
        if len(approx) > 1:
            centroid = np.mean(approx.reshape(-1, 2), axis=0)
            for pt in approx:
                dist = np.linalg.norm(pt[0] - centroid)
                distances.append(dist)
            
            # Normalizar por la distancia máxima
            if max(distances) > 0:
                distances = [d/max(distances) for d in distances]
            distances.sort()
        return distances

    def find_similar_contours(self, contours1, contours2):
      """
      Encuentra pares de contornos similares utilizando máscaras normalizadas y relación de áreas.
      
      Args:
          contours1, contours2: Listas de contornos
          Varios umbrales que serán considerados secundariamente
              
      Returns:
          Lista de tuplas (idx1, idx2, score, area1, perimeter1, area2, perimeter2, bbox1, bbox2)
      """
      similar_pairs = []
      min_contour_area = 200  # Filtrar contornos muy pequeños
      
      # Dimensiones de las mini-imágenes normalizadas
      mask_size = (100, 100)
      
      # Crear máscaras para todos los contornos del primer conjunto
      masks1 = []
      for i, cnt1 in enumerate(contours1):
          area1 = cv2.contourArea(cnt1)
          if area1 < min_contour_area:
              # Agregar una máscara vacía para mantener índices consistentes
              masks1.append(None)
              continue
              
          # Crear máscara normalizada
          mask = np.zeros(mask_size, dtype=np.uint8)
          
          # Normalizar y centrar el contorno
          rect = cv2.boundingRect(cnt1)
          x, y, w, h = rect
          
          # Factor de escala para ajustar al tamaño de la máscara (con un borde)
          scale_x = mask_size[1] * 0.8 / max(w, 1)
          scale_y = mask_size[0] * 0.8 / max(h, 1)
          scale = min(scale_x, scale_y)
          
          # Centrar el contorno en la máscara
          centered_cnt = cnt1 - [x + w/2, y + h/2] + [mask_size[1]/2, mask_size[0]/2]
          scaled_cnt = (centered_cnt * scale).astype(np.int32)
          
          # Dibujar el contorno en la máscara
          cv2.fillPoly(mask, [scaled_cnt], 255)
          masks1.append(mask)
      
      # Para cada contorno en el segundo conjunto, crear máscara y comparar
      for j, cnt2 in enumerate(contours2):
          area2 = cv2.contourArea(cnt2)
          if area2 < min_contour_area:
              continue
              
          # Información para el resultado final
          perimeter2 = cv2.arcLength(cnt2, True)
          x2, y2, w2, h2 = cv2.boundingRect(cnt2)
          
          # Crear máscara normalizada para el contorno 2
          mask2 = np.zeros(mask_size, dtype=np.uint8)
          
          # Normalizar y centrar el contorno 2
          rect2 = cv2.boundingRect(cnt2)
          x, y, w, h = rect2
          
          # Factor de escala para ajustar al tamaño de la máscara (con un borde)
          scale_x = mask_size[1] * 0.8 / max(w, 1)
          scale_y = mask_size[0] * 0.8 / max(h, 1)
          scale = min(scale_x, scale_y)
          
          # Centrar el contorno en la máscara
          centered_cnt2 = cnt2 - [x + w/2, y + h/2] + [mask_size[1]/2, mask_size[0]/2]
          scaled_cnt2 = (centered_cnt2 * scale).astype(np.int32)
          
          # Dibujar el contorno en la máscara
          cv2.fillPoly(mask2, [scaled_cnt2], 255)
          
          # Comparar con todas las máscaras del primer conjunto
          for i, mask1 in enumerate(masks1):
              if mask1 is None:
                  continue
                  
              # Calcular intersección y unión de máscaras
              intersection = cv2.bitwise_and(mask1, mask2)
              union = cv2.bitwise_or(mask1, mask2)
              
              # Contar píxeles para calcular áreas
              area_intersection = np.count_nonzero(intersection)
              area_union = np.count_nonzero(union)
              
              if area_union == 0:  # Evitar división por cero
                  continue
                  
              # Calcular relaciones IoU (Intersection over Union)
              iou = area_intersection / area_union
              
              # Calcular relaciones de área individual respecto a la unión
              cnt1 = contours1[i]
              area1 = cv2.contourArea(cnt1)
              perimeter1 = cv2.arcLength(cnt1, True)
              x1, y1, w1, h1 = cv2.boundingRect(cnt1)
              
              # Usar IoU como principal métrica de similitud
              if iou > 0.6:  # Umbral de IoU ajustable
                  similarity_score = iou
                  
                  similar_pairs.append((
                      i, j, similarity_score,
                      area1, perimeter1, area2, perimeter2,
                      (x1, y1, w1, h1), (x2, y2, w2, h2)
                  ))
              
              # Si es necesario, también podemos usar criterios secundarios como antes
              # (área, perímetro, etc.) para refinar la similitud
      
      # Ordenar por puntuación de similitud (mayor primero)
      similar_pairs.sort(key=lambda x: x[2], reverse=True)
      
      return similar_pairs