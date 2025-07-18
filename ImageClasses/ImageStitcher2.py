from unittest import result
from xml.sax import handler

from ImageClasses.ImageToStitch import ImageToStitch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


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
            return None, None, None, None

        final_image, result_handler = self.stitch_images(images)

        if final_image is None or result_handler is None:
            return None, None, None, None

        contour_hoja = None
        if result_handler.binary is not None:
            contours, _ = cv2.findContours(
                result_handler.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Encontrar el contorno con mayor área
                contour_hoja = max(contours, key=cv2.contourArea)

                # Puedes ajustar 0.08 según tu caso
                epsilon = 0.08 * cv2.arcLength(contour_hoja, True)
                approx_contour = cv2.approxPolyDP(contour_hoja, epsilon, True)
                if len(approx_contour) == 4:
                    contour_hoja = approx_contour
                else:
                    rect = cv2.minAreaRect(contour_hoja)
                    contour_hoja = np.int0(cv2.boxPoints(rect))

        # Extraer la paper mask
        paper_mask = result_handler.binary
        return final_image, contour_hoja, paper_mask, result_handler.text_mask

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
            os.makedirs("debug_imgs", exist_ok=True)
            plt.figure(figsize=(10,3))
            plt.subplot(1,3,1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"Imagen {i+1}")
            plt.axis('off')
            plt.subplot(1,3,2)
            plt.imshow(image_handler.text_mask, cmap='gray')
            plt.title("Máscara texto")
            plt.axis('off')
            plt.subplot(1,3,3)
            plt.imshow(image_handler.binary, cmap='gray')
            plt.title("Máscara binaria")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"debug_imgs/preprocess_{i+1}.png")
            plt.close()

        result_handler = handlers[0]
        for i in range(1, len(handlers)):
            print(f"Stitching image {i+1}/{len(handlers)}")
            handler1 = result_handler
            handler2 = handlers[i]

            img1 = handler1.img
            img2 = handler2.img

            text_area1 = np.count_nonzero(
                handler1.text_mask) if handler1.text_mask is not None else 0
            text_area2 = np.count_nonzero(
                handler2.text_mask) if handler2.text_mask is not None else 0
            paper_area1 = np.count_nonzero(
                handler1.binary) if handler1.binary is not None else 0
            paper_area2 = np.count_nonzero(
                handler2.binary) if handler2.binary is not None else 0

            ratio1 = 0.95 * text_area1 + 0.05 * paper_area1
            ratio2 = 0.95 * text_area2 + 0.05 * paper_area2

            if ratio2 > ratio1:
                handler1, handler2 = handler2, handler1
                img1, img2 = img2, img1

            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]

            scale = self.processing_scale

            text_mask_1_full = handler1.text_mask
            text_mask_2_full = handler2.text_mask

            text_coverage_1 = np.count_nonzero(text_mask_1_full) / (h1 * w1)
            text_coverage_2 = np.count_nonzero(text_mask_2_full) / (h2 * w2)

            text_threshold = 0.001

            has_text1 = text_coverage_1 > text_threshold
            has_text2 = text_coverage_2 > text_threshold

            img1_small = cv2.resize(img1, None, fx=scale, fy=scale)
            img2_small = cv2.resize(img2, None, fx=scale, fy=scale)

            mask1_small = cv2.resize(handler1.mask, None, fx=scale, fy=scale)
            mask2_small = cv2.resize(handler2.mask, None, fx=scale, fy=scale)

            text_mask1 = cv2.resize(
                handler1.text_mask, None, fx=scale, fy=scale)
            text_mask2 = cv2.resize(
                handler2.text_mask, None, fx=scale, fy=scale)

            contours1, _ = cv2.findContours(
                mask1_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours2, _ = cv2.findContours(
                mask2_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            similar_pairs = self.find_similar_contours(
                contours1, contours2
            )
            
            top_contours_mask1 = np.zeros_like(mask1_small)
            top_contours_mask2 = np.zeros_like(mask2_small)
            top_pairs = similar_pairs[:min(15, len(similar_pairs))]
            
            for pair in top_pairs:
                idx1, idx2 = pair[0], pair[1]
                cv2.drawContours(top_contours_mask1, [contours1[idx1]], -1, 255, -1)
                cv2.drawContours(top_contours_mask2, [contours2[idx2]], -1, 255, -1)
            orb_contours = cv2.ORB_create(
                  nfeatures=9000,        # Menos características para contornos
                  scaleFactor=1.3,       # Mayor escala para captar contornos completos
                  nlevels=10,             # Menos niveles para formas más grandes
                  edgeThreshold=31,      # Mayor umbral para formas más completas
                  patchSize=131,          # Patches más grandes para contornos
                  fastThreshold=25,      # Mayor umbral para detectar esquinas más definidas
                  WTA_K=4,               # Aumentado para mayor discriminación
                  scoreType=cv2.ORB_HARRIS_SCORE,
                  firstLevel=0
              )
            kp1_contours, desc1_contours = orb_contours.detectAndCompute(img1_small, top_contours_mask1)
            kp2_contours, desc2_contours = orb_contours.detectAndCompute(img2_small, top_contours_mask2)
            
            all_kp1 = []
            all_desc1 = []
            all_kp2 = []
            all_desc2 = []
            
        
            all_kp1.extend(kp1_contours)
            all_desc1.append(desc1_contours)
            all_kp2.extend(kp2_contours)
            all_desc2.append(desc2_contours)
            if has_text1 and has_text2:
                orb_text = cv2.ORB_create(
                  nfeatures=9000,
                  scaleFactor=1.05,
                  nlevels=15,
                  edgeThreshold=10,
                  patchSize=51,
                  fastThreshold=11,
                  WTA_K=3,
                  scoreType=cv2.ORB_HARRIS_SCORE,
                  firstLevel=0
                )
                kp1_text, desc1_text = orb_text.detectAndCompute(img1_small, text_mask1)
                kp2_text, desc2_text = orb_text.detectAndCompute(img2_small, text_mask2)
                all_kp1.extend(kp1_text)
                all_desc1.append(desc1_text)
                all_kp2.extend(kp2_text)
                all_desc2.append(desc2_text)
            if len(all_desc1) == 0 or len(all_desc2) == 0 or len(all_kp1) < self.min_matches or len(all_kp2) < self.min_matches:
                print("No se pudieron extraer suficientes descriptores")
                return None, None
            else:
                desc1_combined = np.vstack(all_desc1)
                desc2_combined = np.vstack(all_desc2)
                kp1_combined = all_kp1
                kp2_combined = all_kp2
            if has_text1 and has_text2:
                combined_mask1 = cv2.bitwise_or(top_contours_mask1, text_mask1)
                combined_mask2 = cv2.bitwise_or(top_contours_mask2, text_mask2)
            else:
                combined_mask1 = top_contours_mask1
                combined_mask2 = top_contours_mask2
                
            plt.figure(figsize=(8,3))
            plt.subplot(1,2,1)
            plt.imshow(combined_mask1, cmap='gray')
            plt.title(f"Máscara combinada 1 (iter {i})")
            plt.axis('off')
            plt.subplot(1,2,2)
            plt.imshow(combined_mask2, cmap='gray')
            plt.title(f"Máscara combinada 2 (iter {i})")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"debug_imgs/mascara_combinada_{i}.png")
            plt.close()
            
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(desc1_combined, desc2_combined, k=2)
            img1_kp = cv2.drawKeypoints(img1_small, all_kp1, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            img2_kp = cv2.drawKeypoints(img2_small, all_kp2, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.8 * n.distance:
                        good_matches.append(m)
          
            if len(good_matches) < self.min_matches:
                print(f"Not enough matches: {len(good_matches)}/{self.min_matches}")
                return None, None
            plt.figure(figsize=(12,5))
            plt.subplot(1,2,1)
            plt.imshow(img1_kp)
            plt.title(f"Keypoints ORB img1 (iter {i})")
            plt.axis('off')
            plt.subplot(1,2,2)
            plt.imshow(img2_kp)
            plt.title(f"Keypoints ORB img2 (iter {i})")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"debug_imgs/keypoints_{i}.png")
            plt.close()

            # ...existing code...
            img_matches = cv2.drawMatches(
                img1_small, all_kp1, img2_small, all_kp2, good_matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.figure(figsize=(12,6))
            plt.imshow(img_matches)
            plt.title(f"Matches ORB (iter {i})")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"debug_imgs/matches_{i}.png")
            plt.close()
            
            

            src_pts = np.float32([all_kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([all_kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is None:
                print("Unable to find homography.")
                return None, None
              
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
            
            width = x_max - x_min
            height = y_max - y_min
            
            panorama = cv2.warpPerspective(img1, H_adjusted, (width, height))
            
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[max(0, -y_min):max(0, -y_min) + h2, max(0, -x_min):max(0, -x_min) + w2] = 1
            
            y_start = max(0, -y_min)
            y_end = min(height, y_start + h2)
            x_start = max(0, -x_min)
            x_end = min(width, x_start + w2)
            
            img2_y_start = 0 if y_min <= 0 else -y_min
            img2_y_end = img2_y_start + (y_end - y_start)
            img2_x_start = 0 if x_min <= 0 else -x_min
            img2_x_end = img2_x_start + (x_end - x_start)
            
            if (img2_y_end > h2 or img2_x_end > w2 or
                img2_y_start < 0 or img2_x_start < 0 or
                y_start < 0 or x_start < 0 or
                y_end > height or x_end > width):
                print("Error en los límites de recorte")
                return None, None
                
            panorama_region = panorama[y_start:y_end, x_start:x_end]
            img2_region = img2[img2_y_start:img2_y_end, img2_x_start:img2_x_end]
            
            non_black_mask = np.any(panorama_region > 10, axis=2).astype(np.float32)
            alpha_panorama = non_black_mask
            alpha_img2 = 1 - alpha_panorama
            
            for c in range(3):
                panorama_region[:, :, c] = (
                    alpha_panorama * panorama_region[:, :, c] +
                    alpha_img2 * img2_region[:, :, c]
                )
            panorama_bounds = {
                'width': width,
                'height': height,
                'x_min': x_min,
                'y_min': y_min
            }
            panorama_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10,6))
            plt.imshow(panorama_rgb)
            plt.title(f"Panorama iteración {i}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"debug_imgs/panorama_{i}.png")
            plt.close()
            result_handler = ImageToStitch(panorama)
            self.transform_and_combine_masks(result_handler, handler1, handler2, H_adjusted, panorama_bounds)
        plt.figure(figsize=(12,8))
        plt.imshow(cv2.cvtColor(result_handler.img, cv2.COLOR_BGR2RGB))
        plt.title("Panorama final")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("debug_imgs/panorama_final.png")
        plt.close()
        return result_handler.img, result_handler    

    def transform_and_combine_masks(self, result_handler, handler1, handler2, H_adjusted, panorama_bounds):
        width = panorama_bounds['width']
        height = panorama_bounds['height']
        x_min = panorama_bounds['x_min']
        y_min = panorama_bounds['y_min']
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        axs[0, 0].imshow(handler1.mask, cmap='gray')
        axs[0, 0].set_title("handler1.mask")
        axs[0, 1].imshow(handler1.text_mask, cmap='gray')
        axs[0, 1].set_title("handler1.text_mask")
        axs[0, 2].imshow(handler1.binary, cmap='gray')
        axs[0, 2].set_title("handler1.binary")
        axs[1, 0].imshow(handler2.mask, cmap='gray')
        axs[1, 0].set_title("handler2.mask")
        axs[1, 1].imshow(handler2.text_mask, cmap='gray')
        axs[1, 1].set_title("handler2.text_mask")
        axs[1, 2].imshow(handler2.binary, cmap='gray')
        axs[1, 2].set_title("handler2.binary")
        for ax in axs.flat:
            ax.axis('off')
        plt.suptitle("Máscaras antes de combinar")
        plt.tight_layout()
        plt.savefig("debug_imgs/mascaras_antes_combinar.png")
        plt.close()
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        combined_text_mask = np.zeros((height, width), dtype=np.uint8)
        combined_binary = np.zeros((height, width), dtype=np.uint8)

        if handler1.mask is not None:
            mask1_transformed = cv2.warpPerspective(handler1.mask, H_adjusted, (width, height))
            combined_mask = mask1_transformed.copy()
        if handler1.text_mask is not None:
            text_mask1_transformed = cv2.warpPerspective(handler1.text_mask, H_adjusted, (width, height))
            combined_text_mask = text_mask1_transformed.copy()
        if handler1.binary is not None:
            binary1_transformed = cv2.warpPerspective(handler1.binary, H_adjusted, (width, height))
            combined_binary = binary1_transformed.copy()

        h2, w2 = handler2.img.shape[:2]
        y_start = max(0, -y_min)
        y_end = min(height, y_start + h2)
        x_start = max(0, -x_min)
        x_end = min(width, x_start + w2)

        img2_y_start = 0 if y_min <= 0 else -y_min
        img2_y_end = img2_y_start + (y_end - y_start)
        img2_x_start = 0 if x_min <= 0 else -x_min
        img2_x_end = img2_x_start + (x_end - x_start)

        if (img2_y_end <= h2 and img2_x_end <= w2 and
            img2_y_start >= 0 and img2_x_start >= 0 and
            y_start >= 0 and x_start >= 0 and
            y_end <= height and x_end <= width):

            combined_mask[y_start:y_end, x_start:x_end] = 0
            combined_text_mask[y_start:y_end, x_start:x_end] = 0
            combined_binary[y_start:y_end, x_start:x_end] = 0

            if handler2.mask is not None:
                mask2_region = handler2.mask[img2_y_start:img2_y_end, img2_x_start:img2_x_end]
                combined_mask[y_start:y_end, x_start:x_end] = mask2_region
            if handler2.text_mask is not None:
                text_mask2_region = handler2.text_mask[img2_y_start:img2_y_end, img2_x_start:img2_x_end]
                combined_text_mask[y_start:y_end, x_start:x_end] = text_mask2_region
            if handler2.binary is not None:
                binary2_region = handler2.binary[img2_y_start:img2_y_end, img2_x_start:img2_x_end]
                combined_binary[y_start:y_end, x_start:x_end] = binary2_region
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(combined_mask, cmap='gray')
        axs[0].set_title("Máscara combinada")
        axs[1].imshow(combined_text_mask, cmap='gray')
        axs[1].set_title("Máscara texto combinada")
        axs[2].imshow(combined_binary, cmap='gray')
        axs[2].set_title("Máscara binaria combinada")
        for ax in axs.flat:
            ax.axis('off')
        plt.suptitle("Máscaras después de combinar")
        plt.tight_layout()
        plt.savefig("debug_imgs/mascaras_despues_combinar.png")
        plt.close()
        result_handler.mask = combined_mask
        result_handler.text_mask = combined_text_mask
        result_handler.binary = combined_binary
        result_handler.extract_contours()
    
    def find_similar_contours(self, contours1, contours2):
        similar_pairs = []
        min_contour_area = 200
        mask_size = (100, 100)
        masks1 = []
        
        for i, cnt1 in enumerate(contours1):
          area1 = cv2.contourArea(cnt1)
          if area1 < min_contour_area:
              masks1.append(None)
              continue
              
          mask = np.zeros(mask_size, dtype=np.uint8)
          
          rect = cv2.boundingRect(cnt1)
          x, y, w, h = rect
          
          scale_x = mask_size[1] * 0.8 / max(w, 1)
          scale_y = mask_size[0] * 0.8 / max(h, 1)
          scale = min(scale_x, scale_y)
          
          centered_cnt = cnt1 - [x + w/2, y + h/2] + [mask_size[1]/2, mask_size[0]/2]
          scaled_cnt = (centered_cnt * scale).astype(np.int32)
          
          cv2.fillPoly(mask, [scaled_cnt], 255)
          masks1.append(mask)
      
        for j, cnt2 in enumerate(contours2):
            area2 = cv2.contourArea(cnt2)
            if area2 < min_contour_area:
                continue
                
            perimeter2 = cv2.arcLength(cnt2, True)
            x2, y2, w2, h2 = cv2.boundingRect(cnt2)
            
            mask2 = np.zeros(mask_size, dtype=np.uint8)
            
            rect2 = cv2.boundingRect(cnt2)
            x, y, w, h = rect2
            
            scale_x = mask_size[1] * 0.8 / max(w, 1)
            scale_y = mask_size[0] * 0.8 / max(h, 1)
            scale = min(scale_x, scale_y)
            
            centered_cnt2 = cnt2 - [x + w/2, y + h/2] + [mask_size[1]/2, mask_size[0]/2]
            scaled_cnt2 = (centered_cnt2 * scale).astype(np.int32)
            
            cv2.fillPoly(mask2, [scaled_cnt2], 255)
            
            for i, mask1 in enumerate(masks1):
                if mask1 is None:
                    continue
                    
                intersection = cv2.bitwise_and(mask1, mask2)
                union = cv2.bitwise_or(mask1, mask2)
                
                area_intersection = np.count_nonzero(intersection)
                area_union = np.count_nonzero(union)
                
                if area_union == 0:  
                    continue
                    
                iou = area_intersection / area_union
                
                cnt1 = contours1[i]
                area1 = cv2.contourArea(cnt1)
                perimeter1 = cv2.arcLength(cnt1, True)
                x1, y1, w1, h1 = cv2.boundingRect(cnt1)
                
                if iou > 0.6:  # Umbral de IoU ajustable
                    similarity_score = iou
                    
                    similar_pairs.append((
                        i, j, similarity_score,
                        area1, perimeter1, area2, perimeter2,
                        (x1, y1, w1, h1), (x2, y2, w2, h2)
                    ))
                
        
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return similar_pairs
