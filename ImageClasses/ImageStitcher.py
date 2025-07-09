
from datetime import datetime
from turtle import back
from matplotlib.pyplot import gray
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time

from ImageClasses.ImageToStitch import ImageToStitch


class ImageStitcher:
    def __init__(self):
        self.min_matches = 6
        self.text_threshold = 0.001
        self.processing_scale = 0.5

        if hasattr(cv2, 'Stitcher_create'):  # OpenCV 3.x API
            self.opencv_stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        else:  # OpenCV 4.x API
            self.opencv_stitcher = cv2.createStitcher(cv2.Stitcher_SCANS)

        # Store stitcher status codes for better error reporting
        self.status_codes = {
            cv2.Stitcher_OK: "Stitching completed successfully",
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Not enough images for stitching",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameter adjustment failed"
        }

        # Intentar usar GPU si está disponible
        try:
            if hasattr(self.opencv_stitcher, 'setTryUseGpu'):
                self.opencv_stitcher.setTryUseGpu(True)
                print("GPU acceleration enabled for stitching")
        except Exception as e:
            print(f"Could not enable GPU acceleration: {str(e)}")

    def stitch_images(self, images):
        """
        Stitch multiple images using OpenCV's stitcher.

        Args:
            images: List of input images

        Returns:
            Tuple containing (stitched_image, paper_contour, paper_mask)
        """
        if not images:
            print("E: No images to stitch.")
            return None

        if len(images) == 1:
            print("Only one image provided, returning original")
            paper_contour, paper_mask, img_without_background = self.get_paper_polygon_from_stitched_img(
                images[0])
            return images[0], paper_contour, paper_mask
        
        if len(images) == 8:
            print("Procesando 8 imágenes en dos grupos de 4...")
            panorama1 = self.stitch_images(images[:4])
            panorama2 = self.stitch_images(images[4:])
            if panorama1 is None or panorama2 is None:
                print("Error al hacer stitching de los grupos de 4 imágenes")
                return None
            # panorama1 y panorama2 pueden ser tuplas (panorama, contour, mask)
            # Solo nos interesa la imagen panorámica para el siguiente paso
            stitched_imgs = [panorama1[0], panorama2[0]]
            print("Uniendo los dos panoramas resultantes...")
            return self.stitch_images(stitched_imgs)

        print(f"Stitching {len(images)} images using OpenCV stitcher...")
        start_time = time.time()

        try:
            # Formato esperado por el stitcher (BGR)
            formatted_images = []
            for i, img in enumerate(images):
                if img is None:
                    print(f"Warning: Image {i} is None, skipping")
                    continue

                print(f"Processing image {i}: shape={img.shape}, dtype={img.dtype}")

                # Asegurar formato BGR
                if len(img.shape) == 2:  # Escala de grises
                    print(f"  Converting image {i} from grayscale to BGR")
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:  # BGRA
                    print(f"  Converting image {i} from BGRA to BGR")
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                elif img.shape[2] != 3:
                    print(f"  Warning: Image {i} has unusual number of channels: {img.shape[2]}")

                formatted_images.append(img)
                print(f"  Image {i} added to formatted_images, now have {len(formatted_images)} images")
            formatted_images.append(formatted_images[0])  # Añadir la primera imagen al final para cerrar el ciclo
            if len(formatted_images) < 2:
                print(
                    "Error: Se necesitan al menos 2 imágenes válidas para el stitching")
                print(f"Original images: {len(images)}, Formatted images: {len(formatted_images)}")
                return None

            # Aplicar stitching
            status, panorama = self.opencv_stitcher.stitch(formatted_images)

            # Reportar resultados
            elapsed_time = time.time() - start_time
            if status == cv2.Stitcher_OK:
                print(f"Stitching exitoso! ({elapsed_time:.2f} segundos)")
                # Guardar panorama para depuración
                panoramas_dir = "FlowImages/Panoramas"
                os.makedirs(panoramas_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(
                    panoramas_dir, f"panorama_{timestamp}.jpg")
                cv2.imwrite(output_path, panorama)
                print(f"Panorama guardado: {output_path}")

                # Obtener el polígono de papel y máscara
                paper_contour, paper_mask, img_without_background = self.get_paper_polygon_from_stitched_img(
                    panorama)
                return panorama, paper_contour, paper_mask
            else:
                status_message = self.status_codes.get(
                    status, f"Error desconocido (código: {status})")
                print(f"Fallo en stitching: {status_message}")
                return None

        except Exception as e:
            print(f"Error durante el stitching: {str(e)}")
            return None
    def save_debug_masks(self, img1_resized, img2_resized, combined_mask_1, combined_mask_2):
        """
        Save debug images of combined masks and visualizations to the debug_imgs folder.

        Args:
            img1_resized: First resized image
            img2_resized: Second resized image
            combined_mask_1: First combined mask
            combined_mask_2: Second combined mask
        """
        # Create debug directory if it doesn't exist
        debug_dir = "debug_imgs"
        os.makedirs(debug_dir, exist_ok=True)

        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save the raw masks
        mask1_path = os.path.join(debug_dir, f"{timestamp}_mask1.png")
        mask2_path = os.path.join(debug_dir, f"{timestamp}_mask2.png")
        cv2.imwrite(mask1_path, combined_mask_1)
        cv2.imwrite(mask2_path, combined_mask_2)

        # Create visualizations with masks overlaid on images
        vis1 = img1_resized.copy()
        vis2 = img2_resized.copy()

        # Convert to 3-channel if they have 4 channels
        if vis1.shape[2] == 4:
            vis1 = cv2.cvtColor(vis1, cv2.COLOR_BGRA2BGR)
        if vis2.shape[2] == 4:
            vis2 = cv2.cvtColor(vis2, cv2.COLOR_BGRA2BGR)

        # Create colored masks for visualization (blue overlay)
        mask_overlay1 = np.zeros_like(vis1)
        mask_overlay2 = np.zeros_like(vis2)

        # Set blue channel to 255 where mask is non-zero
        mask_overlay1[:, :, 0] = combined_mask_1  # Blue channel
        mask_overlay2[:, :, 0] = combined_mask_2  # Blue channel

        # Overlay masks with transparency
        alpha = 0.5
        vis1 = cv2.addWeighted(vis1, 1, mask_overlay1, alpha, 0)
        vis2 = cv2.addWeighted(vis2, 1, mask_overlay2, alpha, 0)

        # Save visualizations
        vis1_path = os.path.join(debug_dir, f"{timestamp}_vis1.jpg")
        vis2_path = os.path.join(debug_dir, f"{timestamp}_vis2.jpg")
        cv2.imwrite(vis1_path, vis1)
        cv2.imwrite(vis2_path, vis2)

        print(f"Debug images saved to {debug_dir}")

    def get_paper_polygon_from_stitched_img(self, stitched_img):
        
        # Verificar si tiene canal alpha
        has_alpha = len(stitched_img.shape) > 2 and stitched_img.shape[2] == 4
        
        # Convertir a HSV para segmentación por color
        hsv = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2HSV if not has_alpha else cv2.COLOR_BGRA2HSV)
        
        # Detectar áreas rojas (fondo)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Detectar áreas azules (fondo)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Combinar máscaras de fondo
        background_mask = cv2.bitwise_or(red_mask, blue_mask)
        
        # Aplicar máscara para oscurecer el fondo (NUEVA IMPLEMENTACIÓN)
        img_darkened = stitched_img.copy()
        
        if has_alpha:
            black_color = [0, 0, 0, 255]
        else:
            black_color = [0, 0, 0]
        
        background_pixels = np.where(background_mask > 0)
        if len(background_pixels[0]) > 0:  # Check if mask has any non-zero values
            img_darkened[background_pixels] = black_color
        
        # Convertir a escala de grises
        if has_alpha:
            gray = cv2.cvtColor(img_darkened[:,:,0:3], cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(img_darkened, cv2.COLOR_BGR2GRAY)
        
        # Binarizar con umbral 130
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        
        # Aplicar blur para suavizar
        blurred_binary = cv2.GaussianBlur(binary, (101, 101), 0)
        
        # Obtener máscara de papel
        _, paper_mask = cv2.threshold(blurred_binary, 80, 255, cv2.THRESH_BINARY)
        
        # Guardar imágenes para depuración
        debug_dir = "debug_imgs/masks"
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar imágenes del proceso
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_mask_red1.png"), mask_red1)
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_mask_red2.png"), mask_red2)
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_red_mask_combined.png"), red_mask)
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_blue_mask.png"), blue_mask)
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_background_mask.png"), background_mask)
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_darkened.jpg"), img_darkened)
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_gray.png"), gray)
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_binary.png"), binary)
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_blurred_binary.png"), blurred_binary)
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_paper_mask_new.png"), paper_mask)
        
        # También guardar visualizaciones coloreadas superpuestas sobre la imagen original
        vis_img = stitched_img.copy()
        if has_alpha:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGRA2BGR)
        
        # Crear visualización con máscara coloreada
        paper_overlay = np.zeros_like(vis_img)
        paper_overlay[:,:,:] = 255  # Blanco
        paper_pixels = np.where(paper_mask > 0)
        if len(paper_pixels[0]) > 0:
            paper_overlay[paper_pixels] = [0, 255, 0]  # Verde
        
        # Aplicar transparencia
        alpha = 0.5
        paper_vis = cv2.addWeighted(vis_img, 1, paper_overlay, alpha, 0)
        
        # Guardar visualización
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_paper_vis_new.jpg"), paper_vis)
        
        print(f"Máscaras guardadas en {debug_dir}")
        
        # Limpiar la máscara
        kernel = np.ones((15, 15), np.uint8)
        paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Guardar máscara procesada
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_paper_mask_processed.png"), paper_mask)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, paper_mask, stitched_img
        
        # Filtrar por área
        min_area = stitched_img.shape[0] * stitched_img.shape[1] * 0.05
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if not large_contours:
            return None, paper_mask, stitched_img
        
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
        
        # Guardar máscara refinada
        cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_refined_mask.png"), refined_mask)
        
        img_without_background = stitched_img.copy()
        background_color = [255, 255, 255, 0] if has_alpha else [255, 255, 255]
        background_pixels = np.where(refined_mask == 0)
        if len(background_pixels[0]) > 0:
            img_without_background[background_pixels] = background_color
        
        # Guardar imágenes de depuración
        debug_dir = "debug_imgs"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Guardar contorno visualizado sobre la imagen
        vis_img = stitched_img.copy()
        cv2.drawContours(vis_img, [approx_contour], 0, (0, 255, 0), 5)  # Contorno verde grueso
        vis_path = os.path.join(debug_dir, f"{timestamp}_paper_polygon.jpg")
        cv2.imwrite(vis_path, vis_img)
        
        # Guardar imagen sin fondo
        img_path = os.path.join(debug_dir, f"{timestamp}_paper_extracted.png")
        cv2.imwrite(img_path, img_without_background)
        
        print(f"Polígono de papel detectado con {len(approx_contour)} vértices")
        print(f"Imágenes de depuración guardadas en {debug_dir}")
        
        # Verificar y ajustar si no son exactamente 4 vértices
        if len(approx_contour) != 4:
            rect = cv2.minAreaRect(paper_contour)
            box = cv2.boxPoints(rect)
            approx_contour = np.int0(box)
        
        return approx_contour, refined_mask, img_without_background
        
    def visualize_result(self, original_images, panorama=None):
        """
        Visualiza las imágenes originales y el resultado del stitching
        
        Args:
            original_images: Lista de imágenes originales
            panorama: Imagen resultante del stitching (None si falló)
        """
        # First, show the original images in a row
        n_images = len(original_images)
        if n_images == 0:
            return
            
        plt.figure(figsize=(15, 10))
        
        # Show original images in the top row
        for i, img in enumerate(original_images):
            plt.subplot(2, n_images, i+1)
            if img.shape[2] == 4:  # BGRA
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(rgb_img)
            plt.title(f"Imagen {i+1}")
            plt.axis('off')
        
        # Show panorama in the bottom row (spanning all columns)
        if panorama is not None:
            plt.subplot(2, 1, 2)
            if panorama.shape[2] == 4:  # BGRA
                rgb_panorama = cv2.cvtColor(panorama, cv2.COLOR_BGRA2RGB)
            else:
                rgb_panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
            plt.imshow(rgb_panorama)
            plt.title("Resultado del Stitching")
            plt.axis('off')
        else:
            plt.subplot(2, 1, 2)
            plt.text(0.5, 0.5, "Stitching Fallido", 
                    ha='center', va='center', fontsize=20, color='red')
            plt.axis('off')
            
        plt.tight_layout()
        
        # Guardar la visualización
        debug_dir = "debug_imgs"
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(debug_dir, f"{timestamp}_stitching_visualization.png"))
        plt.close() 