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
    def __init__(self, camera_box, images, parameters):
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

    def create_image(self):
        """
        Creates a composite panorama from multiple images using their corner positions.
        Places each image in its correct position in the reference system.

        Raises:
            ValueError: If no images are provided
        """
        if not self.images:
            raise ValueError("No images provided for composite image creation")
        panorama = np.zeros((self.height_px, self.width_px), dtype=np.uint8)
        current_panorama = panorama.copy()
        for idx, img in enumerate(self.images):
            if img.image is None:
                img.process()
            y_start = img.corners_positions_px['top_left'][1]
            x_start = img.corners_positions_px['top_left'][0]
            y_end = y_start + img.height_px
            x_end = x_start + img.width_px
            print(
                f"Placing image {idx + 1} at position: ({x_start}, {y_start}) to ({x_end}, {y_end})")
            current_panorama[y_start:y_end, x_start:x_end] = img.image[:img.height_px, :img.width_px]

        print("Composite image created.")
        self.image = current_panorama
        
        self.visualize_recomposition_process()

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
