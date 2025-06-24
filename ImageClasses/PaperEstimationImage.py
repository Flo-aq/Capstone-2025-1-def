import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from os.path import join
from matplotlib.patches import Rectangle, Polygon
import os
import time
import cv2

from Functions.SecondModule.FirstCaseFunctions import create_lines_from_extremes, get_vertical_and_horizontal_lines, reconstruct_bottom_polygon, reconstruct_top_polygon
from Functions.SecondModule.FourthCaseFunctions import find_polygon_from_two_unclean_intersections
from Functions.SecondModule.SecondCaseFunctions import reconstruct_polygon_from_paralell_lines
from Functions.SecondModule.SecondModuleFunctions import calculate_photo_positions_diagonal, extend_all_lines_and_find_corners, find_polygon_from_intersections, group_lines_by_angle, standardize_polygon
from Functions.SecondModule.ThirdCaseFunctions import find_polygon_from_two_clean_intersections
from ImageClasses.Image import Image

class PaperEstimationImage(Image):
    """
    A specialized Image class that handles the composition and processing of two images
    to detect polygons and calculate optimal camera positions for capture.
    Inherits from the base Image class.
    """
    def __init__(self, camera_box, top_image, bottom_image, parameters=None):
        """
        Initialize ImageFunction2 with required images and parameters.

        Args:
            camera (Camera): Camera object for image acquisition
            top_image (Image): First image to process (top position)
            bottom_image (Image): Second image to process (bottom position)
            parameters (dict): Processing parameters for the second module

        Raises:
            ValueError: If parameters or any image is None
        """
        super().__init__(function=2, image=None, camera_box=camera_box)
        if parameters is None:
            raise ValueError("Parameters cannot be None")
        if top_image is None:
            raise ValueError("Top image cannot be None")
        if bottom_image is None:
            raise ValueError("Bottom image cannot be None")
        self.parameters = parameters["second_module"]
        self.polygon = None
        self.top_image = top_image
        self.top_image.image = cv2.rotate(self.top_image.image, cv2.ROTATE_180)
        self.bottom_image = bottom_image
        self.bottom_image.image = cv2.rotate(self.bottom_image.image, cv2.ROTATE_180)
        self.case = 0
        self.grouped_lines = None
        self.unique_corners = None
    

    def create_image(self):
        """
        Create a composite image by combining top and bottom images.
        Places images at opposite corners of a white canvas.
        """
        print("Creating composite image...")
        
        if len(self.top_image.image.shape) > 2 and self.top_image.image.shape[2] == 4:
            print("Convirtiendo imagen superior de 4 canales a 3 canales...")
            self.top_image.image = cv2.cvtColor(self.top_image.image, cv2.COLOR_BGRA2BGR)
            
        if len(self.bottom_image.image.shape) > 2 and self.bottom_image.image.shape[2] == 4:
            print("Convirtiendo imagen inferior de 4 canales a 3 canales...")
            self.bottom_image.image = cv2.cvtColor(self.bottom_image.image, cv2.COLOR_BGRA2BGR)
            
        composite = np.ones((self.height_px, self.width_px, 3), dtype=np.uint8) * 255
        
        h_top, w_top = self.top_image.image.shape[:2]
        h_bottom, w_bottom = self.bottom_image.image.shape[:2]
        
        composite[0:h_top, 0:w_top] = self.top_image.image
    
        bottom_y = self.height_px - h_bottom
        bottom_x = self.width_px - w_bottom
        
        composite[bottom_y:self.height_px, bottom_x:self.width_px] = self.bottom_image.image

        self.image = composite
    
    def get_lines(self):
        """
        Detect and process lines from both top and bottom images.
        Adjusts bottom image line positions based on offset and determines
        processing case based on line detection results.
        """
        h_bottom = self.bottom_image.height_px
        w_bottom = self.bottom_image.width_px
        bottom_offset = [
            self.width_px - w_bottom,
            self.width_px - h_bottom
        ]

        self.top_image.get_lines()
        self.bottom_image.get_lines()
        
        original_bottom_lines = self.bottom_image.lines.copy()

        self.bottom_image.lines = [
            (
                [p1[0] + bottom_offset[0], p1[1] + bottom_offset[1]],
                [p2[0] + bottom_offset[0], p2[1] + bottom_offset[1]]
            )
            for p1, p2 in original_bottom_lines
        ]
        
        if len(self.top_image.lines) == 0 and (len(original_bottom_lines) + len(self.top_image.lines) != 0):
            self.lines = self.bottom_image.lines
            grouped_lines = group_lines_by_angle(self.lines)
            self.grouped_lines = grouped_lines
            self.case = 1
            
        elif len(original_bottom_lines) == 0 and (len(original_bottom_lines) + len(self.top_image.lines) != 0):
            self.lines = self.top_image.lines
            grouped_lines = group_lines_by_angle(self.lines)
            self.grouped_lines = grouped_lines
            self.case = 1
            
        elif len(self.top_image.lines) != 0 and len(original_bottom_lines) != 0:
            self.lines = self.top_image.lines + self.bottom_image.lines 
            grouped_lines = group_lines_by_angle(self.lines)
            self.grouped_lines = grouped_lines
            if len(grouped_lines.keys()) == 1:
                self.case = 2
            else:
                self.bottom_image.lines_by_angles = group_lines_by_angle(self.bottom_image.lines)
                self.top_image.lines_by_angles = group_lines_by_angle(self.top_image.lines)
                self.unique_corners = extend_all_lines_and_find_corners(self.image, self.lines)
                if len(self.unique_corners) == 2:
                    top_angles = sorted(self.top_image.lines_by_angles.keys())
                    bottom_angles = sorted(self.bottom_image.lines_by_angles.keys())
                    if len(top_angles) == 1 or len(bottom_angles) == 1:
                        self.case = 4
                    else:
                        self.case = 3
        
    def get_polygon(self, width_mm, height_mm):
        """
        Generate polygon representation based on detected lines and case.
        
        Args:
            width_mm (float): Paper width in millimeters
            height_mm (float): Paper height in millimeters
        
        Returns:
            numpy.ndarray: Standardized polygon vertices ordered clockwise
        """
        self.get_lines()
        if self.case == 0:
            polygon = find_polygon_from_intersections(self.unique_corners) if len(self.unique_corners) >= 3 else []
            return standardize_polygon(polygon)
        
        width_px = width_mm / self.camera.mm_per_px_h
        height_px = height_mm / self.camera.mm_per_px_v
        
        if self.case == 1:
            vertical_lines, horizontal_lines = get_vertical_and_horizontal_lines(self.grouped_lines)
            if len(vertical_lines) == 0 or len(horizontal_lines) == 0:
                raise ValueError("No vertical nor horizontal lines found")

            vertical_line, horizontal_line = create_lines_from_extremes(
            vertical_lines, horizontal_lines)
            
            if self.top_image.has_paper:
                polygon =  reconstruct_top_polygon(
                    width_px, height_px,
                    vertical_line, horizontal_line)
            else:
                polygon = reconstruct_bottom_polygon(
                width_px, height_px,
                vertical_line, horizontal_line)
                
            return standardize_polygon(polygon)
        elif self.case == 2:
            polygon = reconstruct_polygon_from_paralell_lines(self.top_image.lines, self.bottom_image.lines, height_px)
            return standardize_polygon(polygon)
        
        elif self.case == 3:
            polygon = find_polygon_from_two_clean_intersections(self.unique_corners, self.top_image.lines_by_angles, self.bottom_image.lines_by_angles, width_px, height_px)
            return standardize_polygon(polygon)

        elif self.case == 4:
            polygon = find_polygon_from_two_unclean_intersections(self.top_image.lines_by_angles, self.bottom_image.lines_by_angles, self.unique_corners, width_px, height_px)
            return standardize_polygon(polygon)

        return []
    
    def process(self, width_mm, height_mm):
        """
        Process images to create composite and detect polygon.
        Measures and reports processing time for each step.

        Args:
            width_mm (float): Paper width in millimeters
            height_mm (float): Paper height in millimeters

        Returns:
            None
        """
        start_time = time.time()
        self.create_image()
        self.save_composite_visualization()
        create_time = time.time() - start_time
        start_time = time.time()
        self.polygon = self.get_polygon(width_mm, height_mm)
        self.save_polygon_visualization()
        polygon_time = time.time() - start_time
        print(f"Tiempo de creación de imagen compuesta: {create_time:.3f} segundos")
        print(f"Tiempo de detección de polígono: {polygon_time:.3f} segundos")
        print(f"Tiempo total de procesamiento: {create_time + polygon_time:.3f} segundos")
        print(f"Caso detectado: {self.case}")
        print("--------------------\n")
        print("Polygon of paper detected")
        
        self.visualize_coverage_strategies()
    
    def calculate_capture_photos_positions(self):
        """
        Calculate optimal camera positions for capturing the detected polygon.
        Tests two diagonal strategies and returns the better one based on
        coverage and number of positions needed.

        Args:
            None

        Returns:
            list: List of (x,y) tuples representing optimal camera positions,
                empty list if no polygon was detected
        """
        if len(self.polygon) == 0:
            return []
            
        fov_width = self.camera.fov_h_px
        fov_height = self.camera.fov_v_px
        margin_px = int(self.parameters["camera_positioning_margin"] / self.camera.mm_per_px_h)  # 5mm margin
        print("Trying first diagonal strategy to get camera positions")
        positions1, coverage1 = calculate_photo_positions_diagonal(
            self.polygon, fov_width, fov_height, margin_px, "topleft-bottomright")
        print("Trying second diagonal strategy to get camera positions")
        positions2, coverage2 = calculate_photo_positions_diagonal(
            self.polygon, fov_width, fov_height, margin_px, "topright-bottomleft")
        print("Deciding which diagonal strategy to use")
        if len(positions1) < len(positions2) or (len(positions1) == len(positions2) and coverage1 > coverage2):
            print("Using first diagonal strategy")
            positions1 = self.convert_positions_to_mm(positions1)
            return positions1
        else:
            print("Using second diagonal strategy")
            positions2 = self.convert_positions_to_mm(positions2)
            return positions2
    
    def convert_positions_to_mm(self, positions):
        """
        Convert camera positions from pixel coordinates to millimeter coordinates.

        Args:
            positions (list): List of (x, y) tuples in pixel coordinates

        Returns:
            list: List of (x_mm, y_mm) tuples in millimeter coordinates
        """
        mm_positions = []
        for x, y in positions:
            
            # Convert to millimeters
            x_mm = x * self.camera.mm_per_px_h
            y_mm = y * self.camera.mm_per_px_v
            
            mm_positions.append((x_mm, y_mm))
        return mm_positions
    
    def save_composite_visualization(self):
        # Crear directorio si no existe
        save_dir = "FlowImages/PaperEstimationImages"
        os.makedirs(save_dir, exist_ok=True)
        
        # Crear timestamp para el nombre de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = join(save_dir, f"CompositeImage_{timestamp}.png")
        
        # Configurar estilo de visualización
        plt.style.use('dark_background')
        plt.figure(figsize=(16, 12))
        
        # Crear la figura con subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('#121212')
        
        # Mostrar imagen superior
        axs[0, 0].imshow(cv2.cvtColor(self.top_image.image, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title("Top Left Image", fontsize=14)
        axs[0, 0].axis('off')
        
        # Mostrar imagen inferior
        axs[0, 1].imshow(cv2.cvtColor(self.bottom_image.image, cv2.COLOR_BGR2RGB))
        axs[0, 1].set_title("Bottom Right Image", fontsize=14)
        axs[0, 1].axis('off')
        
        # Mostrar canvas blanco con posiciones marcadas
        composite_viz = np.ones((self.height_px, self.width_px, 3), dtype=np.uint8) * 255
        
        h_top, w_top = self.top_image.image.shape[:2]
        h_bottom, w_bottom = self.bottom_image.image.shape[:2]
        
        # Dibujar rectángulos para mostrar dónde se ubicarán las imágenes
        top_rect = cv2.rectangle(composite_viz.copy(), (0, 0), (w_top, h_top), (210, 105, 30), 2)
        bottom_rect = cv2.rectangle(top_rect, 
                                  (self.width_px - w_bottom, self.height_px - h_bottom), 
                                  (self.width_px, self.height_px), 
                                  (30, 144, 255), 10)
        
        axs[1, 0].imshow(cv2.cvtColor(bottom_rect, cv2.COLOR_BGR2RGB))
        axs[1, 0].set_title("Images Placement", fontsize=14)
        axs[1, 0].axis('off')
        
        # Mostrar la imagen compuesta final
        axs[1, 1].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        axs[1, 1].set_title("Final Composite Image", fontsize=14)
        axs[1, 1].axis('off')
        
        # Guardar la figura
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(filename, facecolor=fig.get_facecolor(), dpi=150)
        plt.close()

    def save_polygon_visualization(self):
        if len(self.polygon) == 0:
            print("No polygon detected to visualize")
            return
        
        # Crear directorio si no existe
        save_dir = "FlowImages/PaperEstimationImages"
        os.makedirs(save_dir, exist_ok=True)
        
        # Crear timestamp para el nombre de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = join(save_dir, f"PolygonDetection_{timestamp}.png")
        
        # Configurar estilo de visualización
        plt.style.use('dark_background')
        
        # Crear la figura
        fig, ax = plt.subplots(figsize=(16, 12))
        fig.patch.set_facecolor('#121212')
        
        # Mostrar la imagen compuesta
        ax.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        
        # Dibujar el polígono
        polygon_points = self.polygon.reshape(-1, 2)
        ax.add_patch(Polygon(polygon_points, fill=False, edgecolor='#f96927', linewidth=10))
        
        # Dibujar los vértices
        ax.scatter(polygon_points[:, 0], polygon_points[:, 1], color='#eb443b', s=100)
        
        # Etiquetar los vértices
        for i, (x, y) in enumerate(polygon_points):
            ax.annotate(f"P{i}", (x, y), xytext=(5, 5), textcoords='offset points', fontsize=12, weight='bold')
        
        # Añadir título e información
        ax.set_title(f"Polygon Detection - Case: {self.case}", fontsize=18)
        # Desactivar los ejes
        ax.axis('off')
        
        # Guardar la figura
        plt.tight_layout()
        plt.savefig(filename, facecolor=fig.get_facecolor(), dpi=150)
        plt.close()

    def visualize_coverage_strategies(self):
        """
        Visualiza las dos estrategias de cobertura y las guarda en un archivo.
        """
        if len(self.polygon) == 0:
            print("No polygon detected to visualize coverage")
            return
            
        # Crear directorio si no existe
        save_dir = "FlowImages/PaperEstimationImages"
        os.makedirs(save_dir, exist_ok=True)
        
        # Crear timestamp para el nombre de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = join(save_dir, f"CoverageStrategies_{timestamp}.png")
        
        # Parámetros para calcular posiciones (copiar de calculate_capture_photos_positions)
        fov_width = self.camera.fov_h_px
        fov_height = self.camera.fov_v_px
        margin_px = int(self.parameters["camera_positioning_margin"] / self.camera.mm_per_px_h)
        
        # Calcular posiciones para ambas estrategias
        positions1, coverage1 = calculate_photo_positions_diagonal(
            self.polygon, fov_width, fov_height, margin_px, "topleft-bottomright")
        positions2, coverage2 = calculate_photo_positions_diagonal(
            self.polygon, fov_width, fov_height, margin_px, "topright-bottomleft")
        
        # Configurar la figura
        plt.style.use('dark_background')
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        fig.patch.set_facecolor('#121212')
        
        # Preparar datos comunes
        polygon_points = self.polygon.reshape(-1, 2)
        min_x = np.min(polygon_points[:, 0])
        max_x = np.max(polygon_points[:, 0])
        min_y = np.min(polygon_points[:, 1])
        max_y = np.max(polygon_points[:, 1])
        width = max_x - min_x
        height = max_y - min_y
        
        # Función para crear el gráfico de cobertura
        def create_coverage_plot(ax, positions, coverage, strategy_name):
            # Crear máscara para el polígono
            mask = np.zeros((int(height)+1, int(width)+1))
            polygon_for_mask = polygon_points - [min_x, min_y]
            cv2.fillPoly(mask, [polygon_for_mask.astype(np.int32)], 1)
            
            # Crear máscara para las áreas cubiertas
            covered_mask = np.zeros_like(mask)
            for x, y in positions:
                x_norm, y_norm = x - min_x, y - min_y
                x1 = max(0, int(x_norm - fov_width/2))
                y1 = max(0, int(y_norm - fov_height/2))
                x2 = min(mask.shape[1], int(x_norm + fov_width/2))
                y2 = min(mask.shape[0], int(y_norm + fov_height/2))
                covered_mask[y1:y2, x1:x2] = 1
            
            # Mostrar áreas cubiertas y no cubiertas
            uncovered = np.logical_and(mask, np.logical_not(covered_mask))
            covered = np.logical_and(mask, covered_mask)
            
            # Áreas no cubiertas (en rojo)
            ax.imshow(uncovered, origin='lower', extent=[min_x, max_x, min_y, max_y],
                    cmap=plt.cm.colors.ListedColormap(['white', '#FF0000']),
                    alpha=0.3, vmin=0, vmax=1)
            
            # Áreas cubiertas (en verde)
            ax.imshow(covered, origin='lower', extent=[min_x, max_x, min_y, max_y],
                    cmap=plt.cm.colors.ListedColormap(['white', '#00FF00']),
                    alpha=0.3, vmin=0, vmax=1)
            
            # Dibujar el polígono
            ax.add_patch(Polygon(polygon_points, fill=False, edgecolor='#43005c', linewidth=10))
            
            # Dibujar rectángulos para cada posición de la cámara
            for i, (x, y) in enumerate(positions):
                rect = Rectangle((x - fov_width/2, y - fov_height/2),
                              fov_width, fov_height,
                              fill=False, edgecolor='#95005d', linestyle='--', linewidth=10, alpha=0.7)
                ax.add_patch(rect)
                ax.scatter(x, y, s=100, color=f'#eb443b', zorder=5)
            
            # Configurar el aspecto del gráfico
            ax.set_xlim(min_x - 100, max_x + 100)
            ax.set_ylim(min_y - 100, max_y + 100)
            ax.grid(True, color='gray', alpha=0.3)
            ax.set_title(f"{strategy_name}\n{len(positions)} positions, {coverage:.1f}% coverage", fontsize=16)
        
        # Crear los dos gráficos
        create_coverage_plot(axs[0], positions1, coverage1 * 100, "Strategy 1: Top-Left to Bottom-Right")
        create_coverage_plot(axs[1], positions2, coverage2 * 100, "Strategy 2: Top-Right to Bottom-Left")

        # Guardar la figura
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(filename, facecolor=fig.get_facecolor(), dpi=150)
        plt.close()