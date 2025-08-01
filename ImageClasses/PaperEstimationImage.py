import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from os.path import join
from matplotlib.patches import Rectangle, Polygon
import os
import time
import cv2
from typing import List, Tuple, Optional

from Functions.SecondModule.FirstCaseFunctions import create_lines_from_extremes, get_vertical_and_horizontal_lines, reconstruct_bottom_polygon, reconstruct_top_polygon
from Functions.SecondModule.FourthCaseFunctions import find_polygon_from_two_unclean_intersections
from Functions.SecondModule.PhotoPositionsFunctions import calculate_photo_positions_with_tree
from Functions.SecondModule.SecondCaseFunctions import reconstruct_polygon_from_paralell_lines
from Functions.SecondModule.SecondModuleFunctions import calculate_photo_positions_diagonal, calculate_photo_positions_diagonal_with_overlap, extend_all_lines_and_find_corners, find_polygon_from_intersections, group_lines_by_angle, standardize_polygon
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
        self.top_image.image = cv2.rotate(self.top_image.image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.bottom_image = bottom_image
        self.bottom_image.image = cv2.rotate(self.bottom_image.image, cv2.ROTATE_90_COUNTERCLOCKWISE)
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
        print(self.height_px,)
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
            self.height_px - h_bottom
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
            debug_dir = "FlowImages/PaperEstimationImages"
            os.makedirs(debug_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_filename = join(debug_dir, f"debug_case0_{timestamp}.txt")
            with open(debug_filename, "w") as f:
                f.write("unique_corners:\n")
                f.write(str(self.unique_corners) + "\n\n")
                f.write("grouped_lines:\n")
                f.write(str(self.grouped_lines) + "\n\n")
                f.write("bottom_image.lines:\n")
                f.write(str(self.bottom_image.lines) + "\n\n")
                f.write("top_image.lines:\n")
                f.write(str(self.top_image.lines) + "\n\n")
                # Si tienes original_bottom_lines disponible, guárdalo también
                if hasattr(self.bottom_image, "original_lines"):
                    f.write("original_bottom_lines:\n")
                    f.write(str(self.bottom_image.original_lines) + "\n\n")

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
            debug_dir = "FlowImages/PaperEstimationImages"
            os.makedirs(debug_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_filename = join(debug_dir, f"debug_case3_{timestamp}.txt")
            with open(debug_filename, "w") as f:
                f.write("unique_corners:\n")
                f.write(str(self.unique_corners) + "\n\n")
                f.write("grouped_lines:\n")
                f.write(str(self.grouped_lines) + "\n\n")
                f.write("bottom_image.lines:\n")
                f.write(str(self.bottom_image.lines) + "\n\n")
                f.write("top_image.lines:\n")
                f.write(str(self.top_image.lines) + "\n\n")
                # Si tienes original_bottom_lines disponible, guárdalo también
                if hasattr(self.bottom_image, "original_lines"):
                    f.write("original_bottom_lines:\n")
                    f.write(str(self.bottom_image.original_lines) + "\n\n")

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
        print(f"Case detected: {self.case}")
        self.save_polygon_visualization()
        polygon_time = time.time() - start_time
        print(f"Tiempo de creación de imagen compuesta: {create_time:.3f} segundos")
        print(f"Tiempo de detección de polígono: {polygon_time:.3f} segundos")
        print(f"Tiempo total de procesamiento: {create_time + polygon_time:.3f} segundos")
        print(f"Caso detectado: {self.case}")
        print("--------------------\n")
        print("Polygon of paper detected")
        
        
    
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
          
#         positions, coverage = calculate_photo_positions_diagonal_with_overlap(
#         polygon, fov_width, fov_height, mm_to_px_h, mm_to_px_v, 
#         corners="topleft-bottomright", border_mm=30, min_overlap=0.3
#         )
            
        fov_width = self.camera.fov_v_px
        fov_height = self.camera.fov_h_px
        mm_to_px_h = self.camera.mm_per_px_v
        mm_to_px_v = self.camera.mm_per_px_h
        print("Calculating camera positions for polygon coverage")
        print(self.polygon)
        print(f"FOV width in pixels: {fov_width}, FOV height in pixels: {fov_height}")
        print(f"mm to px horizontal: {mm_to_px_h}, mm to px vertical: {mm_to_px_v}")
        print(f"Image height in pixels: {self.height_px}, Image width in pixels: {self.width_px}")
        coverage_positions = calculate_photo_positions_with_tree(self.polygon, fov_width, fov_height, self.height_px, self.width_px, 0.1, 3, 6)
        if len(coverage_positions) == 0:
            print("No camera positions found for polygon coverage")
            return []
        else:
          self.visualize_coverage_strategies(coverage_positions)
          return self.convert_positions_to_mm(coverage_positions[0])
        # print("Trying first diagonal strategy to get camera positions")
        # positions1, coverage1 = calculate_photo_positions_diagonal_with_overlap(
        #     self.polygon, fov_width, fov_height, mm_to_px_h, mm_to_px_v, self.height_px, self.width_px,
        #     corners="topleft-bottomright")
        # print("Trying second diagonal strategy to get camera positions")
        # positions2, coverage2 = calculate_photo_positions_diagonal_with_overlap(
        #     self.polygon, fov_width, fov_height, mm_to_px_h, mm_to_px_v, self.height_px, self.width_px,
        #     corners="topright-bottomleft")
        # margin_px = int(self.parameters["camera_positioning_margin"] / self.camera.mm_per_px_h)  # 5mm margin
        # print("Trying first diagonal strategy to get camera positions")
        # positions1, coverage1 = calculate_photo_positions_diagonal(
        #     self.polygon, fov_width, fov_height, margin_px, "topleft-bottomright")
        # print("Trying second diagonal strategy to get camera positions")
        # positions2, coverage2 = calculate_photo_positions_diagonal(
        #     self.polygon, fov_width, fov_height, margin_px, "topright-bottomleft")
        # print("Deciding which diagonal strategy to use")
        # if len(positions1) < len(positions2) or (len(positions1) == len(positions2) and coverage1 > coverage2):
        #     print("Using first diagonal strategy")
        #     positions1 = self.convert_positions_to_mm(positions1)
        #     return positions1, 0
        # else:
        #     print("Using second diagonal strategy")
        #     positions2 = self.convert_positions_to_mm(positions2)
        #     return positions2, 1
    
    def convert_positions_to_mm(self, positions):
        """
        Convert camera positions from pixel coordinates to millimeter coordinates.

        Args:
            positions (list): List of (x, y) tuples in pixel coordinates

        Returns:
            list: List of (x_mm, y_mm) tuples in millimeter coordinates
        """
        mm_positions = []
        x_offset = self.camera.fov_v_mm / 2
        y_offset = self.camera.fov_h_mm / 2
        print(positions)
        for x, y in positions:
            
            # Convert to millimeters
            x_mm = x * self.camera.mm_per_px_v - x_offset
            y_mm = y * self.camera.mm_per_px_h - y_offset
            
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
        
        composite_img_filename = join(save_dir, f"CompositeOnly_{timestamp}.png")
        cv2.imwrite(composite_img_filename, self.image)

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

    

    def visualize_coverage_strategies(self, positions):
        """
        Visualiza la cobertura acumulada paso a paso, mostrando en cada subplot
        el polígono y la suma de las áreas cubiertas por las imágenes capturadas hasta ese punto.
        """
        if len(self.polygon) == 0:
            print("No polygon detected to visualize coverage")
            return

        # Crear directorio si no existe
        save_dir = "FlowImages/PaperEstimationImages"
        os.makedirs(save_dir, exist_ok=True)

        # Crear timestamp para el nombre de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = join(save_dir, f"CoverageStepByStep_{timestamp}.png")

        fov_width = self.camera.fov_v_px
        fov_height = self.camera.fov_h_px
        
        
        # Convertir polígono al formato esperado
        polygon_points = self.polygon.reshape(-1, 2).tolist()
        composite_img_size = (self.width_px, self.height_px)
        fov_size = (fov_width, fov_height)
        
        # Llamar a la nueva implementación
        self.visualize_capture_plan_progression(
            polygon_points,
            positions[0],
            composite_img_size,
            fov_size,
            filename
        )

    def visualize_capture_plan_progression(
        self,
        polygon_coords: List[Tuple[float, float]],
        capture_positions: List[Tuple[float, float]],
        composite_img_size: Tuple[int, int],
        fov_size: Tuple[int, int],
        save_path: Optional[str] = None
    ):
        """
        Visualiza la progresión del plan de captura mostrando cómo se van agregando las posiciones.
        """
        num_positions = len(capture_positions)
        
        # Calcular número de filas y columnas para subplots
        cols = min(3, num_positions)
        rows = (num_positions + cols - 1) // cols
        
        plt.style.use('dark_background')
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        fig.patch.set_facecolor('#121212')
        
        # Si solo hay una fila o columna, asegurar que axes sea manejable
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        colors = ['#eb443b', '#43005c', '#95005d', '#f96927', '#00ccff', 
                  '#ffcc00', '#00ff99', '#ff66cc', '#9966ff', '#66ccff']
        
        for subplot_idx in range(num_positions):
            row = subplot_idx // cols
            col = subplot_idx % cols
            ax = axes[row, col]
            
            # Configurar fondo y límites
            ax.set_facecolor('#1a1a1a')
            
            # Dibujar polígono
            polygon_patch = Polygon(polygon_coords, closed=True, 
                                      linewidth=3, edgecolor='#43005c', facecolor='#43005c', alpha=0.3)
            ax.add_patch(polygon_patch)
            
            # Dibujar campos de visión hasta la posición actual
            for i in range(subplot_idx + 1):
                pos = capture_positions[i]
                x, y = pos
                w, h = fov_size
                
                # Dibujar rectángulo de campo de visión
                fov_rect = Rectangle((x - w//2, y - h//2), w, h,
                                  linewidth=2, edgecolor=colors[i % len(colors)], 
                                  facecolor=colors[i % len(colors)], alpha=0.2)
                ax.add_patch(fov_rect)
                
                # Marcar centro
                ax.plot(x, y, 'o', color=colors[i % len(colors)], markersize=8)
                ax.text(x, y, f'{i+1}', ha='center', va='center', fontweight='bold', color='white')
            
            # Configurar límites y aspecto
            min_x = min(p[0] for p in polygon_coords)
            max_x = max(p[0] for p in polygon_coords)
            min_y = min(p[1] for p in polygon_coords)
            max_y = max(p[1] for p in polygon_coords)
            
            padding = 100
            ax.set_xlim(min_x - padding, max_x + padding)
            ax.set_ylim(min_y - padding, max_y + padding)
            ax.set_aspect('equal')
            ax.invert_yaxis()  # Para que coincida con coordenadas de imagen
            ax.set_title(f'Paso {subplot_idx + 1}: {subplot_idx + 1} imagen{"es" if subplot_idx > 0 else ""}', 
                        fontsize=14, color='white')
            ax.grid(True, alpha=0.3, color='gray')
        
        # Ocultar subplots vacíos si los hay
        total_subplots = rows * cols
        for subplot_idx in range(num_positions, total_subplots):
            row = subplot_idx // cols
            col = subplot_idx % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, facecolor=fig.get_facecolor(), dpi=150)
        
        plt.close()