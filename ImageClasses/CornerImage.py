import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from os.path import join

from ImageClasses.Image import Image
from Functions.FirstModuleFunctions import (
    first_module_process_image,
    add_border,
    detect_edges,
    process_edges_and_remove_border,
    get_largest_edge,
    approximate_polygon,
    is_border_line
)

COLOR_PALETTE = ['#43005c', '#6e0060', '#95005d', '#b80056', '#d51e4b', '#eb443b', '#f96927', '#ff8d00']


class CornerImage(Image):
    """
    A specialized Image class for processing images using the first module's functionality.
    Implements polygon detection and line extraction from images.
    Inherits from the base Image class.
    """

    def __init__(self, image, camera_box, parameters=None):
        """
        Initialize ImageFunction1 with image processing parameters.

        Args:
            image (array): Input image to process
            camera (Camera): Camera object for image acquisition
            parameters (dict): Dictionary containing processing parameters for the first module
        """
        super().__init__(function=1, image=image, camera_box=camera_box)
        if parameters is None:
            raise ValueError("Parameters cannot be None")
        self.parameters = parameters["first_module"]
        self.polygon = None
        self.has_paper = False
        self.lines = []
        self.lines_by_angles = {}

    def process(self):
        """
        Process the image through multiple steps to detect polygons.

        Processing steps:
        1. Convert image to binary using kernel-based processing
        2. Add border to the binary image
        3. Detect edges using Canny edge detection
        4. Process edges and remove border
        5. Find largest edge contour
        6. Approximate polygon from largest edge
        """
        
        binary = first_module_process_image(
            self.image, self.parameters["kernel_size"], self.parameters["sigma"])
        binary_with_border = add_border(binary, self.parameters["border_size"])
        print("Detecting edges of image...")
        edges = detect_edges(
            binary_with_border, self.parameters["low_threshold"], self.parameters["high_threshold"])
        _, edges_final = process_edges_and_remove_border(
            binary_with_border, edges, self.parameters["border_size"])
        largest_edge = get_largest_edge(edges_final)
        
        image_with_largest_edge = self.image.copy()
        image_with_largest_edge = cv2.rotate(image_with_largest_edge, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        image_with_polygon = self.image.copy()
        image_with_polygon = cv2.rotate(image_with_polygon, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        if largest_edge is not None:
            print("Polygon detected")
            
            hex_color = COLOR_PALETTE[4]
            rgb_color = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])  # Convertir a BGR para OpenCV
            
            cv2.drawContours(image_with_largest_edge, [largest_edge], 0, bgr_color, 10)


            self.polygon = approximate_polygon(largest_edge)
            self.has_paper = True
            
            hex_color = COLOR_PALETTE[6]
            rgb_color = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
            
            polygon_array = np.array(self.polygon)
            cv2.drawContours(image_with_polygon, [polygon_array], 0, bgr_color, 10)
        else:
            self.polygon = []

        self.save_visualization(binary, binary_with_border, edges_final, 
                               image_with_largest_edge, image_with_polygon)
        
        return
    
    def save_visualization(self, binary, binary_with_border, edges, 
                          image_with_largest_edge, image_with_polygon):
        # Crear directorio si no existe
        save_dir = "FlowImages/CornerImages"
        os.makedirs(save_dir, exist_ok=True)
        
        # Crear timestamp para el nombre de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = join(save_dir, f"CornerImage_{timestamp}.png")
        
        # Configurar estilo de la visualización con la paleta personalizada
        plt.style.use('dark_background')
        plt.rcParams['axes.prop_cycle'] = plt.cycler('color', COLOR_PALETTE)
        
        # Crear figura con subplots 2x2 en lugar de 2x3
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('#121212')
        
        # Original Image
        axs[0, 0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title("Original Image")
        axs[0, 0].axis('off')
        
        # Binary Image
        axs[0, 1].imshow(binary, cmap='gray')
        axs[0, 1].set_title("Binary Image")
        axs[0, 1].axis('off')
        
        # Image with Largest Edge
        axs[1, 0].imshow(cv2.cvtColor(image_with_largest_edge, cv2.COLOR_BGR2RGB))
        if self.has_paper:
            axs[1, 0].set_title("Largest Edge Detected")
        else:
            axs[1, 0].set_title("No Edge Detected")
        axs[1, 0].axis('off')
        
        # Image with Polygon
        axs[1, 1].imshow(cv2.cvtColor(image_with_polygon, cv2.COLOR_BGR2RGB))
        if self.has_paper:
            axs[1, 1].set_title("Polygon Detection")
        else:
            axs[1, 1].set_title("No Polygon Detected")
        axs[1, 1].axis('off')
        
        # Añadir título global con información
        plt.suptitle(f"Corner Image Processing - {timestamp}", 
                    fontsize=16, y=0.98)
        
        # Ajustar espaciado y guardar
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename, facecolor=fig.get_facecolor(), dpi=150)
        plt.close()
        

    def get_lines(self, tolerance=10):
        """
        Extract lines from the detected polygon, excluding border lines.

        Args:
            tolerance (int): Tolerance value for border line detection. Defaults to 10.

        Returns:
            list: List of tuples containing start and end points of detected lines.

        Raises:
            ValueError: If no polygon was detected during processing.
        """
        
        lines = []
        if len(self.polygon) != 0:
            for i in range(len(self.polygon)):
                p1 = self.polygon[i][0]
                p2 = self.polygon[(i + 1) % len(self.polygon)][0]

                if not is_border_line(p1, p2, self.image.shape, tolerance):
                    lines.append((p1, p2))
        self.lines = lines
