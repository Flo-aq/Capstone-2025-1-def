from ImagesClasses.Image import Image
from Functions.FirstModuleFunctions import (
    first_module_process_image,
    add_border,
    detect_edges,
    process_edges_and_remove_border,
    get_largest_edge,
    approximate_polygon,
    is_border_line
)

class ImageFunction1(Image):
    """
    A specialized Image class for processing images using the first module's functionality.
    Implements polygon detection and line extraction from images.
    Inherits from the base Image class.
    """

    def __init__(self, image, camera, parameters=None):
        """
        Initialize ImageFunction1 with image processing parameters.

        Args:
            image (array): Input image to process
            camera (Camera): Camera object for image acquisition
            parameters (dict): Dictionary containing processing parameters for the first module
        """
        super().__init__(function=1, image=image, camera=camera)
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
        if largest_edge is not None:
            print("Polygon detected")
            self.polygon = approximate_polygon(largest_edge)
            self.has_paper = True
        else:
            self.polygon = []

        return

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
