import pytesseract
import cv2
import time

class PaperImage():
    """
    Class for processing paper images and extracting text using OCR.
    Handles image orientation, DPI calculation, and text extraction.
    """
    def __init__(self, image, parameters, width_mm, height_mm):
        """
        Initialize PaperImage with image and physical dimensions.

        Args:
            image (numpy.ndarray): Input image array
            parameters (dict): Processing parameters with 'third_module' key
            width_mm (float): Paper width in millimeters
            height_mm (float): Paper height in millimeters
        """
        self.parameters = parameters["third_module"]
        self.text = None
        self.image = image
        self.width_mm = width_mm
        self.height_mm = height_mm
        self.height_px = self.image.shape[0]
        self.width_px = self.image.shape[1]
    
    
    def get_orientation(self):
        """
        Detect if image is rotated 180 degrees using Tesseract OCR.

        Returns:
            bool: True if image is rotated 180 degrees, False otherwise

        Raises:
            RuntimeError: If Tesseract OCR is not properly installed
        """
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        osd = pytesseract.image_to_osd(self.image, output_type=pytesseract.Output.DICT)
        rotation = osd["rotate"]
        return rotation == 180
    
    def process(self):
        """
        Process image by correcting orientation if needed.
        Rotates image 180 degrees if detected as upside down.

        Args:
            None

        Returns:
            None
        """
        if self.get_orientation():
            self.image = cv2.rotate(self.image, cv2.ROTATE_180)

    
    def calculate_dpi(self):
        """
        Calculate average DPI based on image dimensions and physical size.

        Returns:
            int: Average DPI (dots per inch) across horizontal and vertical directions
        """
        dpi_h = self.width_px / (self.width_mm / 25.4)

        dpi_v = self.height_px / (self.height_mm / 25.4)
        
        return int((dpi_h + dpi_v) / 2)

    
    def get_text(self):
        """
        Extract text from image using Tesseract OCR.
        Processes image orientation and sets appropriate DPI before extraction.

        Returns:
            str: Extracted text from the image

        Side effects:
            - Updates self.text with extracted content
            - Prints extraction timing information
        """
        self.process()
        tesseract_config = self.parameters["config"]
        tesseract_config += f" --dpi {self.calculate_dpi()}"
        start_time = time.time()
        self.text = pytesseract.image_to_string(self.image, config=tesseract_config)
        end_time = time.time()
        extraction_time = end_time - start_time
        print("--------------------")
        print("--------------------")
        print(f"Tiempo de extracción de texto: {extraction_time:.4f} segundos")
        print("--------------------")
        print("--------------------")
        return self.text