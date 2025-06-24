class BrailleToCoordinates:
    """
    A class to convert binary Braille text into coordinates for embossing.
    
    This class handles the conversion of binary Braille text (where each character is 
    represented by 6 bits) into physical coordinates for embossing on paper. It supports
    multiple pages and respects Braille spacing standards.
    """
    
    def __init__(self, params):
        """
        Initialize the converter with paper and Braille specifications.

        Args:
            params (dict): Configuration dictionary containing:
                - paper: 
                    - dimensions: Paper dimensions in mm (width_mm, height_mm)
                    - margin: Page margins in mm
                - braille_to_coordinates:
                    - a: Horizontal distance between dots in same cell (mm)
                    - b: Vertical distance between dots in same cell (mm)
                    - c: Distance between identical points in adjacent cells (mm)
                    - d: Distance between identical points in adjacent lines (mm)
                    - e: Base diameter of dots (mm)
                    - f: Recommended height of dots (mm)
                    - s: Space between dot base circumferences (mm)
        """
        self.LETTER_WIDTH = params['paper']['dimensions']['width_mm']
        self.LETTER_HEIGHT = params['paper']['dimensions']['height_mm']
        self.MARGIN = params['paper']['margin']
        
        braille = params['braille_to_coordinates']
        self.a = braille['a']
        self.b = braille['b']
        self.c = braille['c']
        self.d = braille['d']
        self.e = braille['e']
        self.f = braille['f']
        self.s = braille['s']
        
        self.print_width = self.LETTER_WIDTH - (2 * self.MARGIN)
        self.print_height = self.LETTER_HEIGHT - (2 * self.MARGIN)
        self.chars_per_line = int(self.print_width / self.c)

        self.all_coordinates = []
        self.start = "left"
        self.sorted_coordinates = []
    
    def generate_point(self, center_x, center_y):
        """
        Generate coordinates for a single Braille dot.

        Args:
            center_x (float): X coordinate of dot center
            center_y (float): Y coordinate of dot center

        Returns:
            list: List containing one tuple (x, y, z) representing the dot center
                 where z is the height of the dot
        """
        return [(center_x, center_y)]
    
    def binary_to_coordinates(self, binary_text):
        """
        Convert binary Braille text to embossing coordinates.

        The input text should be formatted as space-separated 6-bit binary strings,
        where each string represents one Braille cell. Line breaks are preserved.
        
        Example input:
            "100000 111000\n101000"
            Represents: 
            - First line: 'a' followed by 'b'
            - Second line: 'c'

        Args:
            binary_text (str): Space-separated binary Braille characters with
                             optional line breaks

        Returns:
            list: List of pages, where each page is a list of coordinate tuples
                 [(x, y, z), ...] representing dot positions

        Raises:
            ValueError: If binary_text contains invalid characters or format
        """
        if not binary_text:
            return []
        
        current_page_coordinates = []
        current_x = self.MARGIN
        current_y = self.MARGIN
        current_line_chars = 0
        
        lines = binary_text.split('\n')
        
        for line in lines:
            braille_chars = line.split()
            
            current_x = self.MARGIN
            current_y += self.d
            current_line_chars = 0
            
            for braille_char in braille_chars:
                if current_y > (self.LETTER_HEIGHT - self.MARGIN - self.d):
                    self.all_coordinates.append(current_page_coordinates)
                    current_page_coordinates = []
                    current_y += 2 * self.MARGIN
                    current_x = self.MARGIN
                    current_line_chars = 0
                
                if current_line_chars >= self.chars_per_line:
                    current_y += self.d
                    current_x = self.MARGIN
                    current_line_chars = 0
                
                for i in range(6):
                    if braille_char[i] == '1':
                        col = i // 3
                        row = i % 3
                        
                        center_x = current_x + (col * self.a)
                        center_y = current_y + (row * self.b)
                        
                        current_page_coordinates.extend(self.generate_point(center_x, center_y))
                
                current_x += self.c
                current_line_chars += 1
            
            current_y += self.d
        
        if current_page_coordinates:
            self.all_coordinates.append(current_page_coordinates)
        
        return 
    
    def sort_coordinates(self):
        for page in self.all_coordinates:
            lines = {}
            for coord in page:
                try:
                    lines[coord[1]].append(coord)
                except KeyError:
                    lines[coord[1]] = [coord]
                    
            sorted_y = sorted(lines.keys())
            
            sorted_lines = []
            for y in sorted_y:
                if self.start == "left":
                    sorted_line = sorted(lines[y], key=lambda coord: coord[0])
                    sorted_lines.extend(sorted_line)
                    self.start = "right"
                else:
                    sorted_line = sorted(lines[y], key=lambda coord: coord[0], reverse=True)
                    sorted_lines.extend(sorted_line)
                    self.start = "left"
            self.sorted_coordinates.extend(sorted_lines)
        return


