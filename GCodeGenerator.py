class GCodeGenerator:
    def __init__(self, coordinates):
        self.script = []
        self.coordinates = coordinates
    
    def start_gcode(self):
        self.script.append("G28 X;")
        self.script.append("G28 Y;")
        self.script.append("M204 T1500;")
        self.script.append("G1 F6000;")
        self.script.append("G1 X0.00 Y0.00;")
    
    def end_gcode(self):
        self.script.append("M84;")
    
    def body_gcode(self):
        for coord in self.coordinates:
            x, y = coord
            self.script.append(f"G1 X{x:.2f} Y{y:.2f};")
            self.script.append("M84;")
            self.script.append("M3 S1;")
            self.script.append("M3 S0;")
    
    def generate_gcode(self):
        self.start_gcode()
        self.body_gcode()
        self.end_gcode()
        return self.script
