import serial
import time


class Arduino():
    def __init__(self, baudrate=115200, port=None):
        self.baudrate = baudrate
        self.port = port
        self.serial = None
        self.connected = False
        self.type = ""

    def connect(self, port=None, max_attempts=3, delay=2):
        if port:
            self.port = port

        if not self.port:
            return False

        if self.serial is not None and self.serial.is_open:
            self.connected = True
            return True

        attempts = 0

        while attempts < max_attempts:
            try:
                self.serial = serial.Serial(
                    self.port, self.baudrate, timeout=2)
                if attempts == 0:
                    time.sleep(2) 
                else:
                    time.sleep(0.5)
                self.connected = True
                return True

            except:
                pass

            attempts += 1
            time.sleep(delay)

        self.connected = False
        return False

    def disconnect(self):
        if self.serial and self.serial.is_open:
            self.serial.close()
            self.connected = False

    def is_connected(self):
        return self.connected and self.serial and self.serial.is_open


class ArduinoMegaScanner(Arduino):
    def __init__(self, baudrate=115200, port=None):
        super().__init__(baudrate, port)
        self.type = "Arduino Uno"

    def send_command(self, command):
        if not self.is_connected():
            return "E: Arduino Uno not connected"
        try:
            self.serial.reset_input_buffer()

            if not command.endswith('\n'):
                command += '\n'

            self.serial.write(command.encode('utf-8'))

            start_time = time.time()
            response_lines = []
            
            if self.serial.in_waiting > 0:
                self.serial.reset_input_buffer()

            while time.time() - start_time < 5:
                if self.serial.in_waiting > 0:
                    line = self.serial.readline().decode('utf-8').strip()
                    if line:
                        response_lines.append(line)
                        if line == "COMMAND EXECUTED" or line.startswith("OK"):
                            break
                else:
                    time.sleep(0.01)

            if not response_lines:
                return "E"

            if len(response_lines) == 1:
                return response_lines[0]
            else:
                return "\n".join(response_lines)
              
            return "\n".join(response_lines)  # Return the last line as the response

        except Exception as e:
            return f"E: {str(e)}"

    def home_x(self):
        return self.send_command("HX")

    def home_y(self):
        return self.send_command("HY")

    def get_position_x(self):
        return self.send_command("POSX")

    def get_position_y(self):
        return self.send_command("POSY")

    def move_without_PID(self, axis, distance):
        if axis.upper() not in ["X", "Y"]:
            return "E: Invalid axis"

        command = f"M{axis.upper()} {distance}"
        return self.send_command(command)
    
    def move_with_PID(self, axis, distance):
        if axis.upper() not in ["X", "Y"]:
            return "E: Invalid axis"

        command = f"PID{axis.upper()} {distance}"
        return self.send_command(command)
    
    def full_homing(self):
      return self.send_command("FH")
    
    def set_origin(self):
        return self.send_command("SETORIGIN")

    def home_and_set_origin(self):
        return self.send_command("HOME&SET&ORIGIN")

class ArduinoMegaPrinter(Arduino):
    def __init__(self, baudrate=115200, port=None):
        super().__init__(baudrate, port)
        self.type = "Arduino Mega"

    def send_command(self, command):
        if not self.is_connected():
            return "Error: Arduino Mega no conectado"

        command_type = command.split(" ")[0]
        allowed_commands = ["G28", "M204", "G1", "M3", "M84"]
        if command_type not in allowed_commands:
            return "E: Invalid command"

        try:
            if not command.endswith('\n'):
                command += '\n'
            self.serial.write(command.encode('utf-8'))

            start_time = time.time()
            response_lines = []

            while time.time() - start_time < 20:
                if self.serial.in_waiting > 0:
                    line = self.serial.readline().decode('utf-8').strip()
                    if line:
                        response_lines.append(line)
                        if line == "COMMAND EXECUTED":
                            break
                else:
                    time.sleep(0.1)

            if not response_lines or len(response_lines) == 1:
                return "E"

            if len(response_lines) > 1:
                return "\n".join(response_lines[:-1])

        except Exception as e:
            return f"E: {str(e)}"
