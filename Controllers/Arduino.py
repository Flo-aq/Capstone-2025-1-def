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
        self.command_in_progress = False
        self.last_command_time = 0
        self.command_timeout = 5  # segundos

    def send_command(self, command):
        if not self.is_connected():
            return "E: Arduino Uno not connected"
        
        # Verificar si el último comando es demasiado reciente
        current_time = time.time()
        if current_time - self.last_command_time < 0.5:  # Esperar al menos 0.5 segundos entre comandos
            time.sleep(0.5 - (current_time - self.last_command_time))
        
        # Esperar explícitamente si hay un comando en progreso
        self.wait_for_command_completion()
            
        try:
            print(f"Sending command: {command}")
            self.serial.reset_input_buffer()  # Limpiar buffer antes de enviar
            self.command_in_progress = True
            self.last_command_time = time.time()
            
            if not command.endswith('\n'):
                command += '\n'

            self.serial.write(command.encode('utf-8'))
            self.serial.flush()  # Asegurar que el comando se envía completamente

            # Esperar respuesta con timeout
            start_time = time.time()
            response_lines = []
            
            while (time.time() - start_time) < 20:  # 20 segundos de timeout
                if self.serial.in_waiting > 0:
                    line = self.serial.readline().decode('utf-8').strip()
                    if line:
                        print(f"Received: {line}")
                        response_lines.append(line)
                        
                        # Detectar fin de respuesta
                        if line.startswith("OK") or line.startswith("E:"):
                            # Esperar un poco más por si hay más líneas
                            time.sleep(0.2)
                            # Leer cualquier dato adicional en el buffer
                            while self.serial.in_waiting > 0:
                                extra_line = self.serial.readline().decode('utf-8').strip()
                                if extra_line:
                                    response_lines.append(extra_line)
                            break
                else:
                    time.sleep(0.1)  # Pequeña pausa si no hay datos

            self.command_in_progress = False
            
            if not response_lines:
                return "E: No response from Arduino"
            
            print(f"Response: {response_lines}")
            cleaned_response = []
            for line in response_lines:
                if line.startswith("OK: ") or line.startswith("E: "):
                    cleaned_response.append(line)
            return "\n".join(cleaned_response)
              
        except Exception as e:
            self.command_in_progress = False
            return f"E: {str(e)}"
    
    def wait_for_command_completion(self, timeout=30):
        """Espera hasta que cualquier comando previo se complete."""
        if not self.command_in_progress:
            return True
            
        print("Waiting for previous command to complete...")
        start_time = time.time()
        while self.command_in_progress:
            if time.time() - start_time > timeout:
                print("Command timeout, resetting state")
                self.command_in_progress = False
                self.serial.reset_input_buffer()
                return False
            time.sleep(0.1)
        
        return True

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
    
    def max_limit_x(self):
        """Get maximum X limit from Arduino."""
        response = self.send_command("XLIMIT")
        if response.startswith("OK"):
            return response.split("OK: ")[1].strip()
        return "0"
        
    def max_limit_y(self):
        """Get maximum Y limit from Arduino."""
        response = self.send_command("YLIMIT")
        if response.startswith("OK"):
            return response.split("OK: ")[1].strip()
        return "0"
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
