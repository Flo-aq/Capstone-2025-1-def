from curses import start_color
from tracemalloc import start
import serial
import time
import threading
import queue


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
        self.type = "Arduino Mega"
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.communication_thread = None
        self.running = False
        self.lock = threading.Lock()

    def start_communication_thread(self):
        if self.communicaction_thread is None or not self.communication_thread.is_alive():
            self.running = True
            self.communication_thread = threading.Thread(
                target=self._communication_worker)
            self.communication_thread.daemon = True
            self.communication_thread.start()
            print("Communication thread started")

    def stop_communication_thread(self):
        self.running = False
        if self.communication_thread and self.communication_thread.is_alive():
            self.communication_thread.join(timeout=0.5)
            print("Communication thread stopped")

    def _communication_worker(self):
        while self.running and self.is_connected():
            try:
                command = self.command_queue.get(timeout=0.5)
                print(f"Processing command: {command}")
                with self.lock:
                    self._execute_command(command)
                self.command_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in communication thread: {e}")
                self.response_queue.put(f"E: {str(e)}")

    def _execute_command(self, command):
        if not self.is_connected():
            self.response_queue.put("E: Arduino Mega not connected")
            return
        try:
            self.serial.reset_input_buffer()  # Limpiar buffer antes de enviar

            if not command.endswith('\n'):
                command += '\n'

            self.serial.write(command.encode('utf-8'))
            self.serial.flush()

            start_time = time.time()
            response_lines = []

            while (time.time() - start_time) < 600:  # 20 segundos de timeout
                if self.serial.in_waiting > 0:
                    line = self.serial.readline().decode('utf-8').strip()
                    if line:
                        print(f"Received: {line}")
                        if line == "COMMAND EXECUTED":
                            break  # Solo usamos esto como señal de finalización
                        else:
                            response_lines.append(line)
                else:
                    time.sleep(0.1)
            if not response_lines:
                self.response_queue.put("E: No response from Arduino")
                return
            has_error = any(line.startswith("E:") for line in response_lines)
            if has_error:
                # Filtramos las líneas de error
                error_lines = [
                    line for line in response_lines if line.startswith("E:")]
                self.response_queue.put("\n".join(error_lines))
            else:
                # Si no hay errores, devolvemos todas las líneas con prefijo OK
                self.response_queue.put("\n".join(response_lines))
        except Exception as e:
            self.response_queue.put(f"E: {str(e)}")

    def start_communication_thread(self):
        if self.communication_thread is None or not self.communication_thread.is_alive():
            self.running = True
            self.communication_thread = threading.Thread(
                target=self._communication_worker)
            self.communication_thread.daemon = True
            self.communication_thread.start()
            print("Communication thread started")

    def send_command(self, command, timeout=600):
        """Send a command to Arduino and wait for response."""
        # Start communication thread if needed
        if self.communication_thread is None or not self.communication_thread.is_alive():
            self.start_communication_thread()

        if not self.is_connected():
            return "E: Arduino Mega not connected"

        print(f"Queueing command: {command}")
        # Add command to queue
        self.command_queue.put(command)

        # Wait for response with timeout
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            try:
                response = self.response_queue.get(timeout=0.5)
                self.response_queue.task_done()
                return response
            except queue.Empty:
                # No response yet, keep waiting
                pass

        return "E: Response timeout"

    def connect(self, port=None, max_attempts=3, delay=2):
        result = super().connect(port, max_attempts, delay)
        if result:
            self.start_communication_thread()
        return result

    def disconnect(self):
        self.stop_communication_thread()
        super().disconnect()

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
        if response.startswith("OK:"):
            return response
        return "0"

    def max_limit_y(self):
        """Get maximum Y limit from Arduino."""
        response = self.send_command("YLIMIT")
        if response.startswith("OK:"):
            return response
        return "0"

    def turn_on_leds(self):
        """Turn on the LEDs."""
        return self.send_command("LED ON")

    def turn_off_leds(self):
        """Turn off the LEDs."""
        return self.send_command("LED OFF")

    # def send_command(self, command):
    #     if not self.is_connected():
    #         return "E: Arduino Uno not connected"

    #     # Verificar si el último comando es demasiado reciente
    #     current_time = time.time()
    #     if current_time - self.last_command_time < 0.5:  # Esperar al menos 0.5 segundos entre comandos
    #         time.sleep(0.5 - (current_time - self.last_command_time))

    #     # Esperar explícitamente si hay un comando en progreso
    #     self.wait_for_command_completion()

    #     try:
    #         print(f"Sending command: {command}")
    #         self.serial.reset_input_buffer()  # Limpiar buffer antes de enviar
    #         self.command_in_progress = True
    #         self.last_command_time = time.time()

    #         if not command.endswith('\n'):
    #             command += '\n'

    #         self.serial.write(command.encode('utf-8'))
    #         self.serial.flush()  # Asegurar que el comando se envía completamente

    #         # Esperar respuesta con timeout
    #         start_time = time.time()
    #         response_lines = []

    #         while (time.time() - start_time) < 20:  # 20 segundos de timeout
    #             if self.serial.in_waiting > 0:
    #                 line = self.serial.readline().decode('utf-8').strip()
    #                 if line:
    #                     print(f"Received: {line}")
    #                     response_lines.append(line)

    #                     # Detectar fin de respuesta
    #                     if line.startswith("OK") or line.startswith("E:"):
    #                         # Esperar un poco más por si hay más líneas
    #                         time.sleep(0.2)
    #                         # Leer cualquier dato adicional en el buffer
    #                         while self.serial.in_waiting > 0:
    #                             extra_line = self.serial.readline().decode('utf-8').strip()
    #                             if extra_line:
    #                                 response_lines.append(extra_line)
    #                         break
    #             else:
    #                 time.sleep(0.1)  # Pequeña pausa si no hay datos

    #         self.command_in_progress = False

    #         if not response_lines:
    #             return "E: No response from Arduino"

    #         print(f"Response: {response_lines}")
    #         cleaned_response = []
    #         for line in response_lines:
    #             if line.startswith("OK: ") or line.startswith("E: "):
    #                 cleaned_response.append(line)
    #         return "\n".join(cleaned_response)

    #     except Exception as e:
    #         self.command_in_progress = False
    #         return f"E: {str(e)}"

    # def wait_for_command_completion(self, timeout=30):
    #     """Espera hasta que cualquier comando previo se complete."""
    #     if not self.command_in_progress:
    #         return True

    #     print("Waiting for previous command to complete...")
    #     start_time = time.time()
    #     while self.command_in_progress:
    #         if time.time() - start_time > timeout:
    #             print("Command timeout, resetting state")
    #             self.command_in_progress = False
    #             self.serial.reset_input_buffer()
    #             return False
    #         time.sleep(0.1)

    #     return True

    # def home_x(self):
    #     return self.send_command("HX")

    # def home_y(self):
    #     return self.send_command("HY")

    # def get_position_x(self):
    #     return self.send_command("POSX")

    # def get_position_y(self):
    #     return self.send_command("POSY")

    # def move_without_PID(self, axis, distance):
    #     if axis.upper() not in ["X", "Y"]:
    #         return "E: Invalid axis"

    #     command = f"M{axis.upper()} {distance}"
    #     return self.send_command(command)

    # def move_with_PID(self, axis, distance):
    #     if axis.upper() not in ["X", "Y"]:
    #         return "E: Invalid axis"

    #     command = f"PID{axis.upper()} {distance}"
    #     return self.send_command(command)

    # def full_homing(self):
    #   return self.send_command("FH")

    # def set_origin(self):
    #     return self.send_command("SETORIGIN")

    # def home_and_set_origin(self):
    #     return self.send_command("HOME&SET&ORIGIN")

    # def max_limit_x(self):
    #     """Get maximum X limit from Arduino."""
    #     response = self.send_command("XLIMIT")
    #     if response.startswith("OK"):
    #         return response.split("OK: ")[1].strip()
    #     return "0"

    # def max_limit_y(self):
    #     """Get maximum Y limit from Arduino."""
    #     response = self.send_command("YLIMIT")
    #     if response.startswith("OK"):
    #         return response.split("OK: ")[1].strip()
    #     return "0"


class ArduinoMegaPrinter(Arduino):
    def __init__(self, baudrate=115200, port=None):
        super().__init__(baudrate, port)
        self.type = "Arduino Mega"
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.communication_thread = None
        self.running = False
        self.lock = threading.Lock()

    def start_communication_thread(self):
        if self.communication_thread is None or not self.communication_thread.is_alive():
            self.running = True
            self.communication_thread = threading.Thread(
                target=self._communication_worker)
            self.communication_thread.daemon = True
            self.communication_thread.start()
            print("Printer communication thread started")

    def stop_communication_thread(self):
        self.running = False
        if self.communication_thread and self.communication_thread.is_alive():
            self.communication_thread.join(timeout=0.5)
            print("Printer communication thread stopped")

    def _communication_worker(self):
        while self.running and self.is_connected():
            try:
                command = self.command_queue.get(timeout=0.5)
                with self.lock:
                    self._execute_command(command)
                self.command_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in printer communication thread: {e}")
                self.response_queue.put(f"E: {str(e)}")

    def _execute_command(self, command):
        if not self.is_connected():
            self.response_queue.put("E: Arduino Mega Printer not connected")
            return
        try:
            self.serial.reset_input_buffer()

            if not command.endswith('\n'):
                command += '\n'

            self.serial.write(command.encode('utf-8'))
            self.serial.flush()

            # Para comandos de impresión, esperamos respuesta OK
            start_time = time.time()
            response_lines = []

            while (time.time() - start_time) < 10:  # 10 segundos máximo de espera
                if self.serial.in_waiting > 0:
                    line = self.serial.readline().decode('utf-8').strip()
                    if line:
                        if line == "ok" or line.startswith("ok:") or line.startswith("e") or line.startswith("E"):
                            response_lines.append(line)
                            break
                else:
                    time.sleep(0.1)

            if not response_lines:
                self.response_queue.put("E: No response from Arduino Printer")
            elif any("error" in line.lower() or "unkwown" in line.lower() or "failed" in line.lower() for line in response_lines):
                error_lines = [line for line in response_lines if "error" in line.lower(
                ) or "unkwown" in line.lower() or "failed" in line.lower()]
                self.response_queue.put("\n".join(error_lines))
            else:
                # Devolver solo la primera línea de respuesta
                self.response_queue.put(response_lines[0])

        except Exception as e:
            self.response_queue.put(f"E: {str(e)}")

    def send_command(self, command, timeout=10):
        """Send a command to Arduino Mega Printer and wait for response."""
        # Start communication thread if needed
        if self.communication_thread is None or not self.communication_thread.is_alive():
            self.start_communication_thread()

        if not self.is_connected():
            return "E: Arduino Mega Printer not connected"

        # Add command to queue
        self.command_queue.put(command)

        # Wait for response with timeout
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            try:
                response = self.response_queue.get(timeout=0.5)
                self.response_queue.task_done()
                return response
            except queue.Empty:
                # No response yet, keep waiting
                pass

        return "E: Printer response timeout"

    def connect(self, port=None, max_attempts=3, delay=2):
        result = super().connect(port, max_attempts, delay)
        if result:
            self.start_communication_thread()
        return result

    def disconnect(self):
        self.stop_communication_thread()
        super().disconnect()

    def setup(self):
        """Send setup commands to the printer."""
        commands = [
            "G28 Y",
            "M84",  # Disable motors
            "G28 X",
            "M84",
            "M204 T1500",  # Set acceleration
            "M84",
            "G4 P500",
            "G1 F6000",  # Set feedrate
            "G4 P500",
            "M84"
        ]
        for command in commands:
            response = self.send_command(command)
            if "error" in response.lower() or "unkwown" in response.lower() or "failed" in response.lower():
                return f"E: Setup failed with command '{command}': {response}"
        print("Printer setup complete")
        return "OK"

    def test_setup(self):
        commands = [
              "G28 Y",
              "M84",  # Disable motors
              "G92 X0",
              "M84",
              "M204 T1500",  # Set acceleration
              "M84",
              "G4 P500",
              "G1 F6000",  # Set feedrate
              "G4 P500",
              "M84"
            ]
        for command in commands:
            response = self.send_command(command)
            if "error" in response.lower() or "unkwown" in response.lower() or "failed" in response.lower():
                return f"E: Setup failed with command '{command}': {response}"
        print("Printer setup complete")
        return "OK"
    
    def test_print_braille_point(self, x, y):
        movement_command = f"G1 X{x:.2f} Y{y:.2f}"
        response = self.send_command(movement_command)
        commands = [
            "M84",
            "M5",
            "G4 P250",
            "M3",
            "M5",
            "G4 P250",
            "M3",
            "M5",
            "G4 P250",
            "M3",
            "M5",
            "G4 P250",
            "M3"
        ]
        for command in commands:
            response = self.send_command(command)
            if "error" in response.lower() or "unkwown" in response.lower() or "failed" in response.lower():
                return False
        return True

    def print_braille_point(self, x, y):
        """Print a Braille dot at specified coordinates."""
        movement_command = f"G1 X{x:.2f} Y{y:.2f}"
        response = self.send_command(movement_command)
        commands = [
            "M84",
            "M5",
            "G4 P250",
            "M3"
        ]
        for command in commands:
            response = self.send_command(command)
            if "error" in response.lower() or "unkwown" in response.lower() or "failed" in response.lower():
                return False
        return True

    def print_braille_points(self, coordinates, test_mode=False):
        """Print multiple Braille dots efficiently."""
        success = True
        total_points = len(coordinates)
        printed = 0

        print(f"Starting to print {total_points} Braille points...")

        if not test_mode:
            home_response = self.setup()
            if not home_response.startswith("OK"):
                return False

            for x, y in coordinates:
                if not self.print_braille_point(x, y):
                    print(f"Failed to print point at ({x}, {y})")
                    success = False
                else:
                    printed += 1
                    if printed % 10 == 0:
                        print(
                            f"Progress: {printed}/{total_points} points printed")

            print(
                f"Braille printing complete: {printed}/{total_points} points printed")
            return success
        else:
            home_response = self.test_setup()
            if not home_response.startswith("OK"):
                return False

            for x, y in coordinates:
                if not self.test_print_braille_point(x, y):
                    print(f"Failed to print point at ({x}, {y})")
                    success = False
                else:
                    printed += 1
                    if printed % 10 == 0:
                        print(
                            f"Progress: {printed}/{total_points} points printed")

            print(
                f"Braille printing complete: {printed}/{total_points} points printed")
            return success
