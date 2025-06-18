import serial
from serial.tools import list_ports
import time
from Controllers.Arduino import ArduinoMegaPrinter, ArduinoMegaScanner


class DeviceManager:
    def __init__(self):
        self.devices = {
            "arduino_mega_printer": None,
            "arduino_mega_scanner": None,
            "other": []
        }

        self.ports_info = {}
        self.devices_ports = {"arduino_mega_printer": [],
                              "arduino_mega_scanner": [],
                              "keyboard": [], 
                              "unknown": []}
        
        self.arduino_mega_printer = None
        self.arduino_mega_scanner = None
        
        self.identify_and_categorize_arduinos()
        self.initialize_devices()
        
    def detect_arduinos_and_devices(self):
        available_ports = list(list_ports.comports())

        if not available_ports:
            print("E: No devices found.")
            return self.devices_ports

        identifiers = {
            "arduino_mega": ["arduino mega", "mega", "2560", "atmega2560", "mks"],
            "keyboard": ["keyboard", "teclado", "hid", "input device"]
        }

        arduino_ports = []

        for port in available_ports:
            device_type = "unknown"
            desc = port.description.lower() if port.description else ""
            hwid = port.hwid.lower() if port.hwid else ""
            
            self.ports_info[port.device] = {
                "description": desc,
                "hwid": hwid,
                "type": device_type
            }
            
            if any(id_str in desc or id_str in hwid for id_str in identifiers["arduino_mega"]):
                arduino_ports.append(port.device)
            elif any(id_str in desc or id_str in hwid for id_str in identifiers["keyboard"]):
                device_type = "keyboard"
                self.ports_info[port.device]["type"] = "keyboard"
                self.devices_ports["keyboard"].append(port.device)
            else:
                self.devices_ports["unknown"].append(port.device)
        return arduino_ports
    
    def identify_and_categorize_arduinos(self):
        arduino_ports = self.detect_arduinos_and_devices()
        for port in arduino_ports:
            arduino_type = self.identify_arduino(port)
            if arduino_type == "printer":
                self.devices_ports["arduino_mega_printer"].append(port)
                self.ports_info[port]["type"] = "arduino_mega_printer"
            elif arduino_type == "scanner":
                self.devices_ports["arduino_mega_scanner"].append(port)
                self.ports_info[port]["type"] = "arduino_mega_scanner"
            else:
                self.devices_ports["unknown"].append(port)
                self.ports_info[port]["type"] = "unknown"

    def identify_arduino(self, port):
        try:
            with serial.Serial(port, 115200, timeout=1) as ser:
                time.sleep(2)  # Wait for the Arduino to reset
                ser.write(b'M119\n')
                time.sleep(0.5)
                for i in range(4):
                    line = ser.readline().decode('utf-8').strip()
                    print(f"Line read: {line}")
                    if "ok" in line.lower():
                        return "printer"
                    if "e: cnf" in line.lower():
                        return "scanner"
                return "unknown"
        except serial.SerialException as e:
            print(f"E: Serial error on {port}: {e}")
            return "unknown"
        except Exception as e:
            print(f"E: Unexpected error when identifying {port}: {str(e)}")
            return "unknown"
    
    def get_device_port(self, device_type):
        if device_type in self.devices_ports and self.devices_ports[device_type]:
            return self.devices_ports[device_type][0]
        
        return None

    def initialize_devices(self):
      printer_port = self.get_device_port("arduino_mega_printer")
      if printer_port:
          self.arduino_mega_printer = ArduinoMegaPrinter(port=printer_port)
          self.arduino_mega_printer.connect()
          if not self.arduino_mega_printer.is_connected():
              print("E: Failed to connect to Arduino Mega Printer")
              self.arduino_mega_printer = None
              
      scanner_port = self.get_device_port("arduino_mega_scanner")
      if scanner_port:
          self.arduino_mega_scanner = ArduinoMegaScanner(port=scanner_port)
          self.arduino_mega_scanner.connect()
          if not self.arduino_mega_scanner.is_connected():
              print("E: Failed to connect to Arduino Mega Scanner")
              self.arduino_mega_scanner = None