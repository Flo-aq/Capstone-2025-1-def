from tokenize import cookie_re
from Controllers.Arduino import ArduinoMegaPrinter
from Controllers.DeviceManager import DeviceManager

def test_printer_setup():
    """Test the printer setup process."""
    device_manager = DeviceManager()
    printer = device_manager.arduino_mega_printer
    
    coordinates = [(35.0, 25.0), (35.0, 27.5), (35.0, 31.0), (35.0, 37.0), (35.0, 39.5), (35.0, 43.0), (35.0, 49.0), (35.0, 51.5), (35.0, 57.5), (35.0, 61.0), ]
    
    success = printer.test_print_braille_points(coordinates)
    if success:
        print("Printer setup test passed.")
    else:
        print("Printer setup test failed.")

test_printer_setup()