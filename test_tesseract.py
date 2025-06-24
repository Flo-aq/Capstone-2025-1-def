# Guárdalo como test_tesseract_path.py
import pytesseract

# Configura la ruta
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Verifica que Tesseract se pueda ejecutar
try:
    version = pytesseract.get_tesseract_version()
    print(f"Tesseract funciona correctamente - versión {version}")
except Exception as e:
    print(f"Error: {e}")