import pytesseract
import sys

# Forzamos la ruta que instalaste
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

print("=== DIAGNÓSTICO DE TESSERACT ===")
try:
    version = pytesseract.get_tesseract_version()
    print(f"¡Éxito! Tesseract detectado correctamente.")
    print(f"Versión instalada: {version}")
except Exception as e:
    print(f"❌ ERROR: Python no puede ejecutar Tesseract.")
    print(f"Detalle del error: {e}")
    print("\nRevisa si el archivo existe en: C:\\Program Files\\Tesseract-OCR\\tesseract.exe")