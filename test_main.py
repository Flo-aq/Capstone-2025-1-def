import os
import cv2
import argparse
from main import Main
from picamera2 import Picamera2
import time

def test_homing_and_corners(visit_only=False):
    """
    Prueba completa del sistema con opciones configurables
    
    Args:
        visit_only (bool): Si es True, solo visita esquinas sin capturar imágenes
        output_dir (str): Directorio donde guardar resultados (o None para auto-crear)
        delay (float): Tiempo de espera en segundos para estabilización
    """
    # Crear instancia del sistema
    system = Main()
    
    # Crear directorio para guardar imágenes
    if not visit_only:
        output_dir = f"test_results"
    
        os.makedirs(output_dir, exist_ok=True)
    
    print("\n===== PRUEBA DE HOMING Y ESQUINAS =====\n")
    
    # Paso 1: Homing inicial
    print("PASO 1: Realizando homing inicial...")
    if not system.homing(set_custom_origin=True):
        print("ERROR: Falló el homing inicial. Prueba cancelada.")
        return False
    print("✓ Homing inicial completado exitosamente\n")
    
    
    corners = ["top-left", "top-right", "bottom-right", "bottom-left", "top-left"]
    # Paso 2: Visitar esquinas
    if visit_only:
      print("PASO 2: Visitando esquinas...")
      
      for corner in corners:
          print(f"  Moviendo a {corner}...")
          try:
              system.move_to_corner(corner)
              print(f"  ✓ Llegada a {corner}")
          except Exception as e:
              print(f"  ERROR: No se pudo mover a {corner}: {str(e)}")
      
      print("✓ Visita a esquinas completada\n")
    
    else:
      print("PASO 2: Capturando fotos en esquinas...")
      corners = corners[1:]  # Excluir el último "top-left"
      
      capture_results = {'success': {}, 'images': {}}
      
      for corner in corners:
          print(f"  Moviendo a {corner} para capturar foto...")
          try:
              system.move_to_corner(corner)
              
              print(f"  Capturando imagen en {corner}...")
              img = system.capture_image()
              
              if img is not None:
                  img_path = os.path.join(output_dir, f"{corner}.jpg")
                  cv2.imwrite(img_path, img)
                  
                  capture_results['success'][corner] = True
                  capture_results['images'][corner] = img_path
                  
                  print(f"  ✓ Imagen guardada en: {img_path}")
              else:
                  capture_results['success'][corner] = False
                  print(f"  ✗ Error: No se pudo capturar imagen en {corner}")
          
          except Exception as e:
              capture_results['success'][corner] = False
              print(f"  ✗ Error en {corner}: {str(e)}")
      
      
      successful_captures = sum(1 for status in capture_results['success'].values() if status)
      print(f"\nRESUMEN: Se capturaron {successful_captures} de {len(corners)} imágenes")
      print(f"Imágenes guardadas en: {os.path.abspath(output_dir)}")
      
      return successful_captures == len(corners)

def start_video_feed():
    try:
      camera = Picamera2()
      
      camera_config = camera.create_preview_configuration(main="size": (1640, 1232), "fromat": "RGB888")
      camera.configure(camera_config)
      camera.start()
      
      time.sleep(1)
      
      while True:
          frame = camera.capture_array()
          cv2.imshow("Video Feed", frame)
          
          if cv2.waitKey(1) != -1:
                break
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        if 'camera' in locals():
            camera.stop()
            camera.close()
        cv2.destroyAllWindows()
      

def main():
    """Función principal con opciones de línea de comandos"""
    parser = argparse.ArgumentParser(description="Prueba de homing y captura de esquinas")
    parser.add_argument("command", nargs='?', default="homing",
                      choices=["homing", "video"],
                      help="Comando a ejecutar: 'homing' para test de homing, 'video' para feed de video")
    parser.add_argument("--visit-only", action="store_true",
                      help="Solo visitar esquinas sin capturar imágenes")
    
    args = parser.parse_args()
    
    try:
        if args.command == "video":
            start_video_feed()
        else:  # default es "homing"
            test_homing_and_corners(visit_only=args.visit_only)
    except KeyboardInterrupt:
        print("\n\nPrueba interrumpida por el usuario.")
    except Exception as e:
        print(f"\n\nError inesperado durante la prueba: {str(e)}")

if __name__ == "__main__":
    main()
    # python test_main.py video
    # python test_main.py homing
    # python test_main.py homing --visit-only
    