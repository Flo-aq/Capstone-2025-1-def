import os
import cv2
import argparse
from main import Main
from picamera2 import Picamera2
import time

def test_homing_and_corners():
    """
    Prueba completa del sistema con opciones configurables
    
    Args:
        visit_only (bool): Si es True, solo visita esquinas sin capturar imágenes
        output_dir (str): Directorio donde guardar resultados (o None para auto-crear)
        delay (float): Tiempo de espera en segundos para estabilización
    """
    # Crear instancia del sistema
    system = Main()
    
    
    print("\n===== PRUEBA DE HOMING Y ESQUINAS =====\n")
    
    # Paso 1: Homing inicial
    print("PASO 1: Realizando homing inicial...")
    if not system.homing(set_custom_origin=True):
        print("ERROR: Falló el homing inicial. Prueba cancelada.")
        return False
    print("✓ Homing inicial completado exitosamente\n")
    
    
    system.capture_corners_imgs() 
      
    return

def start_video_feed():
    try:
      camera = Picamera2()
      
      camera_config = camera.create_preview_configuration(main={"size": (1640, 1232), "format": "RGB888"})
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
      

def test_position_commands():
    """
    Prueba de los comandos de posicionamiento (POSX, POSY) 
    y movimiento sin PID (MX, MY)
    """
    # Crear instancia del sistema
    system = Main()

    scanner = system.device_manager.arduino_mega_scanner
    if not scanner:
        print("ERROR: No se detectó Arduino Scanner")
        return False
    
    if not scanner.is_connected():
        print("Conectando a Arduino Scanner...")
        if not scanner.connect():
            print("ERROR: No se pudo conectar al Arduino Scanner")
            return False
    
    # if not system.homing(set_custom_origin=True):
    #     print("ERROR: Falló el homing inicial. Prueba cancelada.")
    #     return False
    
    # print("✓ Homing inicial completado exitosamente\n")
    
    movements = [
        ("X", 30),
        ("X", -40),
        ("Y", 20),
        ("Y", -30)
    ]
    
    for axis, distance in movements:
        print(f"  Moviendo {axis} {distance}mm sin PID...")
        try:
            response = scanner.move_without_PID(axis, distance)
            print(f"  Respuesta: {response}")
            time.sleep(1)  # Esperar a que se complete el movimiento
            
            # Obtener nueva posición
            pos_x = scanner.get_position_x()
            pos_y = scanner.get_position_y()
            print(f"  Nueva posición: X:{pos_x}, Y:{pos_y}")
            
        except Exception as e:
            print(f"  ERROR durante movimiento {axis}{distance}: {str(e)}")
    

    # try:
    #     # Ir a posición conocida
    #     print("  Moviendo a posición X:50, Y:50...")
    #     system.move_to_position(50, 50)
    #     time.sleep(1)
        
    #     # Verificar posición
    #     pos_x = system.device_manager.arduino_mega_scanner.get_position_x()
    #     pos_y = system.device_manager.arduino_mega_scanner.get_position_y()
    #     print(f"  Posición leída: X:{pos_x}, Y:{pos_y}")
        
    # except Exception as e:
    #     print(f"  ERROR en secuencia de movimientos precisos: {str(e)}")
    
    
    print("\nPrueba de comandos de posición completada.")
    return True

def main():
    """Función principal con opciones de línea de comandos"""
    parser = argparse.ArgumentParser(description="Pruebas del sistema de escaneo")
    parser.add_argument("command", nargs='?', default="homing",
                      choices=["homing", "video", "position"],
                      help="Comando a ejecutar: 'homing' para test de esquinas, 'video' para feed de video, 'position' para test de posicionamiento")
    
    args = parser.parse_args()
    
    try:
        if args.command == "video":
            start_video_feed()
        elif args.command == "position":
            test_position_commands()
        else:  # default es "homing"
            test_homing_and_corners()
    except KeyboardInterrupt:
        print("\n\nPrueba interrumpida por el usuario.")
    except Exception as e:
        print(f"\n\nError inesperado durante la prueba: {str(e)}")

if __name__ == "__main__":
    main()
    # python test_main.py video
    # python test_main.py homing
    # python test_main.py homing --visit-only
    # python test_main.py position