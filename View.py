# Importamos las librerías
from ultralytics import YOLO
import cv2

# Leer nuestro modelo
model = YOLO("best2.pt")

# Realizar VideoCaptura
cap = cv2.VideoCapture(1)

# Definir los objetos que deseas contar y sus categorías de reciclaje (agrega los nombres de los objetos y categorías)
object_categories = {
    "botella": {"reciclable": True, "categoria": "Plástico"},
    "papel": {"reciclable": True, "categoria": "Papel"},
    "otros": {"reciclable": False, "categoria": "No reciclable"}
}

# Inicializar un contador para cada objeto
object_counts = {obj: 0 for obj in object_categories}

# Bucle
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()
    
    # Leemos resultados
    resultados = model.predict(frame, imgsz=640, conf=0.90)
  
    # Mostramos resultados
    anotaciones = resultados[0].plot()
    
    # Mostramos nuestros fotogramas
    cv2.imshow("DETECCION Y SEGMENTACION", anotaciones)
    
    resultado = resultados[0]
    num_detections = len(resultado.boxes)
    
    if num_detections > 0:
        # Verificar cada detección para contar los objetos específicos
        for i in range(num_detections):
            box = resultado.boxes[i]
            detected_object = resultado.names[int(box.cls)]
            detected_category = object_categories.get(detected_object, object_categories["otros"])
            
            object_counts[detected_object] += 1
            
    # Cerrar nuestro programa
    if cv2.waitKey(1) == 27:
        break

# Imprimir el número de detecciones para cada objeto
for obj, count in object_counts.items():
    category = object_categories[obj]["categoria"]
    reciclable = "Reciclable" if object_categories[obj]["reciclable"] else "No reciclable"
    print(f"Se detectaron {count} '{obj}' ({category}) - {reciclable}")

cap.release()
cv2.destroyAllWindows()
