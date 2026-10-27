from pathlib import Path
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:27:20 2024

@author: AM4
"""
# Импортируем библиотеки
import os
from pathlib import Path
# Все относительные пути считаются от папки этой лабораторной.
os.chdir(Path(__file__).resolve().parent)

# Для проверки без окон: переменная среды LAB6_NO_GUI=1.
SHOW_WINDOWS = os.environ.get('LAB6_NO_GUI') != '1'
def show_image(title, image):
    if SHOW_WINDOWS:
        cv2.imshow(title, image)
        return cv2.waitKey(1)
    return -1

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import ultralytics
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import torch
torch.set_num_threads(2)

# Проверяем что доступно из оборудования
ultralytics.checks()

# Создаем модель из файла с предобученными весами. 
#!!! При первом вызове загружает веса, требуется интернет/
model = YOLO("yolov8n.pt")

# Запускаем модель. На вход подаем изображения с диска, указав к нему путь.
results = model("image_1.jpg")

# Достаем результаты модели
result = results[0]

# Выводим результаты на экран при помощи OpenCV
show_image("YOLOv8", result.plot())

# Если не работате вывод OpenCV, можно использовать matplotlib
plt.imshow(result.plot()[:, :, ::-1])

# В результатах содержится много полезной информации:
boxes = result.boxes.cpu()       # Рамки объектов по умолчанию в формате YOLO
print(boxes)
boxes = result.boxes.xyxy.cpu()  # Координаты рамок можно преобразовать в пиксели
print(boxes)

# Вероятности для каждого из обнаруженных объектов
# чем больше, тем сеть увереннее что это именно этот объект
confidences = result.boxes.conf
print(confidences)

classes = result.boxes.cls # Номера классов для каждого объекта
print(classes)
class_names = result.names # Сами названия классов  
print(class_names)

# Даже исходное изображение
img = result.orig_img
plt.imshow(img[:, :, ::-1])

# напишем свою функцию для отрисовки прямоугольников на изображении
def draw_bboxes(image, results):
    boxes = results[0].boxes.cpu()
    orig_h, orig_w = results[0].orig_shape # размеры изображения
    class_names = results[0].names             # названия классов  
    for box in boxes:    
        # достаем координаты, название класса и скор
        class_idx = int(box.cls.item())
        confidence = box.conf
        
        # Будем отрисовывать только то в чем сеть хорошо уверена
        if confidence>0.7:
            x1, y1, x2, y2 = box.xyxy[0].numpy()
            # рисуем прямоугольник
            cv2.rectangle(
                image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2, cv2.LINE_AA
            )
            
            # подписываем название класса
            cv2.putText(
                image, class_names[class_idx], (int(x1), int(y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
        
    return image

# вызовем функцию и выведем на экран то чот получилось
annotated_img = draw_bboxes(img.copy(), results)
show_image("YOLOv8", annotated_img)


# Теперь попробуем обучить на собственном датасете
# Набор: 25 изображений онигири, 69 рамок; источник и разбиение в data/manifest.json.

# В masked.yaml пути указаны относительно папки этой лабораторной.
# обучение может занять много времени, особенно на CPU
model = YOLO("yolov8s.pt")
results = model.train(data="masked.yaml", model="yolov8s.pt", epochs=20, batch=4, imgsz=320, workers=0, device="cpu",
                      project=str(Path(__file__).resolve().parent / 'runs'), name='onigiri', exist_ok=True, val=True, verbose=True)

results = model("data/images/test/onigiri_014.jpg", conf=0.85, iou=0.3)

# посмотрим что получилось
result = results[0]
show_image("YOLOv8", result.plot())
cv2.imwrite("result_onigiri.jpg", result.plot())

# Проверка на отдельной тестовой выборке.
metrics = model.val(data="masked.yaml", split="test", imgsz=320, workers=0, device="cpu",
                    project=str(Path(__file__).resolve().parent / "runs"), name="test", exist_ok=True)
print(metrics.results_dict)

# Попробуем обработать видео

# Открываем видеофайл
video_path = "masktrack.mp4"
# Учебное видео содержит людей: для него используем базовую модель.
video_model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(video_path) if SHOW_WINDOWS else cv2.VideoCapture()


while cap.isOpened():
    # Считываем кадр
    success, frame = cap.read()

    if success:
        # Если кадр прочитался успешно, запускаем модель
        results = video_model(frame)
        
        result = results[0]
        key = show_image("YOLOv8", result.plot())

        # По нажатию "q" будем выходить из цикла
        if key & 0xFF == ord("q"):
            break
    else:
        break

# Освобождаем поток видео и закрываем окно отображения
cap.release()
cv2.destroyAllWindows()