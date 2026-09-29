
from ultralytics import YOLO


model = YOLO(r'Nano-UpNet/cell_type_recognition/ultralytics/runs/segment/train/weights/best.pt',task='segment') 

model.predict(source=r'Nano-UpNet/cell_type_recognition/ultralytics/datasets/test_images',save=True,show=True)