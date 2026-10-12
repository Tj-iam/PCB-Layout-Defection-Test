from ultralytics import YOLO

import os
import yaml
import matplotlib.pyplot as plt
from PIL import Image

dataset_path=r"C:\Users\tijan\OneDrive\Desktop\PCB Layout Defection Test";
yaml_path=os.path.join(dataset_path,"data.yaml")

with open(yaml_path, "r") as file:
    data = yaml.safe_load(file)

print(data)

model=YOLO("yolo11n.pt")

results = model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,
    workers=0,
)

metrics = model.val(data=yaml_path)
print(metrics)

results = model.predict(
    source=r"C:\Users\tijan\OneDrive\Desktop\PCB Layout Defection Test\test\images",
    save=True
)