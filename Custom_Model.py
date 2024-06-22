from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # train a new model

results = model.train(data="data.yaml", epochs=30, lr0=0.01, imgsz=640, batch=16,  momentum=0.937)
model.val()

model.save("farm_animal_detector.pt")



