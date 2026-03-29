from ultralytics import YOLO


model = YOLO("yolo11m.pt")

results = model.train(
    data="data.yaml",
    batch=0.7,
    dropout=0.2,
    epochs=500,
    imgsz=640,
    device=0,
    cache=False,
    plots=True,
    save=True,
    resume=False,
)
