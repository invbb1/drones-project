from ultralytics import YOLO

# حمل نموذج YOLOv8n (خفيف وسريع)
model = YOLO("yolov8n.pt")

# درب النموذج على بيانات Roboflow
model.train(
    data="roboflow/data.yaml",  # مسار ملف البيانات
    epochs=15,                  # عدد التكرارات
    imgsz=640,                  # حجم الصورة
    batch=16,                   # حجم الدفعة - غيره حسب قوة جهازك
    name="drone_model_v1"       # اسم مجلد النتائج
)
model = YOLO("weights/elSbLGFAuz", task="detect")
