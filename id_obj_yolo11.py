from ultralytics import YOLO
import os
import json
from datetime import datetime

# 1 Nap model yolo pretrained
model = YOLO("yolo11n.pt")

# 2 Nhap duong dan de them anh cho model
image_path = input("Nhập đường dẫn tới ảnh cần nhận diện: ").strip()
if not os.path.exists(image_path):
    print("Ảnh không tồn tại.")
    exit()

# 3 Vi tri thu muc de luu ket qua
save_dir = "/mnt/d/hcmus_i/lop_nmcntt/search_by_img_project/result_id_obj"
os.makedirs(save_dir, exist_ok=True)

# 4 Thoi gian hien tai
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # ví dụ: 20251025_170520

# 5 Thuc hien nhan dien
results = model.predict(source=image_path, save=False, device="cpu")

# 6 Lay du lieu da nhan dien duoc
data = []
for result in results:
    boxes = result.boxes.xyxy.tolist()        # bounding boxes [x1, y1, x2, y2]
    scores = result.boxes.conf.tolist()       # độ tin cậy
    class_ids = result.boxes.cls.tolist()     # chỉ số lớp
    names = result.names                      # tên lớp (ID → tên)

    for box, score, cls_id in zip(boxes, scores, class_ids):
        data.append({
            "class": names[int(cls_id)],
            "confidence": round(score, 3),
            "bbox": [round(x, 2) for x in box]
        })

# 7 Dua du lieu vao file JSON
output = {
    "image": os.path.basename(image_path),
    "objects": data,
    "timestamp": timestamp
}

# Tạo tên file JSON: timestamp + tên gốc
filename = os.path.splitext(os.path.basename(image_path))[0]
json_filename = f"{timestamp}_{filename}.json"
json_path = os.path.join(save_dir, json_filename)

# Ghi ra file JSON
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print(f"Kết quả đã lưu tại: {os.path.abspath(json_path)}")

