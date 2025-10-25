from flask import Flask, request
import json, datetime, os
from flask_cors import CORS
from PIL import Image
import pytesseract
import yake

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "✅ Flask OCR + Yake API is running!"

@app.route('/ocr', methods=['POST'])
def ocr_image():
    if 'file' not in request.files:
        return {"error": "No file provided"}, 400

    file = request.files['file']
    img = Image.open(file.stream)
    text = pytesseract.image_to_string(img, lang='vi+eng').strip()

    if not text:
        return {"error": "No text detected in image"}, 400

    kw_extractor = yake.KeywordExtractor(lan="vi", n=1, top=10)
    keywords = kw_extractor.extract_keywords(text)
    keyword_list = [kw for kw, score in keywords]

    result = {"text": text, "keywords": keyword_list}

    # 🔹 Tạo thư mục lưu & tên file theo thời gian
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "/mnt/d/hcmus_i/lop_nmcntt/search_by_img_project/result"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/result_{timestamp}.json"

    # 🔹 Ghi file JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"✅ Saved: {output_path}")

    return result

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

