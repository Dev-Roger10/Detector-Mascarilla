"""
app.py — Detector de Mascarillas (local)
=========================================
Ejecutar:
    pip install flask flask-cors tensorflow opencv-python pillow numpy gdown
    python app.py

Luego abrir:  http://localhost:5000
"""

import os, io, urllib.request, base64
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ── TensorFlow / Keras ────────────────────────────────────────────────────
try:
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    TF_OK = True
except ImportError:
    TF_OK = False
    print("⚠  TensorFlow no instalado. Instalar con:  pip install tensorflow")

# ── Rutas ─────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH = os.path.join(BASE, "model", "mask_detector.keras")
HAAR_PATH   = os.path.join(BASE, "model", "haarcascade_frontalface_default.xml")
HAAR_URL    = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
DRIVE_URL   = "https://drive.google.com/drive/folders/113Ccinp1P4rvN9DfVBkDLyS3rEJBxPJM?usp=drive_link"
RUTA_TEMP   = os.path.join(BASE, "_dataset_drive")
EXT         = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

os.makedirs(os.path.join(BASE, "model"), exist_ok=True)

app = Flask(__name__)
CORS(app)

# ── Singletons ────────────────────────────────────────────────────────────
_modelo   = None
_detector = None

def get_detector():
    global _detector
    if _detector: return _detector
    if not os.path.exists(HAAR_PATH):
        print("  Descargando Haar Cascade…")
        urllib.request.urlretrieve(HAAR_URL, HAAR_PATH)
        print("  ✓ Haar Cascade descargado")
    _detector = cv2.CascadeClassifier(HAAR_PATH)
    return _detector

def build_model():
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
    base.trainable = False
    m = keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(2, activation="softmax"),  # 0=con  1=sin
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def get_model():
    global _modelo
    if _modelo: return _modelo
    if not TF_OK: return None
    if os.path.exists(MODELO_PATH):
        print(f"  Cargando modelo: {MODELO_PATH}")
        _modelo = keras.models.load_model(MODELO_PATH)
        print("  ✓ Modelo cargado")
    else:
        print("  ⚠  No se encontró modelo entrenado.")
        print("     Usa el botón 'Descargar dataset y entrenar' en la web.")
        _modelo = build_model()
    return _modelo

def prep_roi(img_bgr):
    det  = get_detector()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = det.detectMultiScale(gray, 1.1, 5, minSize=(50,50))
    found = len(faces) > 0
    if found:
        x,y,w,h = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
        roi = img_bgr[y:y+h, x:x+w]
    else:
        roi = img_bgr
    roi = cv2.resize(roi, (224,224))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = preprocess_input(roi.astype("float32"))
    return roi, found

# ── Rutas Flask ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/model_status")
def model_status():
    return jsonify({"trained": os.path.exists(MODELO_PATH)})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No se recibió imagen"}), 400
    file = request.files["image"]
    nparr = np.frombuffer(file.read(), np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "No se pudo leer la imagen"}), 400
    if not TF_OK:
        return jsonify({"error": "TensorFlow no instalado"}), 500

    modelo = get_model()
    roi, found = prep_roi(img)
    probs  = modelo.predict(np.expand_dims(roi, 0), verbose=0)[0]
    clase  = int(np.argmax(probs))
    conf   = float(probs[clase]) * 100

    return jsonify({
        "label":         "con_mascarilla" if clase == 0 else "sin_mascarilla",
        "confidence":    round(conf, 1),
        "prob_con":      round(float(probs[0]) * 100, 1),
        "prob_sin":      round(float(probs[1]) * 100, 1),
        "face_detected": found,
    })

@app.route("/train", methods=["POST"])
def train():
    if not TF_OK:
        return jsonify({"error": "TensorFlow no instalado"}), 500
    files = request.files.getlist("images[]")
    X, y = [], []
    for f in files:
        n = f.filename.lower()
        if   "con" in n: lbl = 0
        elif "sin" in n: lbl = 1
        else: continue
        nparr = np.frombuffer(f.read(), np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            roi, _ = prep_roi(img)
            X.append(roi); y.append(lbl)
    if len(X) < 4:
        return jsonify({"error": "Necesitas al menos 4 imágenes etiquetadas (nombre con 'con' o 'sin')"}), 400
    global _modelo
    _modelo = build_model()
    hist = _modelo.fit(np.array(X), np.array(y), epochs=10, batch_size=4, verbose=0)
    _modelo.save(MODELO_PATH)
    acc = hist.history["accuracy"][-1]
    return jsonify({"message": "ok", "accuracy": round(float(acc)*100, 1), "samples": len(X)})

@app.route("/train_from_drive", methods=["POST"])
def train_from_drive():
    if not TF_OK:
        return jsonify({"error": "TensorFlow no instalado"}), 500
    try:
        import gdown
    except ImportError:
        return jsonify({"error": "Instala gdown:  pip install gdown"}), 500

    os.makedirs(RUTA_TEMP, exist_ok=True)
    try:
        gdown.download_folder(url=DRIVE_URL, output=RUTA_TEMP, quiet=False, use_cookies=False)
    except Exception as e:
        return jsonify({"error": f"Error al descargar de Drive: {e}"}), 500

    archivos = []
    for r, _, fs in os.walk(RUTA_TEMP):
        for f in fs:
            if os.path.splitext(f)[1].lower() in EXT:
                archivos.append(os.path.join(r, f))
    if not archivos:
        return jsonify({"error": "No se encontraron imágenes en la carpeta de Drive"}), 400

    X, y = [], []
    for ruta in archivos:
        n = os.path.basename(ruta).lower()
        if   "con" in n: lbl = 0
        elif "sin" in n: lbl = 1
        else: continue
        img = cv2.imread(ruta)
        if img is not None:
            roi, _ = prep_roi(img)
            X.append(roi); y.append(lbl)

    if len(X) < 4:
        return jsonify({"error": "Imágenes sin etiqueta en nombre (necesita 'con' o 'sin')"}), 400

    global _modelo
    _modelo = build_model()
    hist = _modelo.fit(np.array(X), np.array(y), epochs=10, batch_size=4, verbose=1)
    _modelo.save(MODELO_PATH)
    acc = hist.history["accuracy"][-1]
    return jsonify({"message": "ok", "accuracy": round(float(acc)*100, 1), "samples": len(X)})

# ── arranque ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "─"*50)
    print("  Detector de Mascarillas")
    print("  http://localhost:5000")
    print("─"*50)
    if os.path.exists(MODELO_PATH):
        print("  ✓ Modelo encontrado — listo para detectar")
    else:
        print("  ⚠  Sin modelo entrenado")
        print("     Usa el botón 'Descargar dataset de Drive y entrenar'")
        print("     o sube tus propias imágenes en la sección de entrenamiento")
    print("─"*50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
