import os, io, urllib.request
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    import torchvision.transforms as transforms
    TF_OK = True
except ImportError:
    TF_OK = False
    print("⚠  PyTorch no instalado. Instalar con:  pip install torch torchvision")

# ── Rutas ─────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH = os.path.join(BASE, "model", "mask_detector.pth")
HAAR_PATH   = os.path.join(BASE, "model", "haarcascade_frontalface_default.xml")
HAAR_URL    = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
DRIVE_URL   = "https://drive.google.com/drive/folders/113Ccinp1P4rvN9DfVBkDLyS3rEJBxPJM?usp=drive_link"
RUTA_TEMP   = os.path.join(BASE, "_dataset_drive")
EXT         = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

os.makedirs(os.path.join(BASE, "model"), exist_ok=True)

app = Flask(__name__)
CORS(app)

DEVICE = torch.device("cpu") if TF_OK else None

# Normalización equivalente a MobileNetV2 preprocess_input
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
]) if TF_OK else None

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
    base = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in base.parameters():
        param.requires_grad = False
    base.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(base.last_channel, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2),
    )
    base.to(DEVICE)
    return base

def get_model():
    global _modelo
    if _modelo: return _modelo
    if not TF_OK: return None
    if os.path.exists(MODELO_PATH):
        print(f"  Cargando modelo: {MODELO_PATH}")
        _modelo = build_model()
        _modelo.load_state_dict(torch.load(MODELO_PATH, map_location=DEVICE))
        _modelo.eval()
        print("  ✓ Modelo cargado")
    else:
        print("  ⚠  No se encontró modelo entrenado.")
        print("     Usa el botón 'Descargar dataset y entrenar' en la web.")
        _modelo = build_model()
        _modelo.eval()
    return _modelo

def prep_roi(img_bgr):
    det   = get_detector()
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = det.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    found = len(faces) > 0
    if found:
        x, y, w, h = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
        roi = img_bgr[y:y+h, x:x+w]
    else:
        roi = img_bgr
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    tensor  = IMG_TRANSFORM(Image.fromarray(roi_rgb))
    return tensor, found

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
    file  = request.files["image"]
    nparr = np.frombuffer(file.read(), np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "No se pudo leer la imagen"}), 400
    if not TF_OK:
        return jsonify({"error": "PyTorch no instalado"}), 500

    modelo        = get_model()
    tensor, found = prep_roi(img)
    with torch.no_grad():
        output = modelo(tensor.unsqueeze(0).to(DEVICE))
        probs  = torch.softmax(output, dim=1)[0].cpu().numpy()

    clase = int(np.argmax(probs))
    conf  = float(probs[clase]) * 100

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
        return jsonify({"error": "PyTorch no instalado"}), 500
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
            tensor, _ = prep_roi(img)
            X.append(tensor); y.append(lbl)
    if len(X) < 4:
        return jsonify({"error": "Necesitas al menos 4 imágenes etiquetadas (nombre con 'con' o 'sin')"}), 400

    global _modelo
    _modelo = build_model()
    _modelo.train()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, _modelo.parameters()), lr=1e-3
    )
    criterion = nn.CrossEntropyLoss()
    X_t = torch.stack(X).to(DEVICE)
    y_t = torch.tensor(y, dtype=torch.long).to(DEVICE)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    for _ in range(10):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(_modelo(xb), yb)
            loss.backward()
            optimizer.step()

    _modelo.eval()
    with torch.no_grad():
        preds = torch.argmax(_modelo(X_t), dim=1)
        acc   = (preds == y_t).float().mean().item()

    torch.save(_modelo.state_dict(), MODELO_PATH)
    return jsonify({"message": "ok", "accuracy": round(acc * 100, 1), "samples": len(X)})

@app.route("/train_from_drive", methods=["POST"])
def train_from_drive():
    if not TF_OK:
        return jsonify({"error": "PyTorch no instalado"}), 500
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
            tensor, _ = prep_roi(img)
            X.append(tensor); y.append(lbl)

    if len(X) < 4:
        return jsonify({"error": "Imágenes sin etiqueta en nombre (necesita 'con' o 'sin')"}), 400

    global _modelo
    _modelo = build_model()
    _modelo.train()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, _modelo.parameters()), lr=1e-3
    )
    criterion = nn.CrossEntropyLoss()
    X_t = torch.stack(X).to(DEVICE)
    y_t = torch.tensor(y, dtype=torch.long).to(DEVICE)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    for _ in range(10):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(_modelo(xb), yb)
            loss.backward()
            optimizer.step()

    _modelo.eval()
    with torch.no_grad():
        preds = torch.argmax(_modelo(X_t), dim=1)
        acc   = (preds == y_t).float().mean().item()

    torch.save(_modelo.state_dict(), MODELO_PATH)
    return jsonify({"message": "ok", "accuracy": round(acc * 100, 1), "samples": len(X)})

# ── arranque ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("─"*50)
    if os.path.exists(MODELO_PATH):
        print("  ✓ Modelo encontrado — listo para detectar")
    else:
        print("  ⚠  Sin modelo entrenado")
        print("     Usa el botón 'Descargar dataset de Drive y entrenar'")
        print("     o sube tus propias imágenes en la sección de entrenamiento")
    print("─"*50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=port)
