import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps

# Formati di ingresso accettati (MPO è il formato con cui Pillow apre molte foto JPEG da smartphone)
ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP"}
# Altezza minima della persona, in rapporto all'altezza dell'immagine, per essere in primo piano
FOREGROUND_HEIGHT_RATIO = 0.45
# Area minima della persona, in rapporto all'area dell'immagine, per essere in primo piano
FOREGROUND_AREA_RATIO = 0.15
# Pixel di margine aggiunti a ogni riquadro rilevato
PAD = 4
# Confidenza minima YOLO per le persone
PERSON_MIN_CONF = 0.3
# Confidenza minima YOLO per gli oggetti
OBJECT_MIN_CONF = 0.5
# Confidenza minima del modello loghi
LOGO_MIN_CONF = 0.4
# Dimensione di inferenza YOLO per persone e oggetti (più alta aiuta con le persone piccole sullo sfondo)
YOLO_IMGSZ = 1280
# Lingue usate da Tesseract
OCR_LANG = "ita+eng"
# Confidenza minima OCR (0-100) per rilevare una parola
OCR_MIN_CONF = 40
# Confidenza minima OCR (0-100) per segnalare testo residuo nella verifica
VERIFY_OCR_MIN_CONF = 60
# Lunghezza minima di una parola per essere segnalata come testo residuo
VERIFY_OCR_MIN_LEN = 3
# Fattore di scala della ricerca volti Haar
FACE_SCALE_FACTOR = 1.1
# Vicini minimi richiesti per accettare un volto Haar
FACE_MIN_NEIGHBORS = 5
# Lato minimo in pixel di un volto
FACE_MIN_SIZE = 24
# Lato massimo dell'immagine usata per la ricerca volti (le immagini più grandi vengono ridotte)
FACE_MAX_SIDE = 1600
# Nome del modello YOLO COCO
MODEL_NAME = "yolov8n.pt"
# Cartella fuori dal repository dove viene scaricato il modello
MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path.home() / ".cache" / "privacy-redactor"))


# Contenitore dei modelli caricati all'avvio, con gli errori di caricamento
@dataclass
class Models:
    yolo: object = None
    logo: object = None
    face: object = None
    errors: list = field(default_factory=list)


# Carica YOLO, il modello loghi facoltativo e il Haar cascade; un errore non blocca gli altri
def load_models():
    models = Models()
    try:
        from ultralytics import YOLO
    except Exception as exc:
        models.errors.append(f"Caricamento ultralytics: {exc}")
        YOLO = None
    if YOLO is not None:
        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            models.yolo = YOLO(str(MODEL_DIR / MODEL_NAME))
        except Exception as exc:
            models.errors.append(f"Caricamento {MODEL_NAME}: {exc}")
        logo_path = os.environ.get("LOGO_MODEL", "").strip()
        if logo_path:
            try:
                models.logo = YOLO(logo_path)
            except Exception as exc:
                models.errors.append(f"Caricamento LOGO_MODEL: {exc}")
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            raise RuntimeError("file cascade non caricato")
        models.face = cascade
    except Exception as exc:
        models.errors.append(f"Caricamento Haar cascade: {exc}")
    return models


# Apre il file, corregge l'orientamento e ricrea l'immagine dai soli pixel, senza metadati
def load_image(path):
    with Image.open(path) as source:
        if source.format not in ALLOWED_FORMATS:
            raise ValueError(f"Formato non supportato: {source.format}. Ammessi: JPEG, PNG, WebP.")
        image = ImageOps.exif_transpose(source)
        if "A" in image.getbands() or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba)
        image = image.convert("RGB")
        return Image.frombytes("RGB", image.size, image.tobytes())


# Allarga il riquadro di PAD pixel e lo limita ai bordi dell'immagine
def pad_box(box, size):
    width, height = size
    x1, y1, x2, y2 = box
    return (
        max(0, int(math.floor(x1)) - PAD),
        max(0, int(math.floor(y1)) - PAD),
        min(width, int(math.ceil(x2)) + PAD),
        min(height, int(math.ceil(y2)) + PAD),
    )


# Crea il dizionario di un rilevamento con il riquadro già allargato
def make_detection(kind, box, label, default_on, size, conf=None):
    return {
        "kind": kind,
        "box": pad_box(box, size),
        "label": label,
        "default_on": default_on,
        "conf": conf,
    }


# Esegue un modello YOLO e restituisce classe, nome, confidenza e riquadro di ogni oggetto
def run_yolo(model, image, conf, imgsz=None):
    options = {"conf": conf, "verbose": False}
    if imgsz:
        options["imgsz"] = imgsz
    items = []
    for result in model.predict(source=image, **options):
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            items.append(
                {
                    "cls": cls_id,
                    "name": str(names[cls_id]),
                    "conf": float(box.conf[0]),
                    "box": tuple(float(v) for v in box.xyxy[0].tolist()),
                }
            )
    return items


# Dice se il riquadro di una persona va considerato in primo piano
def is_foreground(box, size):
    width, height = size
    x1, y1, x2, y2 = box
    box_height = y2 - y1
    box_area = (x2 - x1) * box_height
    return box_height >= FOREGROUND_HEIGHT_RATIO * height or box_area >= FOREGROUND_AREA_RATIO * width * height


# Rileva le persone (classe COCO 0) e le divide in primo piano e sfondo
def detect_persons(items, size):
    found = []
    for item in items:
        if item["cls"] != 0 or item["conf"] < PERSON_MIN_CONF:
            continue
        if is_foreground(item["box"], size):
            found.append(make_detection("persona", item["box"], "persona in primo piano", False, size, item["conf"]))
        else:
            found.append(make_detection("persona_sfondo", item["box"], "persona sullo sfondo", True, size, item["conf"]))
    return found


# Rileva gli oggetti COCO diversi dalle persone
def detect_objects(items, size):
    return [
        make_detection("oggetto", item["box"], item["name"], False, size, item["conf"])
        for item in items
        if item["cls"] != 0 and item["conf"] >= OBJECT_MIN_CONF
    ]


# Rileva i loghi con il modello YOLO personalizzato
def detect_logos(model, image):
    items = run_yolo(model, image, LOGO_MIN_CONF)
    return [make_detection("logo", item["box"], item["name"], True, image.size, item["conf"]) for item in items]


# Cerca volti frontali con Haar e restituisce i riquadri nelle coordinate dell'immagine originale
def raw_faces(cascade, image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    scale = min(1.0, FACE_MAX_SIDE / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=(FACE_MIN_SIZE, FACE_MIN_SIZE),
    )
    return [(x / scale, y / scale, (x + w) / scale, (y + h) / scale) for x, y, w, h in faces]


# Dice se il centro di un riquadro cade dentro uno qualsiasi dei riquadri indicati
def center_in_any(box, boxes):
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes)


# Rileva i volti che non appartengono a una persona in primo piano
def detect_faces(cascade, image, foreground):
    return [
        make_detection("volto", box, "volto", True, image.size)
        for box in raw_faces(cascade, image)
        if not center_in_any(box, foreground)
    ]


# Legge le parole con Tesseract, filtrando per confidenza e lunghezza minime
def ocr_words(image, min_conf, min_len=1):
    data = pytesseract.image_to_data(image, lang=OCR_LANG, output_type=pytesseract.Output.DICT)
    words = []
    for index, raw in enumerate(data["text"]):
        text = (raw or "").strip()
        if len(text) < min_len:
            continue
        try:
            conf = float(data["conf"][index])
        except (TypeError, ValueError):
            continue
        if conf < min_conf:
            continue
        left, top = data["left"][index], data["top"][index]
        box = (left, top, left + data["width"][index], top + data["height"][index])
        words.append({"text": text, "conf": conf, "box": box})
    return words


# Rileva il testo parola per parola
def detect_text(image):
    return [
        make_detection("testo", word["box"], word["text"], True, image.size, word["conf"] / 100)
        for word in ocr_words(image, OCR_MIN_CONF)
    ]


# Esegue un rilevatore; se fallisce registra l'errore e restituisce una lista vuota
def run_step(errors, name, func, *args):
    try:
        return func(*args)
    except Exception as exc:
        errors.append(f"{name}: {exc}")
        return []


# Esegue tutti i rilevatori e restituisce rilevamenti ed errori
def analyze(image, models):
    errors = list(models.errors)
    items = []
    if models.yolo is not None:
        try:
            items = run_yolo(models.yolo, image, min(PERSON_MIN_CONF, OBJECT_MIN_CONF), YOLO_IMGSZ)
        except Exception as exc:
            errors.append(f"Inferenza YOLO: {exc}")
    persons = run_step(errors, "Persone", detect_persons, items, image.size)
    foreground = foreground_boxes(persons)
    faces = []
    if models.face is not None:
        faces = run_step(errors, "Volti", detect_faces, models.face, image, foreground)
    text = run_step(errors, "Testo", detect_text, image)
    objects = run_step(errors, "Oggetti", detect_objects, items, image.size)
    logos = []
    if models.logo is not None:
        logos = run_step(errors, "Loghi", detect_logos, models.logo, image)
    return persons + faces + text + objects + logos, errors


# Estrae i riquadri delle persone in primo piano da una lista di rilevamenti
def foreground_boxes(detections):
    return [d["box"] for d in detections if d["kind"] == "persona"]


# Riempie di nero pieno i riquadri su una copia RGB e conta quelli applicati
def redact(image, boxes):
    pixels = np.array(image.convert("RGB"))
    height, width = pixels.shape[:2]
    applied = 0
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(width, int(x2)), min(height, int(y2))
        if x2 > x1 and y2 > y1:
            pixels[y1:y2, x1:x2] = 0
            applied += 1
    return Image.fromarray(pixels), applied


# Salva il PNG senza metadati in una directory temporanea fuori dal repository
def save_output(image):
    out_dir = tempfile.mkdtemp(prefix="privacy-redactor-")
    path = os.path.join(out_dir, "oscurata.png")
    image.save(path, format="PNG")
    return path


# Rilegge il file salvato e cerca testo residuo, volti residui e metadati
def verify(path, foreground, models):
    report = {"residual_text": [], "residual_faces": 0, "metadata": [], "errors": []}
    try:
        with Image.open(path) as saved:
            if saved.info:
                report["metadata"].append("info: " + ", ".join(sorted(str(key) for key in saved.info)))
            if len(saved.getexif()):
                report["metadata"].append("exif")
            if any("xmp" in str(key).lower() or str(key).startswith("XML:") for key in saved.info):
                report["metadata"].append("xmp")
            pixels = saved.convert("RGB")
    except Exception as exc:
        report["errors"].append(f"Lettura del file salvato: {exc}")
        return report
    try:
        report["residual_text"] = [w["text"] for w in ocr_words(pixels, VERIFY_OCR_MIN_CONF, VERIFY_OCR_MIN_LEN)]
    except Exception as exc:
        report["errors"].append(f"Verifica testo: {exc}")
    if models.face is not None:
        try:
            faces = raw_faces(models.face, pixels)
            report["residual_faces"] = sum(1 for face in faces if not center_in_any(face, foreground))
        except Exception as exc:
            report["errors"].append(f"Verifica volti: {exc}")
    return report


# Oscura i rilevamenti scelti, salva il PNG e restituisce immagine, percorso e rapporto di verifica
def process(image, selected, foreground, models):
    redacted, applied = redact(image, [d["box"] for d in selected])
    path = save_output(redacted)
    report = verify(path, foreground, models)
    report["boxes"] = applied
    return redacted, path, report
