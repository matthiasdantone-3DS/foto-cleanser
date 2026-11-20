import re
import threading

import gradio as gr
from PIL import ImageDraw, ImageFont

import pipeline

# Modelli caricati una sola volta all'avvio
MODELS = pipeline.load_models()
# Evita che due richieste usino i modelli nello stesso momento
MODEL_LOCK = threading.Lock()

# Colore del riquadro per ogni tipo di rilevamento
COLORS = {
    "testo": (255, 165, 0),
    "persona_sfondo": (255, 0, 0),
    "volto": (255, 0, 255),
    "oggetto": (0, 160, 255),
    "logo": (150, 60, 240),
    "persona": (0, 200, 0),
    "manuale": (255, 235, 0),
}

# Legenda con un quadratino colorato per ogni tipo
LEGEND_HTML = (
    "<div>"
    + "".join(
        '<span style="display:inline-block;margin:0 14px 4px 0">'
        f'<span style="display:inline-block;width:12px;height:12px;margin-right:5px;background:rgb{color}"></span>{kind}'
        "</span>"
        for kind, color in COLORS.items()
    )
    + "<br>Linea spessa = area selezionata per l'oscuramento.</div>"
)


# Carica un font leggibile, con ripiego sul font predefinito
def load_font(size):
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


# Neutralizza i caratteri speciali di Markdown nel testo letto dall'OCR
def md_escape(text):
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~])", r"\\\1", text)


# Converte i valori della CheckboxGroup (stringhe) in un insieme di id numerici
def to_ids(values):
    return {int(v) for v in values or []}


# Costruisce l'etichetta di un rilevamento: id, tipo, label e confidenza
def det_label(det):
    conf = f" · {det['conf'] * 100:.0f}%" if det.get("conf") is not None else ""
    return f"#{det['id']} · {det['kind']} · {det['label']}{conf}"


# Aggiorna la CheckboxGroup con tutte le detection e la selezione indicata
def checkbox_update(dets, selected):
    return gr.CheckboxGroup(choices=[(det_label(d), str(d["id"])) for d in dets], value=selected)


# Restituisce gli id dei rilevamenti selezionati in partenza
def default_selection(dets):
    return [str(d["id"]) for d in dets if d["default_on"]]


# Disegna sull'immagine i riquadri colorati per tipo, con linea spessa per quelli selezionati
def draw_preview(image, dets, selected):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    base = max(2, round(max(canvas.size) / 500))
    font = load_font(max(12, base * 6))
    for det in dets:
        color = COLORS.get(det["kind"], (255, 255, 255))
        x1, y1, x2, y2 = det["box"]
        width = base * 3 if det["id"] in selected else base
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        if det["kind"] != "testo":
            tag = f"#{det['id']}"
            draw.rectangle(draw.textbbox((x1, y1), tag, font=font), fill=color)
            draw.text((x1, y1), tag, fill=(0, 0, 0), font=font)
    return canvas


# Testo informativo con dimensioni dell'immagine, numero di aree ed eventuali errori dei rilevatori
def info_text(state):
    if state is None:
        return ""
    width, height = state["image"].size
    lines = [f"Immagine: {width}×{height} px. Aree rilevate: {len(state['dets'])}. Le coordinate manuali sono in pixel."]
    lines += [f"⚠️ {md_escape(error)}" for error in state["errors"]]
    return "\n\n".join(lines)


# Valori che azzerano tutti i componenti dello stato
def empty_outputs():
    return None, None, gr.CheckboxGroup(choices=[], value=[]), "", None, "", None


# Analizza la foto caricata e prepara anteprima, selezione e stato
def on_file(path):
    if not path:
        return empty_outputs()
    try:
        image = pipeline.load_image(path)
    except Exception as exc:
        return None, None, gr.CheckboxGroup(choices=[], value=[]), f"Errore: {md_escape(str(exc))}", None, "", None
    with MODEL_LOCK:
        dets, errors = pipeline.analyze(image, MODELS)
    for index, det in enumerate(dets):
        det["id"] = index
    state = {"image": image, "dets": dets, "errors": errors}
    selected = default_selection(dets)
    preview = draw_preview(image, dets, to_ids(selected))
    return state, preview, checkbox_update(dets, selected), info_text(state), None, "", None


# Aggiunge un'area manuale, già selezionata, e ridisegna l'anteprima
def on_add(x1, y1, x2, y2, state, selected):
    if state is None:
        return None, None, gr.CheckboxGroup(choices=[], value=[]), "Carica prima una foto."
    dets = state["dets"]
    ids = to_ids(selected)
    if None in (x1, y1, x2, y2):
        message = "Compila tutte e quattro le coordinate."
    else:
        width, height = state["image"].size
        left, right = sorted((int(x1), int(x2)))
        top, bottom = sorted((int(y1), int(y2)))
        left, top = max(0, left), max(0, top)
        right, bottom = min(width, right), min(height, bottom)
        if right - left < 1 or bottom - top < 1:
            message = "Area non valida: controlla le coordinate."
        else:
            new_id = max((d["id"] for d in dets), default=-1) + 1
            det = {
                "id": new_id,
                "kind": "manuale",
                "box": (left, top, right, bottom),
                "label": "area manuale",
                "default_on": True,
                "conf": None,
            }
            dets = dets + [det]
            state = {**state, "dets": dets}
            ids.add(new_id)
            message = f"Area #{new_id} aggiunta."
    selected_values = [str(d["id"]) for d in dets if d["id"] in ids]
    preview = draw_preview(state["image"], dets, ids)
    return state, preview, checkbox_update(dets, selected_values), f"{info_text(state)}\n\n{message}"


# Ridisegna l'anteprima quando l'utente cambia la selezione
def on_select(state, selected):
    if state is None:
        return None
    return draw_preview(state["image"], state["dets"], to_ids(selected))


# Costruisce il riepilogo dell'oscuramento e della verifica
def summary_text(report, detection_errors):
    words = report["residual_text"]
    if words:
        shown = ", ".join(md_escape(word) for word in words[:15])
        residual_text = f"{len(words)} parole ({shown})"
    else:
        residual_text = "nessuno"
    metadata = "; ".join(report["metadata"]) if report["metadata"] else "nessuno"
    lines = [
        f"- Box oscurati: {report['boxes']}",
        f"- Testo residuo: {residual_text}",
        f"- Volti residui: {report['residual_faces']}",
        f"- Metadati: {md_escape(metadata)}",
    ]
    lines += [f"- Errore: {md_escape(error)}" for error in detection_errors + report["errors"]]
    return "\n".join(lines)


# Oscura le aree selezionate, verifica il risultato e prepara il file da scaricare
def on_redact(state, selected):
    if state is None:
        return None, "Carica prima una foto.", None
    ids = to_ids(selected)
    chosen = [d for d in state["dets"] if d["id"] in ids]
    try:
        with MODEL_LOCK:
            redacted, path, report = pipeline.process(
                state["image"], chosen, pipeline.foreground_boxes(state["dets"]), MODELS
            )
    except Exception as exc:
        return None, f"Errore durante l'oscuramento: {md_escape(str(exc))}", None
    return redacted, summary_text(report, state["errors"]), path


# Svuota lo stato, il file caricato, i campi manuali e i risultati
def on_reset():
    return (None,) + empty_outputs() + (None, None, None, None)


with gr.Blocks(title="Privacy Redactor") as demo:
    state = gr.State(None)
    gr.Markdown(
        "# Privacy Redactor\n"
        "Carica una foto, scegli le aree da oscurare e scarica il PNG ripulito. "
        "Le persone in primo piano restano visibili; l'oscuramento è sempre nero pieno."
    )
    file_in = gr.File(
        label="Foto (JPEG, PNG, WebP)",
        file_count="single",
        file_types=[".jpg", ".jpeg", ".png", ".webp"],
        type="filepath",
    )
    info = gr.Markdown()
    preview = gr.Image(label="Aree rilevate", type="pil", format="png", interactive=False)
    gr.HTML(LEGEND_HTML)
    checks = gr.CheckboxGroup(label="Aree da oscurare", choices=[], value=[])
    gr.Markdown("**Aggiungi un'area manuale** (pixel dell'immagine originale)")
    with gr.Row():
        x1 = gr.Number(label="x1", precision=0)
        y1 = gr.Number(label="y1", precision=0)
        x2 = gr.Number(label="x2", precision=0)
        y2 = gr.Number(label="y2", precision=0)
        add_btn = gr.Button("Aggiungi")
    with gr.Row():
        redact_btn = gr.Button("Oscura", variant="primary")
        reset_btn = gr.Button("Ricomincia")
    out_image = gr.Image(label="Risultato", type="pil", format="png", interactive=False)
    summary = gr.Markdown()
    out_file = gr.File(label="Immagine oscurata (PNG)", interactive=False)

    outputs = [state, preview, checks, info, out_image, summary, out_file]
    file_in.change(on_file, inputs=file_in, outputs=outputs)
    add_btn.click(on_add, inputs=[x1, y1, x2, y2, state, checks], outputs=[state, preview, checks, info])
    checks.input(on_select, inputs=[state, checks], outputs=preview)
    redact_btn.click(on_redact, inputs=[state, checks], outputs=[out_image, summary, out_file])
    reset_btn.click(on_reset, inputs=None, outputs=[file_in] + outputs + [x1, y1, x2, y2])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
