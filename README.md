# Privacy Redactor

App web (Gradio) per oscurare informazioni sensibili in singole foto. Uso personale: nessuna pubblicazione automatica, nessun servizio a pagamento, nessun database.

Regole di oscuramento:

- persone in primo piano: visibili, volto incluso
- persone sullo sfondo: oscurate per intero
- volti non attribuibili a una persona in primo piano: oscurati
- testo: oscurato ovunque venga rilevato
- loghi e oggetti: proposti, da confermare nella revisione
- riempimento sempre nero pieno, nessuna sfocatura o pixelatura

## Avvio in Codespaces

1. Apri il repository su GitHub.
2. "Code" > "Codespaces" > "Create codespace".
3. Al termine dell'installazione, nel terminale:

```
python app.py
```

4. Apri la porta 7860 dalla scheda "Ports".

Al primo avvio Ultralytics scarica `yolov8n.pt` (in `~/.cache/privacy-redactor`, fuori dal repository).

## Avvio in locale

Installa prima Tesseract:

```
brew install tesseract tesseract-lang
```

Su Linux: `sudo apt-get install -y tesseract-ocr tesseract-ocr-ita`

Poi:

```
pip install -r requirements.txt
python app.py
```

Apri http://localhost:7860.

## Avviso privacy

In Codespaces le immagini passano dai server GitHub: per foto sensibili usa la modalità locale. Gradio conserva copie temporanee dei file caricati nella cartella temporanea del sistema: eliminale a fine lavoro.

## Loghi (LOGO_MODEL)

Il rilevamento loghi è facoltativo. Indica un modello YOLO personalizzato (`.pt`) con la variabile d'ambiente:

```
export LOGO_MODEL=/percorso/al/modello.pt
python app.py
```

In Codespaces puoi impostarla come variabile d'ambiente prima di lanciare l'app. Se `LOGO_MODEL` non è impostata, i loghi non vengono rilevati e non compare alcun errore.

## Limiti noti

- falsi negativi su testo piccolo o sfocato
- volti di profilo non rilevati (Haar cascade frontale)
- persone parzialmente visibili classificate in modo impreciso

Controlla sempre il risultato: la verifica automatica non sostituisce la revisione manuale.
