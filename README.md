# PDF Table Extractor API

API REST construida con **FastAPI** que recibe archivos PDF en adición de un string asociado a la marca (idealmente rockford, hp, crocs), en donde se clasifica cada página mediante una red neuronal convolucional (CNN) exportada a ONNX, y extrae las tablas de las páginas identificadas como *Packing List* usando `img2table` en complemento a PaddleOCRv4.

---
## Tabla de contenidos
- [Descripción general](#descripción-general)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Uso](#uso)
  - [Iniciar el servidor](#iniciar-el-servidor)
  - [Endpoints](#endpoints)
- [Modelos](#modelos)
  - [CNN (ONNX)](#cnn-onnx)
  - [OCR](#ocr)
- [Respuesta de ejemplo](#respuesta-de-ejemplo)
- [Categorías de clasificación](#categorías-de-clasificación)
---

## Descripción general

El flujo de procesamiento es el siguiente:

1. Se recibe un archivo PDF y un string asociada a la marca vía HTTP POST.
2. Cada página del PDF se convierte en imagen (300 DPI).
3. La imagen es redimensionada a un tamaño de 224×224 pixeles y pasada por la CNN para clasificarla.
4. Si la página se clasifica como **PL** (Packing List), se extrae el contenido de sus tablas mediante `img2table` + PaddleOCR.
5. Se retorna la clasificación de cada página y las tablas extraídas en formato JSON.

---

## Requisitos

- Python 3.10.x
- Librerias principales:

```
fastapi
uvicorn
pdfplumber
pymupdf
onnxruntime
img2table[paddle]
numpy
```

> El modelo CNN debe estar presente en `cnn_module/onnx_models/` en formato `.onnx` con el formato de nombre `{accuracy}_{mse}.onnx`.

En caso de no tener las librerias necesarias instaladas, mediante pip hacer.
```
pip install fastapi uvicorn pdfplumber pymupdf onnxruntime "img2table[paddle]" numpy
```

O usar el archivo `requierements.txt` que contiene las versiones especificas usadas para este proyecto
```
pip install -r requierements.txt
```

Opcionalmente, en el caso de querer usar el motor de EasyOCR.
```
pip install "img2table[easyocr]"
```

---
## Estructura del proyecto
```
Extraccion_datos_PL/
├── main.py
├── main_functions.py
├── cnn_module/
|   ├── onnx_models/
|   ├── Train_data/
|   |   ├── Anexo_Corrección/
|   |   ├── Bill_of_Lading/
|   |   ├── Carrier/
|   |   ├── Nada/
|   |   └── Packing_list/
|   ├── Train_PDF/
|   ├── cnn_creation.py
|   └── get_images.py
├── Streamlit_interface/
|   └── inference_page.py
├── README.md
└── requierements.txt

```
---
### Para iniciar el servidor

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

#### `GET /health`

Verifica que el servidor está en línea y los modelos están cargados.

**Respuesta:**
```json
{ "status": "ok" }
```
```
{
  "SKU": [
    "878667",
    "7808774365287",
    "7808774365263",
    "7808774365240",
    "7806774365256",
    "7808774365270",
    "7808774365232"
  ],
  "TotalCases": [
    "23",
    "3",
    "7",
    "13",
    "7",
    "4",
    null
  ],
  "TotalPairs": [
    "138",
    "18",
    "42",
    "78",
    "42",
    "24",
    "6"
  ],
  "Pairs/Case": [
    "6",
    "6",
    "6",
    "6",
    "6",
    "6",
    "6"
  ]
}
```
---

#### `POST /process-pdf`

Procesa el archivo PDF y retorna la clasificación de páginas y las tablas extraídas.

**Request:**

- `Content-Type: multipart/form-data`
- Campo: `file` → archivo `.pdf`

**Respuesta:**
---

## Modelos

### CNN (ONNX)

- Ubicación: `cnn_module/onnx_models/`
- Formato de nombre: `{accuracy}_{mse}.onnx`
- Entrada: imagen RGB de 224×224 px, batch size 1, dtype `float32`
- Salida: vector de probabilidades sobre 5 clases
- Se carga automáticamente el modelo con mayor `accuracy` al iniciar el servidor

### OCR

El OCR utilizado por defecto es **PaddleOCR v4** corriendo en CPU. Las opciones disponibles en `img2table_ocr_model_load` son:

| Selección | Motor |
| --- | --- |
| 1 | EasyOCR |
| 2 | PaddleOCR v3 |
| 3 | PaddleOCR v4 |
| 4 | PaddleOCR v5 |

---

## Categorías de clasificación

| Código | Significado |
|---|---|
| BL | Bill of Lading |
| C | Commercial Invoice |
| AC | Airway Commercial |
| N | No clasificado |
| PL | Packing List *(extracción de tablas)* |

---

## Notas
- Los modelos se cargan **una sola vez** al arrancar el servidor gracias al `lifespan` de FastAPI, minimizando la latencia por request.
- La función `create_tables_pdf` en `main_functions.py` permite generar un PDF con las tablas extraídas como salida, aunque no están siendo usadas como endpoint actualmente.
- El procesamiento es **completamente en CPU**.
