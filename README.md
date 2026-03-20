# PDF Table Extractor API

API REST construida con **FastAPI** que recibe archivos PDF, clasifica cada página mediante una red neuronal convolucional (CNN) exportada a ONNX, y extrae las tablas de las páginas identificadas como *Packing List* usando `img2table` con PaddleOCRv4.

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

1. Se recibe un archivo PDF vía HTTP POST.
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

## Uso

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

---

#### `POST /process-pdf`

Procesa el archivo PDF y retorna la clasificación de páginas y las tablas extraídas.

**Request:**

- `Content-Type: multipart/form-data`
- Campo: `file` → archivo `.pdf`

**Respuesta:**
```json
{
  "total_time": 109.88,
  "class_page_list": [
    "Página. 1 - Clasificación: AC",
    "Página. 2 - Clasificación: AC",
    "Página. 3 - Clasificación: C",
    "Página. 4 - Clasificación: C",
    "Página. 5 - Clasificación: PL"
  ],
  "tables": [
    {
      "columns": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19
      ],
      "data": [
        {
          "0": "Deliver to:\nFORUSSA\nLAS CONDES 11281 LAS CONDES SANTIAGO CHILE\nR.U.T.86.963.200-7",
          "1": "Deliver to:\nFORUSSA\nLAS CONDES 11281 LAS CONDES SANTIAGO CHILE\nR.U.T.86.963.200-7",
          "2": "Deliver to:\nFORUSSA\nLAS CONDES 11281 LAS CONDES SANTIAGO CHILE\nR.U.T.86.963.200-7",
          "3": "Deliver to:\nFORUSSA\nLAS CONDES 11281 LAS CONDES SANTIAGO CHILE\nR.U.T.86.963.200-7",
          "4": "Deliver to:\nFORUSSA\nLAS CONDES 11281 LAS CONDES SANTIAGO CHILE\nR.U.T.86.963.200-7",
          "5": "Deliver to:\nFORUSSA\nLAS CONDES 11281 LAS CONDES SANTIAGO CHILE\nR.U.T.86.963.200-7",
          "6": "Deliver to:\nFORUSSA\nLAS CONDES 11281 LAS CONDES SANTIAGO CHILE\nR.U.T.86.963.200-7",
          "7": "Deliver to:\nFORUSSA\nLAS CONDES 11281 LAS CONDES SANTIAGO CHILE\nR.U.T.86.963.200-7",
          "8": "Deliver to:\nFORUSSA\nLAS CONDES 11281 LAS CONDES SANTIAGO CHILE\nR.U.T.86.963.200-7",
          "9": "Deliver to:\nFORUSSA\nLAS CONDES 11281 LAS CONDES SANTIAGO CHILE\nR.U.T.86.963.200-7",
          "10": "ETD:\nLoading Port: Genoa port in Italy\nETA:\nArrival Port: San Antonio. Chile",
          "11": "ETD:\nLoading Port: Genoa port in Italy\nETA:\nArrival Port: San Antonio. Chile",
          "12": "ETD:\nLoading Port: Genoa port in Italy\nETA:\nArrival Port: San Antonio. Chile",
          "13": "ETD:\nLoading Port: Genoa port in Italy\nETA:\nArrival Port: San Antonio. Chile",
          "14": "ETD:\nLoading Port: Genoa port in Italy\nETA:\nArrival Port: San Antonio. Chile",
          "15": "ETD:\nLoading Port: Genoa port in Italy\nETA:\nArrival Port: San Antonio. Chile",
          "16": null,
          "17": null,
          "18": null,
          "19": null
        },
        {
          "0": "General Description\nCategory.\nGender:\nDescription:\nBrand:\nTotal Cases: 58\nTotal Pairs: 348\nCountry of Origin:\nCarton Detail",
          "1": null,
          "2": null,
          "3": null,
          "4": null,
          "5": null,
          "6": "InvoiceN669/EX\nL/C NO.I095185\n555\n502\n5,16",
          "7": null,
          "8": null,
          "9": null,
          "10": "Shipper:\nCarrier\nVehicle ID:Green Park\nTrip: 548S\nContainer N:\nSVC N:\nNONE\nSeal:\nMovement: LCL\nComments:",
          "11": "Shipper:\nCarrier\nVehicle ID:Green Park\nTrip: 548S\nContainer N:\nSVC N:\nNONE\nSeal:\nMovement: LCL\nComments:",
          "12": "Shipper:\nCarrier\nVehicle ID:Green Park\nTrip: 548S\nContainer N:\nSVC N:\nNONE\nSeal:\nMovement: LCL\nComments:",
          "13": "Shipper:\nCarrier\nVehicle ID:Green Park\nTrip: 548S\nContainer N:\nSVC N:\nNONE\nSeal:\nMovement: LCL\nComments:",
          "14": "Shipper:\nCarrier\nVehicle ID:Green Park\nTrip: 548S\nContainer N:\nSVC N:\nNONE\nSeal:\nMovement: LCL\nComments:",
          "15": "Shipper:\nCarrier\nVehicle ID:Green Park\nTrip: 548S\nContainer N:\nSVC N:\nNONE\nSeal:\nMovement: LCL\nComments:",
          "16": null,
          "17": null,
          "18": null,
          "19": null
        },
        {
          "0": "General Description\nCategory.\nGender:\nDescription:\nBrand:\nTotal Cases: 58\nTotal Pairs: 348\nCountry of Origin:\nCarton Detail",
          "1": null,
          "2": null,
          "3": null,
          "4": null,
          "5": null,
          "6": null,
          "7": null,
          "8": null,
          "9": "Size Range",
          "10": "Size Range",
          "11": "Size Range",
          "12": "Size Range",
          "13": "Size Range",
          "14": null,
          "15": null,
          "16": null,
          "17": null,
          "18": null,
          "19": null
        },
        {
          "0": "PO#",
          "1": "STYLE",
          "2": "FORUS STOC",
          "3": "FORUS STOC",
          "4": "Color\nCode",
          "5": "SKU",
          "6": "Total\nCases",
          "7": "Total\nPairs",
          "8": "Pairs/\nCase",
          "9": "36",
          "10": "37",
          "11": "38",
          "12": "39",
          "13": "40",
          "14": "41",
          "15": "Cases No",
          "16": "G.W.(Kg)",
          "17": "N-W. (Kg)",
          "18": "N-W. (Kg)",
          "19": "Meas (m)"
        },
        {
          "0": "445481",
          "1": "PHILIPPA",
          "2": "RK202011275",
          "3": "RK202011275",
          "4": "KD6",
          "5": "878667",
          "6": "23",
          "7": "138",
          "8": "6",
          "9": "23",
          "10": "23",
          "11": "23",
          "12": "23",
          "13": "23",
          "14": "23",
          "15": "A095008197-A095008219",
          "16": "221,26",
          "17": "200,13",
          "18": "200,13",
          "19": "2,12"
        },
        {
          "0": "445481",
          "1": "PHILIPPA",
          "2": "RK202011275",
          "3": "RK202011275",
          "4": "KD6",
          "5": "7808774365287",
          "6": "3",
          "7": "18",
          "8": "6",
          "9": "18",
          "10": null,
          "11": null,
          "12": null,
          "13": null,
          "14": null,
          "15": "A095008220-A095008222",
          "16": "26.85",
          "17": "24.12",
          "18": "24.12",
          "19": "0.26"
        },
        {
          "0": "445481",
          "1": "PHILIPPA",
          "2": "RK202011275",
          "3": null,
          "4": "KD6",
          "5": "7808774365263",
          "6": "7",
          "7": "42",
          "8": "6",
          "9": null,
          "10": "42",
          "11": null,
          "12": null,
          "13": null,
          "14": null,
          "15": "A095008223-A095008229",
          "16": "65.10",
          "17": null,
          "18": "58.73",
          "19": "0.61"
        },
        {
          "0": "445481",
          "1": "PHILIPPA",
          "2": "RK202011275",
          "3": null,
          "4": "KD6",
          "5": "7808774365240",
          "6": "13",
          "7": "78",
          "8": "6",
          "9": null,
          "10": null,
          "11": "78",
          "12": null,
          "13": null,
          "14": null,
          "15": "A095008230A095008242",
          "16": "123.90",
          "17": "112.07",
          "18": "112.07",
          "19": "1.13"
        },
        {
          "0": "445481",
          "1": "PHILIPPA",
          "2": "RK202011275",
          "3": null,
          "4": "KD6",
          "5": "7806774365256",
          "6": "7",
          "7": "42",
          "8": "6",
          "9": null,
          "10": null,
          "11": null,
          "12": "42",
          "13": null,
          "14": null,
          "15": "A095008243-A095008249",
          "16": "67.76",
          "17": "61.39",
          "18": "61.39",
          "19": "0.61"
        },
        {
          "0": "445481",
          "1": "PHILIPPA",
          "2": "RK202011275",
          "3": null,
          "4": "KD6",
          "5": "7808774365270",
          "6": "4",
          "7": "24",
          "8": "6",
          "9": null,
          "10": null,
          "11": null,
          "12": null,
          "13": "24",
          "14": null,
          "15": "A095008250A095008253",
          "16": "39.76",
          "17": "36.12",
          "18": "36.12",
          "19": "0.35"
        },
        {
          "0": "445481",
          "1": "PHILIPPA",
          "2": "RK202011275",
          "3": null,
          "4": "KD6",
          "5": "7808774365232",
          "6": null,
          "7": "6",
          "8": "6",
          "9": null,
          "10": null,
          "11": null,
          "12": null,
          "13": null,
          "14": "6",
          "15": "A095008254",
          "16": "1037",
          "17": "9.46",
          "18": "9.46",
          "19": "0.10"
        }
      ]
    }
  ]
}
```

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
