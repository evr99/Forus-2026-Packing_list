from fastapi import FastAPI, UploadFile, HTTPException, Form, File
from json import JSONEncoder
from contextlib import asynccontextmanager
from main_functions import (
    cnn_model_selection_and_load,
    img2table_ocr_model_load,
    pdf_plumber_process,
    process_json
)
from io import BytesIO
from time import monotonic

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    cnn_name, cnn_model, input_name, output_name = cnn_model_selection_and_load(
        models_dir_path = "cnn_module/onnx_models"
    )
    if cnn_model is None:
        raise RuntimeError(f"No se pudo cargar el modelo CNN: {cnn_name}")

    ocr_name, ocr_model = img2table_ocr_model_load(selection = 3)
    if ocr_model is None:
        raise RuntimeError(f"No se pudo cargar el modelo OCR: {ocr_name}")

    models["cnn_model"]   = cnn_model
    models["input_name"]  = input_name
    models["output_name"] = output_name
    models["ocr_model"]   = ocr_model

    print(f"CNN cargada:  {cnn_name}")
    print(f"OCR cargado:  {ocr_name}")

    yield
    models.clear()

app = FastAPI(
    title = "PDF Table Extractor",
    description = "Clasifica páginas de un PDF con una CNN y extrae tablas mediante img2table[paddleOCRv4].",
    version = "1.0.0",
    lifespan = lifespan
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...), brand: str = Form(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code = 400, detail = "El archivo debe ser un PDF.")

    pdf_buffer = BytesIO(await file.read())

    start = monotonic()
    tables, class_list = pdf_plumber_process(
        pdf        = pdf_buffer,
        tables     = [],
        class_list = [],
        cnn_model  = models["cnn_model"],
        output_name = models["output_name"],
        input_name  = models["input_name"],
        ocr_model   = models["ocr_model"]
    )
    elapsed = monotonic() - start

    return {
        "total_time":      round(elapsed, 2),
        "class_page_list": class_list,
        "tables":          [{f"data_{i}": df.to_dict(orient = "records")} for i, df in enumerate(tables)]
    }

@app.post("/process-json")
async def process_json_endpoint(file: UploadFile = File(...), brand: str = Form(...)):
    
    brand = brand.lower()
    campos_por_marca = {
        "crocs":    ["SKU", "QtyPerCtn", "TotalQ'ty"],
        "rockford": ["SKU", "Total Cases", "Pairs", "Pairs/Case"],
        "hp":       ["SKU", "Total Cases", "Pairs", "Pairs/Case"],
    }
    campos_de_interes = campos_por_marca.get(brand)
    if campos_de_interes is None:
        raise HTTPException(status_code = 404, detail = f"Marca '{brand}' no reconocida.")
    data = await process_pdf(file, brand)
    lista_de_datos = process_json(data, campos_de_interes)

    header = lista_de_datos[0]
    filas = lista_de_datos[1:]
    
    return {
        header: [fila[i] for fila in filas] for i,header in enumerate(header)
    }
    