# Explicación profunda del código

Este documento detalla el funcionamiento interno de cada componente del proyecto, explicando las decisiones de diseño, el flujo de datos y el propósito de cada bloque de código.

---

## Tabla de contenidos

- [main.py](#mainpy)
  - [Imports y dependencias](#imports-y-dependencias)
  - [Diccionario global `models`](#diccionario-global-models)
  - [Lifespan: carga de modelos al arrancar](#lifespan-carga-de-modelos-al-arrancar)
  - [Instancia de FastAPI](#instancia-de-fastapi)
  - [Endpoint `/health`](#endpoint-health)
  - [Endpoint `/process-pdf`](#endpoint-process-pdf)
- [main_functions.py](#main_functionspy)
  - [cnn_model_selection_and_load](#cnn_model_selection_and_load)
  - [img2table_ocr_model_load](#img2table_ocr_model_load)
  - [pdf_plumber_process](#pdf_plumber_process)
  - [create_tables_pdf](#create_tables_pdf)

---

## `main.py`

### Imports y dependencias

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from main_functions import (
    cnn_model_selection_and_load,
    img2table_ocr_model_load,
    pdf_plumber_process
)
from io import BytesIO
from time import monotonic
```

- **`FastAPI`**: framework principal para construir la API REST.
- **`UploadFile` / `File`**: abstracciones de FastAPI para recibir archivos en requests multipart.
- **`HTTPException`**: permite lanzar errores HTTP con código de estado y mensaje personalizados.
- **`asynccontextmanager`**: decorador de la librería estándar de Python que convierte una función generadora async en un context manager. Se usa para definir el ciclo de vida (`lifespan`) del servidor.
- **`BytesIO`**: permite tratar bytes en memoria como si fueran un archivo, evitando escritura en disco.
- **`monotonic`**: reloj de alta resolución que nunca retrocede, ideal para medir tiempos de ejecución sin interferencia de cambios del reloj del sistema.

---

### Diccionario global `models`

```python
models = {}
```

Este diccionario actúa como un **contenedor de estado en memoria** compartido entre el ciclo de vida del servidor y los endpoints. Al arrancar, se puebla con el modelo CNN y el modelo OCR cargados. Durante los requests, los endpoints acceden a `models["cnn_model"]`, `models["ocr_model"]`, etc., sin tener que recargar los modelos en cada llamada.

Usar un diccionario global mutable es el patrón recomendado por FastAPI para compartir recursos pesados (como modelos ML) entre el lifespan y los endpoints sin recurrir a variables de módulo sueltas ni a instancias de clase.

---

### Lifespan: carga de modelos al arrancar

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    cnn_name, cnn_model, input_name, output_name = cnn_model_selection_and_load(
        models_dir_path = "cnn_module/onnx_models"
    )
    if cnn_model is None:
        raise RuntimeError(f"No se pudo cargar el modelo CNN: {cnn_name}")

    ocr_name, ocr_model = img2table_ocr_model_load(selection=3)
    if ocr_model is None:
        raise RuntimeError(f"No se pudo cargar el modelo OCR: {ocr_name}")

    models["cnn_model"]   = cnn_model
    models["input_name"]  = input_name
    models["output_name"] = output_name
    models["ocr_model"]   = ocr_model

    print(f"CNN cargada:  {cnn_name}")
    print(f"OCR cargado:  {ocr_name}")

    yield  # <-- el servidor está vivo aquí

    models.clear()
```

El **lifespan** es un context manager async que reemplaza a los eventos `startup`/`shutdown` de versiones anteriores de FastAPI. Todo el código **antes del `yield`** se ejecuta cuando el servidor arranca; todo lo que está **después del `yield`** se ejecuta cuando el servidor se apaga.

La secuencia es:

1. Se selecciona y carga el mejor modelo CNN disponible en disco.
2. Se valida que el modelo no sea `None`; si falla, se lanza un `RuntimeError` que impide que el servidor arranque.
3. Se carga el motor OCR (PaddleOCR v4 en este caso).
4. Ambos modelos se almacenan en el diccionario global `models`.
5. El servidor queda en estado `yield` (activo), atendiendo requests.
6. Al recibir señal de cierre, se ejecuta `models.clear()` para liberar referencias.

Este patrón garantiza que los modelos pesados se cargan **una sola vez**, reduciendo la latencia de cada request a solo el tiempo de inferencia.

---

### Instancia de FastAPI

```python
app = FastAPI(
    title = "PDF Table Extractor",
    description = "Clasifica páginas de un PDF con una CNN y extrae tablas mediante img2table[paddleOCRv4].",
    version = "1.0.0",
    lifespan = lifespan
)
```

Se instancia la aplicación FastAPI pasando el `lifespan` definido anteriormente. Los metadatos (`title`, `description`, `version`) se exponen automáticamente en la documentación interactiva de Swagger UI disponible en `/docs`.

---

### Endpoint `/health`

```python
@app.get("/health")
def health():
    return {"status": "ok"}
```

Endpoint síncrono simple que retorna un JSON confirmando que el servidor está en línea. Es útil para healthchecks de orquestadores (Docker, Kubernetes) o balanceadores de carga. No verifica el estado interno de los modelos, solo que el proceso está respondiendo.

---

### Endpoint `/process-pdf`

```python
@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un PDF.")

    pdf_buffer = BytesIO(await file.read())

    start = monotonic()
    tables, class_list = pdf_plumber_process(
        pdf = pdf_buffer,
        tables = [],
        class_list = [],
        cnn_model = models["cnn_model"],
        output_name = models["output_name"],
        input_name = models["input_name"],
        ocr_model = models["ocr_model"]
    )
    elapsed = monotonic() - start

    return {
        "elapsed_seconds": round(elapsed, 2),
        "class_list": class_list,
        "tables": [
            {
                "columns": df.columns.tolist(),
                "data": df.to_dict(orient="records")
            }
            for df in tables
        ]
    }
```

**Flujo detallado:**

1. **Validación de extensión**: se verifica que el archivo tenga extensión `.pdf`. La validación es sobre el nombre del archivo, no sobre el magic number del binario; es una verificación básica pensada para filtrar errores de usuario comunes.

2. **Lectura a memoria**: `await file.read()` lee todos los bytes del archivo subido de forma asíncrona. Luego se envuelven en un `BytesIO` para que las librerías de procesamiento (`pdfplumber`) puedan consumirlo como si fuera un archivo en disco, sin escritura real.

3. **Medición de tiempo**: `monotonic()` captura el tiempo antes y después del procesamiento. La diferencia da el tiempo total en segundos, redondeado a 2 decimales.

4. **Procesamiento**: se delega todo el trabajo pesado a `pdf_plumber_process`, pasando los modelos desde el diccionario global.

5. **Serialización de tablas**: cada tabla extraída es un `DataFrame` de pandas. Se convierte a un diccionario con `columns` (lista de nombres de columna) y `data` (lista de filas como dicts). FastAPI serializa automáticamente este resultado a JSON.

---

## `main_functions.py`

### `cnn_model_selection_and_load`

```python
def cnn_model_selection_and_load(models_dir_path: str):
    try:
        models = listdir(models_dir_path)
        models = [modelos[:-5].split("_") for modelos in models]
        models = {modelo[0]: modelo[1] for modelo in models}
        models = dict(sorted(models.items()))
        best_model = next(reversed(models))
        model_name = f"{best_model}_{models[best_model]}.onnx"

        onnx_path = f"{models_dir_path}/{model_name}"
        onnx_model = InferenceSession(onnx_path)
        input = onnx_model.get_inputs()[0].name
        output = onnx_model.get_outputs()[0].name
        return model_name, onnx_model, input, output
    except:
        return None, None, None, None
```

**Lógica de selección del mejor modelo:**

Los modelos en disco siguen la convención de nombre `{accuracy}_{mse}.onnx`. El proceso de selección es:

1. `listdir` obtiene todos los archivos del directorio.
2. Se elimina la extensión `.onnx` (los últimos 5 caracteres con `[:-5]`) y se separa por `_`, obteniendo `[accuracy, mse]` para cada archivo.
3. Se construye un diccionario `{accuracy: mse}`.
4. Se ordena el diccionario por clave (`accuracy`) de menor a mayor con `dict(sorted(...))`.
5. `next(reversed(models))` obtiene la última clave, es decir, la mayor accuracy.

**Carga con ONNX Runtime:**

`InferenceSession` es la clase principal de `onnxruntime` para ejecutar modelos ONNX. Una vez cargada, se obtienen los nombres del tensor de entrada y salida con `get_inputs()[0].name` y `get_outputs()[0].name`. Estos nombres son necesarios para llamar a `session.run()` correctamente durante la inferencia.

El bloque `try/except` genérico captura cualquier fallo (directorio inexistente, archivos corruptos, etc.) y retorna `None` en todos los campos, delegando el manejo del error al llamador (el lifespan en `main.py`).

---

### `img2table_ocr_model_load`

```python
def img2table_ocr_model_load(selection: int):
    if selection == 1:   # EasyOCR
    elif selection == 2: # PaddleOCR v3
    elif selection == 3: # PaddleOCR v4 (por defecto)
    elif selection == 4: # PaddleOCR v5
```

Esta función actúa como una **fábrica de motores OCR**. El parámetro `selection` permite intercambiar el motor sin modificar el resto del código.

**Por qué PaddleOCR v4 como default (selection=3):**

El comentario en el código lo indica explícitamente: ofrece buenos resultados pero con tiempos de inferencia altos. Es el balance elegido entre precisión y velocidad para el caso de uso (documentos tipo Packing List con tablas estructuradas).

**Configuración común a todos los motores Paddle:**

```python
kw={
    "device": "cpu",
    "enable_mkldnn": False,
    "use_textline_orientation": False,
    "ocr_version": "PP-OCRv4"
}
```

- `device: "cpu"`: fuerza ejecución en CPU, sin dependencia de GPU.
- `enable_mkldnn: False`: desactiva la optimización Intel MKL-DNN; puede causar inestabilidad en ciertos entornos.
- `use_textline_orientation: False`: desactiva la detección de orientación de líneas de texto, reduciendo latencia cuando los documentos siempre están derechos.

Cada `elif` está envuelto en un `try/except` independiente para retornar `None` si el motor no está instalado, permitiendo al llamador manejar el error.

---

### `pdf_plumber_process`

Esta es la función central del sistema. Combina clasificación CNN con extracción OCR página a página.

```python
def pdf_plumber_process(pdf, tables, class_list, cnn_model, output_name, input_name, ocr_model):
    with openpdfplummber(pdf) as pdf:
        categories = ["BL", "C", "AC", "N", "PL"]
        for page in pdf.pages:
            ...
```

**Por qué `pdfplumber`:**

`pdfplumber` está especializado en la extracción de contenido de PDFs (texto, tablas, imágenes de página). Su método `page.to_image(resolution=300).original` devuelve un objeto `PIL.Image` de alta resolución, ideal como entrada para modelos de visión.

**Paso 1 — Clasificación CNN:**

```python
original_image = page.to_image(resolution=300).original
resize_image = original_image.copy().resize(size=(224, 224), resample=2)
image = array(resize_image).astype(float32)
image = expand_dims(image, axis=0)
inference = cnn_model.run([output_name], {input_name: image})[0]
result = argmax(inference[0])
```

- La página se renderiza a 300 DPI para capturar suficiente detalle.
- Se hace una copia antes de redimensionar para preservar `original_image` intacta (se usará luego para el OCR si la página es PL).
- `resample=2` corresponde a `PIL.Image.BILINEAR`, un filtro de interpolación que balancea calidad y velocidad al reducir la imagen a 224×224 px.
- `astype(float32)` convierte los píxeles de uint8 (0–255) a float32, que es el dtype esperado por el modelo ONNX.
- `expand_dims(..., axis=0)` agrega la dimensión de batch: el modelo espera forma `(1, 224, 224, 3)`.
- `cnn_model.run([output_name], {input_name: image})` ejecuta la inferencia. Retorna una lista; `[0]` extrae el único tensor de salida.
- `argmax(inference[0])` obtiene el índice de la clase con mayor probabilidad, que se mapea a la lista `categories`.

**Paso 2 — Extracción de tablas (solo si PL):**

```python
if categories[result] == "PL":
    buffer = BytesIO()
    original_image.save(buffer, format="PNG")
    ocr_image_bytes = buffer.getvalue()
    buffer.close()
    ocr_image = Image(src=ocr_image_bytes)

    extracted_tables = ocr_image.extract_tables(
        ocr=ocr_model,
        implicit_rows=False,
        implicit_columns=False,
        borderless_tables=False,
        min_confidence=85
    )
```

- La imagen original (300 DPI, sin redimensionar) se serializa a PNG en memoria usando un `BytesIO` temporal. Esto evita escritura en disco y mantiene la máxima calidad para el OCR.
- `img2table.document.Image` es la clase que representa una imagen procesable por `img2table`. Acepta bytes directamente en el parámetro `src`.
- **Parámetros de `extract_tables`:**
  - `implicit_rows=False`: no infiere filas sin bordes explícitos.
  - `implicit_columns=False`: ídem para columnas.
  - `borderless_tables=False`: solo detecta tablas con bordes visibles (más precisión, menos falsos positivos).
  - `min_confidence=85`: umbral mínimo de confianza OCR para aceptar una celda; valores más bajos incluyen más texto pero con más errores.

**Manejo de tablas extraídas:**

```python
for table in extracted_tables:
    try:
        tables.append(table.df)
    except Exception as e:
        print(e)
        print(f"error procesando la imagen de la pagina {page.page_number}")
```

Cada elemento de `extracted_tables` es un objeto de `img2table` que expone su contenido como `DataFrame` de pandas mediante `.df`. El `try/except` por tabla permite que un fallo en una tabla específica no interrumpa el procesamiento del resto.

---

### `create_tables_pdf`

```python
def create_tables_pdf(tablas: list):
    css = """<style>...</style>"""
    doc = open_pymupdf()
    for tabla in tablas:
        page = doc.new_page()
        tabla = tabla.to_html()
        rect = Rect(50, 50, 550, 750)
        page.insert_htmlbox(rect=rect, text=tabla, css=css)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes
```

Esta función toma la lista de DataFrames extraídos y genera un nuevo PDF con una tabla por página.

**Flujo:**

1. `open_pymupdf()` sin argumentos crea un documento PDF vacío en memoria.
2. Por cada DataFrame, se agrega una página nueva con `doc.new_page()`.
3. `tabla.to_html()` convierte el DataFrame a una cadena HTML con etiquetas `<table>`.
4. `Rect(50, 50, 550, 750)` define el área de renderizado en la página (en puntos tipográficos; 72 pts = 1 pulgada).
5. `page.insert_htmlbox()` renderiza el HTML dentro del rectángulo, aplicando el CSS inline para estilos de tabla.
6. `doc.tobytes()` serializa el documento completo a bytes, listo para enviar como respuesta HTTP o guardar en disco.

> **Nota:** esta función actualmente no está conectada a ningún endpoint en `main.py`. Está disponible para ser integrada como un endpoint adicional (por ejemplo, `POST /export-pdf`) que reciba las tablas ya procesadas y devuelva el PDF generado.

---

## Consideraciones generales de diseño

### Por qué ONNX para la CNN

Exportar el modelo de Keras/TensorFlow a ONNX permite ejecutarlo con `onnxruntime`, que es más liviano y rápido que cargar el stack completo de TensorFlow solo para inferencia. También desacopla el entorno de entrenamiento del de producción.

### Por qué `pdfplumber` para renderizar páginas e `img2table` para extraer tablas

`pdfplumber` es excelente para obtener imágenes de páginas con control de DPI. `img2table` está especializado en la detección de estructuras de tabla en imágenes mediante morfología (detección de líneas) + OCR por celda, lo que lo hace más robusto que intentar extraer tablas directamente del PDF con herramientas como `tabula` o `camelot`, especialmente cuando el PDF es un escaneo.

### Por qué los modelos se pasan como parámetros y no se acceden globalmente dentro de las funciones

`pdf_plumber_process` y las demás funciones de `main_functions.py` reciben los modelos como parámetros en lugar de acceder al diccionario `models` directamente. Esto las hace más **testeables** (se pueden pasar mocks fácilmente) y **reutilizables** fuera del contexto de FastAPI (por ejemplo, desde un script de línea de comandos o desde Streamlit, como sugieren los comentarios `@cache_resource` en el código).
