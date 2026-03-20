from numpy import array, argmax, expand_dims, float32
from pdfplumber import open as openpdfplummber
from img2table.ocr import PaddleOCR, EasyOCR
from pymupdf import open as open_pymupdf
from onnxruntime import InferenceSession
from img2table.document import Image
from pandas import json_normalize
from pymupdf import Rect
from os import listdir
from io import BytesIO

def cnn_model_selection_and_load(models_dir_path: str):
    """
    - Esta función se encarga de encontrar y cargar el mejor modelo encontrado en el sistema, dejandolo
    disponible para hacer la primera inferencia en todo momento.
    - Retorna respectivamente: el nombre del modelo, el modelo, input_del_modelo, output_del_modelo o 
    None en caso de que no exista el modelo.
    """
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
    except Exception as e:
        print(e)
        return None, None, None, None

def img2table_ocr_model_load(selection: int):
    """
    Dentro de la libreria de img2table, se disponen de distintos OCR para poder trabajar una imagen y extraer
    los respectivos datos dentro de una tabla, dicho esto, en esta función se permite escoger motores OCR, los
    presentes como selección, permiten su instalacion mediante pip y por tanto mediante conexión a internet.
    >> pip install img2table[easyocr]
    o en su defecto
    >> pip install img2table[paddle]
    """
    if selection == 1:
        name = "EasyOCR"
        try:
            ocr = EasyOCR(lang= ["en"], kw={"gpu": False, "download_enabled": True})
            return name, ocr
        except Exception as e:
            print(e)
            return name, None

    elif selection == 2:
        name = "PaddleOCR-v3"
        try:
            ocr = PaddleOCR(
                lang="en",
                kw={
                    "device": "cpu",
                    "enable_mkldnn": False,
                    "use_textline_orientation": False,
                    "ocr_version": "PP-OCRv3"
                    }
                )
            return name, ocr
        except Exception as e:
            print(e)
            return name, None
        
    elif selection == 3: # Usado hasta ahora 10/03/2026, ofrece buenos resultados pero con altos tiempos de inferencia
        name = "PaddleOCR-v4"
        try:
            ocr = PaddleOCR(
                lang="en",
                kw={
                    "device": "cpu",
                    "enable_mkldnn": False,
                    "use_textline_orientation": False,
                    "ocr_version": "PP-OCRv4"
                    }
                )
            return name, ocr
        except Exception as e:
            print(e)
            return name, None

    elif selection == 4:
        name = "PaddleOCR-v5"
        try:
            ocr = PaddleOCR(
                lang="en",
                kw={
                    "device": "cpu",
                    "enable_mkldnn": False,
                    "use_textline_orientation": False,
                    "ocr_version": "PP-OCRv5"
                    }
                )
            return name, ocr
        except Exception as e:
            print(e)
            return name, None
        
def pdf_plumber_process(pdf: BytesIO, tables: list, class_list: list, cnn_model, output_name, input_name, ocr_model):
    """
    Esta función se encarga de realizar el proceso de extracción de datos a las hojas del pdf si corresponde.
    - Primero se obtiene el resultado de la inferencia inicial de la CNN que identifica si la hoja del pdf corresponde a un packing list.
    - Si la hoja se identifica como packing list, pasa por el proceso de la libreria img2table.
        - Identificación de la tabla (lineas), mediante procesamiento de imagenes (filtros morfologicos)
        - Extración de caracteres por celda identificada.
        - Tipo de OCR -> transformer + RCNN
    """
    with openpdfplummber(pdf) as pdf:
        categories = ["BL", "C", "AC", "N", "PL"]
        for page in pdf.pages:
            original_image = page.to_image(resolution = 300).original
            resize_image = original_image.copy().resize(size = (224, 224), resample = 2)
            image = array(resize_image).astype(float32)
            image = expand_dims(image, axis = 0)
            inference = cnn_model.run([output_name], {input_name: image})[0]
            result = argmax(inference[0])
            class_list.append(f"Pagina. {page.page_number} - Clasificacion: {categories[result]}")
            if categories[result] == "PL":
                buffer = BytesIO()
                original_image.save(buffer, format = "PNG")
                ocr_image_bytes = buffer.getvalue()
                buffer.close()
                ocr_image = Image(src = ocr_image_bytes)

                extracted_tables = ocr_image.extract_tables(
                    ocr = ocr_model,
                    implicit_rows = False,
                    implicit_columns = False,
                    borderless_tables = False,
                    min_confidence = 85
                )

                for table in extracted_tables:
                    try:
                        tables.append(table.df)
                    except Exception as e:
                        print(e)
                        print(f"error procesando la imagen de la pagina {page.page_number}")

    return tables, class_list

def create_tables_pdf(tablas: list):
    """
    Funcion encargada de crear el pdf, en base a la selección de las columnas del dataframe desde la interfaz.
    - Transforma el dataframe a una tabla html la cual es despues usada para incorporarla al pdf.
    - Descarga automaticamente el pdf con una tabla por hoja.
    """
    css = """<style>
        table, th, td {
            border: 1px solid black;}
        table {
            border-collapse: collapse;
            width: 100%;}
        th, td {
            padding: 8px;
            background-color: #f2f2 f2;
            text-align: left;}
        </style>"""
    doc = open_pymupdf()
    for tabla in tablas:
        page = doc.new_page()
        tabla = tabla.to_html()
        rect = Rect(50, 50, 550, 750)
        page.insert_htmlbox(rect = rect, text = tabla, css = css)
    pdf_bytes = doc.tobytes()
    doc.close()

    return pdf_bytes

def process_json(json_file: dict, campos_de_interes: list):
    indices    = []
    trigger    = False
    lista_tablas = []

    tablas = json_file["tables"]

    for tabla_dict in tablas:
        for key, filas in tabla_dict.items():
            if type(filas) != list:
                continue

            lista_tabla = []
            for fila_dict in filas:
                # cada fila_dict es {"0": val, "1": val, ...}
                lista = list(fila_dict.values())
                lista = [x.replace("\n", "") if isinstance(x, str) else x for x in lista]

                if any(campo in lista for campo in campos_de_interes):
                    lista_tabla.append(lista)
                    trigger = True
                elif trigger:
                    lista_tabla.append(lista)

            if not lista_tabla:
                trigger = False
                continue

            for campo in campos_de_interes:
                try:
                    indices.append(lista_tabla[0].index(campo))
                except ValueError:
                    pass

            if indices:
                indices = sorted(indices)
                for fila in lista_tabla:
                    lista_tablas.append(fila[min(indices): max(indices) + 1])

            trigger = False
            indices.clear()

    # Filtrar filas donde todos los elementos son None
    lista_tablas = [
        fila for fila in lista_tablas
        if not all(x is None for x in fila)
    ]

    # Filtrar filas donde todos los elementos tienen largo > 13 (basura OCR)
    lista_tablas = [
        fila for fila in lista_tablas
        if not all(len(x) > 13 if isinstance(x, str) else False for x in fila)
    ]

    # Eliminar headers duplicados
    header = lista_tablas[0] if lista_tablas else []
    PL = [header] + [fila for fila in lista_tablas[1:] if fila != header]

    return PL