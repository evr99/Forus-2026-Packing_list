from streamlit import file_uploader, columns, container, session_state, button, write, select_slider
from main_functions import cnn_model_selection_and_load, img2table_ocr_model_load
from main_functions import create_tables_pdf, pdf_plumber_process
from streamlit import set_page_config
from time import monotonic

set_page_config(layout = "wide")
message_container = container(horizontal = True, horizontal_alignment = "distribute", vertical_alignment = "center", border = True)

if "CNN_LOAD" not in session_state:
    models_path = "modelos_onnx_inferencia_1"
    cnn_name, cnn_model, input_name, output_name = cnn_model_selection_and_load(models_dir_path = models_path)
    if cnn_model is None:
        session_state["CNN_LOAD"] = f"Modelo :gray-badge[{cnn_name}] :red-badge[no se pudo cargar correctamente]."
        session_state["ok_status"] = False
    else:
        session_state["CNN_LOAD"] = f"Modelo :gray-badge[{cnn_name}] :green-badge[cargado correctamente]."
        session_state["ok_status"] = True
        session_state["cnn_name"] = cnn_name
        session_state["cnn_model"] = cnn_model
        session_state["input_name"] = input_name
        session_state["output_name"] = output_name

if "OCR_LOAD" not in session_state:
    ocr_name, ocr_model = img2table_ocr_model_load(selection = 3)
    if ocr_model is None:
        session_state["OCR_LOAD"] = f"Modelo :gray-badge[{ocr_name}] :red-badge[no se pudo cargar correctamente]."
        session_state["ok_status"] = False
    else:
        session_state["OCR_LOAD"] = f"Modelo :gray-badge[{ocr_name}] :green-badge[cargado correctamente]."
        session_state["ok_status"] = True
        session_state["ocr_model"] = ocr_model
        session_state["ocr_name"] = ocr_name

if "TIEMPO_TOTAL" not in session_state:
    session_state["TIEMPO_TOTAL"] = ""

try:
    if session_state["pdf_tables"] and session_state["processed"]:
        pass
except:
    session_state["pdf_tables"] = None
    session_state["processed"] = None

if session_state["ok_status"] == True:
    session_state.excel_disabled = False
    session_state.pdf_disabled = False
    message_container.success(session_state["CNN_LOAD"])
    message_container.success(session_state["OCR_LOAD"])
else:
    session_state.excel_disabled = True
    session_state.pdf_disabled = True
    message_container.error(session_state["CNN_LOAD"])
    message_container.error(session_state["OCR_LOAD"])

time_container = container(horizontal = False, border = True)
class_container = container(horizontal = True, border = True)

with container(border = True, horizontal = True):
    col1, col2 = columns([3.5, 1.5])

    with col1:
        with container(horizontal = False):
            dataframe_container = container(
                horizontal = False,
                horizontal_alignment = "distribute",
                vertical_alignment = "center",
                border = True
                )
    with col2:
        with container(horizontal = False, gap = "xsmall"):
            button_container = container(horizontal = True, horizontal_alignment = "distribute")
            with button_container:
                new_upload = button("Cargar PDF")
                make_pdf = button("Crear PDF")
            pdf_file = file_uploader(
                "Subir PDF",
                type = "PDF",
                accept_multiple_files = False,
                key = "PDF",
                disabled = session_state.pdf_disabled
                )
            write(":blue-badge[Desglose Letras de clasificación.]")
            write("AC: Anexo corrección")
            write("C: Terminos y condiciones carrier")
            write("BL: Bill of lading")
            write("PL: Packing list")
            write("N: Nada en la página")

if new_upload and not session_state["processed"]:
    start = monotonic()
    tables = []
    class_list = []
    
    tables, class_list = pdf_plumber_process(
        pdf = pdf_file,
        tables = tables,
        class_list = class_list,
        cnn_model = session_state["cnn_model"],
        output_name = session_state["output_name"],
        input_name = session_state["input_name"],
        ocr_model = session_state["ocr_model"]
        )
    
    end = monotonic()
    session_state["TIEMPO_TOTAL"] = end - start
    session_state["pdf_tables"] = tables
    session_state["class_list"] = class_list
    session_state["processed"] = True
else:
    pass

try:
    time_container.write(f"Tiempo total de respuesta: {session_state['TIEMPO_TOTAL']:.2f} segundos.")
except:
    pass

try:
    for class_item in session_state["class_list"]:
        class_container.write(class_item)
except:
    pass

if session_state["pdf_tables"] is not None:
    for i, table in enumerate(session_state["pdf_tables"]):
        options = [int(x) for x in table.columns.to_list()]
        min_val = min(options)
        max_val = max(options)
        with dataframe_container.container(horizontal = True):
            columns_to_eliminate = select_slider(
                label="Escoger columnas",
                options = options,
                value = (min_val, max_val),
                key = f"key_{i}"
                )
        edited_table = dataframe_container.data_editor(
            table,
            num_rows = "dynamic",
            key = f"table_{i}"
            )
        session_state["pdf_tables"][i] = edited_table

if new_upload:
    print("session state como none")
    session_state["class_list"] = []
    session_state["processed"] = False
    session_state["pdf"] = None

if make_pdf:
    table_data = []
    for i, table in enumerate(session_state["pdf_tables"]):
        start, end = session_state[f"key_{i}"]
        columns = [str(x) for x in range(start, end + 1)]
        data = table[columns]
        table_data.append(data)
        pdf = create_tables_pdf(table_data)
    download_pdf = button_container.download_button(
        label = "Descarga PDF",
        data = pdf,
        file_name = "PL_PDF.pdf"
    )