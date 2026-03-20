from img2table.document import Image
from img2table.ocr import PaddleOCR
from time import monotonic

image = Image("DAC251159604.pdf_13.jpg")
ocr = PaddleOCR(lang="en", kw={"device": "cpu", "enable_mkldnn": False, "use_textline_orientation": False, "ocr_version": "PP-OCRv4"})

start = monotonic()
extracted_tables = image.extract_tables(
       ocr = ocr,
       implicit_rows = False,
       implicit_columns = False,
       borderless_tables = False,
       min_confidence = 85
       )
tables = []
for table in extracted_tables:
    try:
        tables.append(table.df)
    except Exception as e:
        print(e)
        print(f"error procesando la imagen")
end = monotonic()

for table in tables:
    print(table.to_html())
print(f"Tiempo total de procesamiento: {end - start}")