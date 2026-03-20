from img2table.ocr import EasyOCR
from img2table.document import Image
try:
    ocr = EasyOCR(lang=["en"])
    image = Image(src = "DAC251159604.pdf_13.jpg")
    tables =  image.extract_tables(ocr = ocr)
    for table in tables:
        print(table.df)
except Exception as e:
    print(f"El modelo no se pudo cargar, error: {e}")