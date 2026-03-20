"""
Código para leer todos los archivos PDF y obtener una imágen por cada hoja leída.
"""

from pdfplumber import open
from os import listdir
from pathlib import Path

pdf_path = Path(__file__).resolve().parent / "Train_PDF"
images_path = Path(__file__).resolve().parent / "Train_data"

for pdf in listdir(pdf_path):
    with open(pdf_path) as pdf_file:
        for page in pdf_file.pages:
            page2image = page.to_image(resolution = 300).annotated
            page2image = page2image.resize((224, 224))
            save_path = f"{images_path}/{pdf}_{page.page_number}.jpg"
            page2image.save(save_path)