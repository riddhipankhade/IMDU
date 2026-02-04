''' document and image processing python librabry imports: -
1. typing: This is a standard Python module used for type hinting, which helps write 
more maintainable and readable code by specifying the expected types of variables,
 function arguments, and return values (e.g., List, Dict).

2. PIL (Pillow): Python Image Library, widely used for opening,
 manipulating, and saving many different image file formats. 

3. fitz (PyMuPDF): A highly efficient library for working with PDFs and other document formats. It allows you to extract text and images, manipulate pages, and even render pages as images. 
 '''

from typing import List, Dict
from PIL import Image 
import fitz #PyMuPDF

def load_document(file_path: str) -> Dict:#this function returns a dictionary
    """
    Load a doc (pdf or image) and conver it into page images 

    return: dict of pages and number of pages
    """
    pages = []

    #case 1: pdf
    if file_path.lower().endswith(".pdf"):
        doc = fitz.open(file_path)

        for i in range(len(doc)):
            page = doc.load_page(i)
            pix  = page.get_pixmap(dpi=200)

            img = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            pages.append(img)
        
        doc.close()
    
    #case 2: an image
    else:
        img = Image.open(file_path).convert("RGB")
        pages.append(img)

    return {
        "pages" : pages,
        "num_pages" : len(pages)
    }