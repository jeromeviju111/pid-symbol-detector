import os
import fitz  # PyMuPDF


def pdf_to_images(pdf_path, dpi=300, output_dir="pdf_pages"):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    page_image_paths = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        out_path = f"{output_dir}/page_{page_num+1}.png"
        pix.save(out_path)
        page_image_paths.append(out_path)
    doc.close()
    return page_image_paths
