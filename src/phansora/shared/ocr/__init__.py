"""Reading scanned PDFs — one implementation for every product.

``pdf.PdfConverter`` renders a PDF's pages (``render``), reads each with Tesseract
(``tesseract``), and has DeepSeek tidy the recognized text, with a resume cache and a
progress callback. ``pdf.explain_failure`` turns a failure into words a reader can act on.
Callers: SpokenVerse (/pdf-to-txt), Book Alchemy (a scanned book), Talari (a scanned
import). Import from the modules (``phansora.shared.ocr.pdf``), which is where tests patch.
"""
