
# PDF2MKR.py - PDF to Mokuro Format Converter

Converts PDF files into a structured `.mokuro` JSON format and extracts page images. Designed for preparing PDFs (especially comics/magazines) for digital readers or OCR pipelines that use this format.

## Key Features
*   Extracts text blocks, bounding boxes, and font info.
*   Detects horizontal/vertical text orientation.
*   Merges related text fragments.
*   Splits long text lines for readability.
*   Saves each PDF page as a JPG image.
*   Outputs a `.mokuro` JSON file linking text data to page images.

## Prerequisites
*   Python 3.x
*   PyMuPDF (`fitz`), tqdm

## Installation
```bash
pip install PyMuPDF tqdm

Usage
python PDF2MKR.py <input_path> [zoom_factor] [debug_page_index]
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Arguments:

<input_path> (Required): Path to a single PDF or a directory of PDFs.

[zoom_factor] (Optional): Image resolution multiplier (default: 1.0).

[debug_page_index] (Optional): 0-based page index for verbose debug output.

Examples:

Process a single PDF:

python PDF2MKR.py my_document.pdf
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Process all PDFs in a folder with 2x zoom:

python PDF2MKR.py ./pdf_collection/ 2.0
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Process a PDF and debug its first page (index 0):

python PDF2MKR.py book.pdf 1.0 0
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
Output

For each input PDF (e.g., book.pdf):

Image Directory: A folder named book/ containing JPG images of each page (e.g., 000.jpg, 001.jpg).

Mokuro File: A book.mokuro JSON file containing structured text data, bounding boxes, orientation, and links to the page images.

Internal Configuration:
Some constants like text splitting thresholds (LINE_SPLIT_THRESHOLD_SHORT, LINE_SPLIT_THRESHOLD_MEDIUM) can be adjusted directly in the script for fine-tuning.

IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
