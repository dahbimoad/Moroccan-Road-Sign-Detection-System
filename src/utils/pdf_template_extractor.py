import os
import argparse
from pdf2image import convert_from_path
from pathlib import Path
import cv2
import numpy as np

from src.utils.template_extractor import TemplateExtractor
from src.utils.template_manager import TemplateManager


class PDFTemplateExtractor:
    def __init__(self, pdf_path, output_dir):
        """
        Initialize the PDF template extractor.
        
        Args:
            pdf_path (str): Path to the PDF file containing road signs.
            output_dir (str): Directory where the extracted templates will be saved.
        """
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.template_manager = TemplateManager(output_dir)
        self.template_extractor = TemplateExtractor(self.template_manager)
        
    def extract_from_pdf(self, dpi=200):
        """
        Convert PDF pages to images and extract road sign templates.
        
        Args:
            dpi (int): DPI for the PDF to image conversion.
        """
        print(f"Converting PDF: {self.pdf_path} to images...")
        images = convert_from_path(self.pdf_path, dpi=dpi)
        
        print(f"Successfully converted {len(images)} pages from PDF")
        
        for i, image in enumerate(images):
            # Convert PIL image to OpenCV format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            print(f"\nProcessing page {i+1}/{len(images)}")
            self.process_page(img_cv, i+1)
    
    def process_page(self, image, page_num):
        """
        Process a single page image to extract road sign templates.
        
        Args:
            image (numpy.ndarray): OpenCV image of the PDF page.
            page_num (int): Page number for display purposes.
        """
        print(f"Starting template extraction for page {page_num}")
        print("Use the interactive interface to select and extract road signs.")
        print("Press 'q' to move to the next page, 'esc' to cancel.")
        
        # Use the template extractor to extract signs from this image
        self.template_extractor.run(image)
        
        print(f"Completed processing page {page_num}")


def main():
    parser = argparse.ArgumentParser(description="Extract road sign templates from a PDF file")
    parser.add_argument("pdf_path", help="Path to the PDF file containing road signs")
    parser.add_argument("--output_dir", default="data/templates", help="Directory for saving extracted templates")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF conversion")
    args = parser.parse_args()
    
    extractor = PDFTemplateExtractor(args.pdf_path, args.output_dir)
    extractor.extract_from_pdf(dpi=args.dpi)
    
    print("\nTemplate extraction completed.")
    print(f"Templates saved to: {args.output_dir}")


if __name__ == "__main__":
    main()