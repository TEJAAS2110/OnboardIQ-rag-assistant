from typing import Dict, Any
import os
from pypdf import PdfReader

class DocumentProcessor:
    """Process different document types"""
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Extract text from various file types"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            content = self._process_pdf(file_path)
        elif file_ext == '.txt':
            content = self._process_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        return {
            'content': content,
            'file_name': os.path.basename(file_path),
            'file_type': file_ext[1:],
            'file_path': file_path,
            'metadata': {
                'file_name': os.path.basename(file_path),
                'file_type': file_ext[1:]
            }
        }
    
    def _process_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def _process_txt(self, file_path: str) -> str:
        """Read text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
