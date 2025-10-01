import re
from typing import List, Dict, Any
from app.config import get_settings

settings = get_settings()

class SmartChunker:
    """
    Intelligent chunking that preserves semantic boundaries.
    Keeps tables, lists, and sections together.
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Main chunking method that preserves document structure.
        """
        chunks = []
        
        # Split by major sections first
        sections = self._split_into_sections(content)
        
        for section_idx, section in enumerate(sections):
            # Handle tables separately
            if '[TABLE]' in section:
                table_chunks = self._chunk_table(section, metadata, section_idx)
                chunks.extend(table_chunks)
            else:
                # Regular text chunking with overlap
                text_chunks = self._chunk_text(section, metadata, section_idx)
                chunks.extend(text_chunks)
        
        return chunks
    
    def _split_into_sections(self, content: str) -> List[str]:
        """Split document into logical sections"""
        # Split on page markers, major headings, or double newlines
        sections = re.split(r'\n{3,}|\[PAGE \d+\]|(?=\n##)', content)
        return [s.strip() for s in sections if s.strip()]
    
    def _chunk_table(self, table_text: str, metadata: Dict[str, Any], section_idx: int) -> List[Dict[str, Any]]:
        """Keep tables together as single chunks"""
        return [{
            'text': table_text,
            'metadata': {
                **metadata,
                'chunk_type': 'table',
                'section_idx': section_idx,
            }
        }]
    
    def _chunk_text(self, text: str, metadata: Dict[str, Any], section_idx: int) -> List[Dict[str, Any]]:
        """Chunk text with overlap, preserving sentence boundaries"""
        # Extract page number if present
        page_match = re.search(r'\[PAGE (\d+)\]', text)
        page_num = page_match.group(1) if page_match else None
        
        # Remove page markers from text
        clean_text = re.sub(r'\[PAGE \d+\]', '', text).strip()
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size:
                if current_chunk:
                    # Create chunk
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            **metadata,
                            'chunk_type': 'text',
                            'section_idx': section_idx,
                            'page_number': page_num,
                            'char_count': len(chunk_text),
                            'word_count': len(chunk_text.split()),
                        }
                    })
                    
                    # Keep overlap sentences
                    overlap_words = ' '.join(current_chunk).split()[-self.overlap:]
                    current_chunk = [' '.join(overlap_words), sentence]
                    current_length = len(overlap_words) + sentence_length
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_type': 'text',
                    'section_idx': section_idx,
                    'page_number': page_num,
                    'char_count': len(chunk_text),
                    'word_count': len(chunk_text.split()),
                }
            })
        
        return chunks


# Test function
if __name__ == "__main__":
    chunker = SmartChunker(chunk_size=200, overlap=30)
    
    sample_text = """
    [PAGE 1]
    Employee Handbook
    
    ## Leave Policy
    All employees are entitled to 15 days of paid leave per year. 
    Leave requests must be submitted at least 2 weeks in advance.
    Emergency leave can be granted with manager approval.
    
    [TABLE]
    Leave Type | Days | Approval
    Casual | 5 | Manager
    Sick | 7 | HR
    """
    
    metadata = {"file_name": "handbook.pdf", "file_type": "pdf"}
    chunks = chunker.chunk_document(sample_text, metadata)
    
    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({chunk['metadata']['chunk_type']}):")
        print(chunk['text'][:100] + "...")