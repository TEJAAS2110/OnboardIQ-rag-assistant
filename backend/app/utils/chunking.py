from typing import List, Dict, Any

class SmartChunker:
    """Chunk documents intelligently"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(self, text: str, metadata: Dict) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_index': len(chunks),
                    'start_char': start,
                    'end_char': end
                }
            })
            
            start += self.chunk_size - self.overlap
        
        return chunks
