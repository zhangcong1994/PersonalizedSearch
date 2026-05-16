from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class DocumentChunker:
    def __init__(self, chunk_size: int = 256, overlap: int = 20):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(self, text: str) -> List[str]:
        if not isinstance(text, str) or len(text) == 0:
            return []
        
        chunks = []
        text_length = len(text)
        start = 0
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk = text[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
            
            if end == text_length:
                break
            
            start = end - self.overlap
        
        return chunks
    
    def chunk_documents(self, documents: List[str]) -> List[List[str]]:
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.append(chunks)
        return all_chunks
    
    def get_chunk_stats(self, chunks: List[str]) -> Dict:
        if not chunks:
            return {}
        
        lengths = [len(chunk) for chunk in chunks]
        return {
            'total_chunks': len(chunks),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_length': sum(lengths) / len(lengths),
            'median_length': sorted(lengths)[len(lengths) // 2]
        }
    
    def chunk_with_metadata(self, document_id: str, text: str) -> List[Dict]:
        chunks = self.chunk_document(text)
        results = []
        
        for i, chunk in enumerate(chunks):
            results.append({
                'document_id': document_id,
                'chunk_index': i,
                'chunk': chunk,
                'chunk_size': len(chunk),
                'total_chunks': len(chunks)
            })
        
        return results