# ecommerce_agent/domain/document.py

from pydantic import BaseModel
from typing import List, Optional

class Document(BaseModel):
    id: Optional[int] = None
    # 'content' será el page_content del chunk para la búsqueda (embedding y BM25)
    content: str 
    embedding: Optional[List[float]] = None
    # 'window_content' será la ventana ampliada para la recuperación
    window_content: Optional[str] = None 
    # 'source' para el origen del documento
    source: Optional[str] = None 
    
    # Para la búsqueda híbrida y RRF, podemos añadir atributos dinámicos
    # con setattr, pero no son parte del modelo base de Pydantic.
    # Si quieres que se muestren en la respuesta de API o se serialicen,
    # tendrías que considerarlos aquí como Optional o usar un ResponseModel
    # separado en FastAPI. Para una tool interna, no es estrictamente necesario.
    text_rank: Optional[float] = None
    semantic_distance: Optional[float] = None
    rrf_score: Optional[float] = None

    # El campo content_tsv no es parte del modelo de aplicación, es solo de la DB.