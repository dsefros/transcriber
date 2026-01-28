import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

def init_qdrant_client():
    """Инициализация Qdrant клиента"""
    try:
        host = os.getenv('QDRANT_HOST', 'localhost')
        port = int(os.getenv('QDRANT_PORT', '6333'))
        api_key = os.getenv('QDRANT_API_KEY', None)
        
        # Отключение HTTPS для локального подключения
        client = QdrantClient(
            host=host,
            port=port,
            api_key=api_key,
            https=False,
            prefer_grpc=False
        )
        
        # Проверка подключения
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        print(f"✅ Qdrant подключен успешно! Существующие коллекции: {collection_names}")
        
        return client
    except Exception as e:
        print(f"❌ Ошибка подключения к Qdrant: {e}")
        raise

def create_collections_if_not_exists(client):
    """Создание необходимых коллекций если они не существуют"""
    try:
        # Коллекция для семантических фрагментов
        if not any(col.name == "semantic_fragments" for col in client.get_collections().collections):
            client.create_collection(
                collection_name="semantic_fragments",
                vectors_config=VectorParams(
                    size=384,  # Размер эмбеддингов all-MiniLM-L6-v2
                    distance=Distance.COSINE
                )
            )
            print("✅ Коллекция 'semantic_fragments' создана в Qdrant")
        
        # Коллекция для технических терминов
        if not any(col.name == "technical_terms" for col in client.get_collections().collections):
            client.create_collection(
                collection_name="technical_terms",
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )
            print("✅ Коллекция 'technical_terms' создана в Qdrant")
        
        # Коллекция для бизнес-концепций
        if not any(col.name == "business_concepts" for col in client.get_collections().collections):
            client.create_collection(
                collection_name="business_concepts",
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )
            print("✅ Коллекция 'business_concepts' создана в Qdrant")
            
    except Exception as e:
        print(f"❌ Ошибка создания коллекций в Qdrant: {e}")
        raise
