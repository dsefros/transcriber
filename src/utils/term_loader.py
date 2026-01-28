from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import uuid

client = QdrantClient(host='localhost', port=6333)
encoder = SentenceTransformer('all-MiniLM-L6-v2')

term_data = {
    'term_id': 'tms',
    'term_name': 'TMS',
    'full_name': 'Terminal Management System',
    'definition': 'Программное обеспечение для централизованного управления парком POS-терминалов. Обеспечивает удаленную конфигурацию, мониторинг состояния, обновление прошивок и сбор диагностической информации с терминалов.',
    'related_terms': ['База данных', 'POS-терминал', 'Конфигурация', 'Мониторинг', 'Лицензирование', 'Somers.POS'],
    'business_context': ['управление сетью терминалов', 'Развертывание новых функций на терминалах', 'Диагностика проблем с терминалами', 'Обеспечение соответствия требованиям НСПК'],
    'regulatory_info': 'Используется для обеспечения соответствия требованиям регуляторов по безопасности обработки платежных данных и удаленному управлению терминалами.',
    'common_misconceptions': [],
    'examples': [
        'Терминал сходил на TMS и получил конфигурацию.',
        'Терминал использует TMS, что-бы обновлять свои параметры.',
        'На TMS можно отслеживать работу терминалов.',
        'Инженер Сомерс заходит в TMS для выставления настроек СБП на новые терминалы'
    ],
    'importance_level': 9,
    'last_updated': '2026-01-14'
}

embedding = encoder.encode(term_data['definition']).tolist()
# Используем UUID вместо строки
point_id = str(uuid.uuid4())

client.upsert(
    collection_name='technical_terms',
    points=[models.PointStruct(id=point_id, vector=embedding, payload=term_data)]
)
print(f'✅ TMS добавлен в Qdrant! ID: {point_id}')
