import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, func
from sqlalchemy import text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime, timezone
import uuid
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

class Meeting(Base):
    __tablename__ = 'meetings'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String, unique=True, nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=False, default=func.now())
    duration_sec = Column(Integer, nullable=False)
    audio_hash = Column(String(64), unique=True, nullable=False)
    processed_at = Column(DateTime(timezone=True), default=func.now())
    status = Column(String(20), default='raw')
    quality_score = Column(Float)
    context_tags = Column(JSON, default=[])
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

class Speaker(Base):
    __tablename__ = 'speakers'
    
    id = Column(Integer, primary_key=True)
    external_id = Column(String(50), unique=True)
    name = Column(String(100))
    role = Column(String(50))
    voice_embedding = Column(JSON)
    confidence_score = Column(Float)
    last_confirmed = Column(DateTime(timezone=True))
    is_verified = Column(Boolean, default=False)
    voice_samples_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

class Fragment(Base):
    __tablename__ = 'fragments'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    meeting_id = Column(Integer, ForeignKey('meetings.id', ondelete='CASCADE'), nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    speaker_id = Column(Integer, ForeignKey('speakers.id'), nullable=False)
    text = Column(Text, nullable=False)
    raw_text = Column(Text, nullable=False)
    semantic_cluster = Column(Integer)
    importance_score = Column(Float, default=0.5)
    business_value = Column(String(50))
    technical_terms = Column(JSON, default=[])
    qdrant_id = Column(String(36))
    is_edited = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    
    meeting = relationship("Meeting")
    speaker = relationship("Speaker")

def init_db():
    """Инициализация базы данных"""
    try:
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            raise ValueError("DATABASE_URL не установлен в переменных окружения")
        
        engine = create_engine(
            database_url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            connect_args={
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            }
        )
        
        # Создание всех таблиц
        Base.metadata.create_all(engine)
        
        # Проверка подключения
        with engine.connect() as connection:
            version = connection.execute(text("SELECT version()")).scalar()
            print(f"✅ Postgres подключен успешно! Версия: {version}")
        
        return engine
    except Exception as e:
        print(f"❌ Ошибка инициализации базы данных: {e}")
        raise

def get_db_session(engine=None):
    """Получение сессии базы данных"""
    if engine is None:
        engine = init_db()
    
    Session = sessionmaker(bind=engine)
    return Session()
