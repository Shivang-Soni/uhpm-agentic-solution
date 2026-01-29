import os

from sqalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from core.config import settings


db_url = settings.DATABASE_URL

if not db_url:
    raise ValueError("DATABASE_URL not found in environment variables.")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()