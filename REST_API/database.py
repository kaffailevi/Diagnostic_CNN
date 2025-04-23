from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base # assumes your SQLAlchemy models are in models.py

# Replace with your actual PostgreSQL connection string
DATABASE_URL = "postgresql://myuser:mypassword@localhost:5432/mydatabase"

# Create the engine
engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# This function creates the tables in your DB based on your models
def init_db():
    Base.metadata.create_all(bind=engine)
