from sqlalchemy import Column, Integer, String, ForeignKey, LargeBinary
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False)
    images = relationship("Image", back_populates="owner")

class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    data = Column(LargeBinary, nullable=False) # ⬅️ Blob storage here
    user_id = Column(Integer, ForeignKey('users.id'))
    mime_type = Column(String, nullable=False)
    owner = relationship("User", back_populates="images")
