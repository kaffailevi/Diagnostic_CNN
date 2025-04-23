from pydantic import BaseModel
from typing import List, Optional

class ImageOut(BaseModel):
    id: int
    filename: str

    class Config:
        orm_mode = True

class UserOut(BaseModel):
    id: int
    email: str
    images: List[ImageOut]

    class Config:
        orm_mode = True
