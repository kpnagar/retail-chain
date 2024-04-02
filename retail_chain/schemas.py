import uuid
from datetime import datetime
from typing import List

from pydantic import BaseModel, UUID4, Field


class Category(BaseModel):
    id: int
    name: str
    image: str
    creationAt: datetime
    updatedAt: datetime


class Product(BaseModel):
    id: int
    title: str
    price: float
    description: str
    images: List[str]
    creationAt: datetime
    updatedAt: datetime
    category: Category


class Order(BaseModel):
    id: UUID4 = Field(default_factory=uuid.uuid4)
    product: Product
    total_amount: float
    order_date: datetime
    status: str
