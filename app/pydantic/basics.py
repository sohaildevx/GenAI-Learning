from pydantic import BaseModel, Field
from typing import List

class User(BaseModel):
    name: str = ""
    age: int = ""

# user = User(name="Shahid", age="This is age")

# user.name = "sohail"

# print(user)


# Pydantic Fields
class Product(BaseModel):
    title:str = Field(max_length=100)
    price:int = Field(gt=0)
    des:str = Field(min_length=3)

product = Product(title="Vivo",price=30000, des="vivo price")
# print(product)


class Order(BaseModel):
    id:str = Field(min_length=6, max_length=12)
    product: Product

# order = Order(id = "ndj2133", product=product)

# print(order.product.price)

class Carts(BaseModel):
    id:str = Field(min_length=3,max_length=12)
    items: List[Product]
    comment:str = Field(
        default="No Commit"
    )

cart = Carts(id="fjajb",items=product)
print(cart)