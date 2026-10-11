from langchain_openai import ChatOpenAI
from dotenv import load_dotenv;

load_dotenv();

llm = ChatOpenAI(model="gpt-4o");

text = "Hello, My name is Sohail Shaikh, Email is sohail@gmail.com and my age is 20"  


res = llm.invoke(f"Please give me only name, age and email from this text: {text}")

print(res.content);

# Structure output

from pydantic import BaseModel, Field
from typing import List

class ResponseStructure(BaseModel):
    name:str = Field(description="Name of the user")
    email:str = Field(description="Email Address")
    age:int = Field(description="Age")

structure_llm = llm.with_structured_output(ResponseStructure)

res = structure_llm.invoke(f"Please give me only name, age and email from this text: {text}")

print(res.model_dump())

class Movies(BaseModel):
    title:str = Field(description="Movie Title")
    year:str = Field(description="Movie Released Year.")

class AllMovies(BaseModel):
    movies:List[Movies]


Movie_llm = llm.with_structured_output(AllMovies)

res1 = Movie_llm.invoke("Give me List of Resent movie")

print(res1)