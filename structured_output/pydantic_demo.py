from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name: str = "Amey"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(
        ge=0,
        le=10,
        default=5.5,
        description="Decimal value representing the cgpa of the student",
    )


new_student = {"age": "25", "email": "abc@abc.com", "cgpa": 5}

student = Student(**new_student)

print(student.model_dump())
