from typing import TypedDict


class Person(TypedDict):

    name: str
    age: int

p1 : Person = {'name': "Jack", 'age': 199 }

print(p1)