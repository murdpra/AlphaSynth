from pydantic import BaseModel


class QueryIn(BaseModel):
    query: str
    company: str
    k: int = 4
