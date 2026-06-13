from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import ask_question

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'], # GET, POST, etc.
    allow_headers=['*'] # content-type, authorization
)

class ChatRequest(BaseModel):
    '''this ensures that only send str to RAG endpoint'''
    message: str


@app.post('/chat')
def chat_endpoint(req: ChatRequest):

    answer = ask_question(req.message)

    return {'answer': answer}