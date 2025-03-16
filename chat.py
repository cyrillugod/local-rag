import sys
from ollama import chat
from rich.console import Console

from app import get_embedding, console, collection

def main():

    while True:
        query = input('Enter your question: ')
        if (query == 'exit') or (query == 'quit'):
            sys.exit()

        answer = chat_response(query)
        console.print(answer)

def chat_response(query):
    '''
    Embeds query, retrieves context from ChromaDB, and returns response from LLM
    '''

    system_message = '''
    Retrieve relevant information from provided documents and return a concise and informative response.
    If the answer cannot be found, please respond "I'm sorry, I do not have the answer to that question."
    '''

    # embed query
    emb_query = get_embedding(query) # list of embedded queries
    if emb_query is None:
        console.print('[red]Error: Unable to embed query[/red]')
    

    # retrieve context
    try:
        context = collection.query(query_embeddings=emb_query,
                                   n_results=5,
                                   )
        
    except:
        console.print(f'[red]Unable to query ChromaDB.[/red]')

    # generate response
    return chat_complete(emb_query, system_message, context)

def chat_complete(emb_query: str, system_message: str, context: str) -> str:
    '''
    Completes chat response by passing query, system prompt, and context to LLM
    '''
    
    try:
        response = chat(model='deepseek-r1:8b',
                        messages=[
                            {
                            'role': 'system',
                            'content': f'{system_message}\nContext: {context}',
                             }, # system role-content

                            {
                            'role': 'user',
                            'content': emb_query,
                            }, # user role-content
                        ],
                        stream=False,
                        )
        return response['message']['content']
    except:
        console.print('[red]Error: Unable to generate response[/red]')
    
if __name__ == '__main__':
    main()