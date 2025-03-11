import os
import json
import sys
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from chromadb import Client
from chromadb.config import Settings

PERSISTENT_DIRECTORY = './chroma_db'
COLLECTION_NAME = 'local_collection'

# Initialize the console
console = Console()

# Set-up Chroma DB
settings = Settings(persist_directory=PERSISTENT_DIRECTORY, is_persistent=True)

# Initialize the Chroma DB client
client = Client(settings)

# Get or create the index/collection
collection = client.get_or_create_collection(COLLECTION_NAME) # creates at first time/ gets if already exists


# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')


def get_embedding(text: str) -> list:

    try:
        embedding = model.encode(text).tolist()
        return embedding
    except Exception as e:
        console.print(f'[red]Error: {e}[/red]')
        return None
    
PROCESSED_FILES_PATH = 'processed_files.json'


def load_processed_files():
    '''
    Load the processed files from the disk.

    Returns:
        dict: The processed files
    '''
    if os.path.exists(PROCESSED_FILES_PATH):
        with open(PROCESSED_FILES_PATH, 'r') as f:
            return json.load(f)
    else:
        return {}
    

def save_processed_files(processed_files):
    '''
    Save the processed files to disk.

    Args:
        processed_files (dict): The processed files
    '''
    with open(PROCESSED_FILES_PATH, 'w') as f:
        json.dump(processed_files, f)


def read_file(file_path: str) -> str:
    '''
    Read contents of file from docs folder.

    Args:
        file_path (str): The file path

    Returns:
        str: The content of the file
    '''
    try:
        with open(file_path, 'r') as f:
            return f.read()

    except Exception as e:
        console.print(f'[red]Error reading {file_path}: {e}[/red]')
        return ''


def split_text(text: str, chunk_size: int = 300, overlap: int = 100) -> list:
    '''
    Split the text into chunks.

    Args:
        text (str): The text to split
        chunk_size (int): The size of each chunk
        overlap (int): The overlap between chunks

    Returns:
        list: The list of chunks
    '''
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap): # defines the starting index for each chunk
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def process_file(file_path: str) -> None:
    '''
    Document ingestion.
    Process the file and add it to the index.

    Args:
        file_path (str): The file path
    '''
    file_name = os.path.basename(file_path)

    # Read the file
    text = read_file(file_path)
    if not text:
        return None
    
    # Split the text into chunks
    chunks = split_text(text)

    # Get the embeddings for the text
    vector_ids = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        if not embedding:
            continue

        vector_id = f'{file_name}_{i}' # suffixes the file name with the chunk number
        vector_ids.append(vector_id)

        # Generate metadata
        metadata = {
            'file_name': file_name,
            'chunk_number': i,
            'timestamp': datetime.now().isoformat(),
            'preview': chunk[:30] # preview of the chunk
        }

        # Add the embedding to the collection
        try:
            print(embedding)
            collection.add(
                embeddings=[embedding],
                metadata=[metadata],
                documents=[chunk],
                vector_ids=[vector_id]

            )
            console.print(f'[green]Added {vector_id} to collection[/green]')
        except Exception as e:
            console.print(f'[red]Error adding to collection: {e}[/red]')

    processed = load_processed_files()
    processed[file_name] = {
        'modified': os.path.getmtime(file_path), # gets last modified time
        'vector_ids': vector_ids,
        'name': file_name,
    } # keeps track of the processed files
    save_processed_files(processed)

    console.print(f'[green]Processed {file_name} and uploaded to vector store.[/green]')


def delete_vectors(file_name: str):

    processed = load_processed_files()
    file_data = processed.get(file_name)
    if not file_data:
        console.print(f'[red]File {file_name} not found[/red]')
        return
    
    try:
        collection.delete(where={'file_name': file_name}) # from metadata
        console.print(f'[green]Deleted vectors for {file_name}[/green]')
    except Exception as e:
        console.print(f'[red]Error deleting vectors: {e}[/red]')


def list_files():
    ''''
    List the files that have been processed and stored in the local folder.
    '''
    folder_path = 'docs'
    if not os.path.exists(folder_path):
        console.print(f'[red]Folder {folder_path} not found[/red]')
        os.mkdir(folder_path)
        console.print(f'[green]Created folder {folder_path}[/green]')

    processed = load_processed_files()

    try:
        current_files = os.listdir(folder_path)
        for file_name in current_files:
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                processed_data = processed.get(file_name)
                if processed_data:
                    console.print(f'[green]{file_name}[/green]')
                else:
                    console.print(f'[red]{file_name}[/red]')
    except Exception as e:
        console.print(f'[red]Error: {e}[/red]')


def update_files():
    '''
    Update the files in the docs folder.
    '''
    folder_path = "docs"
    if not os.path.exists(folder_path):
        console.print(f"[red]Folder {folder_path} not found[/red]")
        return

    processed = load_processed_files()

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(file_path):
            processed_data = processed.get(file_name, {})
            modified_time = os.path.getmtime(file_path)

            if processed_data and modified_time > processed_data.get("modified", 0):
                console.print(f"[yellow]Updating {file_name}[/yellow]")
                delete_vectors(file_name)

            console.print(f"[yellow]Processing {file_name}[/yellow]")
            process_file(file_path)


def pull():
    '''
    Pull updated files or exit.
    '''
    user_input = input('Type "pull" to update files or "q" to exit.').strip().lower()
    if user_input == 'pull':
        update_files()
    elif user_input == 'q':
        console.print('[green]Exiting...[/green]')
        sys.exit(0)


if 'name' == '__main__':
    while True:
        update_files()
        pull()
