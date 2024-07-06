import pc
import requests
from bs4 import BeautifulSoup
from safetensors import torch


def scrape_books(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    books = []

    book_elements = soup.find_all('article', class_='product_pod')
    for book_element in book_elements:
        title_element = book_element.find('h3').find('a')
        price_element = book_element.find('p', class_='price_color')

        if title_element and price_element:
            title = title_element['title']
            price = price_element.text.strip()
            books.append({'title': title, 'price': price})
    return books


url = 'http://books.toscrape.com/'
scraped_books = scrape_books(url)
print(scraped_books)

# Capture Screenshots With Selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

def capture_screenshot(url, output_path):
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    driver.get(url)
    driver.save_screenshot(output_path)
    driver.quit()

screenshot_path = 'screenshot.png'
capture_screenshot(url, screenshot_path)

# Compare with LLM Results
from transformers import pipeline

# Initialize the LLM pipeline without authentication
llm = pipeline('text-generation', model='gpt2')

def compare_with_llm(scraped_data):
    # Generate text using the LLM
    llm_result = llm("List books with titles and authors", max_length=50, truncation=True, pad_token_id=llm.tokenizer.eos_token_id)
    # For simplicity, we assume llm_result is a list of dicts
    llm_books = [{'title': result['generated_text'].strip(), 'author': 'Unknown'} for result in llm_result]
    return scraped_data == llm_books

# Sample scraped data
scraped_books = [{'title': 'Sample Book', 'author': 'Author Name'}]

# Compare scraped data with LLM result
llm_comparison = compare_with_llm(scraped_books)
print(llm_comparison)



import os
import numpy as np
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone
api_key = 'bcdd5647-f211-4571-af09-2268468cfe0e'
pc = Pinecone(api_key=api_key)

# Create index if it doesn't exist
index_name = 'book-index'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=128,  # Assuming 128 dimensions for simplicity
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',  # Change to the default region
            region='us-east-1'  # Change to the default region
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Sample scraped data
scraped_books = [{'title': 'Sample Book', 'author': 'Author Name'}]

# Generate non-zero embeddings (a list of random floats) for simplicity
dummy_embedding = np.random.rand(128).tolist()

# Store results
for book in scraped_books:
    index.upsert([(book['title'], dummy_embedding, book)])  # Using title as the vector ID and adding dummy embedding

# Querying from the index
query_embedding = np.random.rand(128).tolist()  # Query with a random non-zero embedding
query_result = index.query(vector=query_embedding, top_k=5)  # Use keyword arguments
print(query_result)


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Load model and tokenizer
model_name = 'distilgpt2'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

# Apply LoRA to the model
lora_model = get_peft_model(model, lora_config)

# Quantize the model
lora_model = torch.quantization.quantize_dynamic(
    lora_model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save the quantized model as a whole object
torch.save(lora_model, 'quantized_gpt4_lora.pth')

