import argparse
import json
import os
import logging
from ecommerce_agent.domain.product import Product
from ecommerce_agent.application.services.products_service import ProductsService
from ecommerce_agent.application.services.rag.embeddings import EmbeddingsService
from ecommerce_agent.config import settings

parser = argparse.ArgumentParser(description='Ingest products into the database')
parser.add_argument('--directory', type=str, help='Directory to ingest products from', default=settings.DATA_PRODUCTS_DIR)
args = parser.parse_args()

class IngestProductsTable:
  def __init__(self):
    self.products_service = ProductsService()
    self.embeddings_service = EmbeddingsService()
    
  def extract_products(self, directory: str):
    logging.info(f"Extracting products from {directory}...")
    products = []
    for file in os.listdir(directory):
      with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
        _products = json.load(f)['items']
        for product in _products:
          products.append(Product(
            code=product['code'],
            name=product['name'],
            description=product['description'],
            price=product['price'],
            image_url=product['image_url'],
            stock_level=product['stock_level'],
            is_active=True
          ))
    logging.info(f"Extracted {len(products)} products")
    return products
    
  def add_embedding(self, product: Product):
    logging.info(f"Adding embedding to product {product.code}...")
    embedding = self.embeddings_service.embed_text(product.description)
    product.embedding = embedding
    return product

  def ingest_products_table(self, directory: str):
    products = self.extract_products(directory)
    for product in products:
      product = self.add_embedding(product)
      self.products_service.create_product(product)

ingest_products_table = IngestProductsTable()
ingest_products_table.products_service._create_extensions()
ingest_products_table.products_service._create_table()
ingest_products_table.products_service._create_index()

ingest_products_table.ingest_products_table(args.directory)