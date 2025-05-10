import arxiv
import pandas as pd
import requests
import pypdfium2 as pdfium
from io import BytesIO
import time
from google.cloud import storage
from datetime import datetime, timedelta
import os
import re
from ftfy import fix_text
import json
import arxiv 
from typing import List, Dict, Set
import logging


def remove_abstract(text):
    # Match abstract headings with optional spaces or colons after "Abstract" (case insensitive)
    abstract_patterns = [
        r'(?i)\babstract\b[:\s]*\n?',  # Matches "Abstract", "Abstract:", "abstract" (case insensitive)
    ]
    
    # Compile regex with case insensitivity
    pattern = re.compile('|'.join(abstract_patterns), re.IGNORECASE)

    # Remove the abstract heading (including the optional newline character)
    text = pattern.sub('', text, count=1)  # Only remove the first occurrence

    # Now, remove everything after the abstract up to the next major section (like "1 Introduction")
    text = re.sub(r'(?s)^(.*?)(?=\n\s*(?:1\s*Introduction|\d+\.\s*\w+))', '', text)

    return text.strip()

def clean_text(text):
    # Remove hyphenation from words split across lines
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # Replace newlines within paragraphs with a space
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def fix_encoding(text):
    return fix_text(text)

def remove_headers_footers(text):
    text = re.sub(r'\n?\s*\d+\s*\n', '\n', text)  # Remove standalone numbers (page numbers)
    text = re.sub(r'^\s*arXiv:.*\n', '', text, flags=re.MULTILINE)
    return text

def preprocess_pdf_text(text):
    text = remove_abstract(text)
    text = fix_encoding(text)
    text = remove_headers_footers(text)
    text = clean_text(text)
    
    return text




# Configuration
BUCKET_NAME = "arxiv-pdfs-25"
TARGET_COUNT = 100000
DELAY = 15  # Seconds between requests
DAYS_PER_CHUNK = 90 
EARLIEST_DATE = datetime(1991, 9, 1)  # arXiv launch date

# Setup
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_to_gcs(content, destination_blob_name):
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(content)
    print(f"Uploaded {destination_blob_name}")

class ArXivHarvester:
    def __init__(self):
        self.client = arxiv.Client()
        self.seen_ids: Set[str] = set()
        self.total_processed = 0
        self.current_date = datetime.now()

    def get_date_range_chunk(self) -> tuple[str, str]:
        """Generate the next date range chunk moving backward in time"""
        end_date = self.current_date
        start_date = end_date - timedelta(days=DAYS_PER_CHUNK)
        if start_date < EARLIEST_DATE:
            start_date = EARLIEST_DATE
            
        self.current_date = start_date
        return (start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))

    def fetch_papers(self, start_date: str, end_date: str) -> List[Dict]:
        papers = []
        query = f'submittedDate:[{start_date} TO {end_date}]'
        
        try:
            search = arxiv.Search(
                query=query,
                max_results=1000,  # Max per request
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            for result in self.client.results(search):
                paper_id = result.entry_id.split('/')[-1]
                if paper_id not in self.seen_ids:
                    papers.append({
                        'id': paper_id,
                        'title': result.title,
                        'abstract': result.summary,
                        'published': str(result.published),
                        'pdf_url': result.pdf_url,
                        'categories': result.categories
                    })
        except Exception as e:
            logger.error(f"Error fetching {start_date}-{end_date}: {str(e)}")
            time.sleep(60)
                
        return papers

    def process_paper(self, paper: Dict) -> bool:
        """Process and store individual paper"""
        try:
            start_time = time.time()
            
            # Download PDF
            response = requests.get(paper['pdf_url'], timeout=30)
            response.raise_for_status()
            
            # Extract text
            with BytesIO(response.content) as pdf_file:
                pdf = pdfium.PdfDocument(pdf_file)
                text = ''.join(page.get_textpage().get_text_bounded() for page in pdf)
                cleaned_text = preprocess_pdf_text(text)

            # Store in GCS
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = f"papers/{paper['id']}_{timestamp}"
            
            upload_to_gcs(text, f"{base_path}.txt")
            upload_to_gcs(cleaned_text, f"{base_path}_clean.txt")
            upload_to_gcs(
                json.dumps({
                    **paper,
                    "processed_at": timestamp,
                    "text_length": len(cleaned_text)
                }, indent=2),
                f"{base_path}_meta.json"
            )
            
            # Rate limiting
            elapsed = time.time() - start_time
            time.sleep(max(0, DELAY - elapsed))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed {paper['id']}: {str(e)}")
            return False

    def run(self):
        """Main harvesting loop"""
        logger.info(f"Beginning harvest of {TARGET_COUNT} papers (all categories)")
        
        while self.total_processed < TARGET_COUNT:
            if self.current_date <= EARLIEST_DATE:
                logger.error("Reached arXiv's earliest papers without hitting target!")
                break
                
            start_date, end_date = self.get_date_range_chunk()
            logger.info(f"Processing {start_date} to {end_date}...")
            
            papers = self.fetch_papers(start_date, end_date)
            if not papers:
                continue
                
            for paper in papers:
                if self.total_processed >= TARGET_COUNT:
                    break
                    
                if self.process_paper(paper):
                    self.total_processed += 1
                    self.seen_ids.add(paper['id'])
                    
                    if self.total_processed % 100 == 0:
                        logger.info(
                            f"Progress: {self.total_processed}/{TARGET_COUNT} "
                            f"({self.total_processed/TARGET_COUNT:.1%})"
                        )
            
            # Brief pause between date ranges
            time.sleep(5)
        
        logger.info(f"Harvest complete. Collected {self.total_processed} papers.")

if __name__ == "__main__":
    harvester = ArXivHarvester()
    harvester.run()