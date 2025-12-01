"""
Extract product names and characteristics from scraped webpages using Azure OpenAI.
"""

import asyncio
import csv
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

load_dotenv()

# Configuration
MIN_CHUNK_LINES = 50
MAX_CHUNK_LINES = 100
TEMPERATURE = 1
MAX_TOKENS = 4000

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("extract_products")

# System prompt for product extraction
SYSTEM_PROMPT = """You are a product information extraction specialist.
Analyze the provided text and extract all product names and their characteristics.
For each product, identify:
- Product name
- Key characteristics/features
- Technical specifications (if mentioned)
- Product category/type

Return your findings as a JSON object with this structure:
{
  "products": [
    {
      "name": "product name",
      "category": "product category",
      "characteristics": ["feature1", "feature2", ...],
      "specifications": {"spec_name": "spec_value", ...}
    }
  ]
}
If no products are found, return {"products": []}.
Only return valid JSON."""


def get_azure_client() -> AsyncAzureOpenAI:
    """Build an AsyncAzureOpenAI client from environment variables."""
    required = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", 
                "AZURE_OPENAI_API_VERSION", "AZURE_DEPLOYMENT_NAME"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing Azure OpenAI config: {', '.join(missing)}")

    return AsyncAzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_DEPLOYMENT_NAME"],
    )


def remove_markdown_urls(content: str) -> str:
    """Remove URLs from markdown links, keeping only the link text."""
    # Remove image and link URLs, keep text
    content = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', content)
    content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
    content = re.sub(r'https?://[^\s\n\)]+', '', content)
    content = re.sub(r'www\.[^\s\n\)]+', '', content)
    
    # Clean up spaces but preserve newlines
    return '\n'.join(re.sub(r'[ \t]+', ' ', line).strip() for line in content.split('\n'))


def split_into_pages(content: str) -> List[Tuple[str, str]]:
    """
    Split content into pages based on [@@@]: URL pattern.
    Returns list of (page_url, page_content) tuples.
    """
    pages = []
    lines = content.split('\n')
    current_page_url = None
    current_page_lines = []
    
    for line in lines:
        # Check if this line is a page marker
        match = re.match(r'^\[@@@\]:\s*(.+)$', line.strip())
        if match:
            # Save previous page if it exists
            if current_page_url and current_page_lines:
                page_content = '\n'.join(current_page_lines)
                pages.append((current_page_url, page_content))
            
            # Start new page
            current_page_url = match.group(1).strip()
            current_page_lines = []
        else:
            # Add line to current page
            if current_page_url is not None:
                current_page_lines.append(line)
            elif not pages:
                # Content before first page marker - treat as first page with no URL
                current_page_url = ""
                current_page_lines.append(line)
    
    # Add final page
    if current_page_url is not None and current_page_lines:
        page_content = '\n'.join(current_page_lines)
        pages.append((current_page_url, page_content))
    
    return pages


def log_llm_call(llm_log_file: Path, page_index: int, request: Dict, response: Dict = None, error: str = None):
    """Log LLM call to file."""
    with open(llm_log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Page {page_index} - {datetime.utcnow().isoformat()}\n")
        f.write(f"{'='*80}\n")
        f.write(f"REQUEST:\n{json.dumps(request, indent=2, ensure_ascii=False)}\n")
        if response:
            f.write(f"\nRESPONSE:\n{json.dumps(response, indent=2, ensure_ascii=False)}\n")
        if error:
            f.write(f"\nERROR: {error}\n")
        f.write(f"{'='*80}\n\n")


async def extract_products(client: AsyncAzureOpenAI, page_content: str, page_url: str, page_index: int, 
                           llm_log_file: Optional[Path] = None) -> Dict[str, Any]:
    """Extract products from a page using Azure OpenAI."""
    request_data = {
        "page_index": page_index,
        "page_url": page_url,
        "model": os.environ["AZURE_DEPLOYMENT_NAME"],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "content_length": len(page_content),
        "content_preview": page_content[:500] + "..." if len(page_content) > 500 else page_content
    }
    
    try:
        response = await client.chat.completions.create(
            model=os.environ["AZURE_DEPLOYMENT_NAME"],
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": page_content},
            ],
        )
        
        content = response.choices[0].message.content
        response_data = {
            "page_index": page_index,
            "page_url": page_url,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            },
            "response": content
        }
        
        if llm_log_file:
            log_llm_call(llm_log_file, page_index, request_data, response_data)
            with open(llm_log_file, 'a', encoding='utf-8') as f:
                f.write(f"FULL PAGE CONTENT:\n{'-'*80}\n{page_content}\n{'-'*80}\n\n")
        
        parsed = json.loads(content)
        parsed["page_index"] = page_index
        parsed["page_url"] = page_url
        parsed["page_line_count"] = len(page_content.split('\n'))
        return parsed
        
    except json.JSONDecodeError as exc:
        error_msg = f"JSON parsing failed: {exc}"
        logger.error(f"Page {page_index} ({page_url}): {error_msg}")
        if llm_log_file:
            log_llm_call(llm_log_file, page_index, request_data, error=error_msg)
        return {"page_index": page_index, "page_url": page_url, "error": error_msg, "products": []}
    except Exception as exc:
        error_msg = str(exc)
        logger.error(f"Page {page_index} ({page_url}): {error_msg}")
        if llm_log_file:
            log_llm_call(llm_log_file, page_index, request_data, error=error_msg)
        return {"page_index": page_index, "page_url": page_url, "error": error_msg, "products": []}


def write_csv_row(csv_file: Path, product: Dict, page_url: str, source_file: str, write_header: bool = False):
    """Write a single product row to CSV."""
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'page_url', 'product_name', 'category', 
            'characteristics', 'specifications', 'source_file'
        ])
        if write_header:
            writer.writeheader()
        
        writer.writerow({
            'page_url': page_url,
            'product_name': product.get('name', ''),
            'category': product.get('category', ''),
            'characteristics': ', '.join(product.get('characteristics', [])) if isinstance(product.get('characteristics'), list) else str(product.get('characteristics', '')),
            'specifications': json.dumps(product.get('specifications', {}), ensure_ascii=False) if product.get('specifications') else '',
            'source_file': source_file
        })
        f.flush()
        os.fsync(f.fileno())


async def process_file(input_file: Path, output_file: Path, csv_file: Path, 
                      llm_log_file: Path, min_lines: int, max_lines: int) -> Dict[str, Any]:
    """Process file to extract product information."""
    logger.info(f"Reading {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    logger.info(f"File has {len(content.splitlines())} lines")
    
    # Split into pages
    pages = split_into_pages(content)
    logger.info(f"Found {len(pages)} pages")
    
    # Initialize output files
    if csv_file.exists():
        csv_file.unlink()
    if llm_log_file.exists():
        llm_log_file.unlink()
    
    with open(llm_log_file, 'w', encoding='utf-8') as f:
        f.write(f"LLM Call Log - {datetime.utcnow().isoformat()}\n")
        f.write(f"Source: {input_file}\nPages: {len(pages)}\n{'='*80}\n\n")
    
    # Process pages
    client = get_azure_client()
    results = []
    all_products = []
    first_csv_write = True
    skipped_pages = 0
    
    for i, (page_url, page_content) in enumerate(pages):
        # Clean markdown URLs from page content
        cleaned_content = remove_markdown_urls(page_content)
        
        # Skip pages with less than 100 characters
        if len(cleaned_content.strip()) < 100:
            logger.info(f"Skipping page {i+1} ({page_url}): only {len(cleaned_content.strip())} chars after cleaning")
            skipped_pages += 1
            continue
        
        logger.info(f"Processing page {i+1}/{len(pages)}: {page_url}")
        result = await extract_products(client, cleaned_content, page_url, i, llm_log_file)
        results.append(result)
        
        products = result.get("products", [])
        if products:
            all_products.extend(products)
            for product in products:
                write_csv_row(csv_file, product, page_url, str(input_file), first_csv_write)
                first_csv_write = False
        
        await asyncio.sleep(0.5)  # Rate limiting
    
    # Save JSON output
    output = {
        "source_file": str(input_file),
        "total_pages": len(pages),
        "pages_processed": len(results),
        "pages_skipped": skipped_pages,
        "total_products_found": len(all_products),
        "page_results": results,
        "all_products": all_products
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    return output


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract products from scraped webpages")
    parser.add_argument("input_file", help="Input text file")
    parser.add_argument("-o", "--output", help="JSON output file")
    parser.add_argument("--csv", help="CSV output file")
    parser.add_argument("--llm-log", help="LLM call log file")
    # Note: min-lines and max-lines are kept for backward compatibility but not used
    parser.add_argument("--min-lines", type=int, default=MIN_CHUNK_LINES, help="(Deprecated: pages are processed individually)")
    parser.add_argument("--max-lines", type=int, default=MAX_CHUNK_LINES, help="(Deprecated: pages are processed individually)")
    
    args = parser.parse_args()
    input_file = Path(args.input_file)
    
    if not input_file.exists():
        logger.error(f"File not found: {input_file}")
        return
    
    # Determine output files
    base = input_file.stem
    output_file = Path(args.output) if args.output else input_file.parent / f"{base}_products.json"
    csv_file = Path(args.csv) if args.csv else input_file.parent / f"{base}_products.csv"
    llm_log_file = Path(args.llm_log) if args.llm_log else input_file.parent / f"{base}_llm_calls.log"
    
    try:
        result = await process_file(input_file, output_file, csv_file, llm_log_file, 
                                   args.min_lines, args.max_lines)
        
        print("\n" + "="*60)
        print("EXTRACTION SUMMARY")
        print("="*60)
        print(f"Source: {result['source_file']}")
        print(f"Pages: {result['total_pages']} (processed: {result['pages_processed']}, skipped: {result['pages_skipped']})")
        print(f"Products: {result['total_products_found']}")
        print(f"JSON: {output_file}")
        print(f"CSV: {csv_file}")
        print(f"Log: {llm_log_file}")
        print("="*60)
        
        if result['all_products']:
            print("\nSample products:")
            for i, p in enumerate(result['all_products'][:5], 1):
                print(f"{i}. {p.get('name', 'Unknown')}")
                if p.get('category'):
                    print(f"   Category: {p['category']}")
        
    except Exception as exc:
        logger.error(f"Error: {exc}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
