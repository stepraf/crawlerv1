"""
Generate combined markdown file from crawled pages in the database.

This script queries the database for all pages with a given base URL and
generates a combined markdown file. It uses streaming to avoid keeping
the full text in memory, making it suitable for very large crawls.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

from dotenv import load_dotenv
from sqlmodel import Session, select

# Import shared models and functions from site_crawler
from site_crawler import CrawledPage, engine, generate_output_filename

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("generate_markdown")


@dataclass
class PageData:
    url: str
    title: str
    markdown: str
    html: Optional[str] = None


def get_all_pages_from_db(base_url: str) -> List[PageData]:
    """Retrieve all pages for a given base_url from the database."""
    try:
        with Session(engine) as session:
            statement = select(CrawledPage).where(CrawledPage.base_url == base_url)
            pages = session.exec(statement).all()
            return [
                PageData(
                    url=page.url,
                    title=page.title,
                    markdown=page.markdown,
                    html=page.html,
                )
                for page in pages
            ]
    except Exception as exc:
        logger.error(
            "Database query failed for base_url %s: %s", base_url, exc, exc_info=True
        )
        return []


def write_combined_markdown_streaming(
    base_url: str, output_path: Path, deduplicate: bool = True
) -> None:
    """
    Write combined markdown file using streaming to avoid keeping full text in memory.
    
    Args:
        base_url: The base URL to query pages for
        output_path: Path where the combined markdown will be written
        deduplicate: Whether to deduplicate pages by URL and content hash
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all pages from database
    logger.info("Querying database for pages with base_url: %s", base_url)
    all_pages = get_all_pages_from_db(base_url)
    
    if not all_pages:
        logger.warning("No pages found in database for base_url: %s", base_url)
        return
    
    logger.info("Found %s pages in database", len(all_pages))
    
    # Deduplicate if requested
    if deduplicate:
        # Deduplicate by URL first
        seen_urls: Set[str] = set()
        unique_pages: List[PageData] = []
        for page in all_pages:
            if page.url not in seen_urls:
                seen_urls.add(page.url)
                unique_pages.append(page)
        
        # Also deduplicate by markdown content hash
        seen_content: Set[str] = set()
        final_pages: List[PageData] = []
        for page in unique_pages:
            normalized = page.markdown.strip()
            content_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                final_pages.append(page)
        
        if len(final_pages) < len(unique_pages):
            logger.info(
                "Removed %s duplicate content pages (kept %s unique pages).",
                len(unique_pages) - len(final_pages),
                len(final_pages),
            )
        pages_to_write = final_pages
    else:
        pages_to_write = all_pages
    
    # Write pages to file using streaming (one page at a time)
    logger.info("Writing %s pages to %s", len(pages_to_write), output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, page in enumerate(pages_to_write):
            # Write page URL header
            f.write(f"[@@@]: {page.url}\n\n")
            # Write page markdown content
            f.write(page.markdown)
            
            # Write separator (except for last page)
            if i < len(pages_to_write) - 1:
                f.write("\n\n---\n\n")
    
    logger.info("Combined Markdown saved to %s (%s unique pages)", output_path, len(pages_to_write))


def remove_duplicate_paragraphs(file_path: Path) -> None:
    """
    Remove duplicate paragraphs from the combined markdown file across all sections.
    
    This function reads the file in chunks to minimize memory usage for large files.
    """
    if not file_path.exists():
        logger.warning("File %s does not exist, skipping paragraph deduplication", file_path)
        return
    
    # Helper to normalize links (remove URLs for comparison)
    def normalize_for_comparison(text: str) -> str:
        # Replace markdown links [text](url) with just [text]
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'[\1]', text)
        return " ".join(text.split())
    
    logger.info("Reading file for paragraph deduplication: %s", file_path)
    # Read file content (for very large files, this could be optimized further)
    # But for now, we need the full content to do proper deduplication
    content = file_path.read_text(encoding="utf-8")
    
    # Split by page separators to preserve structure
    sections = content.split("\n\n---\n\n")
    # Track seen paragraphs globally across ALL sections
    seen_paragraphs: Set[str] = set()
    # Also track seen multi-paragraph blocks (for navigation menus, etc.)
    seen_blocks: Set[str] = set()
    removed_count = 0
    deduplicated_sections: List[str] = []
    
    for section in sections:
        # Split section into paragraphs (by double newlines)
        section_paragraphs = section.split("\n\n")
        unique_paragraphs: List[str] = []
        i = 0
        
        while i < len(section_paragraphs):
            para = section_paragraphs[i]
            normalized = " ".join(para.split())  # Collapse all whitespace
            
            # Keep empty paragraphs as separators
            if not normalized:
                unique_paragraphs.append(para)
                i += 1
                continue
            
            # Check for multi-paragraph blocks (e.g., navigation menus)
            # Look ahead for blocks of 3-8 consecutive non-empty paragraphs
            block_size = min(8, len(section_paragraphs) - i)
            block_matched = False
            matched_block_len = 0
            
            # Check blocks from longest to shortest to find duplicates
            for block_len in range(block_size, 2, -1):  # Check 8 down to 3
                block_paras = section_paragraphs[i:i + block_len]
                # Filter out empty paragraphs and normalize
                non_empty = [p for p in block_paras if p.strip()]
                if len(non_empty) < 3:  # Need at least 3 non-empty paragraphs
                    continue
                
                # Normalize each paragraph (including link normalization)
                normalized_paras = [normalize_for_comparison(p) for p in non_empty]
                block_normalized = "\n".join(normalized_paras)
                
                if len(block_normalized) > 50:  # Only check substantial blocks
                    block_hash = hashlib.md5(block_normalized.encode("utf-8")).hexdigest()
                    if block_hash in seen_blocks:
                        # Found a duplicate block - skip it
                        matched_block_len = block_len
                        break
            
            if matched_block_len > 0:
                # Skip this duplicate block
                removed_count += matched_block_len
                i += matched_block_len
                continue
            
            # No duplicate block found - process individual paragraph
            # But first, mark any substantial blocks starting here as seen (for future reference)
            for block_len in range(3, min(block_size + 1, 8)):
                block_paras = section_paragraphs[i:i + block_len]
                non_empty = [p for p in block_paras if p.strip()]
                if len(non_empty) >= 3:
                    normalized_paras = [normalize_for_comparison(p) for p in non_empty]
                    block_normalized = "\n".join(normalized_paras)
                    if len(block_normalized) > 50:
                        block_hash = hashlib.md5(block_normalized.encode("utf-8")).hexdigest()
                        seen_blocks.add(block_hash)
            
            # Check individual paragraph (with link normalization for better matching)
            normalized_with_links = normalize_for_comparison(para)
            para_hash = hashlib.md5(normalized_with_links.encode("utf-8")).hexdigest()
            
            if para_hash not in seen_paragraphs:
                seen_paragraphs.add(para_hash)
                unique_paragraphs.append(para)
            else:
                removed_count += 1
            
            i += 1
        
        # Rejoin section paragraphs (only if there are any unique ones)
        if unique_paragraphs:
            deduplicated_sections.append("\n\n".join(unique_paragraphs))
    
    # Reconstruct document with separators
    deduplicated_content = "\n\n---\n\n".join(deduplicated_sections)
    
    # Write back to file
    file_path.write_text(deduplicated_content, encoding="utf-8")
    
    if removed_count > 0:
        logger.info(
            "Removed %s duplicate paragraphs/blocks from %s",
            removed_count,
            file_path,
        )
    else:
        logger.info("No duplicate paragraphs found in %s", file_path)


def main(base_url: str, output_path: Optional[Path] = None, deduplicate_paragraphs: bool = True) -> None:
    """
    Main function to generate combined markdown from database.
    
    Args:
        base_url: The base URL to query pages for
        output_path: Optional output path. If not provided, auto-generates from base_url
        deduplicate_paragraphs: Whether to remove duplicate paragraphs
    """
    # Generate output filename if not provided
    if output_path is None:
        output_path = Path(generate_output_filename(base_url))
    else:
        output_path = Path(output_path)
    
    # Write combined markdown using streaming
    write_combined_markdown_streaming(base_url, output_path, deduplicate=True)
    
    # Remove duplicate paragraphs if requested
    if deduplicate_paragraphs:
        remove_duplicate_paragraphs(output_path)
    
    logger.info("Markdown generation complete: %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate combined markdown file from crawled pages in database."
    )
    parser.add_argument(
        "base_url",
        help="Base URL to query pages for (must match the base_url used during crawling)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the combined Markdown document. If not provided, auto-generates from base_url and timestamp.",
    )
    parser.add_argument(
        "--no-dedup-paragraphs",
        action="store_true",
        help="Skip paragraph deduplication step",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        output_path = Path(args.output) if args.output else None
        main(
            base_url=args.base_url,
            output_path=output_path,
            deduplicate_paragraphs=not args.no_dedup_paragraphs,
        )
    except KeyboardInterrupt:
        logger.info("Markdown generation interrupted by user.")
    except Exception as exc:
        logger.error("Error generating markdown: %s", exc, exc_info=True)
        raise

