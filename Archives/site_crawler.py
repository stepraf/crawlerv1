"""
Domain-scoped crawler that stores Markdown-rendered pages in SQLite.

Given an entry URL, the script:
1. Crawls all reachable links that share the same base domain (including subdomains).
2. Stores each page's Markdown in a SQLModel table.
3. Concatenates all Markdown into a single text document.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List, Optional, Set
from urllib.parse import urljoin, urlparse, urldefrag

from dotenv import load_dotenv
from sqlmodel import Field, Session, SQLModel, create_engine, select

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "crawl4ai is required. Install dependencies via `pip install -r requirements.txt`."
    ) from exc


load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("site_crawler")


DATABASE_URL = os.getenv("RESEARCH_AGENT_DB_URL", "sqlite:///research_documents.db")
engine = create_engine(DATABASE_URL, echo=False)


class CrawledPage(SQLModel, table=True):
    """Stores Markdown content and HTML for a crawled URL."""

    id: Optional[int] = Field(default=None, primary_key=True)
    base_url: str
    url: str
    title: str
    markdown: str
    html: Optional[str] = Field(default=None)  # Store HTML for link extraction
    created_at: datetime = Field(default_factory=datetime.utcnow)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def generate_output_filename(entry_url: str) -> str:
    """Generate a filename from the entry URL and current timestamp."""
    # Parse the URL to extract domain
    parsed = urlparse(entry_url)
    domain = parsed.netloc or parsed.path
    
    # Remove protocol prefix if present
    domain = domain.replace("https://", "").replace("http://", "")
    
    # Remove www. prefix
    if domain.startswith("www."):
        domain = domain[4:]
    
    # Remove port number if present (e.g., :8080)
    if ':' in domain:
        domain = domain.split(':')[0]
    
    # Clean the domain name for filesystem (remove invalid chars, keep dots, hyphens, underscores)
    domain_clean = re.sub(r'[^\w\-.]', '_', domain)
    
    # Replace dots with underscores for cleaner filenames
    domain_clean = domain_clean.replace('.', '_')
    
    # Remove trailing underscores and clean up
    domain_clean = domain_clean.rstrip('_').lower()
    
    # Generate timestamp in format: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Combine into filename
    filename = f"{domain_clean}_{timestamp}.txt"
    
    return filename


class _LinkExtractor(HTMLParser):
    """Minimal HTML parser that collects href attributes."""

    def __init__(self) -> None:
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, str]]) -> None:
        if tag.lower() != "a":
            return
        for attr, value in attrs:
            if attr.lower() == "href" and value:
                self.links.append(value)
                break


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    clean, _ = urldefrag(url.strip())
    parsed = urlparse(clean)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return parsed.geturl()


def _extract_base_domain(host: str) -> str:
    """Extract base domain from host (e.g., 'www.example.com' -> 'example.com')."""
    parts = host.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def _is_same_domain_or_subdomain(host: str, base_domain: str) -> bool:
    """Check if host is the base domain or a subdomain of it."""
    if host == base_domain:
        return True
    return host.endswith("." + base_domain)


def _filter_links(raw_links: Iterable[str], base_url: str, allowed_host: str) -> List[str]:
    base_domain = _extract_base_domain(allowed_host)
    normalized: List[str] = []
    for link in raw_links:
        absolute = _normalize_url(urljoin(base_url, link))
        if not absolute:
            continue
        parsed = urlparse(absolute)
        if not _is_same_domain_or_subdomain(parsed.netloc, base_domain):
            continue
        normalized.append(absolute)
    return normalized


@dataclass
class PageData:
    url: str
    title: str
    markdown: str
    html: Optional[str] = None


def build_page_data(result: object, url: str) -> Optional[PageData]:
    if not result:
        return None
    markdown = getattr(result, "markdown", "") or ""
    if not markdown.strip():
        return None
    html = getattr(result, "html", "") or ""
    metadata = getattr(result, "metadata", {}) or {}
    title = metadata.get("title") or getattr(result, "title", "") or url
    return PageData(url=url, title=title, markdown=markdown, html=html)


def extract_links(result: object, source_url: str, allowed_host: str) -> List[str]:
    html = getattr(result, "html", "") or ""
    if not html:
        return []
    parser = _LinkExtractor()
    parser.feed(html)
    return _filter_links(parser.links, source_url, allowed_host)


def get_page_from_db(url: str) -> Optional[PageData]:
    """Retrieve a page from the database if it exists."""
    try:
        with Session(engine) as session:
            statement = select(CrawledPage).where(CrawledPage.url == url)
            page = session.exec(statement).first()
            if page:
                logger.debug("Successfully retrieved page from database: %s", url)
                return PageData(
                    url=page.url,
                    title=page.title,
                    markdown=page.markdown,
                    html=page.html,
                )
        logger.debug("Page not found in database: %s", url)
        return None
    except Exception as exc:
        logger.error("Database query failed for URL %s: %s", url, exc, exc_info=True)
        return None


def extract_links_from_page_data(page: PageData, source_url: str, allowed_host: str) -> List[str]:
    """Extract links from a PageData object (using HTML if available, otherwise markdown)."""
    if page.html:
        # Use HTML if available
        parser = _LinkExtractor()
        parser.feed(page.html)
        return _filter_links(parser.links, source_url, allowed_host)
    else:
        # Fallback: extract links from markdown (markdown links [text](url))
        import re
        markdown_links = re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', page.markdown)
        urls = [url for _, url in markdown_links]
        return _filter_links(urls, source_url, allowed_host)


def save_pages(base_url: str, pages: Iterable[PageData]) -> None:
    records = [
        CrawledPage(
            base_url=base_url,
            url=page.url,
            title=page.title,
            markdown=page.markdown,
            html=page.html,
        )
        for page in pages
    ]
    if not records:
        logger.debug("No records to save to database.")
        return
    try:
        with Session(engine) as session:
            session.add_all(records)
            session.commit()
        logger.info("Successfully saved %s pages to SQLite.", len(records))
    except Exception as exc:
        logger.error(
            "Failed to save %s pages to database: %s",
            len(records),
            exc,
            exc_info=True,
        )
        # Don't re-raise - allow crawl to continue even if save fails


def write_combined_markdown(pages: Iterable[PageData], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Deduplicate by URL first
    seen_urls: Set[str] = set()
    unique_pages: List[PageData] = []
    for page in pages:
        if page.url not in seen_urls:
            seen_urls.add(page.url)
            unique_pages.append(page)
    
    # Also deduplicate by markdown content hash to catch identical content at different URLs
    seen_content: Set[str] = set()
    final_pages: List[PageData] = []
    for page in unique_pages:
        # Use stable hash of normalized content
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
    
    # Prepend each page's markdown with its URL
    pages_with_urls = [
        f"[@@@]: {page.url}\n\n{page.markdown}" for page in final_pages
    ]
    joined = "\n\n---\n\n".join(pages_with_urls)
    output_path.write_text(joined, encoding="utf-8")
    logger.info("Combined Markdown saved to %s (%s unique pages)", output_path, len(final_pages))


def remove_duplicate_paragraphs(file_path: Path) -> None:
    """Remove duplicate paragraphs from the combined markdown file across all sections."""
    if not file_path.exists():
        logger.warning("File %s does not exist, skipping paragraph deduplication", file_path)
        return
    
    import re
    
    # Helper to normalize links (remove URLs for comparison)
    def normalize_for_comparison(text: str) -> str:
        # Replace markdown links [text](url) with just [text]
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'[\1]', text)
        return " ".join(text.split())
    
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


def _create_stealth_config() -> BrowserConfig:
    """Create a BrowserConfig with stealth mode enabled to avoid bot detection."""
    return BrowserConfig(
        enable_stealth=True,  # Enable stealth mode to avoid bot detection
        headless=True,  # Run in headless mode (can set to False for debugging)
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0",
        },
        java_script_enabled=True,  # Enable JavaScript execution
        viewport_width=1920,
        viewport_height=1080,
    )


async def crawl_domain(entry_url: str, max_pages: int = 1000) -> List[PageData]:
    normalized_entry = _normalize_url(entry_url)
    if not normalized_entry:
        raise ValueError("Entry URL must include scheme (http/https) and host.")

    parsed = urlparse(normalized_entry)
    allowed_host = parsed.netloc

    queue: deque[str] = deque([normalized_entry])
    visited: Set[str] = set()
    collected: List[PageData] = []
    pages_to_save: List[PageData] = []  # Track newly crawled pages to save

    # Use stealth configuration to avoid bot detection
    config = _create_stealth_config()
    async with AsyncWebCrawler(config=config) as crawler:
        while queue and len(collected) < max_pages:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            # Check if page exists in database first
            cached_page = get_page_from_db(current)
            if cached_page:
                logger.info("Using cached page from database: %s", current)
                collected.append(cached_page)
                # Extract links from cached page
                new_links = extract_links_from_page_data(cached_page, current, allowed_host)
                for link in new_links:
                    if link not in visited:
                        queue.append(link)
                continue

            # Page not in database, need to crawl it
            try:
                logger.info("Crawling: %s", current)
                result = await crawler.arun(url=current)
            except Exception as exc:  # pragma: no cover
                logger.warning("Fetch failed for %s: %s", current, exc)
                continue

            page = build_page_data(result, current)
            if page:
                collected.append(page)
                pages_to_save.append(page)  # Track for saving

            new_links = extract_links(result, current, allowed_host)
            for link in new_links:
                if link not in visited:
                    queue.append(link)

    # Save newly crawled pages to database
    if pages_to_save:
        save_pages(entry_url, pages_to_save)
        # Note: save_pages() logs success/failure internally

    logger.info("Collected %s pages total (requested max %s).", len(collected), max_pages)
    return collected


async def run(entry_url: str, max_pages: int, output_path: Path) -> None:
    init_db()
    pages = await crawl_domain(entry_url, max_pages=max_pages)
    if not pages:
        logger.warning("No pages found for %s", entry_url)
        return
    # Pages are now saved inside crawl_domain, so we just write the combined markdown
    write_combined_markdown(pages, output_path)
    remove_duplicate_paragraphs(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl a domain and store Markdown.")
    parser.add_argument(
        "entry_url",
        nargs="?",
        default="https://crawl4ai.com/",
        help="Starting URL to crawl (default: https://crawl4ai.com/)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1000,
        help="Maximum number of pages to crawl (default: 1000)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the combined Markdown document. If not provided, auto-generates from URL and timestamp.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        # Generate output filename if not provided
        if args.output is None:
            output_path = Path(generate_output_filename(args.entry_url))
        else:
            output_path = Path(args.output)
        
        asyncio.run(run(args.entry_url, args.max_pages, output_path))
    except KeyboardInterrupt:
        logger.info("Crawl interrupted by user.")

