"""
Domain-scoped crawler that stores Markdown-rendered pages in SQLite.

Given an entry URL, the script:
1. Crawls all reachable links that share the same base domain (including subdomains).
2. Stores each page's Markdown in a SQLModel table.

To generate a combined markdown file from the crawled pages, use generate_markdown.py.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from typing import List, Optional, Set
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

# Maximum page size in bytes (10MB)
MAX_PAGE_SIZE = 10 * 1024 * 1024


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
    import re
    parsed = urlparse(entry_url)
    domain = parsed.netloc or parsed.path
    
    # Clean domain name
    domain = domain.replace("https://", "").replace("http://", "")
    if domain.startswith("www."):
        domain = domain[4:]
    if ':' in domain:
        domain = domain.split(':')[0]
    
    domain_clean = re.sub(r'[^\w\-.]', '_', domain).replace('.', '_').rstrip('_').lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{domain_clean}_{timestamp}.txt"


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


def _is_same_domain(host: str, base_host: str) -> bool:
    """Check if host is the same domain or a subdomain of base_host."""
    # Extract base domain (last two parts: example.com)
    base_parts = base_host.split(".")
    base_domain = ".".join(base_parts[-2:]) if len(base_parts) >= 2 else base_host
    
    return host == base_domain or host.endswith("." + base_domain)


def _filter_links(raw_links: List[str], base_url: str, allowed_host: str) -> List[str]:
    """Filter and normalize links to only include same-domain URLs."""
    normalized = []
    for link in raw_links:
        absolute = _normalize_url(urljoin(base_url, link))
        if absolute and _is_same_domain(urlparse(absolute).netloc, allowed_host):
            normalized.append(absolute)
    return normalized


@dataclass
class PageData:
    url: str
    title: str
    markdown: str
    html: Optional[str] = None


def build_page_data(result: object, url: str) -> Optional[PageData]:
    """Extract page data from crawl result."""
    if not result:
        return None
    
    markdown = getattr(result, "markdown", "") or ""
    if not markdown.strip():
        return None
    
    html = getattr(result, "html", "") or ""
    
    # Check total page size (markdown + html)
    total_size = len(markdown.encode("utf-8")) + len(html.encode("utf-8"))
    if total_size > MAX_PAGE_SIZE:
        logger.warning(
            "Skipping page %s: size %s MB exceeds maximum of %s MB",
            url,
            round(total_size / (1024 * 1024), 2),
            MAX_PAGE_SIZE / (1024 * 1024),
        )
        return None
    
    metadata = getattr(result, "metadata", {}) or {}
    title = metadata.get("title") or getattr(result, "title", "") or url
    
    return PageData(url=url, title=title, markdown=markdown, html=html)


def extract_links(html: str, source_url: str, allowed_host: str) -> List[str]:
    """Extract and filter links from HTML."""
    if not html:
        return []
    parser = _LinkExtractor()
    parser.feed(html)
    return _filter_links(parser.links, source_url, allowed_host)


def get_page_from_db(url: str) -> Optional[PageData]:
    """Retrieve a page from the database if it exists."""
    try:
        with Session(engine) as session:
            page = session.exec(
                select(CrawledPage).where(CrawledPage.url == url)
            ).first()
            if page:
                return PageData(
                    url=page.url,
                    title=page.title,
                    markdown=page.markdown,
                    html=page.html,
                )
        return None
    except Exception as exc:
        logger.error("Database query failed for %s: %s", url, exc)
        return None




def extract_links_from_page_data(page: PageData, source_url: str, allowed_host: str) -> List[str]:
    """Extract links from a PageData object."""
    return extract_links(page.html or "", source_url, allowed_host)


def save_page(base_url: str, page: PageData) -> None:
    """Save a single page to the database (upsert)."""
    try:
        with Session(engine) as session:
            existing = session.exec(
                select(CrawledPage).where(CrawledPage.url == page.url)
            ).first()
            
            if existing:
                # Update existing
                existing.title = page.title
                existing.markdown = page.markdown
                existing.html = page.html
            else:
                # Insert new
                session.add(CrawledPage(
                    base_url=base_url,
                    url=page.url,
                    title=page.title,
                    markdown=page.markdown,
                    html=page.html,
                ))
            session.commit()
    except Exception as exc:
        logger.error("Failed to save page %s: %s", page.url, exc)



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


async def crawl_domain(entry_url: str, max_pages: int = 1000) -> int:
    """
    Crawl a domain and save each page to the database immediately.
    Returns the total number of pages crawled (including cached ones).
    """
    normalized_entry = _normalize_url(entry_url)
    if not normalized_entry:
        raise ValueError("Entry URL must include scheme (http/https) and host.")

    parsed = urlparse(normalized_entry)
    allowed_host = parsed.netloc

    queue: deque[str] = deque([normalized_entry])
    visited: Set[str] = set()
    pages_crawled = 0  # Track count instead of storing pages in memory

    # Use stealth configuration to avoid bot detection
    config = _create_stealth_config()
    async with AsyncWebCrawler(config=config) as crawler:
        while queue and pages_crawled < max_pages:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            # Check if page exists in database first
            cached_page = get_page_from_db(current)
            if cached_page:
                logger.info("Using cached page from database: %s", current)
                pages_crawled += 1
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
                save_page(entry_url, page)
                pages_crawled += 1
                
                # Extract links from HTML
                html = getattr(result, "html", "") or ""
                new_links = extract_links(html, current, allowed_host)
                for link in new_links:
                    if link not in visited:
                        queue.append(link)

    logger.info("Crawled %s pages total (requested max %s).", pages_crawled, max_pages)
    return pages_crawled


async def run(entry_url: str, max_pages: int) -> None:
    """
    Run the crawler to collect pages and save them to the database.
    
    Note: To generate the combined markdown file, use the generate_markdown.py script
    separately after crawling is complete.
    """
    init_db()
    pages_crawled = await crawl_domain(entry_url, max_pages=max_pages)
    if pages_crawled == 0:
        logger.warning("No pages found for %s", entry_url)
        return
    
    logger.info(
        "Crawling complete. Use generate_markdown.py to create combined markdown file."
    )


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
        default=10000,
        help="Maximum number of pages to crawl (default: 10000)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(run(args.entry_url, args.max_pages))
    except KeyboardInterrupt:
        logger.info("Crawl interrupted by user.")


