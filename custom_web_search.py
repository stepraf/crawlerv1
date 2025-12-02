"""
Custom web search utility using DuckDuckGo and crawl4ai.

This module is intentionally kept simple and synchronous:
- Uses DuckDuckGo (`DDGS`) to fetch top result URLs.
- Uses crawl4ai to synchronously scrape a subset of those URLs.
- Cleans and truncates the Markdown so it can be injected into LLM prompts.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Dict, Optional

from ddgs import DDGS
from sqlmodel import SQLModel, Field, Session, create_engine, select

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "crawl4ai is required. Install dependencies via `pip install -r requirements.txt`."
    ) from exc


logger = logging.getLogger("custom_web_search")


@dataclass
class SearchPage:
    url: str
    title: str
    content: str


# ---------------------------------------------------------------------------
# Simple SQLite-based cache for search queries and crawled pages
# ---------------------------------------------------------------------------

CACHE_DB_URL = "sqlite:///web_search_cache.db"
cache_engine = create_engine(CACHE_DB_URL, echo=False)


class CachedPage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    url: str = Field(index=True, unique=True)
    title: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SearchCache(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    query: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SearchCachePage(SQLModel, table=True):
    search_id: int = Field(foreign_key="searchcache.id", primary_key=True)
    cached_page_id: int = Field(foreign_key="cachedpage.id", primary_key=True)
    order_index: int = Field(default=0, primary_key=True)


def _init_cache_db() -> None:
    """Initialize cache database tables if they don't exist."""
    SQLModel.metadata.create_all(cache_engine)


_init_cache_db()


def _create_stealth_config() -> BrowserConfig:
    """Create a BrowserConfig with stealth mode enabled to avoid bot detection."""
    return BrowserConfig(
        enable_stealth=True,
        headless=True,
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
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
        java_script_enabled=True,
        viewport_width=1920,
        viewport_height=1080,
    )


def _clean_markdown(content: str) -> str:
    """
    Remove most navigation/menus and strip links from Markdown/HTML-ish content.

    Heuristics used:
    - Remove Markdown links: [text](url) -> text
    - Remove raw URLs
    - Strip basic <a> tags
    - Drop very short nav-like lines and cookie/menu/footer boilerplate
    """
    if not content:
        return ""

    # Strip markdown links [text](url) -> text
    content = re.sub(r"\[([^\]]+)\]\((?:http|https)[^)]+\)", r"\1", content)

    # Strip simple HTML anchors
    content = re.sub(r"<a[^>]*>", "", content, flags=re.IGNORECASE)
    content = re.sub(r"</a>", "", content, flags=re.IGNORECASE)

    # Remove bare URLs
    content = re.sub(r"https?://\S+", "", content)

    # Drop very link-heavy / boilerplate lines
    drop_keywords = [
        "cookie",
        "privacy policy",
        "terms of use",
        "terms and conditions",
        "all rights reserved",
        "navigation",
        "nav menu",
        "main menu",
        "footer",
        "accept all",
        "manage preferences",
    ]

    cleaned_lines: List[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue

        lower = stripped.lower()
        if any(kw in lower for kw in drop_keywords):
            continue

        # Drop lines that are mostly symbols
        if len(re.sub(r"[A-Za-z0-9]", "", stripped)) > len(stripped) * 0.6:
            continue

        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def _truncate(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "\n\n[Content truncated for analysis]"


async def _search_duckduckgo_async(
    query: str,
    max_results: int = 6,
    retries: int = 2,
    backoff_base: float = 2.0,
) -> List[str]:
    """Fetch top DuckDuckGo result URLs with basic retry handling (async helper)."""

    def _sync_search() -> List[Dict]:
        with DDGS(timeout=10) as ddgs:
            return list(
                ddgs.text(
                    query,
                    region="wt-wt",
                    safesearch="moderate",
                    max_results=max_results,
                )
            )

    for attempt in range(1, retries + 1):
        try:
            results: List[Dict] = await asyncio.to_thread(_sync_search)
            urls = [item["href"] for item in results if item.get("href")]
            if urls:
                logger.info("DuckDuckGo: found %s URLs for query '%s'", len(urls), query)
                print(f"[web_search] DuckDuckGo results for '{query}':")
                for i, u in enumerate(urls, start=1):
                    print(f"  [result {i}] {u}")
                return urls
        except Exception as exc:  # pragma: no cover - network dependency
            logger.warning(
                "DuckDuckGo search attempt %s/%s failed: %s",
                attempt,
                retries,
                exc,
            )
        sleep_for = backoff_base * (2 ** (attempt - 1))
        time.sleep(sleep_for + random.uniform(0, 1))

    logger.error("DuckDuckGo search exhausted retries for query '%s'", query)
    return []


async def _scrape_urls_async(
    urls: Iterable[str],
    max_pages: int,
    max_chars_per_page: int,
) -> List[SearchPage]:
    """Scrape URLs concurrently using AsyncWebCrawler, returning cleaned Markdown content."""
    unique_urls = list(dict.fromkeys(urls))[:max_pages]
    if not unique_urls:
        return []

    pages: List[SearchPage] = []

    # First try to satisfy from cache without crawling
    with Session(cache_engine) as session:
        for url in unique_urls:
            cached = session.exec(
                select(CachedPage).where(CachedPage.url == url)
            ).first()
            if cached:
                pages.append(
                    SearchPage(
                        url=cached.url,
                        title=cached.title,
                        content=cached.content,
                    )
                )

        cached_urls = {p.url for p in pages}
        urls_to_crawl = [u for u in unique_urls if u not in cached_urls]

    if urls_to_crawl:
        config = _create_stealth_config()
        async with AsyncWebCrawler(config=config) as crawler:
            async def _scrape_single(url: str) -> None:
                try:
                    result = await crawler.arun(url=url)
                    if not result:
                        return
                    markdown = getattr(result, "markdown", "") or ""
                    if not markdown.strip():
                        return
                    metadata = getattr(result, "metadata", {}) or {}
                    title = getattr(result, "title", "") or metadata.get("title", "") or url
                    cleaned = _clean_markdown(markdown)
                    truncated = _truncate(cleaned, max_chars_per_page)
                    page = SearchPage(url=url, title=title, content=truncated)
                    pages.append(page)

                    # Store/refresh in cache
                    with Session(cache_engine) as session:
                        existing = session.exec(
                            select(CachedPage).where(CachedPage.url == url)
                        ).first()
                        if existing:
                            existing.title = page.title
                            existing.content = page.content
                        else:
                            session.add(
                                CachedPage(
                                    url=url,
                                    title=page.title,
                                    content=page.content,
                                )
                            )
                        session.commit()
                except Exception as exc:  # pragma: no cover - network dependency
                    logger.warning("Scraping failed for %s: %s", url, exc)

            tasks = [asyncio.create_task(_scrape_single(url)) for url in urls_to_crawl]
            await asyncio.gather(*tasks)

    logger.info("Successfully scraped %s/%s URLs", len(pages), len(unique_urls))

    if pages:
        print(f"[web_search] Crawled {len(pages)}/{len(unique_urls)} pages (including cache hits):")
        for i, page in enumerate(pages, start=1):
            snippet = (page.content[:200] + "…") if len(page.content) > 200 else page.content
            print(f"  [page {i}] URL: {page.url}")
            print(f"           Title: {page.title}")
            print(f"           Content length: {len(page.content)} chars")
            print(f"           Snippet: {snippet.replace(chr(10), ' ')[:300]}")

    return pages


def run_web_search(
    query: str,
    max_results: int = 6,
    max_scrape: int = 6,
    max_chars_per_page: int = 10000,
) -> List[Dict[str, str]]:
    """
    Public entry point that returns:
    [{ "url": ..., "title": ..., "content": ... }, ...]
    """
    print("[web_search] ===== New web_search run =====")
    print(f"[web_search] Starting search for query: {query}")

    # Check search cache first
    with Session(cache_engine) as session:
        cached_search = session.exec(
            select(SearchCache).where(SearchCache.query == query)
        ).first()
        if cached_search:
            links = session.exec(
                select(SearchCachePage, CachedPage)
                .where(SearchCachePage.search_id == cached_search.id)
                .join(CachedPage, CachedPage.id == SearchCachePage.cached_page_id)
                .order_by(SearchCachePage.order_index)
            ).all()
            if links:
                print("[web_search] Returning cached search results")
                pages: List[SearchPage] = []
                for link, cached_page in links:
                    pages.append(
                        SearchPage(
                            url=cached_page.url,
                            title=cached_page.title,
                            content=cached_page.content,
                        )
                    )
                print(f"[web_search] Completed web_search (cache hit), {len(pages)} pages collected")
                return [
                    {"url": p.url, "title": p.title, "content": p.content}
                    for p in pages
                ]

    async def _run() -> List[SearchPage]:
        urls = await _search_duckduckgo_async(query, max_results=max_results)
        if not urls:
            print("[web_search] No URLs returned from DuckDuckGo")
            return []
        return await _scrape_urls_async(
            urls, max_pages=max_scrape, max_chars_per_page=max_chars_per_page
        )

    pages = asyncio.run(_run())

    # Store search + page associations in cache
    if pages:
        with Session(cache_engine) as session:
            search_row = session.exec(
                select(SearchCache).where(SearchCache.query == query)
            ).first()
            if not search_row:
                search_row = SearchCache(query=query)
                session.add(search_row)
                session.commit()
                session.refresh(search_row)

            for idx, page in enumerate(pages):
                cached_page = session.exec(
                    select(CachedPage).where(CachedPage.url == page.url)
                ).first()
                if not cached_page:
                    cached_page = CachedPage(
                        url=page.url,
                        title=page.title,
                        content=page.content,
                    )
                    session.add(cached_page)
                    session.commit()
                    session.refresh(cached_page)

                # Link search to page
                link = session.exec(
                    select(SearchCachePage).where(
                        (SearchCachePage.search_id == search_row.id)
                        & (SearchCachePage.cached_page_id == cached_page.id)
                    )
                ).first()
                if not link:
                    session.add(
                        SearchCachePage(
                            search_id=search_row.id,
                            cached_page_id=cached_page.id,
                            order_index=idx,
                        )
                    )
                    session.commit()

    print(f"[web_search] Completed web_search, {len(pages)} pages collected")
    return [
        {"url": page.url, "title": page.title, "content": page.content}
        for page in pages
    ]


