"""
Async research automation script.

Searches DuckDuckGo, scrapes pages with crawl4ai, summarizes via Azure OpenAI,
and stores structured results in SQLite using SQLModel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional

from dotenv import load_dotenv
from duckduckgo_search import DDGS
from openai import AsyncAzureOpenAI
from sqlmodel import Field, Session, SQLModel, create_engine


TEMPERATURE=1
MAX_TOKENS=20000


try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig
except ImportError as exc:  # pragma: no cover - library should exist if requirements installed
    raise SystemExit(
        "crawl4ai is required. Install dependencies via `pip install -r requirements.txt`."
    ) from exc


load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("research_agent")


DATABASE_URL = os.getenv("RESEARCH_AGENT_DB_URL", "sqlite:///research_documents.db")
engine = create_engine(DATABASE_URL, echo=False)


class ResearchDocument(SQLModel, table=True):
    """SQLite-backed record describing a research artifact."""

    id: Optional[int] = Field(default=None, primary_key=True)
    query: str
    url: str
    title: str
    full_markdown_content: str
    ai_summary: str
    key_insights: str  # Stored as JSON array
    created_at: datetime = Field(default_factory=datetime.utcnow)


def init_db() -> None:
    """Create tables if needed."""
    SQLModel.metadata.create_all(engine)


def get_azure_client() -> AsyncAzureOpenAI:
    """Build an AsyncAzureOpenAI client from environment variables."""
    env_names = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_DEPLOYMENT_NAME",
    ]
    missing = [name for name in env_names if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            f"Missing Azure OpenAI configuration for: {', '.join(missing)}. "
            "Populate them in your environment or .env file."
        )

    return AsyncAzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_DEPLOYMENT_NAME"],
    )


async def search_duckduckgo(
    query: str,
    max_results: int = 5,
    retries: int = 3,
    backoff_base: float = 2.0,
) -> List[str]:
    """Fetch top DuckDuckGo URLs with basic retry handling."""
    def _sync_search() -> List[dict]:
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
            results = await asyncio.to_thread(_sync_search)
            urls = [item["href"] for item in results if item.get("href")]
            if urls:
                logger.info("Found %s URLs for query '%s'", len(urls), query)
                return urls
        except Exception as exc:  # pragma: no cover - network dependency
            logger.warning(
                "DuckDuckGo search attempt %s/%s failed: %s",
                attempt,
                retries,
                exc,
            )
        # exponential backoff with jitter
        sleep_for = backoff_base * (2 ** (attempt - 1))
        await asyncio.sleep(sleep_for + random.uniform(0, 1))

    logger.error("DuckDuckGo search exhausted retries for query '%s'", query)
    return []


@dataclass
class ScrapedPage:
    url: str
    title: str
    markdown: str


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


async def _scrape_single(
    crawler: AsyncWebCrawler,
    url: str,
    semaphore: asyncio.Semaphore,
    timeout: float,
) -> Optional[ScrapedPage]:
    async with semaphore:
        try:
            result = await asyncio.wait_for(
                crawler.arun(url=url),
                timeout=timeout,
            )
            if not result:
                return None
            markdown = getattr(result, "markdown", "") or ""
            if not markdown.strip():
                return None
            metadata = getattr(result, "metadata", {}) or {}
            title = getattr(result, "title", "") or metadata.get("title", "") or url
            return ScrapedPage(url=url, title=title, markdown=markdown)
        except asyncio.TimeoutError:
            logger.warning("Scraping timed out for %s", url)
        except Exception as exc:  # pragma: no cover - network dependency
            logger.warning("Scraping failed for %s: %s", url, exc)
        return None


async def scrape_urls(
    urls: Iterable[str],
    max_concurrency: int = 3,
    timeout: float = 30.0,
) -> List[ScrapedPage]:
    """Scrape multiple URLs concurrently returning Markdown content."""
    urls = list(dict.fromkeys(urls))  # dedupe while preserving order
    if not urls:
        return []

    semaphore = asyncio.Semaphore(max_concurrency)
    # Use stealth configuration to avoid bot detection
    config = _create_stealth_config()
    async with AsyncWebCrawler(config=config) as crawler:
        tasks = [
            asyncio.create_task(_scrape_single(crawler, url, semaphore, timeout))
            for url in urls
        ]
        results = await asyncio.gather(*tasks)

    scraped = [page for page in results if page]
    logger.info("Successfully scraped %s/%s URLs", len(scraped), len(urls))
    return scraped


def _truncate_markdown(content: str, max_chars: int = 12000) -> str:
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "\n\n[Content truncated for analysis]"


ASYNC_CLIENT: Optional[AsyncAzureOpenAI] = None


async def analyze_content(markdown: str) -> dict:
    """Send Markdown to Azure OpenAI and parse structured output."""
    global ASYNC_CLIENT
    if ASYNC_CLIENT is None:
        ASYNC_CLIENT = get_azure_client()

    system_prompt = (
        "You are an experienced technology research analyst. "
        "Read the provided Markdown and respond with JSON containing:\n"
        "summary: concise paragraph\n"
        "key_insights: array of 3-5 bullet strings capturing high-signal insights.\n"
        "Only return valid JSON."
    )

    response = await ASYNC_CLIENT.chat.completions.create(
        model=os.environ["AZURE_DEPLOYMENT_NAME"],
        temperature=TEMPERATURE,
        max_completion_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _truncate_markdown(markdown)},
        ],
    )
    content = response.choices[0].message.content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse model JSON: %s\nPayload: %s", exc, content)
        raise
    return parsed


def _serialize_key_insights(value: Iterable[str]) -> str:
    if isinstance(value, str):
        return json.dumps([value])
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    return json.dumps(cleaned, ensure_ascii=False)


def save_documents(documents: List[ResearchDocument]) -> None:
    if not documents:
        return
    with Session(engine) as session:
        session.add_all(documents)
        session.commit()


async def process_query(query: str, max_results: int = 5) -> List[ResearchDocument]:
    """Full workflow orchestration for a single query."""
    urls = await search_duckduckgo(query, max_results=max_results)
    scraped_pages = await scrape_urls(urls)
    documents: List[ResearchDocument] = []

    for page in scraped_pages:
        try:
            analysis = await analyze_content(page.markdown)
        except Exception as exc:  # pragma: no cover
            logger.error("Analysis failed for %s: %s", page.url, exc)
            continue

        summary = analysis.get("summary", "").strip()
        insights = analysis.get("key_insights") or []

        document = ResearchDocument(
            query=query,
            url=page.url,
            title=page.title,
            full_markdown_content=page.markdown,
            ai_summary=summary,
            key_insights=_serialize_key_insights(insights),
        )
        documents.append(document)

    if documents:
        await asyncio.to_thread(save_documents, documents)
        logger.info("Persisted %s documents.", len(documents))
    else:
        logger.warning("No documents to persist for query '%s'.", query)
    return documents


async def main() -> None:
    init_db()
    query = "competitors of crawl4ai"
    await process_query(query)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Research agent interrupted by user.")

