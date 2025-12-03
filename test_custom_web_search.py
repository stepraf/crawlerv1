"""
Simple manual test script for `custom_web_search.run_web_search`.

Usage:
    python3 test_custom_web_search.py "France Orange router"
or:
    python3 test_custom_web_search.py
    (uses a default demo query)
"""

from __future__ import annotations

import sys
from pprint import pprint

from custom_web_search import run_web_search


def main() -> None:
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "France internet service providers ISP broadband list"

    print(f"Running web search for query: {query!r}")
    results = run_web_search(
        query=query,
        max_results=6,
        max_scrape=6,
        max_chars_per_page=2000,
    )

    print("\n===== Search results summary =====")
    print(f"Total pages collected: {len(results)}\n")
    for idx, item in enumerate(results, start=1):
        url = item.get("url")
        title = item.get("title")
        content = item.get("content") or ""
        snippet = (content[:200] + "…") if len(content) > 200 else content

        print(f"[{idx}] URL: {url}")
        print(f"    Title: {title}")
        print(f"    Content length: {len(content)} chars")
        print(f"    Snippet: {snippet.replace(chr(10), ' ')[:300]}")
        print()

    print("===== Raw dicts (for debugging) =====")
    pprint(results)


if __name__ == "__main__":
    main()



