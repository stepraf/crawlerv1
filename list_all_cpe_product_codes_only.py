"""
Simplified LangGraph script to collect CPE product codes/model numbers per ISP.

It reuses:
- Azure OpenAI with structured (Pydantic) JSON output
- custom DuckDuckGo/crawl4ai web search (`run_web_search`)

Output:
- CSV file: `european_isps_cpe_product_codes.csv` with columns:
  Country, ISP, Product Code, Description, Source URLs
"""

import os
import sys
import csv
from typing import TypedDict, List
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from openai import AzureOpenAI

from custom_web_search import run_web_search
from prompts_and_queries import (
    SYSTEM_PROMPT,
    build_user_content_template,
    get_isp_list_prompt,
    get_product_codes_prompt,
    build_isp_search_query,
    build_product_codes_search_query,
)


load_dotenv()

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")

# Countries to process
COUNTRY_LIST: List[str] = [
    "Italy",
    "Spain",
    "Ukraine",
    "Poland",
    "Romania",
    "Netherlands",
    "Belgium",
    "Czechia",
    "Sweden",
    "Portugal",
    "Greece",
    "Hungary",
    "Austria",
    "Belarus",
    "Switzerland",
    "Bulgaria",
    "Serbia",
    "Denmark",
    "Finland",
    "Norway",
    "Slovakia",
    "Ireland",
    "Croatia"
]

# Log file prefix for capturing all stdout; date/time will be added per run.
LOG_FILE_PREFIX = "list_all_cpe_product_codes"


class TeeStdout:
    """File-like object that duplicates writes to multiple streams (e.g., console and log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)
        # Ensure all outputs are flushed promptly
        for s in self.streams:
            s.flush()

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


# Single Azure OpenAI client for all model calls
if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_DEPLOYMENT_NAME]):
    missing = [
        name
        for name, value in [
            ("AZURE_OPENAI_API_KEY", AZURE_OPENAI_API_KEY),
            ("AZURE_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT),
            ("AZURE_OPENAI_API_VERSION", AZURE_OPENAI_API_VERSION),
            ("AZURE_DEPLOYMENT_NAME", AZURE_DEPLOYMENT_NAME),
        ]
        if not value
    ]
    raise RuntimeError(
        f"Missing Azure OpenAI configuration for: {', '.join(missing)}. "
        "Populate them in your environment or .env file."
    )

azure_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ISP(BaseModel):
    name: str = Field(description="The name of the ISP")


class ISPList(BaseModel):
    isps: List[ISP] = Field(description="List of major ISPs in the country")


class ProductCodeList(BaseModel):
    products: List[str] = Field(
        description=(
            "List of CPE product codes/models used by the ISP in the country. "
            "Each item MUST be a single string in the form 'CODE | DESCRIPTION', "
            "where DESCRIPTION can be empty if unknown."
        )
    )


class State(TypedDict):
    countries: List[str]
    isps: List[dict]
    product_codes: List[dict]
    test_mode: bool


# ---------------------------------------------------------------------------
# Core helper: structured output with web search
# ---------------------------------------------------------------------------

def get_structured_output(prompt: str, search_query: str, output_model: type[BaseModel]) -> BaseModel:
    """
    Call Azure OpenAI Chat Completions and parse into the given Pydantic model,
    using our own DuckDuckGo + crawl4ai web search for external context.
    Also logs each LLM prompt and response clearly for debugging.
    
    Args:
        prompt: The LLM prompt text
        search_query: The web search query to use
        output_model: Pydantic model class for structured output
    """
    import json

    print("\n" + "=" * 80)
    print("[LLM CALL] Starting structured output request")
    print(f"[LLM CALL] Pydantic model: {output_model.__name__}")
    print(f"[LLM CALL] Derived web search query: {search_query}")
    print("=" * 80)

    # Collect external web context using our custom search tool
    search_results = run_web_search(
        query=search_query,
        max_results=15,
        max_scrape=15,
        max_chars_per_page=10000,
    )

    web_context_parts: List[str] = []
    for idx, item in enumerate(search_results, start=1):
        url = item.get("url", "")
        title = item.get("title", "")
        content = item.get("content", "")
        web_context_parts.append(
            f"Result {idx}:\nURL: {url}\nTitle: {title}\nContent snippet:\n{content}\n"
        )
    web_context = "\n\n---\n\n".join(web_context_parts) if web_context_parts else "No web results available."

    system_prompt = SYSTEM_PROMPT
    json_schema_str = json.dumps(output_model.model_json_schema(), indent=2)
    user_content = build_user_content_template(web_context, prompt, json_schema_str)

    print("[LLM PROMPT] System prompt:")
    print("-" * 80)
    print(system_prompt)
    print("-" * 80)
    print("[LLM PROMPT] User content (including web context and schema):")
    print("-" * 80)
    print(user_content)
    print("-" * 80)

    response = azure_client.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,
        temperature=1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    content = response.choices[0].message.content
    print("[LLM RESPONSE] Raw JSON from Azure OpenAI:")
    print("-" * 80)
    print(content)
    print("-" * 80)

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}. Text: {content[:500]!r}")

    print("[LLM RESPONSE] Parsed JSON into model:", output_model.__name__)
    print("=" * 80)
    return output_model(**data)


# ---------------------------------------------------------------------------
# Step 2: Fetch ISPs per country (reused, simplified)
# ---------------------------------------------------------------------------

def get_isps_for_countries(state: State) -> State:
    print("\n" + "=" * 80)
    print("STEP 2: FETCHING MAJOR ISPs FOR EACH COUNTRY")
    print("=" * 80)

    countries = state["countries"]
    isps_list: List[dict] = []

    for country in countries:
        print("\n" + "-" * 80)
        print(f"[STEP 2] Processing ISPs for country: {country}")
        print("-" * 80)
        try:
            prompt = get_isp_list_prompt(country)
            search_query = build_isp_search_query(country)
            result = get_structured_output(prompt, search_query, ISPList)
            country_isps = [isp.name for isp in result.isps]

            if state.get("test_mode", False):
                print(f"[STEP 2] Test mode: Limiting to first 2 ISPs for {country}")
                country_isps = country_isps[:2]

            print(f"[STEP 2] ✓ Found {len(country_isps)} ISPs for {country}: {', '.join(country_isps)}")
            for isp in country_isps:
                isps_list.append({"country": country, "isp": isp})
        except Exception as e:
            print(f"[STEP 2] ⚠ Error fetching ISPs for {country}: {e}")

    print("\n" + "=" * 80)
    print(f"[STEP 2] DONE. Total ISPs collected: {len(isps_list)}")
    print("=" * 80)
    return {**state, "isps": isps_list}


# ---------------------------------------------------------------------------
# Step 3: Directly fetch product codes for each ISP
# ---------------------------------------------------------------------------

def get_product_codes_for_isps(state: State) -> State:
    print("\n" + "=" * 80)
    print("STEP 3: FETCHING PRODUCT CODES/MODEL NUMBERS FOR EACH ISP")
    print("=" * 80)

    isps = state["isps"]
    all_products: List[dict] = []
    filename = "european_isps_cpe_product_codes.csv"

    # Initialize CSV with header
    fieldnames = ["Country", "ISP", "Product Code", "Description", "Source URLs"]
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    print(f"[STEP 3] Initialized CSV file: {filename}")

    for isp_info in isps:
        country = isp_info["country"]
        isp = isp_info["isp"]

        print("\n" + "-" * 80)
        print(f"[STEP 3] Processing product codes for ISP: {isp} (Country: {country})")
        print("-" * 80)

        try:
            prompt = get_product_codes_prompt(isp, country)
            search_query = build_product_codes_search_query(isp, country)
            result = get_structured_output(prompt, search_query, ProductCodeList)
            products = result.products or []

            if state.get("test_mode", False):
                print(f"[STEP 3] Test mode: Limiting to first 5 products for {isp} ({country})")
                products = products[:5]

            print(f"[STEP 3] ✓ Found {len(products)} product codes for {isp} ({country})")

            with open(filename, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for raw_entry in products:
                    entry = (raw_entry or "").strip()
                    if " | " in entry:
                        code_part, desc_part = entry.split("|", 1)
                        code = code_part.strip()
                        description = desc_part.strip()
                    else:
                        code = entry
                        description = ""

                    record = {
                        "Country": country,
                        "ISP": isp,
                        "Product Code": code,
                        "Description": description,
                        "Source URLs": "",
                    }
                    writer.writerow(record)
                    all_products.append(
                        {
                            "country": country,
                            "isp": isp,
                            "code": code,
                            "description": description,
                            "source_urls": "",
                        }
                    )
                    print(f"[STEP 3]   ✓ {code}")

        except Exception as e:
            print(f"[STEP 3] ⚠ Error fetching product codes for {isp} ({country}): {e}")

    print("\n" + "=" * 80)
    print(f"[STEP 3] DONE. Total product code entries: {len(all_products)}")
    print("=" * 80)
    return {**state, "product_codes": all_products}


# ---------------------------------------------------------------------------
# Workflow setup
# ---------------------------------------------------------------------------

def create_workflow():
    workflow = StateGraph(State)
    workflow.add_node("get_isps", get_isps_for_countries)
    workflow.add_node("get_product_codes", get_product_codes_for_isps)

    workflow.set_entry_point("get_isps")
    workflow.add_edge("get_isps", "get_product_codes")
    workflow.add_edge("get_product_codes", END)

    return workflow.compile()


def main(test_mode: bool = False):
    # Mirror everything printed to stdout into a log file
    if not isinstance(sys.stdout, TeeStdout):
        original_stdout = sys.__stdout__  # real terminal stdout
        timestamp_for_filename = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{LOG_FILE_PREFIX}_{timestamp_for_filename}.log"
        log_fh = open(log_filename, "a", encoding="utf-8")
        # Add a simple run header with ISO timestamp
        timestamp = datetime.utcnow().isoformat()
        log_fh.write(f"\n\n===== New run at {timestamp} UTC =====\n")
        log_fh.flush()
        sys.stdout = TeeStdout(original_stdout, log_fh)

    print("=" * 80)
    print("EUROPEAN ISPs – CPE PRODUCT CODE COLLECTION WORKFLOW")
    print("=" * 80)

    if test_mode:
        print("\n⚠️  TEST MODE: Processing first items in each loop")
    else:
        print("\n📊 FULL MODE: Processing all items")

    print(f"\nAzure deployment: {AZURE_DEPLOYMENT_NAME}")
    print("=" * 80)

    initial_state: State = {
        "countries": COUNTRY_LIST,
        "isps": [],
        "product_codes": [],
        "test_mode": test_mode,
    }

    app = create_workflow()
    app.invoke(initial_state)

    print("\n" + "=" * 80)
    print("✓ WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect European ISPs and CPE product codes/model numbers"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (process only first items in each loop)",
    )

    args = parser.parse_args()
    main(test_mode=args.test)


