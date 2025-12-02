"""
LangGraph script to collect European ISPs and CPE equipment information.
Uses Azure OpenAI + custom DuckDuckGo/crawl4ai web search for structured output.
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

load_dotenv()

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")

# TODO: Provide your list of countries here.
# Example: COUNTRY_LIST = ["France", "Germany", "Italy"]
COUNTRY_LIST: List[str] = ["France"]

# Log file prefix for capturing all stdout; date/time will be added per run.
LOG_FILE_PREFIX = "list_all_cpe_products"


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


class ISP(BaseModel):
    name: str = Field(description="The name of the ISP")


class ISPList(BaseModel):
    isps: List[ISP] = Field(description="List of major ISPs in the country")


class CPEEquipment(BaseModel):
    name: str = Field(description="The name/model of the CPE equipment")
    source_urls: List[str] = Field(description="List of URLs pointing to sources of information about this equipment (ISP documentation, manufacturer pages, product specifications, etc.)")


class CPEList(BaseModel):
    equipment: List[CPEEquipment] = Field(description="List of major CPE equipment provided by the ISP")


class DeviceModelList(BaseModel):
    models: List[str] = Field(description="List of individual device model names or identifiers")


class EquipmentDetails(BaseModel):
    """Structured details for a single CPE device."""

    product_name: str = Field(description="The product name of the equipment")
    product_code: str = Field(description="The product code or model number")
    product_features: str = Field(description="Key features of the product")
    manufacturer: str = Field(description="The company that manufactures this equipment")
    main_soc: str = Field(description="The main System on Chip (SOC) used in the equipment")
    # Keep these as plain strings for simplicity; the prompt still strongly constrains values.
    device_type: str = Field(description="The type of CPE device (e.g., DOCSIS Gateway, GPON Gateway, DSL Gateway, Wi-Fi Repeater, Ethernet Gateway, FWA Gateway, Set Top Box, Other).")
    wifi_type: str = Field(description="The Wi-Fi standard supported by the device (e.g., Wi-Fi 5, Wi-Fi 6, Wi-Fi 7, Wi-Fi 8, before Wi-Fi 5, None, Unknown).")
    source_urls: List[str] = Field(description="List of URLs pointing to sources of information about this product (manufacturer website, product pages, specifications, reviews, etc.).")


class State(TypedDict):
    countries: List[str]
    isps: List[dict]
    equipment: List[dict]
    equipment_details: List[dict]
    test_mode: bool


def get_structured_output(prompt: str, output_model: type[BaseModel]) -> BaseModel:
    """
    Call Azure OpenAI Chat Completions and parse into the given Pydantic model,
    using our own DuckDuckGo + crawl4ai web search for external context.
    Also logs each LLM prompt and response clearly for debugging.
    """
    import json

    # Build a focused web search query from the original prompt.
    # Keeping this logic here makes the generic web_search tool reusable.
    def _build_search_query(raw_prompt: str) -> str:
        text = " ".join((raw_prompt or "").split())
        if not text:
            return raw_prompt

        # 1) ISP list: "List the major Internet Service Providers (ISPs) in {country}."
        import re

        m = re.search(r"List the major Internet Service Providers \(ISPs\) in ([^.]+)\.", text)
        if m:
            country = m.group(1).strip()
            query = f"{country} internet service providers ISP broadband list"
            print(f"[web_search] ISP query → {query}")
            return query

        # 2) CPE list: "List the major Customer Premises Equipment (CPE) devices that {isp} provides to its subscribers in {country}."
        m = re.search(
            r"List the major Customer Premises Equipment \(CPE\) devices that (.+?) provides to its subscribers in (.+?)\.",
            text,
        )
        if m:
            isp = m.group(1).strip()
            country = m.group(2).strip()
            query = f"{country} {isp} router modem gateway CPE equipment list"
            print(f"[web_search] CPE list query → {query}")
            return query

        # 3) Equipment details:
        # "Provide detailed information about the CPE equipment: {equipment}
        #  This equipment is provided by {isp} in {country}."
        m = re.search(
            r"Provide detailed information about the CPE equipment:\s*(.+?)\s+This equipment is provided by (.+?) in (.+?)\.",
            text,
        )
        if m:
            equipment = m.group(1).strip()
            isp = m.group(2).strip()
            country = m.group(3).strip()
            query = f"{equipment} {isp} {country} router modem gateway CPE specifications"
            print(f"[web_search] Equipment details query → {query}")
            return query

        # Fallback: use the full prompt as-is.
        print(f"[web_search] Raw query → {text}")
        return text

    search_query = _build_search_query(prompt)

    print("\n" + "=" * 80)
    print("[LLM CALL] Starting structured output request")
    print(f"[LLM CALL] Pydantic model: {output_model.__name__}")
    print(f"[LLM CALL] Derived web search query: {search_query}")
    print("=" * 80)

    # Collect external web context using our custom search tool
    search_results = run_web_search(
        query=search_query,
        max_results=5,
        max_scrape=3,
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

    system_prompt = (
        "You are a precise research assistant. "
        "You are given a user request and web search results (cleaned and truncated extracts from relevant web pages). "
        "Use these sources and your general knowledge to answer. "
        "You must respond ONLY with valid JSON conforming to the provided schema."
    )

    user_content = f"""WEB SEARCH RESULTS:
{web_context}

USER REQUEST:
{prompt}

JSON SCHEMA (Pydantic model_json_schema):
{json.dumps(output_model.model_json_schema(), indent=2)}

Instructions:
- Return only the JSON object, with no extra commentary or explanation.
- If you cannot find a value, use "Unknown" for that field but still return a valid JSON object."""

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



def get_isps_for_countries(state: State) -> State:
    print("\n" + "=" * 80)
    print("STEP 2: FETCHING MAJOR ISPs FOR EACH COUNTRY")
    print("=" * 80)
    
    countries = state["countries"]
    isps_list = []
    total_countries = len(countries)

    for country in countries:
        print("\n" + "-" * 80)
        print(f"[STEP 2] Processing ISPs for country: {country}")
        print("-" * 80)
        try:
            prompt = (
                f"List the major Internet Service Providers (ISPs) in {country}. "
                "Include the largest and most significant ISPs that provide broadband and internet services to consumers. "
            )
            result = get_structured_output(prompt, ISPList)
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
    print(f"[STEP 2] DONE. Total ISPs collected: {len(isps_list)} from {total_countries} countries")
    print("=" * 80)
    return {**state, "isps": isps_list}


def extract_individual_models(equipment_description: str, country: str, isp: str) -> List[str]:
    prompt = f"""Extract all individual device models and product names from the following equipment description.

Equipment description: {equipment_description}
This equipment is provided by {isp} in {country}.

This should be a CPE device type such as: DOCSIS Gateway, GPON Gateway, DSL Gateway, Wi-Fi Repeater, Ethernet Gateway, FWA Gateway, Set Top Box, or Other.

Please extract each individual device model, product name, or device identifier as a separate entry.
For example, if the description mentions "Huawei HG8245, HG8145, ZTE F660, Nokia G-010G-A", 
extract these as separate models: ["Huawei HG8245", "Huawei HG8145", "ZTE F660", "Nokia G-010G-A"].

If the description mentions series (like "ZyXEL VMG series"), extract specific models if mentioned, 
or use the series name as a single entry if no specific models are listed.

Return only the individual device models, one per entry in the list."""
    
    try:
        result = get_structured_output(prompt, DeviceModelList)
        models = result.models if result.models else [equipment_description]
        if len(models) > 1:
            print(f"      → Extracted {len(models)} individual models from description")
        return models
    except Exception as e:
        print(f"      ⚠ Warning: Could not extract models, using original description")
        return [equipment_description]


def get_cpe_equipment(state: State) -> State:
    print("\n" + "=" * 80)
    print("STEP 3: FETCHING CPE EQUIPMENT FOR EACH ISP")
    print("=" * 80)
    
    isps = state["isps"]
    equipment_list = []

    for isp_info in isps:
        country = isp_info["country"]
        isp = isp_info["isp"]
        print("\n" + "-" * 80)
        print(f"[STEP 3] Processing CPE equipment for ISP: {isp} (Country: {country})")
        print("-" * 80)

        try:
            prompt = f"""List the major Customer Premises Equipment (CPE) devices that {isp} provides to its subscribers in {country}. 
            
Focus on these types of CPE devices:
- DOCSIS Gateway
- GPON Gateway
- DSL Gateway
- Wi-Fi Repeater
- Ethernet Gateway
- FWA Gateway
- Set Top Box
- Other CPE equipment

Include routers, modems, gateways, FWA devices and other network equipment that ISPs typically provide to customers.
For each type of equipment, provide a description that includes the specific device models when available.
Also provide a list of URLs pointing to sources of information about each equipment type (ISP documentation, manufacturer product pages, specification sheets, etc.)."""

            result = get_structured_output(prompt, CPEList)
            cpe_items = [(eq.name, eq.source_urls) for eq in result.equipment]

            if state.get("test_mode", False):
                print(f"[STEP 3] Test mode: Limiting to first 2 CPE items for {isp} ({country})")
                cpe_items = cpe_items[:2]

            print(f"[STEP 3] ✓ Found {len(cpe_items)} CPE equipment types for {isp} ({country})")

            for eq_idx, (equipment_description, source_urls) in enumerate(cpe_items, 1):
                print(f"[STEP 3]   [{eq_idx}/{len(cpe_items)}] Processing equipment description:")
                print(f"[STEP 3]   {equipment_description}")
                individual_models = extract_individual_models(equipment_description, country, isp)
                for model in individual_models:
                    equipment_list.append({
                        "country": country,
                        "isp": isp,
                        "equipment": model.strip(),
                        "source_urls": "; ".join(source_urls) if source_urls else ""
                    })
                    print(f"[STEP 3]     ✓ Added device model: {model.strip()}")
        except Exception as e:
            print(f"[STEP 3] ⚠ Error fetching CPE equipment for {isp} ({country}): {e}")

    print("\n" + "=" * 80)
    print(f"[STEP 3] DONE. Total CPE equipment entries: {len(equipment_list)}")
    print("=" * 80)
    return {**state, "equipment": equipment_list}


def get_equipment_details(state: State) -> State:
    print("\n" + "=" * 80)
    print("STEP 4: FETCHING DETAILED INFORMATION FOR EACH CPE DEVICE")
    print("=" * 80)
    
    equipment = state["equipment"]
    details_list = []
    filename = "european_isps_cpe.csv"

    # Initialize CSV file with header if starting fresh
    if equipment:
        fieldnames = [
            "Country", "ISP", "Equipment", "Product Name",
            "Product Code", "Product Features", "Manufacturer", "Main SOC", "Device Type", "Wi-Fi Type", "Source URLs"
        ]
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        print(f"[STEP 4] Initialized CSV file: {filename}")

    for eq_info in equipment:
        country = eq_info["country"]
        isp = eq_info["isp"]
        equipment_name = eq_info["equipment"]
        cpe_source_urls = eq_info.get("source_urls", "")

        print("\n" + "-" * 80)
        print(f"[STEP 4] Processing details for device: {equipment_name}")
        print(f"[STEP 4] ISP: {isp} | Country: {country}")
        print("-" * 80)

        try:
            prompt = f"""Provide detailed information about the CPE equipment: {equipment_name}

This equipment is provided by {isp} in {country}.

Please provide:
- Product Name: The official product name
- Product Code: The model number or product code
- Product Features: Key features and capabilities (e.g., WiFi standards, ports, speeds)
- Manufacturer: The company that manufactures this equipment (Please note that Arris, Commscope and Technicolor are the same company today, named Vantiva.)
- Main SOC: The main System on Chip (SOC) or processor used in this equipment
- Device Type: The type of CPE device. Must be one of: DOCSIS Gateway, GPON Gateway, DSL Gateway, Wi-Fi Repeater, Ethernet Gateway, FWA Gateway, Set Top Box, or Other. Choose the most appropriate type based on the equipment's primary function and connection method.
- Wi-Fi Type: The Wi-Fi standard supported by the device. Must be EXACTLY one of: Wi-Fi 5, Wi-Fi 6, Wi-Fi 7, Wi-Fi 8, before Wi-Fi 5, None (if the device does not have Wi-Fi capability), or Unknown (if the WiFi type cannot be determined from available information). Use "Unknown" (not "Unknown — depends on model" or any other variation) if you cannot determine the WiFi type.
- Source URLs: A list of URLs pointing to sources of information about this product (manufacturer website, product specification pages, datasheets, reviews, ISP documentation, etc.)

IMPORTANT: For fields that accept only specific values (Device Type, Wi-Fi Type), you MUST use exactly one of the allowed values. If information is unknown, use "Unknown" (for Wi-Fi Type) or "Other" (for Device Type). Do not use descriptive text like "Unknown — depends on model" - use only the exact literal values specified."""

            result = get_structured_output(prompt, EquipmentDetails)
            # Merge URLs from CPE equipment list and detailed information
            detailed_urls = result.source_urls if result.source_urls else []
            cpe_urls = [url.strip() for url in cpe_source_urls.split(";") if url.strip()] if cpe_source_urls else []
            # Combine and deduplicate URLs
            all_urls = list(dict.fromkeys(detailed_urls + cpe_urls))

            detail_record = {
                "country": country,
                "isp": isp,
                "equipment": equipment_name,
                "product_name": result.product_name,
                "product_code": result.product_code,
                "product_features": result.product_features,
                "manufacturer": result.manufacturer,
                "main_soc": result.main_soc,
                "device_type": result.device_type,
                "wifi_type": result.wifi_type,
                "source_urls": "; ".join(all_urls) if all_urls else ""
            }
            details_list.append(detail_record)

            # Write to CSV immediately
            write_detail_to_csv(detail_record)

            print(f"[STEP 4] ✓ Retrieved details: {result.product_name} ({result.manufacturer})")
            if all_urls:
                print(f"[STEP 4] ✓ Found {len(all_urls)} source URLs")
            print(f"[STEP 4] ✓ Saved to CSV")
        except Exception as e:
            print(f"[STEP 4] ⚠ Error retrieving details for {equipment_name} ({isp}, {country}): {e}")
            # Use CPE URLs if available, even if detailed info failed
            cpe_urls = cpe_source_urls if cpe_source_urls else ""
            detail_record = {
                "country": country,
                "isp": isp,
                "equipment": equipment_name,
                "product_name": equipment_name,
                "product_code": "Unknown",
                "product_features": "Error retrieving details",
                "manufacturer": "Unknown",
                "main_soc": "Unknown",
                "device_type": "Other",
                "wifi_type": "None",
                "source_urls": cpe_urls
            }
            details_list.append(detail_record)

            # Write to CSV even if there was an error
            write_detail_to_csv(detail_record)
            print(f"[STEP 4] ✓ Saved to CSV (with error info)")

    print("\n" + "=" * 80)
    print(f"[STEP 4] DONE. Total equipment details collected: {len(details_list)}")
    print("=" * 80)
    return {**state, "equipment_details": details_list}


def write_detail_to_csv(detail: dict, filename: str = "european_isps_cpe.csv"):
    """Write a single equipment detail row to CSV, creating file with header if needed."""
    fieldnames = [
        "Country", "ISP", "Equipment", "Product Name",
        "Product Code", "Product Features", "Manufacturer", "Main SOC", "Device Type", "Wi-Fi Type", "Source URLs"
    ]
    
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "Country": detail["country"],
            "ISP": detail["isp"],
            "Equipment": detail["equipment"],
            "Product Name": detail["product_name"],
            "Product Code": detail["product_code"],
            "Product Features": detail["product_features"],
            "Manufacturer": detail["manufacturer"],
            "Main SOC": detail["main_soc"],
            "Device Type": detail.get("device_type", "Other"),
            "Wi-Fi Type": detail.get("wifi_type", "None"),
            "Source URLs": detail.get("source_urls", "")
        })


def save_to_csv(state: State) -> State:
    print("\n" + "=" * 80)
    print("STEP 5: FINALIZING CSV FILE")
    print("=" * 80)
    
    equipment_details = state["equipment_details"]
    filename = "european_isps_cpe.csv"
    
    if not equipment_details:
        print("[STEP 5] ⚠ No data to save!")
        return state
    
    # Count existing records in file
    existing_count = 0
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            existing_count = sum(1 for _ in reader)
    
    print(f"[STEP 5] ✓ CSV file '{filename}' contains {existing_count} records")
    print(f"[STEP 5] ✓ All {len(equipment_details)} equipment details have been saved incrementally")
    
    return state


def create_workflow():
    workflow = StateGraph(State)
    workflow.add_node("get_isps", get_isps_for_countries)
    workflow.add_node("get_equipment", get_cpe_equipment)
    workflow.add_node("get_details", get_equipment_details)
    workflow.add_node("save_csv", save_to_csv)
    
    # Start directly from ISPs – countries are provided via COUNTRY_LIST
    workflow.set_entry_point("get_isps")
    workflow.add_edge("get_isps", "get_equipment")
    workflow.add_edge("get_equipment", "get_details")
    workflow.add_edge("get_details", "save_csv")
    workflow.add_edge("save_csv", END)
    

    return workflow.compile()


def main(test_mode: bool = False):
    # Mirror everything printed to stdout into a log file
    # We wrap only once per process invocation to avoid nesting TeeStdout.
    if not isinstance(sys.stdout, TeeStdout):
        original_stdout = sys.__stdout__  # real terminal stdout
        # Build a log filename with UTC date and time, e.g. list_all_cpe_products_20251202_153045.log
        timestamp_for_filename = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{LOG_FILE_PREFIX}_{timestamp_for_filename}.log"
        log_fh = open(log_filename, "a", encoding="utf-8")
        # Add a simple run header with ISO timestamp
        timestamp = datetime.utcnow().isoformat()
        log_fh.write(f"\n\n===== New run at {timestamp} UTC =====\n")
        log_fh.flush()
        sys.stdout = TeeStdout(original_stdout, log_fh)

    print("=" * 80)
    print("EUROPEAN ISPs AND CPE EQUIPMENT DATA COLLECTION WORKFLOW")
    print("=" * 80)
    
    if test_mode:
        print("\n⚠️  TEST MODE: Processing first 2 items in each loop")
    else:
        print("\n📊 FULL MODE: Processing all items")
    
    print(f"\nAzure deployment: {AZURE_DEPLOYMENT_NAME}")
    print("=" * 80)
    
    initial_state: State = {
        # Countries are now provided explicitly instead of being fetched by the workflow
        "countries": COUNTRY_LIST,
        "isps": [],
        "equipment": [],
        "equipment_details": [],
        "test_mode": test_mode
    }
    
    app = create_workflow()
    app.invoke(initial_state)
    
    print("\n" + "=" * 80)
    print("✓ WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Collect European ISPs and CPE equipment information"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (process only first 2 items in each loop)"
    )
    
    args = parser.parse_args()
    main(test_mode=args.test)
