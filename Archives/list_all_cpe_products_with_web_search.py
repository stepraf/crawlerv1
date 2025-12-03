"""
LangGraph script to collect European ISPs and CPE equipment information.
Uses OpenAI API with LangGraph structured output.
"""

import os
import sys
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict, List, Iterable
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1-mini")

# TODO: Provide your list of countries here.
# Example: COUNTRY_LIST = ["France", "Germany", "Italy"]
COUNTRY_LIST: List[str] = ["France"]

# Log file for capturing all stdout
LOG_FILE_PATH = "list_all_cpe_products.log"


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

# Single OpenAI client for all model calls (Responses API with web_search)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


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


def batched(iterable: Iterable, batch_size: int) -> Iterable[list]:
    """Yield lists of up to batch_size items from iterable."""
    iterable = list(iterable)
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def get_structured_output(prompt: str, output_model: type[BaseModel]) -> BaseModel:
    """Call OpenAI Responses API with web search enabled and parse into the given Pydantic model."""
    import json

    enhanced_prompt = f"""{prompt}

IMPORTANT: Use web search to find current and accurate information. Search the web for:
- Official manufacturer websites and product pages
- ISP documentation and support pages
- Product specification sheets and datasheets
- Current product information and reviews
- Official product documentation

Ensure all URLs and information come from authoritative sources.

You must respond with valid JSON matching this schema:
{json.dumps(output_model.model_json_schema(), indent=2)}

Return only the JSON object, with no extra commentary or explanation.
If you cannot find the answer, return 'Unknown' for the answer, but still return the JSON object."""

    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=enhanced_prompt,
        tools=[{"type": "web_search"}],
        tool_choice="auto",
        temperature=1,
        reasoning={"effort": "high"},
    )

    # Responses API exposes output_text for convenience
    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise ValueError("No output_text returned from Responses API")

    try:
        data = json.loads(output_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}. Text: {output_text[:500]!r}")

    # Directly parse into the requested Pydantic model
    return output_model(**data)



def get_isps_for_countries(state: State) -> State:
    print("\n" + "="*60)
    print("STEP 2: Fetching major ISPs for each country...")
    print("="*60)
    
    countries = state["countries"]
    isps_list = []
    total_countries = len(countries)
    batch_size = 10

    for batch_index, country_batch in enumerate(batched(countries, batch_size), start=1):
        print(f"\n  Processing ISP batch {batch_index}: {', '.join(country_batch)}")
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_country = {}
            for country in country_batch:
                print(f"    → Queueing ISPs for {country}...")
                prompt = (
                    f"List the major Internet Service Providers (ISPs) in {country}. "
                    "Include the largest and most significant ISPs that provide broadband and internet services to consumers. "
                )
                future = executor.submit(get_structured_output, prompt, ISPList)
                future_to_country[future] = country

            for future in as_completed(future_to_country):
                country = future_to_country[future]
                try:
                    result = future.result()
                    country_isps = [isp.name for isp in result.isps]

                    if state.get("test_mode", False):
                        print(f"    Test mode: Limiting to first 2 ISPs for {country}")
                        country_isps = country_isps[:2]

                    print(f"    ✓ Found {len(country_isps)} ISPs for {country}: {', '.join(country_isps)}")
                    for isp in country_isps:
                        isps_list.append({"country": country, "isp": isp})
                except Exception as e:
                    print(f"    ⚠ Error fetching ISPs for {country}: {e}")

    print(f"\n  Total ISPs collected: {len(isps_list)} from {total_countries} countries")
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
    print("\n" + "="*60)
    print("STEP 3: Fetching CPE equipment for each ISP...")
    print("="*60)
    
    isps = state["isps"]
    equipment_list = []
    batch_size = 10

    for batch_index, isp_batch in enumerate(batched(isps, batch_size), start=1):
        print(f"\n  Processing CPE batch {batch_index} ({len(isp_batch)} ISPs)...")
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_isp = {}
            for isp_info in isp_batch:
                country = isp_info["country"]
                isp = isp_info["isp"]
                print(f"    → Queueing CPE equipment for {isp} ({country})...")

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

                future = executor.submit(get_structured_output, prompt, CPEList)
                future_to_isp[future] = isp_info

            for future in as_completed(future_to_isp):
                isp_info = future_to_isp[future]
                country = isp_info["country"]
                isp = isp_info["isp"]
                try:
                    result = future.result()
                    cpe_items = [(eq.name, eq.source_urls) for eq in result.equipment]

                    if state.get("test_mode", False):
                        print(f"    Test mode: Limiting to first 2 CPE items for {isp} ({country})")
                        cpe_items = cpe_items[:2]

                    print(f"    ✓ Found {len(cpe_items)} CPE equipment types for {isp} ({country})")

                    for eq_idx, (equipment_description, source_urls) in enumerate(cpe_items, 1):
                        print(f"    [{eq_idx}/{len(cpe_items)}] Processing: {equipment_description[:60]}...")
                        individual_models = extract_individual_models(equipment_description, country, isp)
                        for model in individual_models:
                            equipment_list.append({
                                "country": country,
                                "isp": isp,
                                "equipment": model.strip(),
                                "source_urls": "; ".join(source_urls) if source_urls else ""
                            })
                            print(f"      ✓ Added device: {model.strip()}")
                except Exception as e:
                    print(f"    ⚠ Error fetching CPE equipment for {isp} ({country}): {e}")

    print(f"\n  Total CPE equipment entries: {len(equipment_list)}")
    return {**state, "equipment": equipment_list}


def get_equipment_details(state: State) -> State:
    print("\n" + "="*60)
    print("STEP 4: Fetching detailed information for each CPE equipment...")
    print("="*60)
    
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
        print(f"  Initialized CSV file: {filename}")

    batch_size = 10

    for batch_index, equipment_batch in enumerate(batched(equipment, batch_size), start=1):
        print(f"\n  Processing details batch {batch_index} ({len(equipment_batch)} devices)...")
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_eq = {}
            for eq_info in equipment_batch:
                country = eq_info["country"]
                isp = eq_info["isp"]
                equipment_name = eq_info["equipment"]
                print(f"    → Queueing details for: {equipment_name} ({isp}, {country})")

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

                future = executor.submit(get_structured_output, prompt, EquipmentDetails)
                future_to_eq[future] = eq_info

            for future in as_completed(future_to_eq):
                eq_info = future_to_eq[future]
                country = eq_info["country"]
                isp = eq_info["isp"]
                equipment_name = eq_info["equipment"]
                cpe_source_urls = eq_info.get("source_urls", "")

                try:
                    result = future.result()
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

                    print(f"      ✓ Retrieved details: {result.product_name} ({result.manufacturer})")
                    if all_urls:
                        print(f"      ✓ Found {len(all_urls)} source URLs")
                    print(f"      ✓ Saved to CSV")
                except Exception as e:
                    print(f"      ⚠ Error retrieving details for {equipment_name} ({isp}, {country}): {e}")
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
                    print(f"      ✓ Saved to CSV (with error info)")

    print(f"\n  Total equipment details collected: {len(details_list)}")
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
    print("\n" + "="*60)
    print("STEP 5: Finalizing CSV file...")
    print("="*60)
    
    equipment_details = state["equipment_details"]
    filename = "european_isps_cpe.csv"
    
    if not equipment_details:
        print("  ⚠ No data to save!")
        return state
    
    # Count existing records in file
    existing_count = 0
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            existing_count = sum(1 for _ in reader)
    
    print(f"  ✓ CSV file '{filename}' contains {existing_count} records")
    print(f"  ✓ All {len(equipment_details)} equipment details have been saved incrementally")
    
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
        log_fh = open(LOG_FILE_PATH, "a", encoding="utf-8")
        # Add a simple run header with timestamp
        timestamp = datetime.utcnow().isoformat()
        log_fh.write(f"\n\n===== New run at {timestamp} UTC =====\n")
        log_fh.flush()
        sys.stdout = TeeStdout(original_stdout, log_fh)

    print("="*60)
    print("European ISPs and CPE Equipment Data Collection")
    print("="*60)
    
    if test_mode:
        print("\n⚠️  TEST MODE: Processing first 2 items in each loop")
    else:
        print("\n📊 FULL MODE: Processing all items")
    
    print(f"\nModel: {OPENAI_MODEL}")
    print("="*60)
    
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
    
    print("\n" + "="*60)
    print("✓ Workflow completed successfully!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable is required")
        exit(1)
    
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
