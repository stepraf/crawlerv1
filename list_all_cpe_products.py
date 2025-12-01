"""
LangGraph script to collect European ISPs and CPE equipment information.
Uses OpenAI API with LangGraph structured output.
"""

import os
import csv
from typing import TypedDict, List, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")

llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL, temperature=1)


class Country(BaseModel):
    name: str = Field(description="The name of the country")


class CountryList(BaseModel):
    countries: List[Country] = Field(description="List of all countries in Europe")


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
    product_name: str = Field(description="The product name of the equipment")
    product_code: str = Field(description="The product code or model number")
    product_features: str = Field(description="Key features of the product")
    manufacturer: str = Field(description="The company that manufactures this equipment")
    main_soc: str = Field(description="The main System on Chip (SOC) used in the equipment")
    device_type: Literal[
        "DOCSIS Gateway",
        "GPON Gateway",
        "DSL Gateway",
        "Wi-Fi Repeater",
        "Ethernet Gateway",
        "FWA Gateway",
        "Set Top Box",
        "Other"
    ] = Field(description="The type of CPE device. Must be one of: DOCSIS Gateway, GPON Gateway, DSL Gateway, Wi-Fi Repeater, Ethernet Gateway, FWA Gateway, Set Top Box, or Other")
    wifi_type: Literal[
        "Wi-Fi 5",
        "Wi-Fi 6",
        "Wi-Fi 7",
        "Wi-Fi 8",
        "before Wi-Fi 5",
        "None"
    ] = Field(description="The Wi-Fi standard supported by the device. Must be one of: Wi-Fi 5, Wi-Fi 6, Wi-Fi 7, Wi-Fi 8, before Wi-Fi 5, or None if the device does not have Wi-Fi capability")
    source_urls: List[str] = Field(description="List of URLs pointing to sources of information about this product (manufacturer website, product pages, specifications, reviews, etc.)")


class State(TypedDict):
    countries: List[str]
    isps: List[dict]
    equipment: List[dict]
    equipment_details: List[dict]
    test_mode: bool


def get_structured_output(prompt: str, output_model: type[BaseModel]) -> BaseModel:
    """Get structured output using LangChain's with_structured_output with web search instructions."""
    enhanced_prompt = f"""{prompt}

IMPORTANT: Use web search to find current and accurate information. Search the web for:
- Official manufacturer websites and product pages
- ISP documentation and support pages
- Product specification sheets and datasheets
- Current product information and reviews
- Official product documentation

Ensure all URLs and information come from authoritative sources."""
    
    structured_llm = llm.with_structured_output(output_model)
    return structured_llm.invoke([HumanMessage(content=enhanced_prompt)])



def get_european_countries(state: State) -> State:
    print("\n" + "="*60)
    print("STEP 1: Fetching European countries...")
    print("="*60)
    
    prompt = "List all countries in Europe that have more than 4 million people. List all EU member states and other European countries. Do not list countries of less than 4 million people in your output"
    # prompt = "List all countries in North America."
    print("  Requesting list of countries from OpenAI...")
    result = get_structured_output(prompt, CountryList)
    countries = [country.name for country in result.countries]
    
    if state.get("test_mode", False):
        print(f"  Test mode: Limiting to first 2 countries")
        countries = countries[:2]
    
    print(f"  ✓ Found {len(countries)} countries: {', '.join(countries)}")
    return {**state, "countries": countries}


def get_isps_for_countries(state: State) -> State:
    print("\n" + "="*60)
    print("STEP 2: Fetching major ISPs for each country...")
    print("="*60)
    
    countries = state["countries"]
    isps_list = []
    
    for idx, country in enumerate(countries, 1):
        print(f"\n  [{idx}/{len(countries)}] Processing ISPs for {country}...")
        prompt = f"List the major Internet Service Providers (ISPs) in {country}. Include the largest and most significant ISPs that provide broadband and internet services to consumers. Focus on ISPs that provide CPE equipment such as DOCSIS Gateways, GPON Gateways, DSL Gateways, Wi-Fi Repeaters, Ethernet Gateways, FWA Gateways, Set Top Boxes, and other customer premises equipment."
        result = get_structured_output(prompt, ISPList)
        country_isps = [isp.name for isp in result.isps]
        
        if state.get("test_mode", False):
            print(f"    Test mode: Limiting to first 2 ISPs")
            country_isps = country_isps[:2]
        
        print(f"    ✓ Found {len(country_isps)} ISPs: {', '.join(country_isps)}")
        
        for isp in country_isps:
            isps_list.append({"country": country, "isp": isp})
    
    print(f"\n  Total ISPs collected: {len(isps_list)}")
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
    
    for idx, isp_info in enumerate(isps, 1):
        country = isp_info["country"]
        isp = isp_info["isp"]
        
        print(f"\n  [{idx}/{len(isps)}] Processing CPE equipment for {isp} ({country})...")
        
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
            print(f"    Test mode: Limiting to first 2 CPE items")
            cpe_items = cpe_items[:2]
        
        print(f"    ✓ Found {len(cpe_items)} CPE equipment types")
        
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
    
    for idx, eq_info in enumerate(equipment, 1):
        country = eq_info["country"]
        isp = eq_info["isp"]
        equipment_name = eq_info["equipment"]
        cpe_source_urls = eq_info.get("source_urls", "")
        
        print(f"\n  [{idx}/{len(equipment)}] Processing details for: {equipment_name}")
        print(f"      ISP: {isp}, Country: {country}")
        
        prompt = f"""Provide detailed information about the CPE equipment: {equipment_name}

This equipment is provided by {isp} in {country}.

Please provide:
- Product Name: The official product name
- Product Code: The model number or product code
- Product Features: Key features and capabilities (e.g., WiFi standards, ports, speeds)
- Manufacturer: The company that manufactures this equipment (Please note that Arris, Commscope and Technicolor are the same company today, named Vantiva.)
- Main SOC: The main System on Chip (SOC) or processor used in this equipment
- Device Type: The type of CPE device. Must be one of: DOCSIS Gateway, GPON Gateway, DSL Gateway, Wi-Fi Repeater, Ethernet Gateway, FWA Gateway, Set Top Box, or Other. Choose the most appropriate type based on the equipment's primary function and connection method.
- Wi-Fi Type: The Wi-Fi standard supported by the device. Must be one of: Wi-Fi 5, Wi-Fi 6, Wi-Fi 7, Wi-Fi 8, before Wi-Fi 5, or None if the device does not have Wi-Fi capability.
- Source URLs: A list of URLs pointing to sources of information about this product (manufacturer website, product specification pages, datasheets, reviews, ISP documentation, etc.)

If you don't know specific information indicate 'Unknown'. For source URLs, provide as many relevant URLs as possible, including manufacturer product pages, specification sheets, ISP documentation, or other authoritative sources."""
        
        try:
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
            
            print(f"      ✓ Retrieved details: {result.product_name} ({result.manufacturer})")
            if all_urls:
                print(f"      ✓ Found {len(all_urls)} source URLs")
            print(f"      ✓ Saved to CSV")
        except Exception as e:
            print(f"      ⚠ Error retrieving details: {e}")
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
    workflow.add_node("get_countries", get_european_countries)
    workflow.add_node("get_isps", get_isps_for_countries)
    workflow.add_node("get_equipment", get_cpe_equipment)
    workflow.add_node("get_details", get_equipment_details)
    workflow.add_node("save_csv", save_to_csv)
    
    workflow.set_entry_point("get_countries")
    workflow.add_edge("get_countries", "get_isps")
    workflow.add_edge("get_isps", "get_equipment")
    workflow.add_edge("get_equipment", "get_details")
    workflow.add_edge("get_details", "save_csv")
    workflow.add_edge("save_csv", END)
    

    return workflow.compile()


def main(test_mode: bool = False):
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
        "countries": [],
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
