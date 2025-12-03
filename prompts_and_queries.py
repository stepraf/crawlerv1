"""
Prompt templates and web search query templates for CPE product code collection.

This module centralizes all LLM prompts and search query construction logic
to make them easier to maintain and improve.
"""



# ============================================================================
# System Prompts
# ============================================================================

SYSTEM_PROMPT = (
    "You are a precise research assistant. "
    "You are given a user request and web search results. "
    "Use these sources and your general knowledge to answer. "
    "You must respond ONLY with valid JSON conforming to the provided schema."
)


# ============================================================================
# User Content Templates
# ============================================================================

def build_user_content_template(web_context: str, prompt: str, json_schema: str) -> str:
    """
    Build the user content for LLM requests.
    
    Args:
        web_context: Formatted web search results
        prompt: The user's original prompt/question
        json_schema: JSON schema string from Pydantic model
        
    Returns:
        Formatted user content string
    """
    return f"""WEB SEARCH RESULTS:
{web_context}

USER REQUEST:
{prompt}

JSON SCHEMA (Pydantic model_json_schema):
{json_schema}

Instructions:
- Return only the JSON object, with no extra commentary or explanation.
- If you cannot find a value, use "Unknown" or an empty list for that field but still return a valid JSON object."""


# ============================================================================
# Prompt Templates
# ============================================================================

def get_isp_list_prompt(country: str) -> str:
    """
    Generate prompt for listing ISPs in a country.
    
    Args:
        country: Country name
        
    Returns:
        Prompt string
    """
    return (
        f"List the major Internet Service Providers (ISPs) in {country}. "
        "Include the largest and most significant ISPs that provide broadband and internet services to consumers."
    )


def get_product_codes_prompt(isp: str, country: str) -> str:
    """
    Generate prompt for extracting product codes/model numbers for an ISP.
    
    Args:
        isp: ISP name
        country: Country name
        
    Returns:
        Prompt string
    """
    return f"""
You are researching Customer Premises Equipment (CPE) used by {isp} in {country}.

Your goal is to extract an EXHAUSTIVE list of CPE product codes and model numbers used by this ISP in this country.

Product codes / model numbers look like short alphanumeric tokens, sometimes with symbols, for example:
- "F@st 5359"
- "ZXHN-H3600P"
- "UBC1327"
- "CGA2121"
- "CGM4981COM"

Instructions:
- LIST EVERY DISTINCT PRODUCT CODE OR MODEL NUMBER YOU CAN FIND in the web search results.
- Do NOT stop after a few examples; continue until you have exhausted the information available.
- If a family has many variants (e.g., "CGA2121", "CGA2121N", "CGA2121-ES"), include EACH VARIANT as a separate entry.
- Do NOT group codes as ranges like "CGA2121 series"; always list each specific code separately when possible.
- Exclude:
  - Pure marketing names with no specific code (e.g., "Livebox", "HomeBox" only).
  - Generic technology terms (e.g., "DOCSIS 3.1", "GPON") unless they appear as PART of a model code.
- Deduplicate: if the same code appears in multiple places, return it only once.

For each product, return a JSON object with:
- "code": the exact product code/model identifier string as in the source.
- "description": a short phrase describing the device (e.g., "DOCSIS 3.1 cable gateway", "FTTH GPON router"), or "" if unknown.
- "source_urls": a list of URLs where this exact code appears.

Respond ONLY with valid JSON matching the provided schema, with ONE entry per distinct product code.
"""


# ============================================================================
# Web Search Query Templates
# ============================================================================

def build_isp_search_query(country: str) -> str:
    """
    Build web search query for finding ISPs in a country.
    
    Args:
        country: Country name
        
    Returns:
        Search query string
    """
    return f"{country} internet service providers ISP broadband list"


def build_product_codes_search_query(isp: str, country: str) -> str:
    """
    Build web search query for finding product codes for an ISP.
    
    This can be enhanced with:
    - Site-specific searches (e.g., site:isp-domain.com)
    - File type filters (e.g., filetype:pdf)
    - Language-specific terms
    - Manufacturer domains
    
    Args:
        isp: ISP name
        country: Country name
        
    Returns:
        Search query string
    """
    # Base query
    base_query = f"{country} {isp} router modem gateway model number product code"
    
    # TODO: Enhance with:
    # - Site filters: f'site:{isp_domain} {base_query}'
    # - PDF focus: f'{base_query} filetype:pdf "datasheet" OR "specifications"'
    # - Language-specific terms for country
    
    return base_query



