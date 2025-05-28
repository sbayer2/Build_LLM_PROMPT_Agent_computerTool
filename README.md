Core Concept: Use GPT-4o-mini to generate instructions for the computer-use agent based on natural language queries

What We've Built:

✅ Dynamic prompt generation - GPT-4o-mini creates task-specific instructions
✅ Dynamic model creation - Pydantic models generated on-the-fly for any data structure
✅ WebKit browser automation - Launches and navigates automatically

Proof of Concept:

Understand ANY query ->
Generate its own instructions ->
Navigate websites ->
Extract relevant data ->
Handle different data types ->
Save structured results ->

Result:  successfully navigated Newegg (with all its anti-bot measures) and extracted real product data

User query :  on Newegg find the price of acer laptop under 1000 dollars, list the price and website link for the specific laptop


Agent output length: 871
First 300 chars: {
  "found_items": [
    {
      "title": "Acer Aspire 15 15.6\" FHD Intel Core i9-13900H Laptop 16GB Memory 1TB SSD Windows 11 Home A15-51M-9386",
      "position": "Product page",
      "url": "https://www.newegg.com/p/N82E168343060376?quicklink=true",
      "snippet": "Lowest Price in 30 days",
...

Testing connection to the page...
Page title: DuckDuckGo - Protection. Privacy. Peace of mind.
Current URL: https://duckduckgo.com/
Playwright initialization complete
🔍 Analyzing request: 'on Newegg find the price of acer laptop under 1000 dollars, list the price and website link for the specific laptop'
Generating research plan...
📋 Generated Task Configuration (JSON):
============================================================
{
  "task_name": "Find Acer Laptops under $1000 on Newegg",
  "search_terms": [
    "Acer laptop under 1000",
    "Acer laptop price",
    "buy Acer laptop"
  ],
  "target_websites": [
    "newegg.com"
  ],
  "data_to_extract": [
    {
      "field_name": "laptop_name",
      "field_type": "string",
      "description": "The name or model of the Acer laptop"
    },
    {
      "field_name": "price",
      "field_type": "string",
      "description": "The price of the Acer laptop"
    },
    {
      "field_name": "link",
      "field_type": "string",
      "description": "The URL link to the specific laptop on Newegg"
    }
  ],
  "success_criteria": "Found at least 1 item with partial data",
  "example_output": {
    "laptop_name": "Acer Aspire 5 A515-45-R14K",
    "price": "$699.99",
    "link": "https://www.newegg.com/acer-aspire-5-a515-45-r14k/p/N82E16834315402"
  }
}
============================================================
Generated task configuration: Find Acer Laptops under $1000 on Newegg
✅ Research agent configured!
📋 Task: Find Acer Laptops under $1000 on Newegg
🔎 Search terms: Acer laptop under 1000, Acer laptop price, buy Acer laptop
📊 Will extract: laptop_name, price, link
🎯 Success criteria: Found at least 1 item with partial data
