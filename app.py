import os
import asyncio
import json
import time
import base64
import re
import openai
import argparse
from datetime import datetime
from agents import Agent, Runner, ComputerTool, ModelSettings
from agents.computer import AsyncComputer, Environment, Button
from pydantic import BaseModel, Field, create_model
from typing import List, Optional, Dict, Any, Tuple, Literal, Union, Type
from playwright.async_api import async_playwright, Browser, Page

# Check and print API key for debugging (masking most of it)
api_key = os.environ.get("OPENAI_API_KEY", "")
if api_key:
    masked_key = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
    print(f"OpenAI API key found: {masked_key}")
else:
    print("WARNING: No OpenAI API key found in environment variables!")


class PromptGenerator:
    """Generate dynamic instructions for the computer-use agent"""

    def __init__(self, openai_api_key: str = None):
        # Use provided key or fall back to environment variable
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.client = openai.OpenAI(api_key=api_key)

    async def generate_research_instructions(self, user_query: str) -> Tuple[str, Type[BaseModel], dict]:
        """Generate instructions and output model from user query"""

        # Define the schema for our research task
        schema = {
            "type": "object",
            "properties": {
                "task_name": {"type": "string", "description": "Short name for this research task"},
                "search_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords to search for"
                },
                "target_websites": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional specific websites to check"
                },
                "data_to_extract": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field_name": {"type": "string"},
                            "field_type": {"type": "string", "enum": ["string", "number", "array"]},
                            "description": {"type": "string"}
                        }
                    },
                    "description": "Fields to extract from results"
                },
                "success_criteria": {"type": "string", "description": "When to stop searching"},
                "example_output": {"type": "object", "description": "Example of expected output"}
            },
            "required": ["task_name", "search_terms", "data_to_extract", "success_criteria", "example_output"]
        }

        prompt = f"""
        Based on this user request: "{user_query}"

        Generate a research task configuration that includes:
        1. Clear search terms to use
        2. What specific data fields to extract (with their types)
        3. When to stop searching (success criteria - be flexible and achievable)
        4. An example of the expected output structure

        Important: 
        - Success criteria should be IMMEDIATELY achievable (e.g., "Found at least 1 relevant item")
        - Include explicit instructions about extracting data AS SOON AS it's visible
        - Emphasize recording partial data rather than waiting for perfect matches
        
        Guidelines for success_criteria:
        - Use phrases like: "IMMEDIATELY extract after finding ANY product with a visible price"
        - Or: "STOP and extract data from the FIRST relevant result"
        - Or: "Extract from the FIRST page showing products, do not navigate further"
        - Success = Speed of extraction, not quality of results
        - Never require more than 1 item or complete data

        Remember: The agent should extract visible data immediately, not search endlessly for perfect matches.

        Examples:
        - For queries about FINDING LISTS OR COLLECTIONS:
          - search_terms: [main topic, topic + location/qualifier]
          - data_to_extract: relevant fields based on what user wants to know
          - success_criteria: "STOP and extract from FIRST page showing relevant items"

        - For queries about INFORMATION OR RECOMMENDATIONS:
          - search_terms: [topic, topic + "best"/"top"/"reviews"]
          - data_to_extract: name/title (string), key details (string), any ratings if visible (string)
          - success_criteria: "Extract IMMEDIATELY upon finding ANY information about the topic"

        - For queries about PRICES OR PRODUCTS:
          - search_terms: [product, product + "prices"/"buy"]
          - data_to_extract: name (string), price (string), details (string)
          - success_criteria: "STOP searching after finding FIRST item with a price"

        Remember: 
        - Keep success criteria flexible and achievable
        - Extract whatever useful data is visible
        - Don't require specific data formats (like numeric ratings)
        - If the query mentions reviews/ratings, make them optional string fields
        """

        # Call LLM with schema enforcement
        task_config = await self._call_llm(schema, prompt)

        # Generate Pydantic model dynamically based on the task
        output_model = self._create_dynamic_model(task_config['data_to_extract'], task_config['task_name'])

        # Generate agent instructions
        instructions = self._generate_agent_instructions(task_config)

        return instructions, output_model, task_config

    async def _call_llm(self, schema: dict, prompt: str):
        """Call OpenAI with schema enforcement (this is our prompt_llm_for_json!)"""

        for i in range(3):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": f"You MUST produce output that adheres to the following JSON schema:\n\n{json.dumps(schema, indent=4)}. Output your JSON in a ```json markdown block."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3  # Lowered for more consistent results
                )

                response_text = response.choices[0].message.content

                # Extract JSON from markdown
                if "```json" in response_text:
                    start = response_text.find("```json")
                    end = response_text.rfind("```")
                    response_text = response_text[start + 7:end].strip()

                parsed = json.loads(response_text)
                # DEBUG: Show the generated JSON
                print("\n📋 Generated Task Configuration (JSON):")
                print("=" * 60)
                print(json.dumps(parsed, indent=2))
                print("=" * 60)
                print(f"Generated task configuration: {parsed['task_name']}")

                return parsed

            except Exception as e:
                if i == 2:
                    raise e
                print(f"Retry {i + 1}/3: {str(e)}")

    def _create_dynamic_model(self, fields_config: List[dict], task_name: str) -> Type[BaseModel]:
        """Create a Pydantic model dynamically based on the fields configuration"""

        # Map string types to Python types
        type_mapping = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
            "array": List[str]  # Default to list of strings
        }

        # Build field definitions for the found items
        item_fields = {}
        for field in fields_config:
            field_name = field['field_name']
            field_type = type_mapping.get(field['field_type'], str)
            description = field['description']

            item_fields[field_name] = (field_type, Field(..., description=description))

        # Add standard fields that every item should have
        item_fields['title'] = (str, Field(..., description="Name or title of the item"))
        item_fields['position'] = (str, Field(..., description="Position on the webpage"))
        item_fields['url'] = (str, Field(..., description="URL where the item was found"))
        item_fields['snippet'] = (str, Field(..., description="Brief description"))

        # Create the item model
        ItemModel = create_model(
            f'{task_name.replace(" ", "")}Item',
            **item_fields,
            __base__=BaseModel
        )

        # Create the output model that contains a list of items
        OutputModel = create_model(
            f'{task_name.replace(" ", "")}Output',
            found_items=(List[ItemModel], Field(default_factory=list, description="List of found items")),
            search_summary=(str, Field(..., description="Summary of the research results")),
            search_complete=(bool, Field(False, description="Whether the research is complete")),
            timestamp=(str, Field(default_factory=lambda: datetime.now().isoformat())),
            __base__=BaseModel
        )

        return OutputModel

    def _generate_agent_instructions(self, task_config: dict) -> str:
        """Generate detailed instructions for the computer-use agent"""

        # Build the data extraction format
        fields_to_extract = []
        for field in task_config['data_to_extract']:
            fields_to_extract.append(f"- {field['field_name']}: {field['description']}")

        # Enhanced instructions with Google search navigation
        instructions = f"""
        You are a specialized research agent performing: {task_config['task_name']}

        IMPORTANT - DUCKDUCKGO SEARCH INSTRUCTIONS:
        - The DuckDuckGo search box is in the CENTER of the page
        - It has placeholder text "Search without being tracked"
        - Click INSIDE the search box (not on the logo or anywhere else)
        - The search box is a wide rectangular input field
        - After clicking in the search box, type your search terms
        - Then press Enter to search

        Your task is to:
        1. Click in the DuckDuckGo search box that says 'search without being tracked' (the box in the upper center of page)
        2. Type your search query
        3. Press Enter to see results
        4. Click on relevant search results
        5. Extract the requested information

        

        SEARCH TERMS TO USE:
        {', '.join(f'"{term}"' for term in task_config['search_terms'])}

        NAVIGATION HELPERS:
        - You should start on duckduckgo.com
        - Search for: {task_config['search_terms'][0]}
        - Look for and click on relevant search results
        - Common search patterns:
          * Direct product searches: "{task_config['search_terms'][0]}"
          * Store-specific: "[Store Name] {task_config['search_terms'][0]}"
          * Shopping searches: "buy {task_config['search_terms'][0]} online"
        - Click on official store links or shopping results
        - Once on a relevant site, look for the requested data

        DATA TO EXTRACT:
        For each item found, extract:
        {chr(10).join(fields_to_extract)}

        IMPORTANT GUIDELINES:
        - Take your time to observe what's on screen
        - Wait 2-3 seconds after page loads before interacting
        - Move mouse naturally before clicking
        - Add small delays between actions to appear more human
        - Start with a duckduckgo search using: {task_config['search_terms'][0]}
        - Click on promising search results
        - Work within the single browser page provided
        - Navigate through pages as needed to find information
        - Extract data that you can see on screen (don't make up information)

        -If you encounter a CAPTCHA or verification, try returning to search results and clicking a different link

        SUCCESS CRITERIA: {task_config['success_criteria']}

        TIMEOUT PREVENTION: You have a maximum of 20 turns. If you find ANY relevant items before the Maximum , record them in the Response Format.
        
        🚨 CRITICAL DATA EXTRACTION RULES - MUST FOLLOW 🚨:
        1. STOP IMMEDIATELY when you see ANY product with a visible price
        2. DO NOT continue searching after finding the first relevant item
        3. Extract data from the FIRST page that shows products/prices
        4. Product listings, search results, or product pages - ALL are valid for extraction
        5. If you see a price tag, that's your signal to STOP and EXTRACT
        6. DO NOT navigate away from a page showing products
        7. Record what you see RIGHT NOW, not what might be on another page
        
        ⏰ TURN LIMITS:
        - Turn 1-5: Search and navigate to find products
        - Turn 6+: MUST extract and return data, no more searching
        - Turn 10+: EMERGENCY MODE - Return whatever you have immediately

        
        FLEXIBLE DATA EXTRACTION:
        - If you find partial information, record it anyway
        - Don't wait for perfect matches to all fields
        - If a field isn't visible, mark it as "Not found" or "N/A"
        - Any relevant information is better than no information
        - A single product with just name and price meets the success criteria

        🛑 EXTRACTION TRIGGERS - STOP AND EXTRACT WHEN YOU SEE:
        - ANY price (e.g., $19.99, £50, €30)
        - Product names with prices
        - "Add to cart" or "Buy now" buttons
        - Product listings or grids
        - Search results showing products
        - ANY combination of product name + price
        
        ⚡ IMMEDIATE ACTION REQUIRED:
        When you see ANY of the above → STOP navigating and extract NOW!
    

        RESPONSE FORMAT:
        Return a JSON object with this structure:
        {{
            "found_items": [
                {{
                    "title": "Name/title of the item",
                    "position": "Position on page (e.g., '1st result')",
                    "url": "Current page URL",
                    "snippet": "Brief description",
                    {chr(10).join(f'"{field["field_name"]}": <extracted {field["description"]}>,' for field in task_config['data_to_extract'])}
                }}
            ],
            "search_summary": "Summary of what was found",
            "search_complete": true/false
        }}

        🚨 FINAL REMINDER: Your PRIMARY directive is to EXTRACT data, not to find the "best" result.
        The FIRST relevant product you see should be extracted and returned.
        DO NOT continue searching after finding relevant data.
        SUCCESS = Fast extraction, not perfect results.
        """

        return instructions


# Define a simpler playwright-based computer implementation
class PlaywrightComputer(AsyncComputer):
    """A simplified Playwright-based computer implementation for the ComputerTool."""

    def __init__(self):
        """Initialize the PlaywrightComputer."""
        self._width = 1280
        self._height = 720
        self._device_pixel_ratio = 1.0
        self._user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        
        # Add turn tracking for better debugging
        self.turn_count = 0
        self.actions_log = []

    @property
    def environment(self) -> str:
        """Return the environment as a string.

        The OpenAI API expects 'windows', 'mac', 'linux', or 'browser', not an object.
        Based on the error message, we're returning 'mac' since we're on a Mac.
        """
        return 'mac'  # Return as a string, not an object

    async def search_and_navigate(self, search_query: str) -> None:
        """Search via Google and help navigate to results - perfect for WebKit browser."""
        if not self.page:
            raise RuntimeError("Playwright page not initialized")

        try:
            # Check if we're already on Google, if not, navigate there
            current_url = self.page.url
            if "google.com" not in current_url:
                print("Navigating to Google...")
                await self.page.goto("https://www.google.com", wait_until="networkidle")
                await asyncio.sleep(1)

            # Clear any existing search
            search_box = await self.page.query_selector('input[name="q"]')
            if search_box:
                await search_box.click()
                # Clear existing text (Cmd+A for Mac, then Delete)
                await self.page.keyboard.press("Meta+A")  # Select all
                await self.page.keyboard.press("Delete")

            # Type the search query
            print(f"Searching for: {search_query}")
            await self.page.keyboard.type(search_query, delay=50)

            # Press Enter to search
            await self.page.keyboard.press("Enter")

            # Wait for search results
            await self.page.wait_for_selector('div#search', timeout=5000)
            print("Search results loaded")

            # Give results time to fully render
            await asyncio.sleep(2)

        except Exception as e:
            print(f"Error during search: {str(e)}")
            # Fallback: try to navigate to Google again
            try:
                await self.page.goto("https://www.google.com")
                print("Navigated to Google as fallback")
            except:
                print("Could not navigate to Google")

    async def __aenter__(self):
        """Set up Playwright resources based on user's browser preference."""
        try:
            print("Starting Playwright...")
            self.playwright = await async_playwright().start()

            print("\n==== BROWSER SELECTION ====")
            print("Please choose your preferred browser approach:")
            print(
                "1. Connect to existing Chrome with remote debugging (requires Chrome to be running with --remote-debugging-port=9222)")
            print("2. Launch a new WebKit (Safari-like) browser")

            choice = input("Enter 1 or 2: ").strip()

            if choice == "1":
                # Chrome with remote debugging approach
                print("\n==== CHROME REMOTE DEBUGGING ====")
                print("IMPORTANT: Before proceeding, make sure you have:")
                print("1. Launched Chrome with remote debugging using this command in a separate terminal:")
                print(
                    "   /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug-profile")
                print("2. Navigated to a grocery store website (Walmart, Target, Kroger, etc.)")
                print("\nThe script will connect to your existing Chrome session.")
                input("Press Enter when the above steps are completed...")

                # Connect to the existing Chrome browser via CDP
                try:
                    print("Connecting to Chrome with remote debugging...")
                    self.browser = await self.playwright.chromium.connect_over_cdp("http://localhost:9222")
                    print("Successfully connected to Chrome!")

                    # Get all pages
                    all_pages = self.browser.contexts[0].pages
                    print(f"Found {len(all_pages)} pages in the browser")

                    # Use the first available page
                    if all_pages:
                        self.page = all_pages[0]
                        self.context = self.browser.contexts[0]
                        url = self.page.url
                        print(f"Using existing page: {url}")
                    else:
                        print("No open pages found. Creating a new page...")
                        self.context = self.browser.contexts[0]
                        self.page = await self.context.new_page()

                        # Navigate to a grocery website
                        print("Please navigate to a grocery store website to search for egg prices.")
                        input("Press Enter when you've navigated to a grocery website...")
                except Exception as e:
                    print(f"Error connecting to Chrome: {str(e)}")
                    print("Make sure Chrome is running with remote debugging enabled.")
                    print("Falling back to WebKit browser...")
                    choice = "2"  # Fall back to WebKit

            if choice == "2":
                # WebKit (Safari-like) browser approach
                print("\n==== LAUNCHING WEBKIT (SAFARI-LIKE) BROWSER ====")
                input("Press Enter to launch the browser...")

                # Launch a new WebKit browser (Safari-like)
                print("Launching WebKit browser...")
                self.browser = await self.playwright.webkit.launch(
                    headless=False  # Make it visible
                )
                print("Successfully launched WebKit browser!")

                # Create a new context and page
                self.context = await self.browser.new_context(
                    viewport={"width": self._width, "height": self._height},
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
                )
                self.page = await self.context.new_page()

                # Start with Google instead of blank page
                print("Navigating to DuckDuckgo...")
                # Change this line in __aenter__:
                await self.page.goto("https://duckduckgo.com")  # No CAPTCHAs!

                print("\n✅ Browser ready! The agent will:")
                print("1. Search for relevant websites")
                print("2. Click on search results")
                print("3. Extract the requested information")
                print("\nNo manual navigation needed!")

                await asyncio.sleep(2)  # Give user time to see Google loaded


            # Verify we can interact with the page (for both approaches)
            try:
                print("Testing connection to the page...")
                title = await self.page.title()
                url = self.page.url
                print(f"Page title: {title}")
                print(f"Current URL: {url}")
            except Exception as e:
                print(f"Warning: Could not get page information: {str(e)}")

            print("Playwright initialization complete")
            return self

        except Exception as e:
            print(f"ERROR in PlaywrightComputer.__aenter__: {str(e)}")
            import traceback
            traceback.print_exc()
            if hasattr(self, 'playwright') and self.playwright:
                await self.__aexit__(type(e), e, None)
            raise

    async def _is_human_verification_present(self) -> bool:
        """Check if there's a human verification or CAPTCHA on the page."""
        try:
            # Look for common text or elements that might indicate a verification
            captcha_texts = ["human", "captcha", "verify", "robot", "bot check"]

            # Get the page content
            content = await self.page.content()
            content = content.lower()

            # Check if any of the keywords are in the page content
            for text in captcha_texts:
                if text in content:
                    return True

            # Also check for common CAPTCHA elements
            captcha_elements = await self.page.query_selector_all(
                'iframe[src*="captcha"], div[class*="captcha"], div[class*="recaptcha"]')
            if captcha_elements:
                return True

            return False
        except Exception as e:
            print(f"Error checking for human verification: {str(e)}")
            return False

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up Playwright resources."""
        print("Cleaning up Playwright resources...")

        if self.page:
            try:
                await self.page.close()
                print("Page closed")
            except Exception as e:
                print(f"Error closing page: {str(e)}")
            self.page = None

        if self.context:
            try:
                await self.context.close()
                print("Context closed")
            except Exception as e:
                print(f"Error closing context: {str(e)}")
            self.context = None

        if self.browser:
            try:
                await self.browser.close()
                print("Browser closed")
            except Exception as e:
                print(f"Error closing browser: {str(e)}")
            self.browser = None

        if self.playwright:
            try:
                await self.playwright.stop()
                print("Playwright stopped")
            except Exception as e:
                print(f"Error stopping playwright: {str(e)}")
            self.playwright = None

    async def screenshot(self) -> str:
        """Take a screenshot of the current state."""
        self.turn_count += 1
        print(f"\n📸 Turn {self.turn_count}: Taking screenshot")
        
        if not self.page:
            raise RuntimeError("Playwright page not initialized")

        try:
            # Use only basic parameters that are universally supported
            screenshot_bytes = await self.page.screenshot(
                type="jpeg",  # Try JPEG instead of PNG (faster)
                quality=70  # Medium quality for speed
            )

            # Encode to base64
            base64_image = base64.b64encode(screenshot_bytes).decode("utf-8")

            print(f"Screenshot captured successfully (length: {len(base64_image)} chars)")

            return base64_image
        except Exception as e:
            print(f"Error taking screenshot: {str(e)}")
            import traceback
            traceback.print_exc()

            # Try an even simpler fallback
            try:
                print("Attempting basic fallback screenshot...")
                screenshot_bytes = await self.page.screenshot()  # No parameters at all
                base64_image = base64.b64encode(screenshot_bytes).decode("utf-8")
                print(f"Fallback screenshot successful (length: {len(base64_image)} chars)")
                return base64_image
            except Exception as fallback_error:
                print(f"Fallback screenshot also failed: {str(fallback_error)}")
                raise RuntimeError(f"Failed to capture screenshot: {str(e)}")

    async def click(self, x: int, y: int, button: str = "left") -> None:
        """Click at the specified coordinates with the specified button.

        Args:
            x: The x coordinate to click at
            y: The y coordinate to click at
            button: The mouse button to use ("left", "right", or "middle")
        """
        if not self.page:
            raise RuntimeError("Playwright page not initialized")

        # Validate button parameter
        valid_buttons = ["left", "right", "middle"]
        if button not in valid_buttons:
            print(f"Warning: Invalid button '{button}' requested, defaulting to 'left'")
            button = "left"

        try:
            await self.page.mouse.click(x, y, button=button)
            action = f"Clicked at ({x}, {y}) with {button} button"
            print(f"🖱️  Turn {self.turn_count}: {action}")
            self.actions_log.append({"turn": self.turn_count, "action": action})
            
            # Check current page content for products/prices
            current_url = self.page.url
            if "search" in current_url or "product" in current_url or "item" in current_url:
                print(f"📍 Turn {self.turn_count}: On potential product page - {current_url}")
                if self.turn_count >= 8:
                    print(f"🚨 Turn {self.turn_count}: CRITICAL - Should extract NOW from {current_url}")
            
            # Add a small delay after clicking to allow page to respond
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Error performing click at ({x}, {y}) with {button} button: {str(e)}")
            raise

    async def double_click(self, x: int, y: int, button: str = "left") -> None:
        """Double click at the specified coordinates with the specified button.

        Args:
            x: The x coordinate to click at
            y: The y coordinate to click at
            button: The mouse button to use ("left", "right", or "middle")
        """
        if not self.page:
            raise RuntimeError("Playwright page not initialized")

        # Validate button parameter
        valid_buttons = ["left", "right", "middle"]
        if button not in valid_buttons:
            print(f"Warning: Invalid button '{button}' requested, defaulting to 'left'")
            button = "left"

        try:
            await self.page.mouse.dblclick(x, y, button=button)
            print(f"Double-clicked at coordinates: ({x}, {y}) with {button} button")
            # Add a small delay after double-clicking
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Error performing double-click at ({x}, {y}) with {button} button: {str(e)}")
            raise

    async def keypress(self, keys: Union[str, List[str]]) -> None:
        """Press one or more keys."""
        if not self.page:
            raise RuntimeError("Playwright page not initialized")

        try:
            # Handle both string and list input
            if isinstance(keys, list):
                # Map common key names to Playwright format
                mapped_keys = []
                for key in keys:
                    if key.upper() == "CTRL":
                        mapped_keys.append("Control")
                    elif key.upper() == "CMD":
                        mapped_keys.append("Meta")
                    elif key.upper() == "ALT":
                        mapped_keys.append("Alt")
                    elif key.upper() == "SHIFT":
                        mapped_keys.append("Shift")
                    elif key.upper() == "ENTER":
                        mapped_keys.append("Enter")
                    elif key.upper() == "TAB":
                        mapped_keys.append("Tab")
                    elif key.upper() == "ESCAPE":
                        mapped_keys.append("Escape")
                    else:
                        mapped_keys.append(key)

                if len(mapped_keys) == 1:
                    # Single key from list
                    await self.page.keyboard.press(mapped_keys[0])
                    print(f"Pressed key: {mapped_keys[0]}")
                else:
                    # Key combination
                    key_combination = "+".join(mapped_keys)
                    await self.page.keyboard.press(key_combination)
                    print(f"Pressed key combination: {key_combination}")
            else:
                # Fix common case issues for single keys
                if keys.upper() == "ENTER":
                    keys = "Enter"
                elif keys.upper() == "TAB":
                    keys = "Tab"
                elif keys.upper() == "ESCAPE":
                    keys = "Escape"
                elif keys.upper() == "CTRL":
                    keys = "Control"
                elif keys.upper() == "CMD":
                    keys = "Meta"

                await self.page.keyboard.press(keys)
                print(f"Pressed key: {keys}")
        except Exception as e:
            print(f"Error pressing keys {keys}: {str(e)}")
            raise  # Make sure this says 'raise' not 'raiseexit'

    async def drag(self, from_x: int, from_y: int, to_x: int, to_y: int) -> None:
        """Drag from one set of coordinates to another."""
        if not self.page:
            raise RuntimeError("Playwright page not initialized")

        try:
            await self.page.mouse.move(from_x, from_y)
            await self.page.mouse.down()
            await self.page.mouse.move(to_x, to_y)
            await self.page.mouse.up()
            print(f"Dragged from ({from_x}, {from_y}) to ({to_x}, {to_y})")
        except Exception as e:
            print(f"Error dragging from ({from_x}, {from_y}) to ({to_x}, {to_y}): {str(e)}")
            raise

    async def type(self, text: str, delay: int = 0) -> None:
        """Type the specified text with an optional delay between keystrokes.

        Args:
            text: The text to type
            delay: Optional delay between keystrokes in milliseconds (default: 0)
        """
        if not self.page:
            raise RuntimeError("Playwright page not initialized")

        try:
            if delay > 0:
                await self.page.keyboard.type(text, delay=delay)
            else:
                await self.page.keyboard.type(text)
            action = f"Typed: {text}"
            print(f"⌨️  Turn {self.turn_count}: {action}")
            self.actions_log.append({"turn": self.turn_count, "action": action})
            
            # Check if we're past turn 6 and should be extracting
            if self.turn_count >= 6:
                print(f"⚠️  Turn {self.turn_count}: Agent should be extracting data by now!")
        except Exception as e:
            print(f"Error typing text: {str(e)}")
            raise

    async def press(self, key: str) -> None:
        """Press a specific key."""
        if not self.page:
            raise RuntimeError("Playwright page not initialized")

        try:
            await self.page.keyboard.press(key)
            print(f"Pressed key: {key}")
        except Exception as e:
            print(f"Error pressing key {key}: {str(e)}")
            raise



    async def navigate(self, url: str) -> None:
        """Navigate to a URL."""
        if not self.page:
            raise RuntimeError("Playwright page not initialized")

        try:
            await self.page.goto(url, wait_until="domcontentloaded")
            print(f"Navigated to: {url}")
            # Wait for page to stabilize after navigation
            await asyncio.sleep(2)
        except Exception as e:
            print(f"Error navigating to {url}: {str(e)}")
            raise

    async def move(self, x: int, y: int) -> None:
        """Move the mouse to the specified coordinates."""
        if not self.page:
            raise RuntimeError("Playwright page not initialized")

        try:
            await self.page.mouse.move(x, y)
            print(f"Moved mouse to: ({x}, {y})")
        except Exception as e:
            print(f"Error moving mouse to ({x}, {y}): {str(e)}")
            raise

    async def scroll(self, x: int, y: int, scroll_x: int = 0, scroll_y: int = 0) -> None:
        """Scroll by the specified amount.

        Args:
            x: The x coordinate to position the mouse at before scrolling
            y: The y coordinate to position the mouse at before scrolling
            scroll_x: The amount to scroll horizontally (positive = right, negative = left)
            scroll_y: The amount to scroll vertically (positive = down, negative = up)
        """
        if not self.page:
            raise RuntimeError("Playwright page not initialized")

        try:
            # First move to the specified position
            if x > 0 and y > 0:
                await self.page.mouse.move(x, y)

            # Then scroll using the wheel event
            await self.page.mouse.wheel(scroll_x, scroll_y)
            print(f"Scrolled at position ({x}, {y}) by ({scroll_x}, {scroll_y})")

            # Add a small delay after scrolling to allow page to update
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Error scrolling at ({x}, {y}) by ({scroll_x}, {scroll_y}): {str(e)}")
            raise

    async def wait(self, ms: int = 1000) -> None:
        """Wait for the specified number of milliseconds.

        Args:
            ms: The number of milliseconds to wait (default: 1000ms = 1 second)
        """
        try:
            await asyncio.sleep(ms / 1000.0)
            print(f"Waited for {ms} milliseconds")
        except Exception as e:
            print(f"Error waiting for {ms} ms: {str(e)}")
            raise





    @property
    def dimensions(self) -> tuple:
        """Return the dimensions of the browser window as a tuple."""
        return (self._width, self._height)


# After the PlaywrightComputer class ends (after the dimensions property)

async def debug_screenshot(computer):
    """Debug function to test screenshot capture and formatting."""
    import base64  # Make sure this import is added if needed

    try:
        # Take a screenshot
        screenshot_data = await computer.screenshot()

        # Print basic info about the screenshot
        print(f"Screenshot data type: {type(screenshot_data)}")
        print(f"Screenshot data length: {len(screenshot_data)}")
        print(f"Screenshot data starts with: {screenshot_data[:50]}...")

        # Validate the data URL format
        if screenshot_data.startswith("data:image/"):
            print("✓ Screenshot has correct data URL format")
        else:
            print("✗ Screenshot does NOT have correct data URL format")

        # Check for valid base64 content
        try:
            parts = screenshot_data.split(",", 1)
            if len(parts) == 2:
                header, content = parts
                # Try decoding the base64 content
                decoded = base64.b64decode(content)
                print(f"✓ Base64 content is valid (decoded length: {len(decoded)} bytes)")
            else:
                print("✗ Screenshot data does not have proper data URL structure with comma separator")
        except Exception as e:
            print(f"✗ Base64 content is invalid: {str(e)}")

        return screenshot_data
    except Exception as e:
        print(f"Debug screenshot error: {str(e)}")
        return None


# Then the SimpleComputer class follows


# A simplified computer for testing without browser automation
class SimpleComputer(AsyncComputer):
    """A simple computer implementation for the ComputerTool (for testing)."""

    def __init__(self):
        """Initialize the SimpleComputer."""
        self._width = 1280
        self._height = 720
        self._device_pixel_ratio = 1.0
        self._user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

    @property
    def environment(self) -> str:
        """Return the environment as a string.

        The OpenAI API expects 'windows', 'mac', 'linux', or 'browser', not an object.
        Based on the error message, we're returning 'mac' since we're on a Mac.
        """
        return 'mac'  # Return as a string, not an object

    async def screenshot(self) -> str:
        """Take a screenshot of the current state."""
        print("Taking screenshot (simulated)")
        return json.dumps({"base64": "simulated_base64_image_data"})

    async def click(self, x: int, y: int) -> None:
        """Click at the specified coordinates."""
        print(f"Clicked at ({x}, {y}) (simulated)")

    async def double_click(self, x: int, y: int) -> None:
        """Double click at the specified coordinates."""
        print(f"Double-clicked at ({x}, {y}) (simulated)")

    async def type(self, text: str) -> None:
        """Type the specified text."""
        print(f"Typed: {text} (simulated)")

    async def press(self, key: str) -> None:
        """Press a specific key."""
        print(f"Pressed key: {key} (simulated)")

    async def keypress(self, key: str) -> None:
        """Press a specific key (alias for press)."""
        await self.press(key)

    async def navigate(self, url: str) -> None:
        """Navigate to a URL."""
        print(f"Navigated to: {url} (simulated)")

    async def move(self, x: int, y: int) -> None:
        """Move the mouse to the specified coordinates."""
        print(f"Moved mouse to: ({x}, {y}) (simulated)")

    async def scroll(self, x: int, y: int) -> None:
        """Scroll by the specified amount."""
        print(f"Scrolled by: ({x}, {y}) (simulated)")

    async def drag(self, from_x: int, from_y: int, to_x: int, to_y: int) -> None:
        """Drag from one set of coordinates to another."""
        print(f"Dragged from ({from_x}, {from_y}) to ({to_x}, {to_y}) (simulated)")

    async def wait(self, ms: int) -> None:
        """Wait for the specified number of milliseconds."""
        print(f"Waited for {ms} milliseconds (simulated)")

    @property
    def dimensions(self) -> tuple:
        """Return the dimensions of the browser window as a tuple."""
        return (self._width, self._height)


class DynamicResearchAgent:
    """General-purpose research agent with dynamic instructions"""

    def __init__(self, api_key: Optional[str] = None, computer: Optional[AsyncComputer] = None):
        """Initialize the Dynamic Research Agent"""
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        self.computer = computer
        self.prompt_generator = PromptGenerator()
        self.current_task = None
        self.output_model = None
        self.agent = None

    async def setup_task(self, user_query: str) -> dict:
        """Setup a new research task based on user query"""
        print(f"\n🔍 Analyzing request: '{user_query}'")
        print("Generating research plan...")

        # Generate instructions and output model
        instructions, output_model, task_config = await self.prompt_generator.generate_research_instructions(user_query)

        self.current_task = task_config['task_name']
        self.output_model = output_model

        # Create the agent with dynamic instructions
        self.agent = self._create_agent(instructions)

        print(f"\n✅ Research agent configured!")
        print(f"📋 Task: {self.current_task}")
        print(f"🔎 Search terms: {', '.join(task_config['search_terms'])}")
        print(f"📊 Will extract: {', '.join(f['field_name'] for f in task_config['data_to_extract'])}")
        print(f"🎯 Success criteria: {task_config['success_criteria']}")

        return task_config

    def _create_agent(self, instructions: str) -> Agent:
        """Create agent with dynamic instructions"""
        if self.computer is None:
            self.computer = SimpleComputer()

        model_settings = ModelSettings(truncation="auto")

        return Agent(
            name=f"Dynamic Research Agent - {self.current_task}",
            instructions=instructions,
            tools=[ComputerTool(computer=self.computer)],
            model="computer-use-preview",
            model_settings=model_settings
        )

    async def search(self) -> BaseModel:
        """Run the agent to perform the research"""
        if not self.agent:
            raise RuntimeError("Must call setup_task() before search()")

        try:
            print(f"\n🚀 Starting {self.current_task}...")
            print("Agent is working (this may take a few minutes)...")
            start_time = time.time()

            # Create a more assertive extraction prompt
            extraction_prompt = (
                f"TASK: {self.current_task}\n\n"
                "🚨 CRITICAL INSTRUCTION: The MOMENT you see ANY product with a price, "
                "STOP searching and extract that data immediately. "
                "DO NOT navigate to other pages once you find products. "
                "Extract from the FIRST relevant page you find. "
                "Your success is measured by extraction speed, not finding the 'perfect' result."
            )
            
            result = await asyncio.wait_for(
                Runner.run(self.agent, extraction_prompt, max_turns=20),
                timeout=600  # 10 minutes
            )

            end_time = time.time()
            print(f"\n✅ Research completed in {end_time - start_time:.2f} seconds")
            
            # Print action summary if using PlaywrightComputer
            if hasattr(self.computer, 'actions_log') and self.computer.actions_log:
                print(f"\n📊 Action Summary - Total turns: {self.computer.turn_count}")
                print("=" * 60)
                for action in self.computer.actions_log[-10:]:  # Show last 10 actions
                    print(f"Turn {action['turn']}: {action['action']}")
                print("=" * 60)

            # Parse the result
            output_text = result.final_output

            # DEBUG: Let's see what the agent actually returned
            print(f"\n🔍 DEBUG - Raw agent output:")
            print("=" * 60)
            print(output_text[:500] + "..." if len(output_text) > 500 else output_text)
            print("=" * 60)
            # Add:
            print(f"\n🔍 DEBUG - Agent output length: {len(output_text)}")
            print(f"First 300 chars: {output_text[:300]}...")

            # Extract JSON
            json_start = output_text.find("{")
            json_end = output_text.rfind("}")

            if json_start >= 0 and json_end > json_start:
                json_str = output_text[json_start:json_end + 1]
                print(f"\n🔍 DEBUG - Found JSON, length: {len(json_str)}")
                try:
                    data = json.loads(json_str)
                    print(f"Parsed data has keys: {list(data.keys())}")
                    print(f"Number of found_items: {len(data.get('found_items', []))}")
                except Exception as e:
                    print(f"JSON parse error: {e}")
                data = json.loads(json_str)
                return self.output_model(**data)
            else:
                # Return empty result if no JSON found
                return self.output_model(
                    search_summary="No structured data found in agent output",
                    search_complete=True
                )

        except asyncio.TimeoutError:
            return self.output_model(
                search_summary="Research timed out after 10 minutes",
                search_complete=False
            )
        except Exception as e:
            print(f"Error during research: {str(e)}")
            return self.output_model(
                search_summary=f"Error during research: {str(e)}",
                search_complete=False
            )


async def save_search_results(output: BaseModel, filename: Optional[str] = None) -> str:
    """Save search results to a JSON file.

    Args:
        output: The research results to save (any Pydantic model)
        filename: Optional custom filename, otherwise auto-generated with timestamp

    Returns:
        filepath: The path to the saved file
    """
    try:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_{timestamp}.json"

        # Ensure the results directory exists
        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", filename)

        # Save the results
        print(f"Saving results to {filepath}...")
        with open(filepath, "w") as f:
            # Use model_dump() for any Pydantic model
            json.dump(output.model_dump(), f, indent=2)

        print(f"Results saved successfully")
        return filepath
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        return f"Error saving to {filename}: {str(e)}"


async def main_async(user_query: str, save_to_file: bool = True, use_playwright: bool = True):
    """Dynamic research based on user query"""

    print("=" * 60)
    print("🤖 DYNAMIC RESEARCH AGENT")
    print("=" * 60)

    # Create computer implementation
    if use_playwright:
        try:
            async with PlaywrightComputer() as computer:
                agent = DynamicResearchAgent(computer=computer)

                # Setup the task
                task_config = await agent.setup_task(user_query)

                # No more manual navigation needed!
                print("\n🔍 Agent will now:")
                print("1. Search Google for relevant sites")
                print("2. Navigate to appropriate results")
                print("3. Extract the requested information")
                print("\n🚀 Starting automated research...")


                # Run the search
                result = await agent.search()

                # Display results
                print(f"\n📊 RESULTS: Found {len(result.found_items)} items")
                print(f"📝 Summary: {result.search_summary}")

                # Display found items
                for i, item in enumerate(result.found_items, 1):
                    print(f"\n--- Item {i} ---")
                    print(f"Title: {item.title}")
                    print(f"Position: {item.position}")
                    # Print dynamic fields
                    for field in task_config['data_to_extract']:
                        field_name = field['field_name']
                        if hasattr(item, field_name):
                            print(f"{field_name.title()}: {getattr(item, field_name)}")
                    print(f"URL: {item.url}")

                # Save results
                if save_to_file and result.found_items:
                    task_name = task_config['task_name'].replace(" ", "_")[:30]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{task_name}_{timestamp}.json"
                    filepath = await save_search_results(result, filename)
                    print(f"💾 Results saved to: {filepath}")

                return result
        except Exception as e:
            print(f"Error with Playwright: {str(e)}")
            print("Falling back to simple computer...")

    # Fallback to simple computer
    agent = DynamicResearchAgent()
    await agent.setup_task(user_query)
    return await agent.search()


def main():
    """Main entry point with natural language input"""
    parser = argparse.ArgumentParser(description="Dynamic Web Research Tool")
    parser.add_argument(
        "query",
        nargs='?',  # Make it optional
        help="What to research (e.g., 'cat food prices', 'laptop deals under $1000')"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run without saving results to a file"
    )
    parser.add_argument(
        "--simple-computer",
        action="store_true",
        help="Use simple computer implementation (no browser automation)"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (will override environment variable)"
    )

    args = parser.parse_args()

    # Set API key if provided
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    # Interactive mode if no query provided
    if not args.query:
        print("🤖 DYNAMIC RESEARCH AGENT")
        print("\nExamples:")
        print("  - 'find prices for organic eggs'")
        print("  - 'search for cat food prices at pet stores'")
        print("  - 'outdoor birdhouse kits under $50'")
        print("  - 'laptop deals for students'")
        print("  - 'coffee makers with good reviews'")
        query = input("\nWhat would you like me to research? ")
    else:
        query = args.query

    if query:
        asyncio.run(main_async(
            query,
            save_to_file=not args.no_save,
            use_playwright=not args.simple_computer  # Note: changed from args.simple
        ))
    else:
        print("No query provided. Exiting.")


if __name__ == "__main__":
    main()