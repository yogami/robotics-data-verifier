import os
import re
from playwright.sync_api import Page, expect

BASE_URL = os.getenv("PLAYWRIGHT_TEST_BASE_URL", "https://robotics-data-verifier-production.up.railway.app")

def test_homepage_redirects(page: Page):
    """Verify the root URL redirects to the static index (or questionnaire)."""
    page.goto(BASE_URL)
    # The current main.py redirects '/' to '/static/index.html'
    expect(page).to_have_url(re.compile(r".*/static/index\.html.*"))

def test_questionnaire_page_loads(page: Page):
    """Verify the questionnaire UI is accessible and rendering."""
    page.goto(f"{BASE_URL}/questionnaire")
    
    # Wait for the main container to load
    expect(page.locator("body")).to_be_visible()
    
    # Ensure the title exists
    expect(page).to_have_title(re.compile(".*Customer Discovery.*|.*Robotics.*", re.IGNORECASE))

def test_diagnostic_demo_runs(page: Page):
    """Verify the Raw Edge-Compute Quality Gate runs successfully and renders the plot."""
    page.goto(f"{BASE_URL}/diagnostic")
    
    # Verify we are on the diagnostic page
    expect(page.locator("h1")).to_contain_text("Data Quality Gate")
    
    # Click the Run button
    run_btn = page.locator("button#runBtn")
    expect(run_btn).to_be_visible()
    run_btn.click()
    
    # Wait for the backend to process the raw HDF5 math and generate the matplotlib plot.
    terminal_content = page.locator("#terminal-content")
    
    # We expect to see the successful JSON response message
    expect(terminal_content).to_contain_text("All episodes passed quality gates", timeout=15000)
    
    # SUBSTANCE TEST: Assert that the Deep-Tech matplotlib distribution PNG successfully renders in the DOM
    plot_img = terminal_content.locator("img[alt='Deep-Tech Kinematic Entropy Distribution']")
    expect(plot_img).to_be_visible()
    expect(plot_img).to_have_attribute("src", "/static/raw_telemetry_entropy_plot.png")
    
    print("✅ Edge-Compute Playwright substance tests passed successfully.")
