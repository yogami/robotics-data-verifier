import re
from playwright.sync_api import Page, expect

def test_verifier_node_dashboard(page: Page):
    # 1. Navigate to the live Railway URL
    page.goto("https://robotics-data-verifier-production.up.railway.app")
    
    # 2. Assert Title
    expect(page).to_have_title(re.compile("Verifier Node"))
    
    # 3. Assert the Upload Section is visible initially
    upload_section = page.locator("#uploadSection")
    expect(upload_section).to_be_visible()
    
    # 4. Upload the local quality_report.json
    # Note: page.set_input_files allows uploading to an <input type="file">
    page.locator("#reportInput").set_input_files("quality_report.json")
    
    # 5. The upload section should fade out and grid should become visible
    grid = page.locator("#dashboardGrid")
    expect(grid).to_be_visible(timeout=5000)
    
    # 6. Verify the Dataset ID populated correctly
    dataset_id = page.locator("#valDatasetId")
    expect(dataset_id).to_have_text("lerobot/aloha_mobile_cabinet")
    
    # 7. Verify the Slashing Action Logic works based on our data
    action = page.locator("#kpiAction")
    expect(action).to_contain_text("SLASH 50%")
    
    # 8. Verify Flags are rendered
    flags_list = page.locator("#flagsList > li")
    expect(flags_list).to_have_count(2)
    
    print("All E2E UI tests passed successfully on the live Railway deployment!")
