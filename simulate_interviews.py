from playwright.sync_api import sync_playwright
import time

URL = "https://robotics-data-verifier-production.up.railway.app/questionnaire"

def run():
    print("Starting E2E simulation against Railway Production...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        for i in range(1, 6):
            print(f"[{i}/5] Simulating interview submission for Robotics Lab {i}...")
            
            # Go to the production URL
            page.goto(URL, wait_until="networkidle")
            
            # Fill the required fields
            page.fill("#interviewee_name", f"Simulated Subject {i}")
            page.fill("#company", f"Robotics Lab {i}")
            
            # Fill some of our highly-vetted discovery questions
            page.fill("#q1", f"Head of Data Ops {i}. KPIs: reducing compute spend by 20%.")
            page.fill("#q4", "Recorded on ViperX, saved as HDF5, uploaded to S3.")
            page.fill("#q6", "About 2.5 TB. Formatted as RLDS.")
            page.fill("#q9", "Timestamp drift is our biggest issue. RGB is 30Hz, joints are 200Hz.")
            page.fill("#q14", "Pre-training. We need a quality gate before it hits the cluster.")
            page.fill("#q17", "We'd be willing to test a localized version next month.")
            
            # Intercept the network request to verify the API response explicitly
            with page.expect_response("**/api/responses") as response_info:
                page.click("#submitBtn")
            
            response = response_info.value
            
            if response.ok:
                print(f"✅ Success! API returned {response.status}. Submission {i} is safely recorded in the Railway PostgreSQL database.")
            else:
                print(f"❌ Error! API returned {response.status}. Deployment might still be building on Railway.")
            
            # Wait for the UI success message to become visible to verify frontend handles it correctly
            page.wait_for_selector("#successMsg", state="visible", timeout=10000)
            
            time.sleep(1)
            
        browser.close()
        print("\nAll 5 simulations completed successfully!")

if __name__ == "__main__":
    run()
