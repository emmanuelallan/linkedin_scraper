# app.py
import uuid
from flask import Flask, request, render_template, send_file, jsonify, Response
import pandas as pd
import threading
import time
import os
from dotenv import load_dotenv # pip install python-dotenv
from linkedin_scraper import Person, Company, actions
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from google import genai
from selenium.webdriver.chrome.service import Service

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# --- Configuration (Load from .env for security) ---
CHROMEDRIVER_PATH = os.getenv("CHROMEDRIVER_PATH")
LINKEDIN_EMAIL = os.getenv("LINKEDIN_EMAIL")
LINKEDIN_PASSWORD = os.getenv("LINKEDIN_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Initialize Gemini (2025 API) ---
gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Failed to initialize Gemini client: {e}")
        gemini_client = None
else:
    print("WARNING: GEMINI_API_KEY not set. Gemini generation will be skipped.")

# Global dictionary to store processing status for different sessions/uploads
processing_status = {}

# Helper function for retry logic

def scrape_with_retries(scrape_func, max_retries=3, delay=8, label="scrape"):  # delay in seconds
    for attempt in range(max_retries):
        try:
            return scrape_func()
        except Exception as e:
            print(f"{label} attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
    print(f"All {max_retries} {label} attempts failed.")
    return ""

def process_csv_in_background(input_filepath, output_filepath, session_id):
    """
    This function will run in a separate thread.
    It contains the core logic for CSV reading, LinkedIn scraping,
    Gemini API calls, and writing back to CSV.
    """
    global processing_status
    processing_status[session_id] = {"status": "Starting...", "progress": 0, "error": None, "output_file": None}

    driver = None
    try:
        df = pd.read_csv(input_filepath)
        df['Message 1'] = ""
        df['Message 2'] = ""
        df['Message 3'] = ""

        total_rows = len(df)
        if total_rows == 0:
            processing_status[session_id].update({"status": "Completed! (Empty CSV)", "progress": 100})
            return

        # --- Initialize Selenium WebDriver ---
        print("Setting up ChromeDriver...")
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        try:
            if CHROMEDRIVER_PATH:
                service = Service(CHROMEDRIVER_PATH)
                driver = webdriver.Chrome(service=service, options=chrome_options)
            else:
                driver = webdriver.Chrome(options=chrome_options)
            if LINKEDIN_EMAIL and LINKEDIN_PASSWORD:
                actions.login(driver, LINKEDIN_EMAIL, LINKEDIN_PASSWORD)
                print("Logged into LinkedIn.")
            else:
                print("LinkedIn credentials not provided. Scraping might be limited.")
        except Exception as e:
            print(f"Error initializing WebDriver or logging into LinkedIn: {e}")
            processing_status[session_id]["error"] = f"WebDriver setup failed: {e}. Check chromedriver installation and path."
            processing_status[session_id]["status"] = "Failed"
            return

        for index, row in enumerate(df.itertuples(index=False), 0):
            person_linkedin_url = getattr(row, 'defaultProfileUrl', None)
            company_linkedin_url = getattr(row, 'regularCompanyUrl', None)
            person_name = getattr(row, 'fullName', None)
            if not person_name or pd.isna(person_name):
                person_name = f"Lead {index+1}"

            current_progress = int(((index + 1) / total_rows) * 100)
            processing_status[session_id]["progress"] = current_progress
            processing_status[session_id]["status"] = f"Processing {person_name} ({index+1}/{total_rows})..."
            print(f"Processing: {person_name} ({index+1}/{total_rows})")

            person_about = ""
            company_about = ""

            # --- LinkedIn Scraping Logic with Retry ---
            try:
                if driver:
                    if person_linkedin_url and isinstance(person_linkedin_url, str) and person_linkedin_url.startswith('http') and pd.notna(person_linkedin_url):
                        def scrape_person():
                            person = Person(person_linkedin_url, driver=driver, scrape=False)
                            person.scrape(close_on_complete=False)
                            return person.about if person.about else ""
                        person_about = scrape_with_retries(scrape_person, max_retries=3, delay=8, label=f"person scrape for {person_name}")
                        if not person_about:
                            print(f"Failed to scrape person for {person_name} after retries.")
                        else:
                            print(f"Person about: {person_about[:100]}...")
                    else:
                        print(f"Skipping invalid person LinkedIn URL for {person_name}")

                    if company_linkedin_url and isinstance(company_linkedin_url, str) and company_linkedin_url.startswith('http') and pd.notna(company_linkedin_url):
                        def scrape_company():
                            company = Company(company_linkedin_url, driver=driver, scrape=False)
                            company.scrape(close_on_complete=False)
                            return company.about_us if company.about_us else ""
                        company_about = scrape_with_retries(scrape_company, max_retries=3, delay=8, label=f"company scrape for {person_name}")
                        if not company_about:
                            print(f"Failed to scrape company for {person_name} after retries.")
                        else:
                            print(f"Company about: {company_about[:100]}...")
                    else:
                        print(f"Skipping invalid company LinkedIn URL for {person_name}")
                else:
                    print("WebDriver not available, skipping LinkedIn scraping.")
            except Exception as e:
                print(f"Error during LinkedIn scraping for {person_name}: {e}")

            # --- Fallback to mock data if scraping fails ---
            if not person_about:
                person_about = f"Mock: {person_name} is a highly accomplished RF Engineer with a background in designing advanced wireless communication systems. They have worked on several key projects involving 5G infrastructure and satellite communications. Their 'About' section highlights a passion for innovation and pushing the boundaries of radio frequency technology."
            if not company_about:
                company_about = f"Mock: {getattr(row, 'companyName', 'The Company')} is a leading firm in advanced materials and components for the RF and microwave industry. Their mission is to enable next-generation connectivity through cutting-edge RF solutions. They emphasize innovation, quality, and a commitment to solving complex engineering challenges for their clients."

            # --- Gemini Prompt Engineering and Generation ---
            generated_messages = ["", "", ""]
            if (isinstance(person_about, str) and person_about and not person_about.startswith("Mock:")) or (isinstance(company_about, str) and company_about and not company_about.startswith("Mock:")):
                combined_data = f"Individual Profile Summary:\n{person_about}\n\nCompany Profile Summary:\n{company_about}"
                gemini_prompt = f"""
                You are an AI assistant specialized in crafting \"cold\" email introductions with a sharp, analytical, and subtly arrogant tone.
                Your goal is to generate 1-3 distinct, concise observations about the lead or their company that demonstrate deep understanding of their profile/work,
                are direct, and carry a confident, perhaps subtly challenging or superior tone. Avoid generic compliments or flattery.

                Based on the following information for {person_name}:

                {combined_data}

                Generate 3 distinct introductory observations. Each observation should be a standalone sentence or short paragraph,
                designed to cut through the noise and provoke thought. Format them as a numbered list (1., 2., 3.) with no extra text or introduction.
                """
                if gemini_client is not None:
                    try:
                        response = gemini_client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[{"parts": [{"text": gemini_prompt}]}]
                        )
                        raw_output = response.text if hasattr(response, 'text') and response.text is not None else ""
                        parsed_messages = []
                        for line in raw_output.split('\n'):
                            line = line.strip()
                            if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                                parsed_messages.append(line[2:].strip())
                        generated_messages = parsed_messages[:3]
                        while len(generated_messages) < 3:
                            generated_messages.append("")
                        print(f"Gemini output: {generated_messages}")
                    except Exception as e:
                        print(f"Error generating content with Gemini for {person_name}: {e}")
                        generated_messages = ["Gemini Error: " + str(e)[:50], "", ""]
                else:
                    print("Gemini client not initialized, skipping generation.")
                    generated_messages = ["Gemini not initialized.", "", ""]
            else:
                print(f"Skipping Gemini generation due to insufficient data for {person_name}.")

            # --- Consolidate messages back to DataFrame ---
            df.at[index, 'Message 1'] = generated_messages[0]
            df.at[index, 'Message 2'] = generated_messages[1]
            df.at[index, 'Message 3'] = generated_messages[2]

            time.sleep(8) # Increased delay to avoid rate limiting and blocking

        if driver:
            driver.quit()
            print("Browser closed.")

        df.to_csv(output_filepath, index=False)
        processing_status[session_id].update({"status": "Completed!", "progress": 100, "output_file": os.path.basename(output_filepath)})
        print(f"Processing complete for session {session_id}. Data saved to {output_filepath}")

    except Exception as e:
        processing_status[session_id].update({"status": "Failed", "error": str(e), "progress": 0})
        print(f"Critical error in background processing for session {session_id}: {e}")
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


@app.route('/')
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles CSV file upload and starts background processing."""
    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400
    file = request.files['file']
    if not file or not file.filename:
        return jsonify({"message": "No selected file"}), 400
    try:
        session_id = str(uuid.uuid4())
        input_filename = file.filename
        if not input_filename:
            return jsonify({"message": "Invalid filename"}), 400
        input_filepath = os.path.join(UPLOAD_FOLDER, input_filename)
        output_filename = f"enriched_{input_filename}"
        output_filepath = os.path.join(RESULTS_FOLDER, output_filename)
        file.save(input_filepath)
        thread = threading.Thread(target=process_csv_in_background, args=(input_filepath, output_filepath, session_id))
        thread.start()
        return jsonify({
            "message": "File uploaded and processing started.",
            "session_id": session_id,
            "filename": input_filename
        }), 202
    except Exception as e:
        return jsonify({"message": f"Upload failed: {str(e)}"}), 500

@app.route('/status/<session_id>')
def get_status(session_id):
    """Provides the current processing status for a given session ID."""
    status_info = processing_status.get(session_id, {"status": "Waiting...", "progress": 0, "error": None, "output_file": None})
    return jsonify(status_info)

@app.route('/download/<filename>')
def download_file(filename):
    """Allows users to download the processed CSV file."""
    try:
        return send_file(os.path.join(RESULTS_FOLDER, filename), as_attachment=True)
    except FileNotFoundError:
        return jsonify({"message": "File not found."}), 404
    except Exception as e:
        return jsonify({"message": f"Download failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Create a .env file in the same directory as app.py with your credentials:
    # CHROMEDRIVER_PATH=/usr/local/bin/chromedriver # Or wherever your chromedriver is
    # LINKEDIN_EMAIL=your_linkedin_email@example.com
    # LINKEDIN_PASSWORD=your_linkedin_password
    # GEMINI_API_KEY=YOUR_GEMINI_API_KEY
    
    app.run(debug=True, port=5000) # Run on port 5000, set debug=False for production