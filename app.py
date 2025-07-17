import uuid
import re
import logging
from flask import Flask, request, render_template, send_file, jsonify, Response
import pandas as pd
import threading
import time
import os
from dotenv import load_dotenv
from linkedin_scraper import Person, Company, actions
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from google import genai
from selenium.webdriver.chrome.service import Service
from werkzeug.utils import secure_filename
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import secrets
import json
import openai

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
limiter.init_app(app)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# --- Configuration (Load from .env for security) ---
CHROMEDRIVER_PATH = os.getenv("CHROMEDRIVER_PATH")
DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Global dictionary to store processing status for different sessions/uploads
processing_status = {}

# Default prompt template
DEFAULT_PROMPT_TEMPLATE = """You are an AI assistant specialized in crafting "cold" email introductions with a sharp, analytical, and subtly arrogant tone.
Your goal is to generate 1-3 distinct, concise observations about the lead or their company that demonstrate deep understanding of their profile/work,
are direct, and carry a confident, perhaps subtly challenging or superior tone. Avoid generic compliments or flattery.

Based on the following information for {person_name}:

{combined_data}

Generate 3 distinct introductory observations. Each observation should be a standalone sentence or short paragraph,
designed to cut through the noise and provoke thought. Format them as a numbered list (1., 2., 3.) with no extra text or introduction."""

# Helper function for retry logic
def scrape_with_retries(scrape_func, max_retries=3, delay=8, label="scrape"):
    for attempt in range(max_retries):
        try:
            return scrape_func()
        except Exception as e:
            logger.warning(f"{label} attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
    logger.error(f"All {max_retries} {label} attempts failed.")
    return ""

# Updated helper function to resolve LinkedIn company URL and convert to ca.linkedin.com format
def resolve_linkedin_company_url(url, driver, wait_time=5):
    """
    Resolves LinkedIn company URL from numeric ID to actual company URL
    and converts it to ca.linkedin.com format expected by linkedin_scraper library
    """
    try:
        # First, navigate to the URL to get the redirected/resolved URL
        driver.get(url)
        time.sleep(wait_time)
        final_url = driver.current_url
        
        # Check if we got a valid company URL
        if '/company/' in final_url:
            # Convert www.linkedin.com to ca.linkedin.com format expected by the library
            if 'www.linkedin.com/company/' in final_url:
                # Extract the company slug from the URL
                company_part = final_url.split('www.linkedin.com/company/')[-1]
                # Remove any trailing parameters or fragments
                company_slug = company_part.split('?')[0].split('#')[0].rstrip('/')
                # Convert to ca.linkedin.com format
                resolved_url = f"https://ca.linkedin.com/company/{company_slug}"
                logger.info(f"Converted URL from {final_url} to {resolved_url}")
                return resolved_url
            elif 'ca.linkedin.com/company/' in final_url:
                # Already in correct format
                logger.info(f"URL already in correct format: {final_url}")
                return final_url
            else:
                logger.warning(f"Unexpected LinkedIn company URL format: {final_url}")
                return final_url
        else:
            logger.warning(f"Resolved URL is not a valid company URL: {final_url}")
            # Try to convert the original URL if it contains company info
            if '/company/' in url:
                company_part = url.split('/company/')[-1]
                company_slug = company_part.split('?')[0].split('#')[0].rstrip('/')
                fallback_url = f"https://ca.linkedin.com/company/{company_slug}"
                logger.info(f"Using fallback URL: {fallback_url}")
                return fallback_url
            return url
            
    except Exception as e:
        logger.error(f"Error resolving company URL: {e}")
        # Try to create a fallback URL if possible
        if '/company/' in url:
            try:
                company_part = url.split('/company/')[-1]
                company_slug = company_part.split('?')[0].split('#')[0].rstrip('/')
                fallback_url = f"https://ca.linkedin.com/company/{company_slug}"
                logger.info(f"Using fallback URL after error: {fallback_url}")
                return fallback_url
            except:
                pass
        return url

# Alternative approach: Use company name to construct URL if numeric ID fails
def get_company_url_from_name(company_name, driver, wait_time=5):
    """
    Construct LinkedIn company URL from company name
    This can be used as a fallback when numeric ID resolution fails
    """
    if not company_name or pd.isna(company_name):
        return None
    
    try:
        # Clean company name and convert to URL slug format
        # Remove common suffixes and special characters
        clean_name = company_name.lower()
        clean_name = re.sub(r'\b(inc|llc|ltd|corporation|corp|company|co)\b\.?', '', clean_name)
        clean_name = re.sub(r'[^\w\s-]', '', clean_name)  # Remove special chars except hyphens
        clean_name = re.sub(r'\s+', '-', clean_name.strip())  # Replace spaces with hyphens
        clean_name = re.sub(r'-+', '-', clean_name)  # Replace multiple hyphens with single
        clean_name = clean_name.strip('-')  # Remove leading/trailing hyphens
        
        # Construct URL
        company_url = f"https://ca.linkedin.com/company/{clean_name}"
        logger.info(f"Constructed company URL from name '{company_name}': {company_url}")
        return company_url
        
    except Exception as e:
        logger.error(f"Error constructing company URL from name '{company_name}': {e}")
        return None

# Updated company scraping logic with fallback
def scrape_company_with_fallback(company_linkedin_url, company_name, driver, delay_seconds, person_name):
    """
    Scrape company information with fallback to company name-based URL
    """
    company_about = ""
    
    # Try with the original/resolved URL first
    if company_linkedin_url and isinstance(company_linkedin_url, str) and company_linkedin_url.startswith('http') and pd.notna(company_linkedin_url):
        try:
            resolved_url = resolve_linkedin_company_url(company_linkedin_url, driver)
            logger.info(f"Attempting to scrape company with resolved URL: {resolved_url}")
            
            def scrape_company():
                company = Company(resolved_url, driver=driver, scrape=False)
                company.scrape(close_on_complete=False)
                return company.about_us if company.about_us else ""
            
            company_about = scrape_with_retries(scrape_company, max_retries=1, delay=delay_seconds, label=f"company scrape for {person_name}")
            
            if company_about:
                logger.info(f"Successfully scraped company info: {company_about[:100]}...")
                return company_about
            else:
                logger.warning(f"No company info found with resolved URL: {resolved_url}")
                
        except Exception as e:
            logger.error(f"Error scraping company with resolved URL: {e}")
    
    # Fallback: Try constructing URL from company name
    if not company_about and company_name:
        try:
            name_based_url = get_company_url_from_name(company_name, driver)
            if name_based_url:
                logger.info(f"Attempting to scrape company with name-based URL: {name_based_url}")
                
                def scrape_company_by_name():
                    company = Company(name_based_url, driver=driver, scrape=False)
                    company.scrape(close_on_complete=False)
                    return company.about_us if company.about_us else ""
                
                company_about = scrape_with_retries(scrape_company_by_name, max_retries=1, delay=delay_seconds, label=f"company scrape by name for {person_name}")
                
                if company_about:
                    logger.info(f"Successfully scraped company info using name-based URL: {company_about[:100]}...")
                else:
                    logger.warning(f"No company info found with name-based URL: {name_based_url}")
                    
        except Exception as e:
            logger.error(f"Error scraping company with name-based URL: {e}")
    
    return company_about

def suggest_column_mapping(df):
    """Suggest column mappings based on content analysis"""
    suggestions = {
        'name_column': None,
        'profile_column': None,
        'company_column': None,
        'company_name_column': None  # Add this new suggestion
    }
    
    columns = df.columns.tolist()
    
    # Analyze first few rows for patterns
    sample_size = min(10, len(df))
    sample_df = df.head(sample_size)
    
    for col in columns:
        col_lower = col.lower()
        sample_values = sample_df[col].astype(str).str.lower()
        
        # Check for name patterns
        if any(keyword in col_lower for keyword in ['name', 'full', 'contact', 'person']):
            if not suggestions['name_column']:
                suggestions['name_column'] = col
        
        # Check for LinkedIn profile patterns
        if any('/in/' in str(val) for val in sample_values) or 'linkedin.com/in' in col_lower:
            suggestions['profile_column'] = col
        
        # Check for company LinkedIn URL patterns
        if any('/company/' in str(val) for val in sample_values) or ('linkedin' in col_lower and 'company' in col_lower):
            suggestions['company_column'] = col
        
        # Check for company name patterns
        if any(keyword in col_lower for keyword in ['company', 'organization', 'employer', 'firm']):
            # Prioritize exact matches
            if col_lower in ['company', 'companyname', 'company_name', 'organization', 'employer']:
                suggestions['company_name_column'] = col
            elif not suggestions['company_name_column'] and 'name' in col_lower:
                suggestions['company_name_column'] = col
    
    return suggestions

def initialize_gemini_client(api_key):
    """Initialize Gemini client with provided API key"""
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        return None

def initialize_openai_client(api_key):
    """Initialize OpenAI client with provided API key"""
    try:
        return openai.OpenAI(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return None

def generate_content_with_ai(prompt, config):
    """Generate content using either Gemini or OpenAI based on config"""
    ai_provider = config.get('ai_provider', 'gemini')
    
    if ai_provider == 'openai':
        # Use OpenAI GPT
        openai_api_key = config.get('openai_api_key') or DEFAULT_OPENAI_API_KEY
        if not openai_api_key:
            return ["OpenAI API key required", "", ""]
        
        try:
            client = initialize_openai_client(openai_api_key)
            if not client:
                return ["Failed to initialize OpenAI client", "", ""]
            
            model = config.get('openai_model', 'gpt-4o')
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an AI assistant specialized in crafting cold email introductions with a sharp, analytical tone."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            raw_output = response.choices[0].message.content if response.choices else ""
            parsed_messages = []
            for line in raw_output.split('\n'):
                line = line.strip()
                if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                    parsed_messages.append(line[2:].strip())
            
            # Ensure we have exactly 3 messages
            while len(parsed_messages) < 3:
                parsed_messages.append("")
            
            return parsed_messages[:3]
            
        except Exception as e:
            logger.error(f"Error generating content with OpenAI: {e}")
            return [f"OpenAI Error: {str(e)[:50]}", "", ""]
    
    else:
        # Use Gemini (default)
        gemini_api_key = config.get('gemini_api_key') or DEFAULT_GEMINI_API_KEY
        if not gemini_api_key:
            return ["Gemini API key required", "", ""]
        
        try:
            client = initialize_gemini_client(gemini_api_key)
            if not client:
                return ["Failed to initialize Gemini client", "", ""]
            
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=[{"parts": [{"text": prompt}]}]
            )
            
            raw_output = response.text if hasattr(response, 'text') and response.text is not None else ""
            parsed_messages = []
            for line in raw_output.split('\n'):
                line = line.strip()
                if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                    parsed_messages.append(line[2:].strip())
            
            # Ensure we have exactly 3 messages
            while len(parsed_messages) < 3:
                parsed_messages.append("")
            
            return parsed_messages[:3]
            
        except Exception as e:
            logger.error(f"Error generating content with Gemini: {e}")
            return [f"Gemini Error: {str(e)[:50]}", "", ""]

def process_csv_in_background(input_filepath, output_filepath, session_id, config):
    """
    This function will run in a separate thread.
    It contains the core logic for CSV reading, LinkedIn scraping,
    Gemini API calls, and writing back to CSV.
    """
    global processing_status
    processing_status[session_id] = {"status": "Starting...", "progress": 0, "error": None, "output_file": None, "log_message": "Initializing process..."}

    driver = None
    gemini_client = None
    
    try:
        # Initialize Gemini client with user's API key
        if config.get('gemini_api_key'):
            gemini_client = initialize_gemini_client(config['gemini_api_key'])
        elif DEFAULT_GEMINI_API_KEY:
            gemini_client = initialize_gemini_client(DEFAULT_GEMINI_API_KEY)
        
        df = pd.read_csv(input_filepath)
        df['Message 1'] = ""
        df['Message 2'] = ""
        df['Message 3'] = ""

        total_rows = len(df)
        if total_rows == 0:
            processing_status[session_id].update({"status": "Completed! (Empty CSV)", "progress": 100})
            return

        # Extract column mappings from config
        name_col = config.get('name_column')
        profile_col = config.get('profile_column')
        company_col = config.get('company_column')
        
        # Get delay setting
        delay_seconds = max(int(config.get('delay_seconds', 8)), 8)  # Minimum 8 seconds
        
        # --- Initialize Selenium WebDriver ---
        logger.info("Setting up ChromeDriver...")
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        try:
            if CHROMEDRIVER_PATH:
                service = Service(CHROMEDRIVER_PATH)
                driver = webdriver.Chrome(service=service, options=chrome_options)
            else:
                driver = webdriver.Chrome(options=chrome_options)
            
            # Use user's LinkedIn credentials if provided
            linkedin_email = config.get('linkedin_email')
            linkedin_password = config.get('linkedin_password')
            
            if linkedin_email and linkedin_password:
                actions.login(driver, linkedin_email, linkedin_password)
                logger.info("Logged into LinkedIn with user credentials.")
            else:
                logger.warning("LinkedIn credentials not provided. Scraping might be limited.")
                
        except Exception as e:
            logger.error(f"Error initializing WebDriver or logging into LinkedIn: {e}")
            processing_status[session_id]["error"] = f"WebDriver setup failed: {e}"
            processing_status[session_id]["status"] = "Failed"
            return

        for index, row in df.iterrows():
            # Get values using configured column mappings
            person_name = row.get(name_col, f"Lead {index+1}") if name_col else f"Lead {index+1}"
            person_linkedin_url = row.get(profile_col) if profile_col else None
            company_linkedin_url = row.get(company_col) if company_col else None
            
            # Get company name from configured column
            company_name_col = config.get('company_name_column')
            company_name = row.get(company_name_col) if company_name_col else None
            
            if pd.isna(person_name) or not person_name:
                person_name = f"Lead {index+1}"

            current_progress = int(((index + 1) / total_rows) * 100)
            processing_status[session_id]["progress"] = current_progress
            processing_status[session_id]["status"] = f"Processing {person_name} ({index+1}/{total_rows})..."
            processing_status[session_id]["log_message"] = f"Starting to process lead: {person_name} ({index+1}/{total_rows})"
            logger.info(f"Processing: {person_name} ({index+1}/{total_rows})")

            person_about = ""
            company_about = ""

            # --- LinkedIn Scraping Logic with Retry ---
            try:
                if driver:
                    # Scrape person profile
                    if person_linkedin_url and isinstance(person_linkedin_url, str) and person_linkedin_url.startswith('http') and pd.notna(person_linkedin_url):
                        def scrape_person():
                            person = Person(person_linkedin_url, driver=driver, scrape=False)
                            person.scrape(close_on_complete=False)
                            return person.about if person.about else ""
                        processing_status[session_id]["log_message"] = f"Fetching LinkedIn profile for {person_name}..."
                        person_about = scrape_with_retries(scrape_person, max_retries=2, delay=delay_seconds, label=f"person scrape for {person_name}")
                        if person_about:
                            processing_status[session_id]["log_message"] = f"Successfully retrieved profile data for {person_name}"
                            logger.info(f"Person about: {person_about[:100]}...")
                        else:
                            processing_status[session_id]["log_message"] = f"No profile data found for {person_name}"

                    # Scrape company profile using improved logic
                    processing_status[session_id]["log_message"] = f"Fetching company information for {person_name}..."
                    company_about = scrape_company_with_fallback(
                        company_linkedin_url, 
                        company_name, 
                        driver, 
                        delay_seconds, 
                        person_name
                    )
                    if company_about:
                        processing_status[session_id]["log_message"] = f"Successfully retrieved company data for {person_name}"
                    else:
                        processing_status[session_id]["log_message"] = f"No company data found for {person_name}"
                            
            except Exception as e:
                logger.error(f"Error during LinkedIn scraping for {person_name}: {e}")

            # --- AI Content Generation ---
            generated_messages = ["", "", ""]
            if person_about or company_about:
                combined_data = f"Individual Profile Summary:\n{person_about}\n\nCompany Profile Summary:\n{company_about}"
                
                # Use custom prompt if provided
                prompt_template = config.get('custom_prompt', DEFAULT_PROMPT_TEMPLATE)
                ai_prompt = prompt_template.format(
                    person_name=person_name,
                    combined_data=combined_data
                )
                
                # Generate content using the selected AI provider
                processing_status[session_id]["log_message"] = f"Generating AI content for {person_name} using {config.get('ai_provider', 'gemini')}..."
                generated_messages = generate_content_with_ai(ai_prompt, config)
                processing_status[session_id]["log_message"] = f"Successfully generated messages for {person_name}"
                logger.info(f"Generated messages for {person_name} using {config.get('ai_provider', 'gemini')}")

            # --- Update DataFrame ---
            df.at[index, 'Message 1'] = generated_messages[0]
            df.at[index, 'Message 2'] = generated_messages[1]
            df.at[index, 'Message 3'] = generated_messages[2]

            time.sleep(delay_seconds)

        if driver:
            driver.quit()
            logger.info("Browser closed.")

        df.to_csv(output_filepath, index=False)
        processing_status[session_id].update({"status": "Completed!", "progress": 100, "output_file": os.path.basename(output_filepath)})
        logger.info(f"Processing complete for session {session_id}. Data saved to {output_filepath}")

    except Exception as e:
        processing_status[session_id].update({"status": "Failed", "error": str(e), "progress": 0})
        logger.error(f"Critical error in background processing for session {session_id}: {e}")
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

@app.route('/analyze-csv', methods=['POST'])
@limiter.limit("10 per minute")
def analyze_csv():
    """Analyze CSV structure and suggest column mappings"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400
    
    if not file.filename.lower().endswith('.csv'):
        return jsonify({"error": "Please upload a CSV file"}), 400
    
    try:
        # Read first few rows to analyze structure
        df = pd.read_csv(file, nrows=10)
        columns = df.columns.tolist()
        suggestions = suggest_column_mapping(df)
        
        # Get sample data for preview and handle NaN values
        # First convert DataFrame to use Python native types
        sample_df = df.head(3).copy()
        
        # Create a JSON-safe dictionary
        sample_data = []
        for _, row in sample_df.iterrows():
            row_dict = {}
            for col in sample_df.columns:
                val = row[col]
                # Convert NaN/None to empty string
                if pd.isna(val):
                    row_dict[col] = ""
                # Convert other types to string if not basic JSON types
                elif not isinstance(val, (int, float, str, bool)) or (isinstance(val, float) and pd.isna(val)):
                    row_dict[col] = str(val)
                else:
                    row_dict[col] = val
            sample_data.append(row_dict)
        
        # Clean suggestions to handle any NaN values
        clean_suggestions = {}
        for key, value in suggestions.items():
            clean_suggestions[key] = value if pd.notna(value) else None
        
        # Test JSON serialization before returning
        try:
            json.dumps({
                "columns": columns,
                "suggestions": clean_suggestions,
                "sample_data": sample_data,
                "total_columns": len(columns),
                "sample_rows": len(sample_data)
            })
        except TypeError as e:
            logger.error(f"JSON serialization error: {e}")
            # If serialization fails, convert everything to strings
            for row in sample_data:
                for key in row:
                    row[key] = str(row[key])
        
        return jsonify({
            "columns": columns,
            "suggestions": clean_suggestions,
            "sample_data": sample_data,
            "total_columns": len(columns),
            "sample_rows": len(sample_data)
        })
    except Exception as e:
        logger.error(f"Error analyzing CSV: {e}")
        return jsonify({"error": f"Failed to analyze CSV: {str(e)}"}), 500

@app.route('/upload', methods=['POST'])
@limiter.limit("5 per minute")
def upload_file():
    """Handles CSV file upload and starts background processing."""
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if not file or not file.filename:
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({"error": "Please upload a CSV file"}), 400
        
        # Get configuration from form data
        config = {
            'name_column': request.form.get('name_column'),
            'profile_column': request.form.get('profile_column'),
            'company_column': request.form.get('company_column'),
            'company_name_column': request.form.get('company_name_column'),
            'custom_prompt': request.form.get('custom_prompt', '').strip(),
            'delay_seconds': max(int(request.form.get('delay_seconds', 8)), 8),
            'linkedin_email': request.form.get('linkedin_email', '').strip(),
            'linkedin_password': request.form.get('linkedin_password', '').strip(),
            'ai_provider': request.form.get('ai_provider', 'gemini'),
            'gemini_api_key': request.form.get('gemini_api_key', '').strip(),
            'openai_api_key': request.form.get('openai_api_key', '').strip(),
            'openai_model': request.form.get('openai_model', 'gpt-4o')
        }
        
        # Validate required fields
        if not config['name_column'] or not config['profile_column']:
            return jsonify({"error": "Name and Profile columns are required"}), 400
        
        # Secure filename
        filename = secure_filename(file.filename)
        session_id = str(uuid.uuid4())
        
        input_filepath = os.path.join(UPLOAD_FOLDER, f"{session_id}_{filename}")
        output_filename = f"enriched_{filename}"
        output_filepath = os.path.join(RESULTS_FOLDER, f"{session_id}_{output_filename}")
        
        file.save(input_filepath)
        
        # Start background processing
        thread = threading.Thread(
            target=process_csv_in_background, 
            args=(input_filepath, output_filepath, session_id, config)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "message": "File uploaded and processing started.",
            "session_id": session_id,
            "filename": filename
        }), 202
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/status/<session_id>')
def get_status(session_id):
    """Provides the current processing status for a given session ID."""
    if not re.match(r'^[a-f0-9\-]{36}$', session_id):
        return jsonify({"error": "Invalid session ID"}), 400
    
    status_info = processing_status.get(session_id, {
        "status": "Session not found", 
        "progress": 0, 
        "error": None, 
        "output_file": None
    })
    return jsonify(status_info)

@app.route('/download/<filename>')
def download_file(filename):
    """Allows users to download the processed CSV file."""
    try:
        # Security: validate filename - allow session_id_enriched_originalname.csv
        if not re.match(r'^[a-f0-9\-]{36}_enriched_.+\.csv$', filename):
            return jsonify({"error": "Invalid filename"}), 400
        
        filepath = os.path.join(RESULTS_FOLDER, filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404
        
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return jsonify({"error": f"Download failed: {str(e)}"}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "active_sessions": len(processing_status)
    })

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)