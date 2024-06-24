import pandas as pd
import string
import re
from collections import Counter
import nltk
import joblib
import io
import fitz
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from serpapi import GoogleSearch
from nltk.tokenize import word_tokenize
import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
from parsel import Selector
import mysql.connector
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
from mysql.connector import Error
from datetime import datetime

# Initialization
if 'login_status' not in st.session_state:
    st.session_state.login_status = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "role_selection"
if 'user_type' not in st.session_state:
    st.session_state.user_type = ""
if 'predicted_job_category' not in st.session_state:
    st.session_state.predicted_job_category = ""
if 'job_cat' not in st.session_state:
    st.session_state.job_cat = ""
if 'selected_job' not in st.session_state:
    st.session_state.selected_job = ""
if 'similarity_score' not in st.session_state:
    st.session_state.similarity_score = 0
if 'page' not in st.session_state:
    st.session_state.page = ""
    
# st.set_page_config(layout="wide")
def create_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="kevwe",
            password="umukoro3056#",
            database="resume_job_files"
        )
        if connection.is_connected():
            return connection
    except Error as e:
        st.error(f"Error: '{e}'")
        return None

def close_connection(connection):
    if connection.is_connected():
        connection.close()
   
def save_user_to_db(user_type, name, email, number, password):
    table = "candidate" if user_type == "Candidate" else "employer"
    query = f"INSERT INTO {table} (name, email, number, password) VALUES (%s, %s, %s, %s)"
    connection = create_connection()
    if connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (name, email, number, password))
            connection.commit()
        close_connection(connection)
        
def save_resume(candidate_id, resume_file, resume_text):
    connection = create_connection()
    if connection:
        with connection.cursor() as cursor:
            resume_binary = resume_file.getvalue()
            cursor.execute("""
                UPDATE candidate
                SET resume_file = %s, resume_text = %s
                WHERE id = %s
            """, (resume_binary, resume_text, candidate_id))
            connection.commit()
        connection.close()

def is_email_registered(user_type, email):
    table = "candidate" if user_type == "Candidate" else "employer"
    query = f"SELECT * FROM {table} WHERE email = %s"
    connection = create_connection()
    if connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (email,))
            user = cursor.fetchone()
        close_connection(connection)
        return user is not None
    return False

def add_job(title, description, company, location, category, link, employer_id):
    connection = create_connection()
    if connection:
        with connection.cursor() as cursor:
            cursor.execute("INSERT INTO job (title, description, company, location, category, link, employer_id) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                           (title, description, company, location, category, link, employer_id))
            connection.commit()
        close_connection(connection)

def fetch_jobs():
    connection = create_connection()
    if connection:
        with connection.cursor(dictionary=True) as cursor:
            cursor.execute("SELECT * FROM job")
            jobs = cursor.fetchall()
        close_connection(connection)
        return jobs

def fetch_jobs_by_employer(employer_id):
    connection = create_connection()
    if connection:
        with connection.cursor(dictionary=True) as cursor:
            cursor.execute("SELECT * FROM job WHERE employer_id = %s", (employer_id,))
            jobs = cursor.fetchall()
        close_connection(connection)
        return jobs

def fetch_applications_for_job(job_id):
    connection = create_connection()
    applications = []
    if connection:
        with connection.cursor(dictionary=True) as cursor:
            cursor.execute("""
                SELECT a.candidate_id, c.email, c.id, a.similarity_score, a.application_date, c.resume_file
                FROM application a
                JOIN candidate c ON a.candidate_id = c.id
                WHERE a.job_id = %s
            """, (job_id,))
            applications = cursor.fetchall()
        connection.close()
    return applications

def db_apply(candidate_id, job_id, similarity_score):
    connection = create_connection()
    if connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO application (candidate_id, job_id, application_date, similarity_score)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE similarity_score = VALUES(similarity_score), application_date = VALUES(application_date)
            """, (candidate_id, job_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), similarity_score))
            connection.commit()
        connection.close()

def display_applications(job_id):
    applications = fetch_applications_for_job(job_id)
    table_data = []
    for application in applications:
        table_data.append({
            "Candidate ID": application['id'],
            "Email": f"[{application['email']}](mailto:{application['email']})",
            "Similarity Score": f"{application['similarity_score']:.2f}",
            "Application Date": application['application_date']
        })
    
    # Convert the table_data into a DataFrame
    df = pd.DataFrame(table_data)
    # Use the 'unsafe_allow_html' parameter to allow rendering of HTML links
    st.markdown(df.to_markdown(), unsafe_allow_html=True)

def verify_login(user_type, email, password):
    table = "candidate" if user_type == "Candidate" else "employer"
    query = f"SELECT * FROM {table} WHERE email = %s AND password = %s"
    connection = create_connection()
    user = None
    if connection:
        with connection.cursor(dictionary=True) as cursor:
            cursor.execute(query, (email, password))
            user = cursor.fetchone()
        close_connection(connection)
    return user

def role_selection_page():
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://img.freepik.com/premium-photo/resumes-applicants-magnifying-glass-beige-background-top-view-with-copy-space-job-search-concept_35674-14216.jpg");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    st.title("Welcome To JobSync")
    st.subheader("Your One-Stop Platform for Job Matching and Skill Gap Analysis")
    st.write("**Select your role:**")
    st.session_state.user_type = st.selectbox("", ["Candidate", "Employer"])
    if st.button("Proceed"):
        st.session_state.current_page = "login"
        
    st.write("### For Candidates:")
    st.write("""
        - **Find Jobs:** Discover opportunities matching your skills.
        - **Skill Gap Analysis:** Identify and bridge skill gaps.
        - **Resume Upload:** Manage your resumes for better matching.
    """)
    
    st.write("### For Employers:")
    st.write("""
        - **Post Jobs:** Create and manage job postings.
        - **Review Applications:** Access and review candidate applications.
        - **Candidate Matching:** Find the best candidates based on skills and experience.
    """)
    
def is_valid_email(email):
    regex = r'^\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.match(regex, email)

def is_valid_phone_number(phone_number):
    return phone_number.isdigit()

def sign_up_page():
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{   
    background-image: url("https://static.vecteezy.com/system/resources/previews/017/396/233/non_2x/fashion-style-template-with-abstract-shapes-in-pastel-and-plant-colors-neutral-background-with-minimalistic-theme-vector.jpg");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    </style>
    """
    st.title("Sign Up")
    name = st.text_input("Name")
    email = st.text_input("Email")
    number = st.text_input("Phone Number")
    password = st.text_input("Create a password", type="password")
    st.markdown(page_bg_img, unsafe_allow_html=True)

    if st.button("Complete Sign Up"):
        if not is_valid_email(email):
            st.warning("Please enter a valid email address.")
        elif not is_valid_phone_number(number):
            st.warning("Please enter a valid phone number (digits only).")
        elif is_email_registered(st.session_state.user_type, email):
            st.warning("This email is already registered. Please go back and log in.")
        else:
            save_user_to_db(st.session_state.user_type, name, email, number, password)
            st.success("Signed up successfully! Please log in.")
            st.session_state.current_page = "login"

def login_page():
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSMAAjNlH7VmgF5Zd4ypShnXmWwZn7SnZ82Cw&s");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    </style>
    """
    st.title("Login 🔒")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("**Login**"):
        user = verify_login(st.session_state.user_type, email, password)
        if user:
            st.success("Logged in successfully!")
            st.session_state.login_status = True
            st.session_state.user_id = user['id']  # Store the candidate ID in the session state
            st.session_state.user_email = user['email']
            st.session_state.user_name = user['name']
            st.session_state.current_page = "file_submission" if st.session_state.user_type == "Candidate" else "employer_dashboard"
        else:
            st.error("Invalid email or password")
    
    st.markdown(page_bg_img, unsafe_allow_html=True) 
    st.write("Not a user? Sign up!")
    if st.button("Sign Up"):
        st.session_state.current_page = "sign_up"

# File submission page
def file_submission_page():
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{   
    background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSMAAjNlH7VmgF5Zd4ypShnXmWwZn7SnZ82Cw&s");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.title("📁 Resume Upload")
    st.write("Please submit your resume in PDF format")
    st.write("Choose to either know your job category or search for a job straight away")
    uploaded_file = st.file_uploader("Upload Resume")

    if uploaded_file is None:
        st.warning("Please upload a file to begin")

    else:
        st.write("File submitted successfully!")
        st.session_state.filep = io.BytesIO(uploaded_file.read())
        st.session_state.filepdf = fitz.open(stream=st.session_state.filep, filetype="pdf")
        with st.sidebar:
            st.header("Candidate Actions")
            if st.button("Know Your Job Category"):
                st.session_state.current_page = "job_category"

            elif st.button("Search for a Job"):
                st.session_state.current_page = "job_search"
    
    if st.session_state.current_page != "login":
        if st.button("**Go Back**"):
            st.session_state.current_page = "file_submission"

# Job category page
def know_your_job_category_page():
    st.title("Job Category ✅")
    st.write("Here you can find your suitable job category.")
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{   
    background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSMAAjNlH7VmgF5Zd4ypShnXmWwZn7SnZ82Cw&s");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    cleaned_text = preprocess_pdf(st.session_state.filepdf)
    save_resume(st.session_state.user_id,st.session_state.filep,cleaned_text)
    # Loading necessary files
    joblib_file = "job_category_model.pkl"
    model = joblib.load(joblib_file)
    joblib_file_vectorizer = "job_tfidf_vectorizer.pkl"
    tfidf_vectorizer = joblib.load(joblib_file_vectorizer)
    joblib_file_le = "job_label_encoder.pkl"
    le = joblib.load(joblib_file_le)
    pdf_vec = tfidf_vectorizer.transform([cleaned_text])
    pdf_pred = model.predict(pdf_vec)
    pdf_pred = le.inverse_transform(pdf_pred)
    predicted_job_category = pdf_pred[0].upper()
    st.session_state.predicted_job_category = predicted_job_category
    st.write(f"Your Predicted Job Category: {predicted_job_category}")
    st.write(f"Hurray! Let's Proceed")
    if st.button("Proceed to Job Search"):
        st.session_state.current_page = "job_search"
    if st.session_state.current_page != "login":
        if st.button("**Go Back**"):
            st.session_state.current_page = "file_submission"

# Preprocess PDF
def preprocess_pdf(pdf_doc):
    text = extract_text_from_pdf(pdf_doc)
    cleaned_text = preprocess_text(text)
    return cleaned_text

# Extract text from PDF
def extract_text_from_pdf(pdf_doc):
    text = ""
    for page_num in range(len(pdf_doc)):
        page = pdf_doc.load_page(page_num)
        text += page.get_text()
    return text

# Preprocess text
def preprocess_text(text):
    
    # Lower casing
    text = text.lower()
    # Punctuation removal
    text = "".join([i for i in text if i not in string.punctuation])
    # Stopwords removal
    STOPWORDS = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    # URL removal
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    # Stemming
    stemmer = PorterStemmer()
    words = " ".join([stemmer.stem(word) for word in text.split()])
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
    pos_tagged_text = nltk.pos_tag(words.split())
    words = " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])
                    
    return words

def parse_job_page(job_url):
    response = requests.get(job_url)
    if response.status_code == 200:
        selector = Selector(response.text)
        script_data = json.loads(selector.xpath("//script[@type='application/ld+json']/text()").get())
        description = []
        for element in selector.xpath("//div[contains(@class, 'show-more')]/ul/li/text()").getall():
            text = element.replace("\n", "").strip()
            if len(text) != 0:
                description.append(text)
        script_data["jobDescription"] = description
        script_data.pop("description") # remove the key with the encoded HTML
        return script_data
    else:
        return None

def linkedin_job_search(job_title, location):
    head = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Connection": "keep-alive",
        "Accept-Language": "en-US,en;q=0.9,lt;q=0.8,et;q=0.7,de;q=0.6",
    }

    url = f"https://www.linkedin.com/jobs/search/?keywords={job_title.replace(' ', '%20')}&location={location.replace(' ', '%20')}&origin=JOB_SEARCH_PAGE_SEARCH_BUTTON&refresh=true"
    response = requests.get(url, headers=head)
    job_lists = []

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html5lib')
        job_listings = soup.find_all('div', {'class':'job-search-card'})
        for job in job_listings:
            title = job.find('h3', {'class': 'base-search-card__title'}).text.strip()
            company = job.find('a', {'class': 'hidden-nested-link'}).text.strip()
            location = job.find('span', {'class': 'job-search-card__location'}).text.strip()
            anchor_tag = job.find('a', class_='base-card__full-link')
            href_link = anchor_tag['href']
            job_data = parse_job_page(href_link)
            if job_data:
                job_description = job_data.get("jobDescription", [])
                job_listing = {
                "title": title,
                "company": company,
                "location": location,
                "link": href_link,
                "description": job_description
                }
                job_lists.append(job_listing)
            else:
                job_listing = {
                "title": title,
                "company": company,
                "location": location,
                "link": href_link,
                "description": "Not Available"
                }
                job_lists.append(job_listing)
                
    return job_lists

def indeed_job_search(job_title, location):
    head= {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Connection": "keep-alive",
    "Accept-Language": "en-US,en;q=0.9,lt;q=0.8,et;q=0.7,de;q=0.6"}

    url = f"https://ng.indeed.com/jobs?q={job_title.replace(' ', '+')}&l={location.replace(' ', '+')}"
    response = requests.get(url, headers=head)

    job_listings_dict = []  # List to store job listings in string form

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html5lib')
        joblistings = soup.find_all("div",{"class":"cardOutline"})
        for job in joblistings:
            title = job.find("a",{"class":"jcs-JobTitle css-jspxzf eu4oa1w0"}).text.strip()
            company = job.find("div",{"class":"company_location"}).find("span",{"class":"css-63koeb eu4oa1w0"}).text.strip()
            location = job.find("div",{"class":"company_location"}).find("div",{"class":"css-1p0sjhy eu4oa1w0"}).text.strip()
            link = job.find('a',{'class':'jcs-JobTitle css-jspxzf eu4oa1w0'}).get('href')
            href_link = 'https://ng.indeed.com' + link
            job_response = requests.get(href_link, headers=head)
            job_soup = BeautifulSoup(job_response.text, 'html5lib')
            job_description_tag = job_soup.find('div',{'class':'jobsearch-JobComponent-description css-16y4thd eu4oa1w0'}) 
            job_desc = job_description_tag.text.strip() if job_description_tag else "N/A"
            job_listing_ = {"title": title, "company": company, "location": location, "link": href_link, "description": job_desc}
            job_listings_dict.append(job_listing_)

    return job_listings_dict

def job_match_calc(resume_text, job_listings):
    preprocessed_resume_text = preprocess_text(resume_text)
    preprocessed_job_listings = []

    for job_listing in job_listings:
        job_description = job_listing["description"]
        if isinstance(job_description, list):  # Convert list of job descriptions to string
            job_description = " ".join(job_description)
        preprocessed_job_description = preprocess_text(job_description)
        preprocessed_job_listings.append(preprocessed_job_description)

    # Add preprocessed resume text to the list
    preprocessed_job_listings.append(preprocessed_resume_text)

    # Create tagged data for Doc2Vec
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(preprocessed_job_listings)]

    # Train Doc2Vec model
    model = Doc2Vec(vector_size=30, min_count=2, epochs=80)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    # Infer vectors for resume and job listings
    resume_vec = model.infer_vector(word_tokenize(preprocessed_resume_text.lower()))
    job_vecs = [model.infer_vector(word_tokenize(job.lower())) for job in preprocessed_job_listings[:-1]]  # Exclude the resume text

    # Calculate cosine similarity between resume and job listings
    similarities = [cosine_similarity([resume_vec], [job_vec])[0][0] for job_vec in job_vecs]

    # Rank job listings based on similarity
    ranked_job_listings = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    return ranked_job_listings, job_listings

def get_youtube_videos(search_query):
    params = {
      "engine": "youtube",
      "search_query": search_query,
      "api_key": "ac8720049b1feff6d3e036e0c162f37785027a22e68709ffb084d44924bf9cad"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    videos = []
    for result in results['video_results']:
        title = result['title']
        link = result['link']
        videos.append((title, link))
        
    return videos[:5]

def get_coursera_courses(search_query):
    coursera_search_url = f"https://www.coursera.org/search?query={search_query.replace(' ', '+')}"
    response = requests.get(coursera_search_url)
    soup = BeautifulSoup(response.text, 'html5lib') 
    courses = []
    for course in soup.find_all('div', {"class":"cds-ProductCard-header"}):
        url = course.find("a" ,{"class":"cds-119 cds-113 cds-115 cds-CommonCard-titleLink css-si869u cds-142"}).get("href")
        course_url = f"https://www.coursera.org{url}"
        course_host = course.find("p", {"class":"cds-ProductCard-partnerNames css-vac8rf"}).text.strip()
        course_title = course.find('h3', {'class':'cds-CommonCard-title css-6ecy9b'}).text.strip()
        courses.append((course_title, course_url, course_host))
    
    return courses[:5] 

def job_apply_page():
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{   
    background-image: url("https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvcm0yNDYta2F0aWUtMDQtZy5qcGc.jpg");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    selected_job = st.session_state.get("selected_job", {})
    similarity_score = st.session_state.get("similarity_score", 0) * 100
    job_title = selected_job.get("title", "")
    job_company = selected_job.get("company", "")
    job_location = selected_job.get("location", "")
    job_link = selected_job.get("link", "")
    search_query = st.session_state.job_cat
    
    st.title(job_title)
    st.write(f"Company: {job_company}")
    st.write(f"Location: {job_location}")
    st.write(f"**Link**: {job_link}")
    st.write(f"Similarity Score: {similarity_score}%")
    
    if similarity_score > 70:
        st.write("Based on your similarity score, you have a high chance of getting this job!")
    elif 50 < similarity_score <= 70:
        st.write("Based on your similarity score, you have a moderate chance of getting this job.")
    else:
        st.write("Based on your similarity score, you have a low chance of getting this job. Consider improving your skills or applying for other positions.")

    st.write("Here are some recommendations to help you improve your skills and increase your chances:")

    st.subheader("Recommended Courses on Coursera:")
    courses = get_coursera_courses(search_query+" Courses")
    for title, url, host in courses:
        st.write(f"**Title**: [{title}]({url})")
        st.write(f"**Host**: {host}") 
        st.write(f"-----------------") 

    st.subheader("Recommended YouTube Videos:")
    videos = get_youtube_videos(search_query+" Courses")
    for title, link in videos:
        st.write(f"**Title**: [{title}]({link})")
        
    if st.radio("Would you like interview preparation materials?", options=["Yes", "No"]) == "Yes":
        st.write(f"Here are some interview preparation videos on {search_query} jobs:")
        intvideos = get_youtube_videos(search_query+" interview prep")
        for title, link in intvideos:
            st.write(f"**Title**: [{title}]({link})")
    
    st.write("**If you wish to apply go ahead and click the job link at the top, then click the button below**")
    if st.button("Done Applying"):
        candidate_id = st.session_state.get("user_id")  # Assuming candidate_id is stored in session state
        job_id = selected_job.get("id","")  
        if job_id:
            db_apply(candidate_id, job_id, similarity_score)
            st.success("Application submitted successfully!")
        else:
            st.success("Application submitted successfully and good luck!")
                
    col1, col2 = st.columns(2)
    with col2:
        if st.button("**Logout**"):
                st.session_state.login_status = False
                st.session_state.current_page = "role_selection"
    with col1:
        if st.session_state.current_page != "login":
            if st.button("**Go Back**"):
                st.session_state.current_page = "job_search"

def apply_for_job(job,similarity):
    st.session_state.selected_job = job
    st.session_state.similarity_score = similarity
    st.session_state.current_page = "job_apply"
    
def job_search_page():
    st.title("Job Search")
    st.write("Please enter your job category and select a location to search for jobs.")
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{   
    background-image: url("https://static.vecteezy.com/system/resources/previews/017/396/233/non_2x/fashion-style-template-with-abstract-shapes-in-pastel-and-plant-colors-neutral-background-with-minimalistic-theme-vector.jpg");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    job_category_key = "job_category_input"

    # Retrieve the job category from the session state or default to an empty string
    job_category = st.session_state.get(job_category_key, st.session_state.get("predicted_job_category", ""))

    job_category_input = st.text_input("**Job Category**", job_category)

    new_job_category = job_category_input
    with st.expander("Suggestions:"):
        if st.button("Intern"):
            new_job_category += " intern"
        if st.button("Manager"):
            new_job_category += " manager"
        if st.button("Senior"):
            new_job_category = "senior " + new_job_category
        if st.button("Junior"):
            new_job_category = "junior " + new_job_category
        if st.button("Entry Level"):
            new_job_category = "entry level " + new_job_category

    # Update the job category in the session state only if it has changed
    if new_job_category != job_category:
        st.session_state[job_category_key] = new_job_category

    job_category = new_job_category.upper()
    st.session_state.job_cat = job_category
    st.text_input("**Updated Job Category**", job_category)

    predefined_locations = ["Abia", "Aba", "Abuja", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", "Borno", "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu",  "Gombe", "Imo", "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun", "Ondo", "Osun", "Oyo", "Plateau", "Port Harcourt", "Rivers", "Owerri", "Sokoto", "Taraba", "Yobe", "Zamfara", "Nigeria", "Other"]
    selected_state = st.selectbox("**Select location**", predefined_locations)

    if selected_state == "Other":
        customized_location = st.text_input("**Enter your preferred location**", key="preferred_location")
    else:
        customized_location = selected_state
        
    
    sorting_order = st.radio("**Filter**:", ["Best match to Least match", "Least match to Best match"])
    st.write("**Note: Similarity score indicates how closely your resume matches the job requirements, with 1 being a perfect match.**")
    
    if st.button("**SEARCH**"):
        st.write("Searching on platform:")
        # Fetch jobs from the database
        jobs = fetch_jobs()

        # Filter jobs based on job category and location
        matched_jobs = [job for job in jobs if job_category_input.lower() in job['category'].lower() and (customized_location.lower() in job['location'].lower())]

        # Preprocess the resume
        resume_text = preprocess_pdf(st.session_state.filepdf)

        # Calculate job match scores
        ranked_job_listings, job_listings_strings = job_match_calc(resume_text, matched_jobs)

        # Sort job listings based on the selected sorting order
        if sorting_order == "Best match to Least match":
            ranked_job_listings.sort(key=lambda x: x[1], reverse=True)
        else:
            ranked_job_listings.sort(key=lambda x: x[1])

        if len(ranked_job_listings) == 0:
            st.write(f"{'*'*10} Sorry no results found on platform {'*'*10}")
        else:
            st.write(f"Displaying {len(ranked_job_listings)} job listings:")
            with st.expander("Job Listings"):
                for idx, (index, similarity_score) in enumerate(ranked_job_listings):
                    job_listing = job_listings_strings[index]
                    st.write(f"Rank: {idx+1}")
                    st.markdown(f"**Title:** [{job_listing['title']}]({job_listing['link']})")
                    st.markdown(f"**Company:** {job_listing['company']}")
                    st.markdown(f"**Location:** {job_listing['location']}")
                    st.markdown(f"**Similarity Score:** {similarity_score:.2f}")
                    st.button("Apply for job", key=f"apply_{idx}", on_click=apply_for_job, args=(job_listing, similarity_score))
                    st.write("---")

    
    
    # Check if both job category and location are provided
    if job_category and customized_location:
        st.write("Select another platform to search for jobs:")
        st.write("For international locations, Indeed is your best option.")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("**LINKEDIN**"):
                st.write(f"Searching for {job_category} jobs in {customized_location.upper()} on LinkedIn...")
                linkedin_job_listings = linkedin_job_search(job_category, customized_location.upper())
                resume_text = preprocess_pdf(st.session_state.filepdf)
                ranked_job_listings, job_listings_strings = job_match_calc(resume_text, linkedin_job_listings)
                # Sort job listings based on selected sorting order
                if sorting_order == "Best match to Least match":
                    ranked_job_listings.sort(key=lambda x: x[1], reverse=True)
                else:
                    ranked_job_listings.sort(key=lambda x: x[1])
                if len(ranked_job_listings) == 0:
                    st.write(f"{'*'*10} Sorry no results found on platform {'*'*10}")
                else:
                    st.write(f"Displaying {len(ranked_job_listings)} job listings:")
                    with st.expander("Job Listings"):
                        for idx, (index, similarity_score) in enumerate(ranked_job_listings):
                            job_listing = job_listings_strings[index]
                            st.write(f"Rank: {idx+1}")
                            st.markdown(f"**Title:** [{job_listing['title']}]({job_listing['link']})")
                            st.markdown(f"**Company:** {job_listing['company']}")
                            st.markdown(f"**Location:** {job_listing['location']}")
                            st.markdown(f"**Similarity Score:** {similarity_score:.2f}")
                            st.button("Apply for job", key=f"apply_{idx}", on_click=apply_for_job, args=(job_listing,similarity_score))
                            st.write("---")

        with col2:
            if st.button("**INDEED**"):
                st.write(f"Searching for {job_category} jobs in {customized_location.upper()} on Indeed...")
                indeed_job_listings = indeed_job_search(job_category, customized_location.upper())
                resume_text = preprocess_pdf(st.session_state.filepdf)
                ranked_job_listings, job_listings_strings = job_match_calc(resume_text, indeed_job_listings)
                # Sort job listings based on selected sorting order
                if sorting_order == "Best match to Least match":
                    ranked_job_listings.sort(key=lambda x: x[1], reverse=True)
                else:
                    ranked_job_listings.sort(key=lambda x: x[1])
                if len(ranked_job_listings) == 0:
                    st.write(f"{'*'*10} Sorry no results found on platform {'*'*10}")
                else:
                    st.write(f"Displaying {len(ranked_job_listings)} job listings:")
                    with st.expander("Job Listings"):
                        for idx, (index, similarity_score) in enumerate(ranked_job_listings):
                            job_listing = job_listings_strings[index]
                            st.write(f"Rank: {idx+1}")
                            st.markdown(f"**Title:** [{job_listing['title']}]({job_listing['link']})")
                            st.markdown(f"**Company:** {job_listing['company']}")
                            st.markdown(f"**Location:** {job_listing['location']}")
                            st.markdown(f"**Similarity Score:** {similarity_score:.2f}")
                            st.button("Apply for job", key=f"apply_{idx}", on_click=apply_for_job, args=(job_listing,similarity_score))
                            st.write("---")
                    

    if st.session_state.current_page != "login":
        if st.button("**Go Back**"):
            st.session_state.current_page = "file_submission"
          
def employer_dashboard_page():
    st.title("Employer Dashboard")
    st.write("Welcome to the employer dashboard. Here you can manage your job postings, review applications, and track your hiring progress. Use the sidebar to navigate between different sections.")
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{   
    background-image: url("https://i.pinimg.com/736x/5e/25/01/5e25011fdd8ee087e21d733b9df66af8.jpg");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.subheader("Tips & Recommendations")
    st.write("**Creating Effective Job Postings:** Provide clear and concise job descriptions, highlight key responsibilities, and specify required qualifications.")
    st.write("**Optimizing the Hiring Process:** Use structured interviews, standardize evaluation criteria, and provide timely feedback to candidates.")
    st.write("**Selecting the Best Candidates:** Focus on both technical skills and cultural fit, and consider the potential for growth and development.")
    with st.sidebar:
        st.header("Employer Actions")
        if st.button("Add Job"):
            st.session_state.page = "add_job"
        if st.button("View Applications"):
            st.session_state.page = "view_applications"
        if st.button("**Logout**"):
            st.session_state.login_status = False
            st.session_state.current_page = "role_selection"
            
    if st.session_state.page == "add_job":
        add_job_page()
    elif st.session_state.page == "view_applications":
        view_applications_page()
     
def add_job_page():
    st.header("Add a Job")
    st.write("Looking for new employees?")
    st.write("Fill in the job description below:")
    job_title = st.text_input("Job Title", "e.g., Software Engineer")
    job_company = st.text_input("Company", "e.g., ABC Corp")
    job_location = st.text_input("Location", "e.g., Lagos, Nigeria")
    job_category = st.text_input("Category", "e.g., Software Engineering, Accountant")
    job_link = st.text_input("Job Link", "e.g., https://abc.com/jobs/123")
    job_description = st.text_area("Job Description", "Enter the job description here...")
    if st.button("Submit"):
        add_job(job_title, job_description, job_company, job_location, job_category, job_link, st.session_state.user_id)
        st.success("Job added successfully!")
  
def view_applications_page():
    st.header("View Applications")
    employer_id = st.session_state.user_id
    jobs = fetch_jobs_by_employer(employer_id)
    if not jobs:
        st.write("No jobs found for this employer.")
    else:
        job_id = st.selectbox("Select Job", [job['id'] for job in jobs])
        st.write(f"Applications for Job ID {job_id}")
        display_applications(job_id)

# Page navigation
if st.session_state.current_page == "role_selection":
    role_selection_page()
elif st.session_state.current_page == "sign_up":
    sign_up_page()
elif st.session_state.current_page == "login":
    login_page()
elif st.session_state.current_page == "file_submission":
    file_submission_page()
elif st.session_state.current_page == "job_category":
    know_your_job_category_page()
elif st.session_state.current_page == "job_search":
    job_search_page()
elif st.session_state.current_page == "employer_dashboard":
    employer_dashboard_page()
elif st.session_state.current_page == "job_apply":
    job_apply_page()