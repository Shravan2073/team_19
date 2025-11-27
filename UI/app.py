import streamlit as st
import numpy as np
import pandas as pd
import time
import requests
import tldextract
import socket
import validators
import whois
import datetime
import re
from urllib.parse import urlparse

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle

# Page configuration with enhanced styling
st.set_page_config(
    page_title="🔒 Phishing URL Detector",
    page_icon="🔒",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Override Streamlit's default white backgrounds */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Prevent white boxes in all containers */
    .stContainer, .stColumn, .stDataFrame, .stMetric {
        background: transparent !important;
    }
    
    /* Header styling */
    .stApp header {
        background: transparent !important;
    }
    
    /* Title styling */
    h1 {
        color: white !important;
        font-size: 3rem !important;
        font-weight: 700 !important;
        text-align: center !important;
        margin-bottom: 1rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.2rem !important;
        text-align: center !important;
        margin-bottom: 3rem !important;
        font-weight: 300 !important;
    }
    
    /* Card styling - REMOVED to prevent white box issues */
    /* .stCard {
        background: white !important;
        border-radius: 15px !important;
        padding: 2rem !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2) !important;
        margin-bottom: 2rem !important;
        border: none !important;
    } */
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background: white !important;
        border: 2px solid #e1e5e9 !important;
        border-radius: 10px !important;
        padding: 0.8rem 1rem !important;
        font-size: 1.1rem !important;
        color: #2c3e50 !important;
        transition: all 0.3s ease !important;
        caret-color: black !important; /* Black blinking cursor */
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        outline: none !important;
        caret-color: black !important; /* Ensure black cursor on focus */
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.8rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background: white !important;
        border: 2px solid #e1e5e9 !important;
        border-radius: 10px !important;
    }
    
    /* Result card styling - SIMPLIFIED to prevent white box */
    .result-card {
        border-radius: 10px !important;
        margin: 1rem 0 !important;
    }
    
    .legitimate-result {
        border-left: 5px solid #27ae60 !important;
    }
    
    .phishing-result {
        border-left: 5px solid #e74c3c !important;
    }
    
    .warning-result {
        border-left: 5px solid #f39c12 !important;
    }
    
    /* Feature table styling */
    .dataframe {
        background: white !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        text-align: left !important;
        padding: 1rem !important;
        border: none !important;
    }
    
    .dataframe td {
        padding: 0.8rem 1rem !important;
        border-bottom: 1px solid #f1f2f6 !important;
        color: #2c3e50 !important;
    }
    
    .dataframe tr:nth-child(even) {
        background: #f8f9fa !important;
    }
    
    /* Loading spinner styling */
    .stSpinner > div {
        text-align: center !important;
        color: white !important;
        font-size: 1.1rem !important;
    }
    
    /* Success/Error message styling */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 10px !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
        font-weight: 500 !important;
    }
    
    /* Footer styling */
    .footer {
        text-align: center !important;
        color: rgba(255,255,255,0.8) !important;
        margin-top: 3rem !important;
        font-size: 0.9rem !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main {
            padding: 1rem !important;
        }
        
        h1 {
            font-size: 2rem !important;
        }
        
        .subtitle {
            font-size: 1rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models_and_preprocessors():
    base = r"C:\Users\shrav\Downloads\Deep-Learning-Phishing-Website-Detection-main\Deep-Learning-Phishing-Website-Detection-main"
    # Load tokenizer
    with open(base + r"\models\preprocessing\tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Load scaler
    with open(base + r"\models\preprocessing\std_sc.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load imputer
    with open(base + r"\models\preprocessing\lr_imputer.pkl", "rb") as f:
        imputer = pickle.load(f)

    # Load CNN model
    cnn = None
    try:
        cnn = tf.keras.models.load_model(base + r"\models\deep_learining\CNN.keras")
    except Exception:
        cnn = None

    # Load DNN model if available
    dnn = None
    try:
        dnn = tf.keras.models.load_model(base + r"\models\deep_learining\DNN.tf")
    except Exception:
        dnn = None

    return tokenizer, scaler, imputer, cnn, dnn


def get_domain(url):
    try:
        if not isinstance(url, str):
            url = str(url)
        url_for_parse = url if "://" in url else "http://" + url
        parsed = urlparse(url_for_parse)
        hostname = parsed.hostname or parsed.path
        if not hostname:
            return "", ""
        ext = tldextract.extract(hostname)
        sub = ext.subdomain or ""
        dom = ext.domain or ""
        suf = ext.suffix or ""
        if dom == "" and suf == "":
            return hostname, hostname
        domain_str = f"{dom}.{suf}" if suf else dom
        subdomain_str = f"{sub}.{domain_str}" if sub else domain_str
        return subdomain_str, domain_str
    except Exception:
        return "", ""


def validate_url(url):
    try:
        return validators.url(url) == True
    except Exception:
        return False


def get_request(url):
    """
    Use HTTPS as the default protocol if user didn't provide one.
    Keep this function responsible only for making the request — the explicit-HTTP
    check is performed on the raw user input in main().
    """
    try:
        validation = validate_url(url)
    except Exception:
        validation = False

    # Default to HTTPS if protocol not present or input not a valid full URL
    if not validation:
        url = "https://" + url

    try:
        response = requests.get(url, timeout=20)
    except Exception:
        response = None
    return response


def get_soup(response):
    from bs4 import BeautifulSoup
    if response is None:
        return None
    return BeautifulSoup(response.text, "html.parser")


def get_login_time(url):
    start_time = time.time()
    response = get_request(url)
    end_time = time.time()
    load_time_in_seconds = end_time - start_time
    return load_time_in_seconds, response


def get_external_link(soup, url):
    if soup is None:
        return 0
    external_links = 0
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and url not in href:
            external_links += 1
    return external_links


def get_redirects(response):
    if response is None:
        return 0
    try:
        num_redirects = len(response.history)
    except Exception:
        num_redirects = 0
    return num_redirects


def get_num_image(soup):
    if soup is None:
        return 0
    return len(soup.find_all("img"))


def get_ip_address_reputation(response):
    try:
        parsed = urlparse(response.url if isinstance(response.url, str) else str(response.url))
        hostname = parsed.hostname or parsed.path
        ip_str = socket.gethostbyname(hostname)
        response_ip = requests.get(f"http://checkip.dyndns.org/?ip={ip_str}", timeout=10)
        reputation = "safe" if "OK" in response_ip.text else "malicious"
    except Exception:
        reputation = None
    return reputation


def get_num_iframes(soup):
    if soup is None:
        return 0
    return len(soup.find_all("iframe"))


def get_num_hidden_text(soup):
    if soup is None:
        return 0
    num_hidden_text = 0
    try:
        for element in soup.find_all():
            style = element.get("style")
            if style and "display:none" in style.lower():
                num_hidden_text += 1
    except Exception:
        num_hidden_text = 0
    return num_hidden_text


def get_alexa_rank(soup):
    try:
        alexa_rank = None
        for meta in soup.find_all("meta"):
            if "name" in meta.attrs and meta.attrs["name"].lower() == "alexa":
                alexa_rank = int(meta.attrs["content"])
    except Exception:
        alexa_rank = None
    return alexa_rank


def get_page_rank(url):
    try:
        GOOGLE_PR_CHECK_URL = "http://toolbarqueries.google.com/tbr?client=navclient-auto&features=Rank&ch=%s&q=info:%s"
        domain = url.split("//")[-1].split("/")[0]
        hsh = hash(domain.encode("utf-8")) & 0xEFFFFFFF
        response = requests.get(GOOGLE_PR_CHECK_URL % (hsh, domain))
        if response.status_code == 200:
            page_rank = int(response.content.strip().split(":")[-1])
        else:
            page_rank = None
    except Exception:
        page_rank = None
    return page_rank


def get_ext_tot_ratio(soup, url):
    if soup is None:
        return 0
    try:
        num_internal_links = 0
        num_external_links = 0
        for link in soup.find_all("a"):
            href = link.get("href")
            if href:
                if url in href:
                    num_internal_links += 1
                else:
                    num_external_links += 1
        if num_internal_links > 0:
            external_to_internal_ratio = num_external_links / num_internal_links
        else:
            external_to_internal_ratio = num_external_links
    except Exception:
        external_to_internal_ratio = 0
    return external_to_internal_ratio


def get_response_features(url):
    time_and_response = get_login_time(url)
    response = time_and_response[1]
    if response is None:
        return [0] * 7
    soup = get_soup(response)
    time_load = time_and_response[0]
    num_ex_links = get_external_link(soup, url)
    num_redirects = get_redirects(response)
    num_img = get_num_image(soup)
    num_iframe = get_num_iframes(soup)
    num_hidden = get_num_hidden_text(soup)
    ext_tot_ratio = get_ext_tot_ratio(soup, url)
    return [time_load, num_ex_links, num_redirects, num_img, num_iframe, num_hidden, ext_tot_ratio]


def get_suspicious_words(url):
    keywords = ["login", "password", "verify", "account", "security", "wp", "admin", "content",
                "site", "images", "js", "alibaba", "css", "myaccount", "dropbox", "themes",
                "plugins", "signin", "view"]
    found_keywords = 0
    for keyword in keywords:
        if keyword in url:
            found_keywords += 1
    return found_keywords


def has_ip_address(url):
    ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    return 1 if ip_pattern.search(url) else 0


def is_url_shortened(url):
    url_shortening_services = ["bit.ly", "tinyurl.com", "goo.gl", "ow.ly"]
    return 1 if any(service in url for service in url_shortening_services) else 0


def get_count_features(url):
    length = len(url)
    subdomain, domain = get_domain(url)
    subdomain_len = len(subdomain) if subdomain else 0
    subdomain_ratio = subdomain_len / length if length > 0 else 0
    hyphen_in_d = 1 if "-" in subdomain else 0
    num_dots = url.count('.')
    num_www = url.count('www')
    num_dcom = url.count('.com')
    num_http = url.count('http')
    num_https = url.count('https')
    num_2slash = url.count('//')
    num_quest = url.count('?')
    num_prtc = url.count('%')
    num_equal = url.count('=')
    num_star = url.count('*')
    num_dollar = url.count('$')
    num_under = url.count('_')
    num_space = url.count('%20') + url.count(' ')
    num_slash = url.count('/')
    num_dash = url.count('-')
    num_at = url.count('@')
    num_tile = url.count('~')
    num_line = url.count('|')
    num_colon = url.count(':')
    num_semic = url.count(';')
    num_comma = url.count(',')
    return [length, subdomain_ratio, num_dots,
            num_www, num_dcom, num_http, num_https, num_2slash, num_quest, num_prtc, num_equal,
            num_star, num_dollar, num_under, num_space, num_slash, num_dash, num_at, num_tile,
            num_line, num_colon, num_semic, num_comma]


def get_url_features(url):
    return [get_suspicious_words(url)] + [has_ip_address(url)] + [is_url_shortened(url)] + get_count_features(url)


def get_age(domain):
    try:
        creation_date = whois.whois(domain).creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date is None:
            return 0
        age = (datetime.datetime.now() - creation_date).days / 365
    except Exception:
        age = 0
    return age


feature_names = ['age', 'num_suspicious_words', 'has_ip_address',
                 'is_url_shortened', 'length', 'subdomain_ratio', 'num_dots', 'num_www',
                 'num_dcom', 'num_http', 'num_https', 'num_2slash', 'num_quest',
                 'num_prtc', 'num_equal', 'num_star', 'num_dollar', 'num_under',
                 'num_space', 'num_slash', 'num_dash', 'num_at', 'num_tile', 'num_line',
                 'num_colon', 'num_semic', 'num_comma', 'login_time', 'num_ex_links',
                 'num_redirects', 'num_img', 'num_iframe', 'num_hidden', 'ext_tot_ratio']


def extract_features_only(url, tokenizer, scaler, imputer):
    # Extract raw features
    age = get_age(url)
    response_features = get_response_features(url)
    url_features = get_url_features(url)

    X_num = [age] + url_features + response_features
    # Create DataFrame for display
    feature_df = pd.DataFrame(np.array(X_num).reshape(1, -1), columns=feature_names)

    # Prepare numeric array (handle imputer and scaler)
    try:
        if any(x is None for x in X_num):
            X_num_imputed = imputer.transform([X_num])
            X_num_norm = scaler.transform(X_num_imputed)
        else:
            X_num_norm = scaler.transform(np.array([X_num]))
    except Exception:
        X_num_norm = scaler.transform(np.zeros((1, len(feature_names))))

    # Prepare text input
    text_input_shape = (100,)
    X_text = np.array(pad_sequences(tokenizer.texts_to_sequences([url]), maxlen=text_input_shape[0], padding="post"))

    return feature_df, X_text, X_num_norm


def predict_from_prepared(X_text, X_num_norm, cnn, dnn, model_name='CNN'):
    prob = None
    pred = "Unknown"
    if model_name == 'CNN' and cnn is not None:
        prob = float(cnn.predict([X_text, X_num_norm], verbose=0).squeeze())
        pred = "Phishing" if prob > 0.5 else "Legitimate"
    elif model_name == 'DNN' and dnn is not None:
        prob = float(dnn.predict(X_num_norm).squeeze())
        pred = "Phishing" if prob > 0.5 else "Legitimate"
    else:
        pred = "Model not available"
    return pred, prob


def main():
    # Header section with enhanced styling
    st.markdown("<h1>🔒 Phishing Website Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Advanced AI-powered URL analysis to protect you from phishing attempts</p>", unsafe_allow_html=True)
    
    # Load models
    tokenizer, scaler, imputer, cnn, dnn = load_models_and_preprocessors()
    
    # Set default model to CNN
    model_choice = 'CNN'
    
    # URL input section - simplified layout
    url_input = st.text_input("🌐 Enter Website URL", value="https://www.example.com", placeholder="Enter URL to analyze...")
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Prediction button with enhanced styling
    if st.button("🔍 Analyze URL", use_container_width=True):
        raw_input = url_input.strip()
        lower_in = raw_input.lower()
        
        if not raw_input:
            st.error("⚠️ Please enter a URL to analyze")
            return
            
        # Create a progress container
        with st.container():
            # Feature extraction with enhanced UI
            st.markdown("### 📊 Feature Extraction")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress steps
            status_text.text("🔍 Initializing analysis...")
            progress_bar.progress(10)
            time.sleep(0.5)
            
            status_text.text("🌐 Extracting URL features...")
            progress_bar.progress(30)
            
            # Extract features
            feat, X_text, X_num_norm = extract_features_only(raw_input, tokenizer, scaler, imputer)
            
            status_text.text("📈 Processing response data...")
            progress_bar.progress(70)
            time.sleep(0.3)
            
            status_text.text("✅ Features extracted successfully!")
            progress_bar.progress(100)
            
            # Display features in an enhanced table
            st.markdown("#### Feature Analysis")
            st.dataframe(feat.T, use_container_width=True)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
        
        # Results section with enhanced UI
        st.markdown("### 🎯 Analysis Results")
        

        # Normal model flow for https:// or protocol-less inputs
        with st.spinner("🧠 Running AI analysis..."):
            pred, prob = predict_from_prepared(X_text, X_num_norm, cnn, dnn, model_choice)
        
        if prob is None:
            st.warning("⚠️ Model not available or prediction failed")
        else:
            # Enhanced result display with color coding
            if pred == "Legitimate" and prob < 0.98:
                st.success("✅ SAFE - Website appears legitimate")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence Level", f"{(1-prob)*100:.1f}%")
                with col2:
                    st.metric("Risk Score", f"{prob*100:.1f}%")
                
                if prob < 0.3:
                    st.info("🟢 Low risk - Safe to proceed")
                else:
                    st.warning("🟡 Moderate risk - Exercise caution")
            else:
                st.error("🚨 PHISHING DETECTED")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence Level", f"{prob*100:.1f}%")
                with col2:
                    st.metric("Risk Score", f"{prob*100:.1f}%")
                st.warning("🔴 High risk - Do not proceed")
    
    # Enhanced footer with additional information
    st.markdown("---")
    st.markdown("""
    <div class='footer'>
        <p>🔒 <strong>Security Tips:</strong></p>
        <p>• Always check for HTTPS protocol before entering sensitive information</p>
        <p>• Be cautious of URLs with suspicious characters or misspellings</p>
        <p>• Verify the domain name matches the legitimate website</p>
        <p>• Model analysis may take a few seconds for comprehensive scanning</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
