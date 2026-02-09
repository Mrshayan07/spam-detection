import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import nltk

# Configure page
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main background and text colors */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Title styling */
    .main h1 {
        color: white;
        text-align: center;
        font-size: 3em;
        margin-bottom: 0.5em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: rgba(255,255,255,0.9);
        font-size: 1.1em;
        margin-bottom: 2em;
    }
    
    /* Input box styling */
    .stTextArea textarea {
        border: 2px solid #667eea !important;
        border-radius: 10px !important;
        font-size: 16px !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: bold !important;
        padding: 12px 40px !important;
        border: none !important;
        border-radius: 25px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3) !important;
    }
    
    /* Result styling - Spam */
    .spam-result {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff4757 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 2em;
        font-weight: bold;
        box-shadow: 0 8px 25px rgba(255, 75, 87, 0.3);
        margin-top: 20px;
    }
    
    /* Result styling - Not Spam */
    .not-spam-result {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 2em;
        font-weight: bold;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.3);
        margin-top: 20px;
    }
    
    /* Card styling */
    .info-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.95);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def download_nltk():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

download_nltk()

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            if i not in stop_words and i not in string.punctuation:
                y.append(ps.stem(i))
    return " ".join(y)

# Load models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Header Section
st.markdown("""
    <div style='text-align: center; padding: 2em 0;'>
        <h1>🚨 Spam Detector</h1>
        <p class='subtitle'>Intelligent Email & SMS Spam Classification</p>
    </div>
""", unsafe_allow_html=True)

# Create columns for layout
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    # Info box
    st.markdown("""
        <div class='info-card'>
            <h3>📬 How it works</h3>
            <p>This advanced classifier uses machine learning to detect spam messages with high accuracy. 
            Simply paste your message below and click "Analyze Message" to get results.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Input section
    st.markdown("### ✉️ Enter Your Message")
    input_sms = st.text_area(
        "Paste the email or SMS content here:",
        height=150,
        placeholder="Type or paste your message here...",
        label_visibility="collapsed"
    )
    
    # Button and prediction
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        predict_button = st.button('🔍 Analyze Message', use_container_width=True)
    
    with col_btn2:
        clear_button = st.button('🔄 Clear', use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    if predict_button:
        if input_sms.strip():
            with st.spinner('Analyzing message...'):
                # Preprocess
                transformed_sms = transform_text(input_sms)
                
                # Vectorize
                vector_input = tfidf.transform([transformed_sms])
                
                # Predict
                result = model.predict(vector_input)[0]
                probability = model.predict_proba(vector_input)[0]
                
                # Display results
                st.markdown("<br>", unsafe_allow_html=True)
                
                if result == 1:
                    st.markdown(
                        '<div class="spam-result">🚨 SPAM DETECTED</div>',
                        unsafe_allow_html=True
                    )
                    st.error(f"⚠️ This message is likely **SPAM**")
                    st.warning(f"Confidence: {probability[1]*100:.2f}%")
                else:
                    st.markdown(
                        '<div class="not-spam-result">✅ NOT SPAM</div>',
                        unsafe_allow_html=True
                    )
                    st.success(f"✓ This message appears to be **LEGITIMATE**")
                    st.info(f"Confidence: {probability[0]*100:.2f}%")
        else:
            st.warning("⚠️ Please enter a message to analyze!")

with col2:
    # Sidebar info
    st.markdown("""
        <div class='info-card'>
            <h3>📊 Quick Stats</h3>
            <p>
            <strong>Classifier:</strong> TF-IDF + ML Model<br>
            <strong>Language:</strong> English<br>
            <strong>Processing:</strong> Real-time<br>
            <strong>Accuracy:</strong> High
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='info-card'>
            <h3>💡 Tips</h3>
            <ul>
            <li>Paste full message</li>
            <li>Include subject line</li>
            <li>Check results</li>
            <li>Report false positives</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr style='margin: 2em 0; border: 1px solid rgba(255,255,255,0.2);'>
    <div style='text-align: center; color: rgba(255,255,255,0.7); padding: 1em;'>
        <small>🔒 Your messages are processed locally and not stored | Built with ❤️ using Streamlit</small>
    </div>
""", unsafe_allow_html=True)