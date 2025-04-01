import os
import re
import json
import textwrap
from io import BytesIO

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

# Streamlit for UI and web app framework
import streamlit as st

# Cohere for text generation (ensure you have your API key configured in st.secrets)
import cohere  

# Transformers pipeline for zero-shot classification using Hugging Face model
from transformers import pipeline

# Google Translate API wrapper for text translation
from googletrans import Translator


# ----------------------------
# Custom CSS for UI Enhancements
# ----------------------------
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2 {
        text-align: center;
        color: #4a90e2;
    }
    .sidebar .sidebar-content {
        background-color: #f7f7f7;
        padding: 1rem;
    }
    .save-button, .download-button, .submit-button {
        background-color: #4a90e2;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
    }
    .download-button {
        background-color: #50c878;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ----------------------------
# Caching: Load Models and Translator
# ----------------------------
@st.cache_resource
def load_classifier():
    """
    Load and cache the zero-shot classifier model from Hugging Face.
    Using 'facebook/bart-large-mnli' for text classification.
    """
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return classifier


classifier = load_classifier()


@st.cache_resource
def load_translator():
    """
    Load and cache the Google Translator instance.
    """
    translator = Translator()
    return translator


translator = load_translator()


@st.cache_resource
def load_generator():
    """
    Load and cache the Cohere text generator client using the API key from secrets.
    """
    token = st.secrets["cohere"]["api_key"] 
    client = cohere.Client(token)
    return client


generator = load_generator()


# ----------------------------
# Utility Functions
# ----------------------------
def classify_query(query):
    """
    Classify the given query into one of the candidate resource types.
    
    Parameters:
        query (str): The user's query text.
    
    Returns:
        str: The label (resource type) with the highest score.
    """
    candidate_labels = ["PDF", "Quiz", "Audio Lesson"]
    result = classifier(query, candidate_labels)
    return result["labels"][0]


def translate_text(text, target_lang):
    """
    Translate the provided text to a target language if necessary.
    
    Parameters:
        text (str): The original text.
        target_lang (str): The target language code (e.g., 'en', 'hi').
    
    Returns:
        str: Translated text if target language is not English; otherwise, returns original text.
    """
    if target_lang != "en":
        translation = translator.translate(text, dest=target_lang)
        return translation.text
    return text


def process_inline_formatting(text):
    """
    Process basic inline markdown formatting and convert it into HTML-like tags for ReportLab.
    
    - **bold** becomes <b>bold</b>
    - *italic* becomes <i>italic</i>
    - [text](url) becomes <a href="url" color="blue">text</a>
    
    Parameters:
        text (str): The input text with markdown.
    
    Returns:
        str: Text with converted formatting.
    """
    # Convert bold markdown to HTML <b> tags
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Convert italic markdown to HTML <i> tags
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    # Convert markdown links to HTML <a> tags with blue color
    text = re.sub(r"\[(.+?)\]\((https?://[^\)]+)\)", r'<a href="\2" color="blue">\1</a>', text)
    return text


def generate_resource_content(query, subject):
    """
    Generate educational resource content using Cohere's text generation API.
    
    Parameters:
        query (str): The student's question.
        subject (str): The subject context (e.g., Math, Science, etc.).
    
    Returns:
        str: The generated resource content.
    """
    # Prepare a dynamic prompt for the generator including structured requirements
    dynamic_prompt = (
        f"You are an expert teacher in {subject}. A student asked: '{query}'.\n\n"
        "Generate a world-class, comprehensive educational resource on this topic that is interactive, easy-to-understand, and engaging. The resource must include the following sections, formatted with clear headers and bullet points where appropriate:\n\n"
        "1. **Introduction:**\n"
        "   - Provide a concise yet thorough explanation of the key concepts, background, and context for the topic.\n"
        "   - Explain any specialized terminology in simple language.\n\n"
        "2. **Practice Problems:**\n"
        "   - List at least 5 well-designed practice problems related to the topic.\n"
        "   - For each problem, include:\n"
        "       * The problem statement.\n"
        "       * A detailed, step-by-step solution.\n"
        "       * Explanatory notes to help understand the solution.\n\n"
        "3. **Tips & Tricks:**\n"
        "   - Provide at least 3 actionable tips or strategies to master the topic.\n"
        "   - Ensure the tips are practical and easy to implement in a classroom setting.\n\n"
        "4. **Additional Resources:**\n"
        "   - Include a list of 2-3 free, high-quality online resources (such as websites, articles, or video tutorials) for further learning.\n"
        "   - Include hyperlinks if possible.\n\n"
        "5. **Subject-Specific Advice:**\n"
        "   - Offer any additional recommendations, insights, or innovative teaching strategies tailored specifically to {subject}.\n"
        "   - This may include methods for making lessons more interactive or ideas for increasing student engagement.\n\n"
        "Ensure the output is well-structured with clear headers for each section, uses bullet points or numbered lists where appropriate, and is written in a style that is both engaging and immediately applicable for a teacher in the classroom."
    )
    
    # Generate text using the Cohere API
    response = generator.generate(
        prompt=dynamic_prompt,
        max_tokens=1000,
        temperature=0.7,
    )
    # Print the generated content for debugging purposes
    print(response.generations[0].text)
    return response.generations[0].text


def create_pdf_reportlab(content, file_name, heading_to_content_space=24):
    """
    Create a styled PDF using ReportLab.
    
    Features:
    - Lines starting with "##" are treated as headings.
    - Any text before the first heading is ignored.
    - Spacing between headings and content is configurable.
    - If a section's first non-empty line starts with a bullet marker, the content is rendered as a bullet list.
    - Inline markdown formatting is converted to HTML-like tags.
    - Uses DejaVu Sans font for full Unicode support.
    
    Parameters:
        content (str): The content to be included in the PDF.
        file_name (str): Name of the generated PDF file.
        heading_to_content_space (int): Vertical space between headings and their content.
    
    Returns:
        bytes: The generated PDF in bytes.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    
    # Path to the DejaVuSans.ttf file (ensure this path is correct in your project)
    font_path = os.path.join("dejavu-fonts-ttf-2.37", "dejavu-fonts-ttf-2.37", "ttf", "DejaVuSans.ttf")
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"TTF Font file not found: {font_path}")
    
    # Register the DejaVuSans font for ReportLab
    pdfmetrics.registerFont(TTFont("DejaVuSans", font_path))
    
    styles = getSampleStyleSheet()
    
    # Define custom style for headings
    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        fontName='DejaVuSans',
        fontSize=18,
        leading=22,
        spaceBefore=20,
        spaceAfter=6,
        textColor='#4a90e2'
    )
    
    # Define custom style for normal body text
    body_style = ParagraphStyle(
        'Body',
        parent=styles['BodyText'],
        fontName='DejaVuSans',
        fontSize=12,
        leading=16,
        spaceAfter=12
    )
    
    # Define custom style for bullet list items
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['BodyText'],
        fontName='DejaVuSans',
        fontSize=12,
        leading=16,
        leftIndent=20,
        bulletIndent=10,
        spaceAfter=6
    )
    
    elements = []
    
    # Split content into sections based on headings (lines starting with "##")
    parts = re.split(r'(^##\s*.+$)', content, flags=re.MULTILINE)
    
    # Skip any text before the first heading (if present)
    if parts and not parts[0].strip().startswith("##"):
        pass
    
    # Process each heading-content pair
    for i in range(1, len(parts), 2):
        heading_line = parts[i].strip()  # e.g., "## Introduction"
        heading_text = heading_line.lstrip("#").strip()  # Remove hashes and whitespace
        heading_text = process_inline_formatting(heading_text)  # Process inline markdown
        elements.append(Paragraph(heading_text, heading_style))
        elements.append(Spacer(1, heading_to_content_space))
        
        # Get the corresponding section content
        section_content = parts[i+1].strip() if i+1 < len(parts) else ""
        section_content = process_inline_formatting(section_content)
        
        # Determine if the section should be rendered as a bullet list
        bullet_match = None
        for line in section_content.splitlines():
            if line.strip():
                bullet_match = re.match(r'^(?:[\dA-Za-z]+\.\s+|\-\s+)', line)
                break
        
        if bullet_match:
            bullet_items = []
            # Process each non-empty line as a bullet item
            for line in section_content.splitlines():
                line = line.strip()
                if line:
                    # Remove bullet marker (if present) from the line
                    line = re.sub(r'^(?:[\dA-Za-z]+\.\s+|\-\s+)', '', line)
                    bullet_items.append(Paragraph(line, bullet_style))
            elements.append(ListFlowable(bullet_items, bulletType='bullet', leftIndent=20))
        else:
            if section_content:
                elements.append(Paragraph(section_content, body_style))
        
        elements.append(Spacer(1, 12))
    
    # Build the PDF document with the assembled elements
    doc.build(elements)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data


# ----------------------------
# Session Persistence Functions
# ----------------------------
def load_saved_resources():
    """
    Load previously saved resources from a JSON file.
    
    Returns:
        list: List of saved resource strings.
    """
    if os.path.exists("saved_resources.json"):
        with open("saved_resources.json", "r") as f:
            return json.load(f)
    return []


def save_saved_resources(resources):
    """
    Save the list of resources to a JSON file.
    
    Parameters:
        resources (list): List of resource strings.
    """
    with open("saved_resources.json", "w") as f:
        json.dump(resources, f)


def load_peer_suggestions():
    """
    Load peer suggestions from a JSON file.
    
    Returns:
        list: List of peer suggestion dictionaries.
    """
    if os.path.exists("peer_suggestions.json"):
        with open("peer_suggestions.json", "r") as f:
            return json.load(f)
    return []


def save_peer_suggestions(suggestions):
    """
    Save the list of peer suggestions to a JSON file.
    
    Parameters:
        suggestions (list): List of suggestion dictionaries.
    """
    with open("peer_suggestions.json", "w") as f:
        json.dump(suggestions, f)


# Initialize session state variables if they don't exist
if "saved_resources" not in st.session_state:
    st.session_state.saved_resources = load_saved_resources()
if "peer_suggestions" not in st.session_state:
    st.session_state.peer_suggestions = load_peer_suggestions()


def save_resource(resource):
    """
    Save a generated resource if it's not already saved.
    
    Parameters:
        resource (str): The generated resource content.
    """
    if resource not in st.session_state.saved_resources:
        st.session_state.saved_resources.append(resource)
        save_saved_resources(st.session_state.saved_resources)
        st.success(f"Saved resource: {resource}")
    else:
        st.info("Resource already saved.")


def add_peer_suggestion(suggestion):
    """
    Add a new peer suggestion and update the JSON file.
    
    Parameters:
        suggestion (dict): Dictionary containing suggestion details.
    """
    st.session_state.peer_suggestions.append(suggestion)
    save_peer_suggestions(st.session_state.peer_suggestions)
    st.success("Thank you for your suggestion!")


# ----------------------------
# FAQ Chatbot Section
# ----------------------------
FAQS = {
    "What is the purpose of this app?": "This app helps users access AI-curated educational resources optimized for low-connectivity areas.",
    "How can I save a resource?": "Simply click the 'Save this Resource' button to save a resource for later use.",
    "How do I download a resource?": "Use the 'Download Resource' button to download the content for offline access.",
    "What languages are supported?": "Currently, the app supports English, Hindi, Tamil, and Bengali.",
}


def display_faq():
    """
    Display the FAQ section in the Streamlit app.
    """
    st.subheader("FAQs")
    faq_question = st.selectbox("Select a question", options=list(FAQS.keys()))
    if st.button("Get Answer", key="faq"):
        answer = FAQS.get(faq_question, "Sorry, I don't have an answer for that.")
        st.info(answer)


# ----------------------------
# Main App UI Layout
# ----------------------------
st.title("AI-Powered Educational Assistant")
st.markdown("<h2 style='text-align: center;'>Empowering Low-Connectivity Areas with AI</h2>", unsafe_allow_html=True)

# Sidebar for user preferences and saved resources
with st.sidebar:
    st.header("User Preferences")
    grade = st.selectbox("Select Grade Level", options=["Grade 1", "Grade 2"])
    subject = st.selectbox("Select Subject", options=["Math", "Science", "English"])
    # Language selection returns a tuple; we extract the language code
    language = st.selectbox(
        "Select Language",
        options=[("English", "en"), ("Hindi", "hi"), ("Tamil", "ta"), ("Bengali", "bn")],
        format_func=lambda x: x[0]
    )[1]
    st.markdown("---")
    st.subheader("Saved Resources")
    if st.session_state.saved_resources:
        for res in st.session_state.saved_resources:
            st.write(res)
    else:
        st.write("No resources saved yet.")

st.subheader("Ask a Question or Get Learning Resources")
user_query = st.text_input("Type your query (e.g., 'I need practice problems on fractions'):")

st.info("Voice input feature coming soon!")


# Place the file name text input outside of the button click handler.
# This sets a default value of "generated_resource" if the user does not enter anything.
file_name = st.text_input("Enter PDF file name:", value="generated_resource", key="pdf_file_name")


if st.button("Get Recommendation"):
    if not user_query.strip():
        st.warning("Please enter a query.")
    else:
        # Translate query to English if necessary
        translated_query = translate_text(user_query, "en")
        if language != "en":
            st.info(f"Translated Query: {translated_query}")
        
        # Classify query to determine resource type (PDF, Quiz, or Audio Lesson)
        resource_type = classify_query(translated_query)
        st.success(f"Recommended Resource Type: **{resource_type}**")
        
        # Generate educational content using Cohere's text generation API
        generated_content = generate_resource_content(translated_query, subject)
        st.markdown("### Generated Resource Content")
        st.markdown(generated_content)
        
        # Save the generated resource on button click
        if st.button("Save this Resource"):
            save_resource(generated_content)
        
                
        # Create the PDF using the user-provided file name
        pdf_data = create_pdf_reportlab(generated_content, f"{file_name}.pdf")
        st.download_button(
            label="Download Generated Resource as PDF",
            data=pdf_data,
            file_name=f"{file_name}.pdf",
            mime="application/pdf"
        )

st.markdown("---")
display_faq()

st.markdown("---")
st.subheader("Peer-Learning Module")
st.markdown("Suggest a new learning resource for others:")

# Peer suggestion form inputs
suggest_subject = st.selectbox("Subject", options=["Math", "Science", "English"], key="peer_subject")
suggest_grade = st.selectbox("Grade Level", options=["Grade 1", "Grade 2"], key="peer_grade")
suggest_resource_type = st.selectbox("Resource Type", options=["PDF", "Quiz", "Audio Lesson"], key="peer_type")
suggest_description = st.text_area("Describe the resource and why it would be helpful:")

if st.button("Submit Suggestion", key="submit_suggestion"):
    suggestion = {
        "subject": suggest_subject,
        "grade": suggest_grade,
        "resource_type": suggest_resource_type,
        "description": suggest_description,
    }
    add_peer_suggestion(suggestion)

st.markdown("---")
st.subheader("Peer Suggestions")
if st.session_state.peer_suggestions:
    for idx, sugg in enumerate(st.session_state.peer_suggestions, start=1):
        st.markdown(f"**Suggestion {idx}:**")
        st.write(f"Subject: {sugg.get('subject')}, Grade: {sugg.get('grade')}, Type: {sugg.get('resource_type')}")
        st.write(f"Description: {sugg.get('description')}")
        st.markdown("---")
else:
    st.write("No peer suggestions yet.")

st.markdown("---")
st.subheader("Feedback")
feedback = st.radio("How useful was this recommendation?", options=["Excellent", "Good", "Average", "Poor"])
if st.button("Submit Feedback", key="feedback_submit"):
    st.write(f"Thank you for your feedback: **{feedback}**")

st.markdown("<hr>", unsafe_allow_html=True)
st.caption(
    "Prototype built using Streamlit, Cohere, Hugging Face Transformers, Google Translate, and ReportLab. "
    "Offline caching is simulated. Future enhancements include voice input and persistent session storage."
)

"""
To run the code in a IDE terminal
python -m streamlit run app.py
"""