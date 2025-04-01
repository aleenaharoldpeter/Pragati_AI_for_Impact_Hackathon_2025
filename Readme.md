# AI-Powered Educational Resource Generator

## Problem Statement
Creating high-quality educational resources requires time and expertise. This app leverages AI to generate well-structured learning materials, including PDFs, quizzes, and audio lessons, based on user queries. It also supports translation and classification, making education more accessible worldwide.

## 🚀 Live App
[Access the Live App](https://dummy-link.com)

## 🛠️ Tech Used
- **Python**: Core programming language
- **Streamlit**: Web UI framework
- **Cohere API**: AI-powered text generation
- **Hugging Face Transformers**: Zero-shot classification
- **Google Translate API**: Language translation
- **ReportLab**: PDF generation

## 📋 Prerequisites
- Python 3.8+
- API keys for:
  - Cohere (for text generation)
  - Google Translate (optional, for translation)
- `pip` installed

## ⚙️ Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/ai-edu-resource-generator.git
   cd ai-edu-resource-generator
   ```
2. **Install dependencies:**
   ```bash
    pip install -r requirements.txt
   ```
3. **Set up API keys:**

    Create a `.streamlit/secrets.toml` file:

   ```toml
   [cohere]
    api_key = "your_cohere_api_key"
   ```     
4. **Run the Streamlit app:**
   ```bash
    streamlit run app.py
   ```   
## 📦 Dependencies   
```bash
pip install streamlit cohere googletrans==4.0.0-rc1 reportlab transformers
```   
## 📝 Features Overview
# 🚀 AI-Education Project  

## 1️⃣ AI-Generated Resources  
- Uses Cohere AI to generate structured educational content based on queries.  
- Outputs interactive study materials with explanations, practice problems, and resources.  

## 2️⃣ Smart Classification  
- Hugging Face Transformers classify queries into PDF, quiz, or audio lesson.  
- Ensures users receive the most suitable format.  

## 3️⃣ Language Translation  
- Google Translate API enables multilingual support for global accessibility.  

## 4️⃣ PDF Generation  
- ReportLab converts AI-generated content into professionally styled PDFs.  

## 5️⃣ UI Enhancements  
- Custom Streamlit styling for an engaging user experience.  
- Includes a chatbot-like FAQ section.  

---

## 🏗️ Code Explanation  

### `load_classifier()`  
- Loads a zero-shot classifier to categorize user queries.  

### `load_generator()`  
- Initializes Cohere's AI model for text generation.  

### `generate_resource_content(query, subject)`  
- Uses a structured AI prompt to generate high-quality educational content.  

### `create_pdf_reportlab(content, file_name)`  
- Formats AI-generated text into a well-structured PDF using ReportLab.  

### `translate_text(text, target_lang)`  
- Utilizes Google Translate API for multilingual support.  

### `save_resource(resource)`  
- Saves generated content locally for future reference.  

---

## 📖 Future Improvements  
✅ Implement voice-to-text for spoken queries.  
✅ Add adaptive learning paths based on user history.  
✅ Integrate interactive quizzes using AI.  

---

## 📬 Feedback & Contributions  
We welcome feedback and contributions! Feel free to submit issues or pull requests.  
---
© 2025 **AI-Education Project** | Made with ❤️ using AI  
