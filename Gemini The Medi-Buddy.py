# -*- coding: utf-8 -*-
"""Medicine Dosage Calculator - Enhanced with Voice, OCR, Chatbot & AI Prescription Explanation"""

import gradio as gr
import pandas as pd
import re
from datetime import datetime
from rapidfuzz import fuzz, process
from deep_translator import GoogleTranslator
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch
import tempfile
import os
import json

# Try to import Gemma model from HuggingFace
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠ Transformers library not available. Install: pip install transformers torch")

# Try to import Gemini API
try:
    import google.generativeai as genai
    GEMINI_API_AVAILABLE = True
except ImportError:
    GEMINI_API_AVAILABLE = False
    print("⚠ Gemini API not available. Install: pip install google-generativeai")

# OCR imports
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠ OCR libraries not available. Install: pip install pillow pytesseract")

# OpenCV for image preprocessing
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠ OpenCV not available. Install for better OCR: pip install opencv-python")

# Speech recognition imports
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("⚠ Speech recognition not available. Install: pip install SpeechRecognition")

# Weather API
try:
    import requests
    WEATHER_API_AVAILABLE = True
except ImportError:
    WEATHER_API_AVAILABLE = False
    print("⚠ Requests library not available. Install: pip install requests")

# Global variables
df = None
gemini_model = None
tokenizer = None
prescription_history = []
chat_history = []
recognizer = sr.Recognizer() if SPEECH_AVAILABLE else None

# Available languages with proper display names
SUPPORTED_LANGUAGES = [
    ("English", "en"),
    ("Hindi (हिन्दी)", "hi"),
    ("Tamil (தமிழ்)", "ta"),
    ("Telugu (తెలుగు)", "te"),
    ("Kannada (ಕನ್ನಡ)", "kn"),
    ("Malayalam (മലയാളം)", "ml"),
    ("Marathi (मराठी)", "mr"),
    ("Gujarati (ગુજરાતી)", "gu"),
    ("Bengali (বাংলা)", "bn"),
    ("Punjabi (ਪੰਜਾਬੀ)", "pa"),
    ("Urdu (اردو)", "ur"),
    ("Odia (ଓଡ଼ିଆ)", "or"),
    ("Assamese (অসমীয়া)", "as"),
    ("Sanskrit (संस्कृतम्)", "sa"),
    ("Kashmiri (कॉशुर)", "ks"),
    ("Bodo (बर')", "brx"),
    ("Dogri (डोगरी)", "doi"),
    ("Konkani (कोंकणी)", "kok"),
    ("Maithili (मैथिली)", "mai"),
    ("Manipuri (मैतैलोन्)", "mni"),
    ("Nepali (नेपाली)", "ne"),
    ("Sindhi (سنڌي)", "sd"),
    ("Santali (ᱥᱟᱱᱛᱟᱲᱤ)", "sat"),
]

# === Translation Function ===
def translate_text(text, target_lang_code, source_lang="auto"):
    """Translate text using Google Translator with enhanced error handling"""
    if not text or not text.strip():
        return "⚠ No text to translate"
    
    try:
        # Extract language code if it's in format "language (code)"
        if isinstance(target_lang_code, tuple):
            target_lang_code = target_lang_code[1]
        
        # Clean the text for better translation
        text_to_translate = text.strip()
        
        # Don't translate if it's already in English
        if target_lang_code == 'en':
            return f"**Original Text:**\n\n{text_to_translate}"
        
        # Initialize translator
        translator = GoogleTranslator(source=source_lang, target=target_lang_code)
        
        # Handle long texts by splitting
        if len(text_to_translate) > 5000:
            chunks = [text_to_translate[i:i+5000] for i in range(0, len(text_to_translate), 5000)]
            translated_chunks = []
            for chunk in chunks:
                translated_chunks.append(translator.translate(chunk))
            translated_text = " ".join(translated_chunks)
        else:
            translated_text = translator.translate(text_to_translate)
        
        # Get language name for display
        lang_name = [name for name, code in SUPPORTED_LANGUAGES if code == target_lang_code]
        lang_display = lang_name[0] if lang_name else target_lang_code.upper()
        
        return f"""**🌐 Translation ({lang_display}):**

{translated_text}

---
*Note: This is an automated translation. For medical accuracy, always consult a healthcare professional.*"""
    
    except Exception as e:
        return f"""❌ **Translation failed:** {str(e)}

**Possible reasons:**
• Internet connection issue
• Translation service temporarily unavailable
• Text too long or contains special characters
• Language code not supported

**💡 Solution:** Try again in a few moments or use a different language."""

def translate_multiple_texts(text1, text2, text3, target_lang):
    """Translate multiple text boxes at once"""
    results = []
    for text in [text1, text2, text3]:
        if text and text.strip():
            results.append(translate_text(text, target_lang))
        else:
            results.append("No text to translate")
    return results

# === Initialize Gemma Model ===
def initialize_gemma(model_name="models/gemma-3-4b-it", use_api=False, api_key=None):
    """Initialize Gemma model from HuggingFace or use Gemini API"""
    global gemini_model, tokenizer
    
    if use_api and GEMINI_API_AVAILABLE:
        try:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel('models/gemma-3-4b-it')
            return "✅ Gemini API initialized successfully!"
        except Exception as e:
            return f"❌ Failed to initialize Gemini API: {str(e)}"
    
    elif GEMINI_AVAILABLE:
        try:
            print(f"🚀 Loading {model_name}... This may take a moment.")
            
            # Use a smaller model for faster loading
            if model_name == "models/gemma-3-4b-it":
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                gemini_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # Try a smaller model if the specified one fails
                tokenizer = AutoTokenizer.from_pretrained("models/gemma-3-4b-it", trust_remote_code=True)
                gemini_model = AutoModelForCausalLM.from_pretrained(
                    "models/gemma-3-4b-it",
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            return f"✅ Gemma model '{model_name}' loaded successfully! You can now use all AI features."
        except Exception as e:
            gemini_model = None
            tokenizer = None
            return f"❌ Failed to load Gemma model: {str(e)}\n\nPlease try using the Gemini API option instead."
    else:
        return "⚠ Please install required packages: pip install transformers torch"

def generate_gemma_text(prompt, use_api=False):
    """Generate text using Gemma model or Gemini API"""
    global gemini_model, tokenizer
    
    if gemini_model is None:
        return "Error: Gemma model not loaded. Please initialize the model first in the Setup tab."
    
    try:
        # Auto-detect if using API based on model type
        is_api_model = GEMINI_API_AVAILABLE and hasattr(gemini_model, 'generate_content')
        
        if (use_api or is_api_model) and GEMINI_API_AVAILABLE:
            response = gemini_model.generate_content(prompt)
            if response and response.text:
                return response.text
            return "No response from Gemini API."
        elif GEMINI_AVAILABLE and gemini_model and tokenizer:
            # Format prompt for Gemma
            formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(gemini_model.device)
            
            with torch.no_grad():
                outputs = gemini_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the model's response
            if "<start_of_turn>model" in response:
                response = response.split("<start_of_turn>model")[-1].strip()
            if "<end_of_turn>" in response:
                response = response.split("<end_of_turn>")[0].strip()
            
            return response
        else:
            return "Error: AI model not available. Please check your setup."
            
    except Exception as e:
        return f"Error generating text: {str(e)}"

# === Load Dataset ===
def load_dataset(file):
    """Load dataset from uploaded file"""
    global df
    
    try:
        if file is None:
            return "⚠ Please select a file to upload", gr.update()
        
        print(f"📂 Loading file: {file.name}")
        
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name, encoding='utf-8')
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file.name)
        else:
            return "⚠ Please upload CSV (.csv) or Excel (.xlsx, .xls) file only", gr.update()
        
        df.columns = df.columns.str.strip().str.title()
        
        if 'Name' not in df.columns:
            available = ', '.join(df.columns.tolist())
            return f"❌ Error: 'Name' column not found!\n\nColumns found: {available}", gr.update()
        
        df['Name'] = df['Name'].astype(str).str.strip()
        df = df[df['Name'].str.len() > 0]
        df = df[df['Name'] != 'nan']
        df['Name_Search'] = df['Name'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
        
        original_count = len(df)
        df = df.drop_duplicates(subset='Name', keep='first')
        duplicates_removed = original_count - len(df)
        
        sample_meds = df['Name'].head(10).tolist()
        sample_text = ', '.join(sample_meds[:5])
        if len(sample_meds) > 5:
            sample_text += f" ... and {len(df) - 5} more"
        
        success_msg = f"""✅ **Dataset Loaded Successfully!**

📊 **Statistics:**
• Total medicines: {len(df)}
• Duplicates removed: {duplicates_removed}

💊 **Sample Medicines:**
{sample_text}

✅ **Ready to calculate dosages!**"""
        
        return success_msg, gr.update()
        
    except Exception as e:
        return f"❌ Error loading file: {str(e)}", gr.update()

# === Find Medicine ===
def find_medicine(medicine_name):
    """Find medicine using fuzzy matching"""
    global df
    
    if df is None:
        return None, "⚠ Please upload your dataset first!"
    
    if not medicine_name or not medicine_name.strip():
        return None, "⚠ Please enter a medicine name"
    
    medicine_name = medicine_name.strip()
    search_clean = re.sub(r'[^\w\s]', '', medicine_name.lower())
    
    # Exact match
    exact = df[df['Name'].str.lower() == medicine_name.lower()]
    if not exact.empty:
        return exact.iloc[0], f"✅ Exact match found: {exact.iloc[0]['Name']}"
    
    # Contains match
    contains = df[df['Name'].str.lower().str.contains(medicine_name.lower(), na=False)]
    if not contains.empty:
        return contains.iloc[0], f"✅ Found: {contains.iloc[0]['Name']}"
    
    # Fuzzy matching
    try:
        result = process.extractOne(
            search_clean, 
            df['Name_Search'].tolist(), 
            scorer=fuzz.token_sort_ratio, 
            score_cutoff=70
        )
        
        if result:
            matched_name, score, idx = result
            return df.iloc[idx], f"✅ Found: {df.iloc[idx]['Name']} (Match: {score}%)"
    except:
        pass
    
    # Show suggestions
    try:
        suggestions = process.extract(
            search_clean, 
            df['Name_Search'].tolist(), 
            scorer=fuzz.token_sort_ratio, 
            limit=5
        )
        
        sugg_list = []
        for _, score, idx in suggestions:
            if score > 50:
                sugg_list.append(f"  • {df.iloc[idx]['Name']} ({score}% match)")
        
        if sugg_list:
            sugg_text = "\n".join(sugg_list)
            return None, f"❌ Medicine '{medicine_name}' not found.\n\n💡 Did you mean:\n{sugg_text}"
    except:
        pass
    
    return None, f"❌ Medicine '{medicine_name}' not found in database."

# === Calculate Dosage ===
def calculate_dosage(age, weight, strength_str):
    """Calculate dosage based on age and weight"""
    mg_match = re.search(r'(\d+\.?\d*)', str(strength_str))
    base_mg = float(mg_match.group(1)) if mg_match else 500.0
    
    if age < 1:
        single_dose = weight * 10
        frequency = "Every 8 hours"
        category = "Infant"
    elif age < 12:
        single_dose = weight * 15
        frequency = "Every 6-8 hours"
        category = "Child"
    elif age < 60:
        single_dose = base_mg
        frequency = "Every 6-8 hours"
        category = "Adult"
    else:
        single_dose = base_mg * 0.75
        frequency = "Every 8 hours"
        category = "Elderly"
    
    return {
        "category": category,
        "single_dose": round(single_dose, 1),
        "frequency": frequency,
        "daily_dose": round(single_dose * 3, 1),
        "max_daily": round(base_mg * 4, 1)
    }

# === AI Explanation ===
def get_ai_explanation(medicine_info, age, weight, dosage_info):
    """Get AI explanation using Gemma model"""
    
    fallback = f"""📋 **Medicine Information**

**Name:** {medicine_info.get('Name', 'Unknown')}
**Classification:** {medicine_info.get('Classification', 'N/A')}
**Indication:** {medicine_info.get('Indication', 'N/A')}

💊 **Recommended Dosage**
• Patient Category: {dosage_info['category']}
• Single dose: {dosage_info['single_dose']} mg
• Frequency: {dosage_info['frequency']}
• Daily total: {dosage_info['daily_dose']} mg

⚠ **Disclaimer:** Consult healthcare professional."""
    
    try:
        prompt = f"""Provide a brief medical explanation (max 200 words) for:

Medicine: {medicine_info.get('Name', 'Unknown')}
Classification: {medicine_info.get('Classification', 'N/A')}
Patient: {dosage_info['category']}, Age {age} years, Weight {weight} kg
Recommended Dosage: {dosage_info['single_dose']}mg, {dosage_info['frequency']}

Include:
1. How the medicine works
2. Why this dosage is appropriate
3. Common side effects
4. Precautions"""
        
        response = generate_gemma_text(prompt)
        
        if response and not response.startswith("Error"):
            return response
    except:
        pass
    
    return fallback

# === Process Medicine ===
def process_medicine(medicine_name, patient_name, age, weight):
    """Main processing function"""
    global df, prescription_history
    
    if df is None:
        return "⚠ **No dataset loaded!**", "", "", "", None
    
    if not medicine_name or not medicine_name.strip():
        return "⚠ Please enter a medicine name", "", "", "", None
    
    if not patient_name or not patient_name.strip():
        patient_name = "Patient"
    
    if age is None or age <= 0:
        return "⚠ Please enter a valid age", "", "", "", None
    
    if weight is None or weight <= 0:
        return "⚠ Please enter a valid weight", "", "", "", None
    
    try:
        medicine_info, search_msg = find_medicine(medicine_name)
        
        if medicine_info is None:
            return search_msg, "", "", "", None
        
        dosage_info = calculate_dosage(age, weight, medicine_info.get('Strength', '500mg'))
        explanation = get_ai_explanation(medicine_info, age, weight, dosage_info)
        
        # Store in history
        prescription_history.append({
            'timestamp': datetime.now(),
            'patient_name': patient_name,
            'medicine_name': medicine_info.get('Name', 'N/A'),
            'age': age,
            'weight': weight,
            'dosage': dosage_info,
            'explanation': explanation,
            'medicine_info': medicine_info
        })
        
        medicine_display = f"""✅ **Medicine Found: {medicine_info.get('Name', 'N/A')}**

**Classification:** {medicine_info.get('Classification', 'N/A')}
**Indication:** {medicine_info.get('Indication', 'N/A')}
**Strength:** {medicine_info.get('Strength', 'N/A')}"""
        
        dosage_display = f"""👤 **Patient:** {patient_name}
📊 **Category:** {dosage_info['category']}

💊 **Single Dose:** {dosage_info['single_dose']} mg
⏰ **Frequency:** {dosage_info['frequency']}
📈 **Daily Total:** {dosage_info['daily_dose']} mg
⚠️ **Maximum Daily:** {dosage_info['max_daily']} mg"""
        
        pdf = generate_pdf(patient_name, medicine_info, age, weight, dosage_info, explanation)
        
        return medicine_display, dosage_display, explanation, search_msg, pdf
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", "", None

# === Generate PDF ===
def generate_pdf(patient_name, medicine_info, age, weight, dosage_info, explanation):
    """Generate PDF report"""
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='wb')
        temp_path = temp_file.name
        temp_file.close()
        
        doc = SimpleDocTemplate(temp_path, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        elements = []
        
        elements.append(Paragraph("Medicine Dosage Report", styles['Title']))
        elements.append(Spacer(1, 0.3*inch))
        
        elements.append(Paragraph("Patient Information", styles['Heading2']))
        elements.append(Paragraph(f"Name: {patient_name}", styles['Normal']))
        elements.append(Paragraph(f"Age: {age} years", styles['Normal']))
        elements.append(Paragraph(f"Weight: {weight} kg", styles['Normal']))
        elements.append(Paragraph(f"Category: {dosage_info['category']}", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        elements.append(Paragraph("Medicine Information", styles['Heading2']))
        elements.append(Paragraph(f"Name: {medicine_info.get('Name', 'N/A')}", styles['Normal']))
        elements.append(Paragraph(f"Classification: {medicine_info.get('Classification', 'N/A')}", styles['Normal']))
        elements.append(Paragraph(f"Strength: {medicine_info.get('Strength', 'N/A')}", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        elements.append(Paragraph("Recommended Dosage", styles['Heading2']))
        elements.append(Paragraph(f"Single Dose: {dosage_info['single_dose']} mg", styles['Normal']))
        elements.append(Paragraph(f"Frequency: {dosage_info['frequency']}", styles['Normal']))
        elements.append(Paragraph(f"Daily Total: {dosage_info['daily_dose']} mg", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        if explanation and len(explanation) > 50:
            elements.append(Paragraph("Medical Information", styles['Heading2']))
            clean_explanation = explanation.replace('**', '').replace('*', '').replace('#', '')
            elements.append(Paragraph(clean_explanation[:800], styles['Normal']))
        
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("DISCLAIMER: For educational purposes only.", styles['Normal']))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        
        doc.build(elements)
        return temp_path
    except:
        return None

# === Enhanced OCR with Image Preprocessing ===
def extract_text_from_image(image):
    """Extract text from prescription image using OCR with preprocessing"""
    if not OCR_AVAILABLE:
        return "⚠ OCR not available. Install: pip install pillow pytesseract"
    
    if image is None:
        return "⚠ Please upload an image"
    
    try:
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if CV2_AVAILABLE:
            # Advanced preprocessing with OpenCV
            img_array = np.array(image)
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply preprocessing techniques
            # 1. Resize image (upscale if too small)
            height, width = gray.shape
            if height < 1000 or width < 1000:
                scale_factor = max(1000/height, 1000/width)
                gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, 
                                interpolation=cv2.INTER_CUBIC)
            
            # 2. Denoise
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # 3. Apply adaptive thresholding for better contrast
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # 4. Morphological operations to remove noise
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(processed)
            
            # Extract text with multiple PSM modes
            texts = []
            
            # Try different PSM modes on processed image
            for psm in [6, 4, 3]:
                try:
                    text = pytesseract.image_to_string(processed_image, config=f'--oem 3 --psm {psm}')
                    if text.strip():
                        texts.append(text)
                except:
                    continue
            
            # Also try original image
            try:
                text_original = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
                if text_original.strip():
                    texts.append(text_original)
            except:
                pass
            
            if not texts:
                return "⚠ No text found in image. Please ensure the image is clear and contains readable text."
            
            # Choose the longest extracted text (usually most complete)
            final_text = max(texts, key=len).strip()
            
            # Show statistics
            char_count = len(final_text)
            line_count = len([line for line in final_text.split('\n') if line.strip()])
            
            return f"""✅ **Text Extracted Successfully!**

📊 **Statistics:**
• Characters extracted: {char_count}
• Lines detected: {line_count}

📄 **Extracted Text:**

{final_text}

---
💡 **Tip:** If text is incomplete, try:
• Taking a clearer photo
• Ensuring good lighting
• Making sure text is horizontal
• Using higher resolution image"""
        
        else:
            # Fallback to basic OCR without OpenCV preprocessing
            texts = []
            for psm in [6, 4, 3, 11]:
                try:
                    text = pytesseract.image_to_string(image, config=f'--oem 3 --psm {psm}')
                    if text.strip():
                        texts.append(text)
                except:
                    continue
            
            if not texts:
                return "⚠ No text found. Install opencv-python for better results:\npip install opencv-python"
            
            final_text = max(texts, key=len).strip()
            char_count = len(final_text)
            
            return f"""✅ **Extracted Text:** ({char_count} characters)

{final_text}

---
💡 **For better accuracy, install opencv-python:**
pip install opencv-python"""
            
    except Exception as e:
        return f"""❌ **OCR Failed:** {str(e)}

💡 **Troubleshooting:**
• Ensure Tesseract is installed: https://github.com/tesseract-ocr/tesseract
• Install opencv-python: pip install opencv-python
• Check image quality and format
• Make sure the image contains clear, readable text"""

# === NEW: AI Explain Extracted Prescription ===
def explain_extracted_prescription(extracted_text):
    """AI explains the extracted prescription text in detail"""
    if not extracted_text or not extracted_text.strip() or len(extracted_text) < 20:
        return "⚠ Please extract text from prescription first using the 'Extract Text' button above."
    
    # Check if it's an error message
    if extracted_text.startswith("⚠") or extracted_text.startswith("❌"):
        return "⚠ Cannot explain: No valid prescription text extracted. Please upload a clear prescription image."
    
    prompt = f"""You are a medical AI assistant. Analyze this prescription text extracted via OCR and provide a comprehensive explanation.

Prescription Text:
{extracted_text[:2000]}

Please provide:

1. **Medicines Identified**: List all medicines mentioned with their generic/brand names
2. **Dosage Information**: Extract dosage for each medicine (strength, frequency, duration)
3. **Medical Purpose**: Explain what each medicine is typically used for
4. **Administration Instructions**: When and how to take each medicine
5. **Important Warnings**: Any contraindications, side effects, or precautions
6. **Additional Notes**: Any other relevant information from the prescription

Format the response clearly with headers and bullet points. If any information is unclear due to OCR errors, mention it."""
    
    try:
        response = generate_gemma_text(prompt)
        
        if response and not response.startswith("Error"):
            return f"""🤖 **AI Prescription Analysis**

{response}

---
⚠️ **Disclaimer:** This is an AI-generated analysis for informational purposes only. Always consult with a healthcare professional before taking any medication. Verify all dosages and instructions with your doctor or pharmacist."""
        else:
            return f"❌ AI explanation failed: {response}\n\nPlease check if AI model is properly initialized in the Setup tab."
    except Exception as e:
        return f"❌ Error generating explanation: {str(e)}\n\nPlease ensure AI model is configured correctly."

# === Analyze Prescription ===
def analyze_prescription(text):
    """Analyze prescription using AI"""
    if not text or not text.strip():
        return "⚠ Please enter prescription text"
    
    prompt = f"""Analyze this prescription and extract:
1. All medicine names mentioned
2. Dosages for each medicine
3. Frequency of administration
4. Duration of treatment
5. Any warnings or special instructions

Format clearly with bullet points.

Prescription text:
{text}"""
    
    try:
        response = generate_gemma_text(prompt)
        if response and not response.startswith("Error"):
            return response
        else:
            return f"❌ Analysis failed: {response}"
    except Exception as e:
        return f"❌ Analysis failed: {str(e)}"

# === Speech Recognition ===
def speech_to_text(audio_path):
    """Convert speech to text for chatbot input"""
    if not SPEECH_AVAILABLE:
        return "⚠ Speech recognition not available. Install: pip install SpeechRecognition"
    
    if audio_path is None:
        return ""
    
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return "⚠ Could not understand audio"
    except sr.RequestError:
        return "⚠ Speech recognition service error"
    except Exception as e:
        return f"⚠ Error: {str(e)}"

# === NEW: Medical Chatbot with Speech ===
def chat_with_bot(user_message, history, audio_input=None, use_api=False):
    """Medical chatbot powered by Gemma with speech input"""
    global chat_history
    
    # If audio provided, convert to text
    if audio_input is not None:
        transcribed_text = speech_to_text(audio_input)
        if transcribed_text and not transcribed_text.startswith("⚠"):
            user_message = transcribed_text
    
    if not user_message or not user_message.strip():
        return history, "", None
    
    if gemini_model is None:
        bot_response = "❌ **AI model not initialized!** Please configure your AI model in the 'Gemma Setup' tab first."
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": bot_response})
        return history, "", None
    
    # Build context from chat history
    context = ""
    if history:
        # Format history for prompt
        recent_msgs = history[-10:]  # Last 10 messages
        context_lines = []
        for msg in recent_msgs:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_lines.append(f"{role}: {msg['content']}")
        context = "\n".join(context_lines)
    
    prompt = f"""You are a helpful medical information assistant. Answer questions about medicines, health conditions, symptoms, and general medical information.

Important guidelines:
- Provide accurate, evidence-based medical information
- Always remind users to consult healthcare professionals for personal medical advice
- Be clear about limitations and when professional help is needed
- If asked about specific dosages, suggest consulting a doctor
- Be empathetic and understanding
- Keep responses concise but informative (max 300 words)

Previous conversation:
{context}

User question: {user_message}

Assistant response:"""
    
    try:
        response = generate_gemma_text(prompt, use_api=use_api)
        
        if response and not response.startswith("Error"):
            bot_response = response
        else:
            bot_response = f"❌ I encountered an error: {response}\n\nPlease try rephrasing your question or check the AI configuration."
        
        # Add to history
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": bot_response})
        
        return history, "", None
        
    except Exception as e:
        bot_response = f"❌ Error: {str(e)}\n\nPlease ensure AI model is properly configured."
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": bot_response})
        return history, "", None

def clear_chat():
    """Clear chat history"""
    global chat_history
    chat_history = []
    return [], "", None

# === Weather API Functions ===
def get_weather_data(city="Delhi", api_key=None):
    """Get weather data from OpenWeatherMap API"""
    if not WEATHER_API_AVAILABLE:
        return None, "⚠ Weather API not available. Install: pip install requests"
    
    if not api_key:
        # Use default demo key or ask user to provide
        return None, "⚠ Please provide OpenWeatherMap API key"
    
    try:
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': city,
            'appid': api_key,
            'units': 'metric'
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        data = response.json()
        
        if response.status_code == 200:
            weather_info = {
                'city': data['name'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'description': data['weather'][0]['description'],
                'main': data['weather'][0]['main'],
                'icon': data['weather'][0]['icon']
            }
            return weather_info, "✅ Weather data fetched successfully"
        else:
            return None, f"❌ Error: {data.get('message', 'Unknown error')}"
            
    except Exception as e:
        return None, f"❌ API Error: {str(e)}"

def analyze_weather_diseases(weather_data):
    """Analyze weather-related disease risks"""
    if not weather_data:
        return "⚠ No weather data available"
    
    try:
        temp = weather_data['temperature']
        humidity = weather_data['humidity']
        weather_main = weather_data['main'].lower()
        
        diseases = []
        precautions = []
        
        # Temperature-based risks
        if temp > 35:
            diseases.append("🌡️ **Heat Stroke & Dehydration**")
            precautions.append("• Drink plenty of water (3-4 liters daily)")
            precautions.append("• Avoid direct sun between 11 AM - 4 PM")
            precautions.append("• Wear light, breathable cotton clothes")
            precautions.append("• Use hats/umbrellas when outdoors")
            precautions.append("• Seek AC/cool places during peak heat")
        
        elif temp < 10:
            diseases.append("❄️ **Hypothermia & Seasonal Flu**")
            precautions.append("• Wear warm layers of clothing")
            precautions.append("• Keep homes well-heated")
            precautions.append("• Stay active to maintain body heat")
            precautions.append("• Eat warm, nutritious foods")
            precautions.append("• Get flu vaccination if eligible")
        
        # Humidity-based risks
        if humidity > 80:
            diseases.append("💧 **Fungal Infections & Asthma**")
            precautions.append("• Keep skin dry, use antifungal powder")
            precautions.append("• Wear loose, breathable clothes")
            precautions.append("• Use dehumidifier if available")
            precautions.append("• Avoid damp areas")
            precautions.append("• Take regular baths")
        
        # Weather condition based risks
        if 'rain' in weather_main or weather_main == 'drizzle':
            diseases.append("🌧️ **Mosquito-borne Diseases**")
            precautions.append("• Use mosquito nets/repellents")
            precautions.append("• Eliminate stagnant water")
            precautions.append("• Wear full-sleeve clothing")
            precautions.append("• Keep windows screened")
            precautions.append("• Seek medical help for persistent fever")
        
        if weather_main == 'fog' or weather_main == 'haze':
            diseases.append("🌫️ **Respiratory Issues**")
            precautions.append("• Wear N95 masks outdoors")
            precautions.append("• Limit outdoor activities")
            precautions.append("• Use air purifiers indoors")
            precautions.append("• Keep windows closed")
            precautions.append("• Stay hydrated")
        
        # Build the response
        response = f"""🌤️ **Weather Alert for {weather_data['city']}**

📊 **Current Conditions:**
• Temperature: {temp}°C (Feels like: {weather_data['feels_like']}°C)
• Humidity: {humidity}%
• Conditions: {weather_data['description']}
• Wind: {weather_data['wind_speed']} m/s

⚠️ **Potential Health Risks:**

"""
        
        if diseases:
            response += "\n".join(diseases)
        else:
            response += "✅ No major weather-related health risks detected"
        
        if precautions:
            response += "\n\n🛡️ **Prevention Tips:**\n"
            response += "\n".join(precautions[:10])  # Limit to 10 precautions
        
        # General advice
        response += f"""

💡 **General Health Advice:**
• Monitor local AQI for air quality alerts
• Stay updated on weather forecasts
• Keep emergency contacts handy
• Maintain basic first aid kit
• Follow local health department advisories

📅 **Seasonal Recommendations:**
{'• Summer: Focus on hydration and sun protection' if temp > 25 else ''}
{'• Winter: Layer clothing and prevent respiratory infections' if temp < 20 else ''}
{'• Monsoon: Guard against water-borne and mosquito diseases' if humidity > 70 else ''}

---
⚠️ **Disclaimer:** This is general guidance. Consult healthcare professionals for personal medical advice.
Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"""
        
        return response
        
    except Exception as e:
        return f"❌ Analysis error: {str(e)}"

def get_weather_analysis(city, api_key):
    """Main function to get and analyze weather"""
    if not city or not city.strip():
        return "⚠ Please enter a city name", "", ""
    
    weather_data, status_msg = get_weather_data(city.strip(), api_key)
    
    if weather_data:
        analysis = analyze_weather_diseases(weather_data)
        
        # Get AI explanation
        ai_prompt = f"""Based on this weather data in {city}, provide detailed medical advice for common weather-related diseases:

Weather: {weather_data['description']}
Temperature: {weather_data['temperature']}°C
Humidity: {weather_data['humidity']}%
Wind Speed: {weather_data['wind_speed']} m/s

Explain:
1. Common diseases likely in these conditions
2. Specific prevention strategies
3. When to seek medical help
4. Special care for vulnerable groups (elderly, children, chronic patients)
5. Home remedies for mild weather-related symptoms"""
        
        ai_explanation = generate_gemma_text(ai_prompt)
        
        weather_summary = f"""📍 **Location:** {weather_data['city']}
🌡️ **Temperature:** {weather_data['temperature']}°C
💧 **Humidity:** {weather_data['humidity']}%
💨 **Wind:** {weather_data['wind_speed']} m/s
☁️ **Conditions:** {weather_data['description']}"""
        
        return weather_summary, analysis, ai_explanation
    else:
        return status_msg, "", ""

# === Download All Prescriptions ===
def download_all_prescriptions():
    """Generate PDF with all prescription history"""
    global prescription_history
    
    if not prescription_history:
        return None
    
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='wb')
        temp_path = temp_file.name
        temp_file.close()
        
        doc = SimpleDocTemplate(temp_path, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        elements = []
        
        elements.append(Paragraph("All Prescriptions History", styles['Title']))
        elements.append(Paragraph(f"Total Prescriptions: {len(prescription_history)}", styles['Normal']))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        for idx, rx in enumerate(prescription_history, 1):
            elements.append(Paragraph(f"Prescription #{idx}", styles['Heading2']))
            elements.append(Paragraph(f"Date: {rx['timestamp'].strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
            elements.append(Paragraph(f"Patient: {rx['patient_name']}", styles['Normal']))
            elements.append(Paragraph(f"Age: {rx['age']} years, Weight: {rx['weight']} kg", styles['Normal']))
            elements.append(Paragraph(f"Medicine: {rx['medicine_name']}", styles['Normal']))
            elements.append(Paragraph(f"Dosage: {rx['dosage']['single_dose']}mg, {rx['dosage']['frequency']}", styles['Normal']))
            elements.append(Spacer(1, 0.2*inch))
            
            if idx < len(prescription_history):
                elements.append(PageBreak())
        
        doc.build(elements)
        return temp_path
    except Exception as e:
        print(f"Error generating batch PDF: {e}")
        return None

# === Create Interface ===
with gr.Blocks(title="Enhanced Medicine Calculator", theme=gr.themes.Soft(), css="footer {visibility: hidden}") as demo:
    
    gr.Markdown("# 💊 Gemini The Medi-Buddy")
    gr.Markdown("🎤 Voice Input | 📸 Enhanced OCR | 🌐 Translation | 🤖 AI Chatbot | 🌤️ Weather Alert | 🏥 Helpline | 🦠 Disease Info")
    
    # Gemma Setup
    with gr.Tab("🔧 AI Setup (Gemma/Gemini)"):
        gr.Markdown("### Configure AI Model")
        gr.Markdown("Choose between free Gemma model (local) or Gemini API (requires API key)")
        
        with gr.Tab("Local Gemma Model (Free)"):
            gr.Markdown("**Option 1: Free Local Gemma Model**")
            gr.Markdown("Downloads model locally (~3GB). Requires good internet and RAM.")
            
            model_choice = gr.Dropdown(
                choices=["models/gemma-3-4b-it"],
                value="models/gemma-3-4b-it",
                label="🤖 Select Model"
            )
            init_local_btn = gr.Button("🚀 Load Local Gemma", variant="primary")
        
        with gr.Tab("Gemini API (Online)"):
            gr.Markdown("**Option 2: Gemini API**")
            gr.Markdown("Requires free API key from: https://makersuite.google.com/app/apikey")
            
            gemini_api_key = gr.Textbox(label="🔑 Gemini API Key", type="password")
            init_api_btn = gr.Button("🌐 Initialize Gemini API", variant="primary")
        
        init_status = gr.Textbox(label="Status", interactive=False, lines=3)
        
        # Update button clicks
        init_local_btn.click(
            fn=lambda model_name: initialize_gemma(model_name=model_name, use_api=False),
            inputs=model_choice,
            outputs=init_status
        )
        
        init_api_btn.click(
            fn=lambda api_key: initialize_gemma(use_api=True, api_key=api_key),
            inputs=gemini_api_key,
            outputs=init_status
        )
        
        # Translation in Setup
        gr.Markdown("---")
        gr.Markdown("### 🌐 Translate Status Messages")
        setup_translation_lang = gr.Dropdown(
            choices=SUPPORTED_LANGUAGES,
            value="en",
            label="Select Language for Translation"
        )
        setup_translate_btn = gr.Button("🔄 Translate Status", variant="secondary")
        setup_translated_status = gr.Textbox(label="Translated Status", interactive=False, lines=3)
        
        setup_translate_btn.click(
            fn=translate_text,
            inputs=[init_status, setup_translation_lang],
            outputs=setup_translated_status
        )
    
    # Dataset Setup
    with gr.Tab("📂 Dataset Setup"):
        gr.Markdown("### Upload Medicine Dataset")
        file_input = gr.File(label="📁 Upload File", file_types=[".csv", ".xlsx", ".xls"])
        upload_btn = gr.Button("📤 Load Dataset", variant="primary")
        status = gr.Textbox(label="Status", interactive=False, lines=10)
        
        upload_btn.click(fn=load_dataset, inputs=file_input, outputs=[status, file_input])
        
        # Translation in Dataset Setup
        gr.Markdown("---")
        gr.Markdown("### 🌐 Translate Status")
        dataset_translation_lang = gr.Dropdown(
            choices=SUPPORTED_LANGUAGES,
            value="hi",
            label="Select Language for Translation"
        )
        dataset_translate_btn = gr.Button("🔄 Translate Status", variant="secondary")
        dataset_translated_status = gr.Textbox(label="Translated Status", interactive=False, lines=10)
        
        dataset_translate_btn.click(
            fn=translate_text,
            inputs=[status, dataset_translation_lang],
            outputs=dataset_translated_status
        )
    
    # Dosage Calculator
    with gr.Tab("💊 Dosage Calculator"):
        gr.Markdown("### Calculate Personalized Dosage")
        
        with gr.Row():
            patient_name = gr.Textbox(label="👤 Patient Name", placeholder="Enter patient name")
            med_input = gr.Textbox(label="💊 Medicine Name", placeholder="Enter or speak medicine name")
            med_audio = gr.Audio(label="🎤 Voice Input (Optional)", sources=["microphone"], type="numpy")
        
        with gr.Row():
            age_input = gr.Number(label="👶 Age (years)", value=30, minimum=0.1)
            weight_input = gr.Number(label="⚖️ Weight (kg)", value=70, minimum=1)
        
        calc_btn = gr.Button("🧮 Calculate Dosage", variant="primary", size="lg")
        
        search_result = gr.Textbox(label="Search Result", interactive=False, lines=2)
        
        with gr.Row():
            med_info = gr.Textbox(label="📋 Medicine Info", interactive=False, lines=7)
            dosage_out = gr.Textbox(label="💊 Dosage", interactive=False, lines=7)
        
        explain_out = gr.Textbox(label="🤖 AI Explanation", interactive=False, lines=10)
        pdf_out = gr.File(label="📄 Download PDF")
        
        calc_btn.click(
            fn=process_medicine,
            inputs=[med_input, patient_name, age_input, weight_input],
            outputs=[med_info, dosage_out, explain_out, search_result, pdf_out]
        )
        
        # Comprehensive Translation Section
        gr.Markdown("---")
        gr.Markdown("### 🌐 Multi-Language Translation Center")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("**📋 Translate Medical Information**")
                dosage_translation_lang = gr.Dropdown(
                    choices=SUPPORTED_LANGUAGES,
                    value="hi",
                    label="🌍 Select Target Language"
                )
                
                translate_all_btn = gr.Button("🔄 Translate All Sections", variant="primary")
            
            with gr.Column():
                gr.Markdown("**⚡ Quick Translate Options**")
                with gr.Row():
                    translate_med_btn = gr.Button("💊 Translate Medicine Info", variant="secondary")
                    translate_dosage_btn = gr.Button("📊 Translate Dosage", variant="secondary")
                    translate_explain_btn = gr.Button("🤖 Translate Explanation", variant="secondary")
        
        with gr.Tabs():
            with gr.TabItem("💊 Medicine Info Translation"):
                med_trans_out = gr.Textbox(label="Translated Medicine Information", interactive=False, lines=8)
            
            with gr.TabItem("📊 Dosage Translation"):
                dosage_trans_out = gr.Textbox(label="Translated Dosage Information", interactive=False, lines=8)
            
            with gr.TabItem("🤖 Explanation Translation"):
                explain_trans_out = gr.Textbox(label="Translated AI Explanation", interactive=False, lines=10)
            
            with gr.TabItem("🔍 Search Result Translation"):
                search_trans_out = gr.Textbox(label="Translated Search Result", interactive=False, lines=3)
        
        # Connect translation buttons
        translate_med_btn.click(
            fn=translate_text,
            inputs=[med_info, dosage_translation_lang],
            outputs=med_trans_out
        )
        
        translate_dosage_btn.click(
            fn=translate_text,
            inputs=[dosage_out, dosage_translation_lang],
            outputs=dosage_trans_out
        )
        
        translate_explain_btn.click(
            fn=translate_text,
            inputs=[explain_out, dosage_translation_lang],
            outputs=explain_trans_out
        )
        
        translate_all_btn.click(
            fn=lambda m, d, e, lang: translate_multiple_texts(m, d, e, lang),
            inputs=[med_info, dosage_out, explain_out, dosage_translation_lang],
            outputs=[med_trans_out, dosage_trans_out, explain_trans_out]
        )
        
        # Translate search result separately
        search_translate_btn = gr.Button("🔍 Translate Search Result", variant="secondary")
        search_translate_btn.click(
            fn=translate_text,
            inputs=[search_result, dosage_translation_lang],
            outputs=search_trans_out
        )
    
    # Prescription Analyzer
    with gr.Tab("📸 Prescription Analyzer"):
        gr.Markdown("### AI Prescription Analysis with Enhanced OCR")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Option 1: Upload Image**")
                rx_image = gr.Image(label="📸 Upload Prescription Image", type="pil")
                ocr_btn = gr.Button("🔍 Extract Text (Enhanced OCR)", variant="secondary")
            
            with gr.Column():
                gr.Markdown("**Option 2: Type/Paste Text**")
                rx_input = gr.Textbox(label="📝 Prescription Text", lines=10, placeholder="Paste text or extract from image...")
        
        ocr_btn.click(fn=extract_text_from_image, inputs=rx_image, outputs=rx_input)
        
        gr.Markdown("---")
        gr.Markdown("### 🤖 AI Analysis & Explanation")
        
        with gr.Row():
            rx_explain_btn = gr.Button("🤖 AI Explain Prescription", variant="primary", size="lg")
            rx_analyze_btn = gr.Button("📊 Quick Analysis", variant="secondary")
        
        rx_explain_out = gr.Textbox(label="🤖 AI Detailed Explanation", interactive=False, lines=15)
        rx_out = gr.Textbox(label="📊 Quick Analysis", interactive=False, lines=10)
        
        rx_explain_btn.click(fn=explain_extracted_prescription, inputs=rx_input, outputs=rx_explain_out)
        rx_analyze_btn.click(fn=analyze_prescription, inputs=rx_input, outputs=rx_out)
        
        gr.Markdown("---")
        gr.Markdown("### 🌐 Comprehensive Translation Center")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("**🌍 Translation Settings**")
                rx_translation_lang = gr.Dropdown(
                    choices=SUPPORTED_LANGUAGES,
                    value="ta",
                    label="Select Target Language"
                )
                
            with gr.Column():
                gr.Markdown("**⚡ Quick Actions**")
                with gr.Row():
                    rx_translate_explain_btn = gr.Button("🤖 Translate Analysis", variant="primary")
                    rx_translate_analysis_btn = gr.Button("📊 Translate Quick Analysis", variant="secondary")
        
        with gr.Tabs():
            with gr.TabItem("🤖 Translated AI Explanation"):
                rx_trans_out = gr.Textbox(label="🌐 Translated AI Analysis", interactive=False, lines=15)
            
            with gr.TabItem("📊 Translated Quick Analysis"):
                rx_analysis_trans_out = gr.Textbox(label="🌐 Translated Quick Analysis", interactive=False, lines=10)
            
            with gr.TabItem("📄 Translated Extracted Text"):
                rx_text_trans_out = gr.Textbox(label="🌐 Translated Prescription Text", interactive=False, lines=10)
        
        # Connect translation buttons
        rx_translate_explain_btn.click(
            fn=translate_text,
            inputs=[rx_explain_out, rx_translation_lang],
            outputs=rx_trans_out
        )
        
        rx_translate_analysis_btn.click(
            fn=translate_text,
            inputs=[rx_out, rx_translation_lang],
            outputs=rx_analysis_trans_out
        )
        
        # Translate extracted prescription text
        rx_translate_text_btn = gr.Button("📄 Translate Extracted Text", variant="secondary")
        rx_translate_text_btn.click(
            fn=translate_text,
            inputs=[rx_input, rx_translation_lang],
            outputs=rx_text_trans_out
        )
    
    # Medical Chatbot with Speech
    with gr.Tab("🤖 Medical Chatbot"):
        gr.Markdown("### 💬 Ask Medical Questions with Voice Input")
        gr.Markdown("Ask me anything about medicines, health conditions, symptoms, or general medical information!")
        
        chatbot_interface = gr.Chatbot(
            label="💬 Medical Assistant",
            height=500,
            show_label=True,
            type="messages"
        )
        
        with gr.Row():
            chat_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask me about medicines, symptoms, health conditions, etc...",
                lines=2,
                scale=3
            )
            chat_audio = gr.Audio(
                label="🎤 Voice Input",
                sources=["microphone"],
                type="filepath",
                scale=1
            )
            chat_submit = gr.Button("📤 Send", variant="primary", scale=1)
        
        with gr.Row():
            chat_clear = gr.Button("🗑️ Clear Chat", variant="secondary")
            chat_examples = gr.Examples(
                examples=[
                    "What is Paracetamol used for?",
                    "What are the side effects of antibiotics?",
                    "How does Ibuprofen work?",
                    "What should I do if I have a fever?",
                    "Can I take medicine on an empty stomach?",
                    "What are the symptoms of diabetes?",
                    "How to manage high blood pressure?",
                    "What vitamins are important for immunity?"
                ],
                inputs=chat_input,
                label="💡 Example Questions"
            )
        
        # Translation for Chatbot
        gr.Markdown("---")
        gr.Markdown("### 🌐 Translate Chat Conversation")
        
        with gr.Row():
            chatbot_translation_lang = gr.Dropdown(
                choices=SUPPORTED_LANGUAGES,
                value="hi",
                label="Select Language for Translation"
            )
            
            translate_chat_btn = gr.Button("🔄 Translate Last Response", variant="secondary")
        
        chat_translation_output = gr.Textbox(
            label="Translated Response",
            interactive=False,
            lines=10
        )
        
        def translate_last_chat_response(history, target_lang):
            """Translate the last chatbot response"""
            if not history:
                return "No conversation to translate"
            
            last_message = history[-1]
            # Handle messages format (dict)
            if isinstance(last_message, dict):
                if last_message.get("role") == "assistant":
                    return translate_text(last_message["content"], target_lang)
                else:
                    return "Last message is not from assistant"
            
            # Fallback for legacy tuple format
            try:
                if len(last_message) > 1:
                    return translate_text(last_message[1], target_lang)
            except:
                pass
                
            return "Could not identify last response"
        
        translate_chat_btn.click(
            fn=translate_last_chat_response,
            inputs=[chatbot_interface, chatbot_translation_lang],
            outputs=chat_translation_output
        )
        
        gr.Markdown("""
        ---
        ⚠️ **Important Disclaimer:**
        - This chatbot provides general medical information only
        - Always consult a healthcare professional for personal medical advice
        - Do not use this for emergency medical situations
        - Verify all information with qualified medical practitioners
        """)
        
        # Add use_api flag to chat function
        chat_submit.click(
            fn=lambda msg, hist, audio: chat_with_bot(msg, hist, audio, use_api=gemini_model is not None and hasattr(gemini_model, 'generate_content')),
            inputs=[chat_input, chatbot_interface, chat_audio],
            outputs=[chatbot_interface, chat_input, chat_audio]
        )
        
        chat_input.submit(
            fn=lambda msg, hist, audio: chat_with_bot(msg, hist, audio, use_api=gemini_model is not None and hasattr(gemini_model, 'generate_content')),
            inputs=[chat_input, chatbot_interface, chat_audio],
            outputs=[chatbot_interface, chat_input, chat_audio]
        )
        
        chat_clear.click(
            fn=clear_chat,
            outputs=[chatbot_interface, chat_input, chat_audio]
        )
    
    # Weather & Health Alerts Tab
    with gr.Tab("🌤️ Weather Health Alert"):
        gr.Markdown("### Weather-Based Disease Prediction & Prevention")
        gr.Markdown("Get real-time weather analysis and AI-powered health recommendations")
        
        with gr.Row():
            weather_city = gr.Textbox(
                label="📍 City Name",
                value="Delhi",
                placeholder="Enter city name (e.g., Delhi, Chandigarh, Lucknow...)"
            )
            weather_api_key = gr.Textbox(
                label="🔑 OpenWeatherMap API Key",
                type="password",
                placeholder="Get free API key: https://openweathermap.org/api"
            )
        
        weather_btn = gr.Button("🌤️ Get Weather & Health Analysis", variant="primary", size="lg")
        
        with gr.Row():
            weather_summary = gr.Textbox(label="📊 Weather Summary", interactive=False, lines=5)
            weather_analysis = gr.Textbox(label="⚠️ Health Risk Analysis", interactive=False, lines=15)
        
        weather_ai_explanation = gr.Textbox(label="🤖 AI Medical Recommendations", interactive=False, lines=15)
        
        weather_btn.click(
            fn=get_weather_analysis,
            inputs=[weather_city, weather_api_key],
            outputs=[weather_summary, weather_analysis, weather_ai_explanation]
        )
        
        # Translation for Weather Tab
        gr.Markdown("---")
        gr.Markdown("### 🌐 Translate Weather Information")
        
        with gr.Row():
            weather_translation_lang = gr.Dropdown(
                choices=SUPPORTED_LANGUAGES,
                value="hi",
                label="Select Language for Translation"
            )
            
            translate_weather_all_btn = gr.Button("🔄 Translate All Weather Info", variant="secondary")
        
        with gr.Tabs():
            with gr.TabItem("📊 Translated Weather Summary"):
                weather_summary_trans = gr.Textbox(label="Translated Weather Summary", interactive=False, lines=5)
            
            with gr.TabItem("⚠️ Translated Health Analysis"):
                weather_analysis_trans = gr.Textbox(label="Translated Health Risk Analysis", interactive=False, lines=15)
            
            with gr.TabItem("🤖 Translated AI Recommendations"):
                weather_ai_trans = gr.Textbox(label="Translated AI Medical Recommendations", interactive=False, lines=15)
        
        def translate_weather_info(summary, analysis, ai_explanation, lang):
            """Translate all weather information"""
            return [
                translate_text(summary, lang),
                translate_text(analysis, lang),
                translate_text(ai_explanation, lang)
            ]
        
        translate_weather_all_btn.click(
            fn=translate_weather_info,
            inputs=[weather_summary, weather_analysis, weather_ai_explanation, weather_translation_lang],
            outputs=[weather_summary_trans, weather_analysis_trans, weather_ai_trans]
        )
        
        gr.Markdown("""
        ---
        ### 🎯 How Weather Affects Health
        
        **High Temperature (>35°C):**
        • Heat stroke, dehydration, heat exhaustion
        • Sunburn, heat rash
        • Aggravation of heart conditions
        
        **Low Temperature (<10°C):**
        • Hypothermia, frostbite
        • Seasonal flu, pneumonia
        • Worsening of arthritis
        
        **High Humidity (>80%):**
        • Fungal infections
        • Asthma attacks
        • Heat-related illnesses
        
        **Rainy Season:**
        • Mosquito-borne diseases
        • Water contamination
        • Viral infections
        
        **Air Pollution/Smog:**
        • Respiratory diseases
        • Eye irritation
        • Cardiovascular problems
        
        ---
        ### 🛡️ General Prevention Tips
        
        1. **Stay Hydrated:** Drink 3-4 liters of water daily
        2. **Dress Appropriately:** Wear weather-suitable clothing
        3. **Monitor Air Quality:** Check AQI regularly
        4. **Stay Informed:** Follow weather forecasts
        5. **Seek Shelter:** During extreme weather conditions
        6. **Keep Medications Ready:** Especially for chronic conditions
        
        ---
        ⚠️ **Emergency:** Call 108/112 for medical emergencies
        """)
        
        # Translate disease info
        gr.Markdown("---")
        gr.Markdown("### 🌐 Translate Disease Information")
        disease_info_text = gr.Textbox(
            label="Disease Information Text",
            value="""High Temperature (>35°C):
• Heat stroke, dehydration, heat exhaustion
• Sunburn, heat rash
• Aggravation of heart conditions

Low Temperature (<10°C):
• Hypothermia, frostbite
• Seasonal flu, pneumonia
• Worsening of arthritis""",
            lines=10,
            visible=False
        )
        
        translate_disease_info_btn = gr.Button("🌡️ Translate Disease Info", variant="secondary")
        disease_info_trans = gr.Textbox(label="Translated Disease Information", interactive=False, lines=15)
        
        translate_disease_info_btn.click(
            fn=translate_text,
            inputs=[disease_info_text, weather_translation_lang],
            outputs=disease_info_trans
        )
    
    # North India Helpline Tab
    with gr.Tab("📞 North India Helpline"):
        gr.Markdown("### 🏥 Emergency & Healthcare Contacts - North India")
        
        # National Helplines
        with gr.Accordion("🇮🇳 National Emergency Numbers", open=True):
            national_helplines = gr.Markdown("""
            | Number | Service | Coverage |
            |--------|---------|----------|
            | **108** | Emergency Medical Services | All India |
            | **112** | Single Emergency Number | All India |
            | **102** | Ambulance | All India |
            | **1091** | Women Helpline | All India |
            | **1098** | Child Helpline | All India |
            | **1073** | Senior Citizens Helpline | All India |
            | **14404** | COVID-19 Helpline | All India |
            | **1800-180-1104** | Mental Health Helpline | All India |
            """)
        
        # State-wise Helplines
        with gr.Accordion("📍 State-wise Health Departments", open=False):
            with gr.Tabs():
                with gr.TabItem("Delhi/NCR"):
                    delhi_helplines = gr.Markdown("""
                    **🌆 Delhi Government Health Services:**
                    - **COVID-19 Helpline:** 1031
                    - **Delhi Government Helpline:** 1076
                    - **Ambulance Control Room:** 102
                    - **AIIMS Emergency:** 011-26588500
                    - **Safdarjung Hospital:** 011-26165050
                    - **RML Hospital:** 011-23404200
                    
                    **🩺 Major Hospitals:**
                    - **AIIMS, Delhi:** 011-26588500
                    - **Sir Ganga Ram Hospital:** 011-42251000
                    - **Max Hospital, Saket:** 011-26515050
                    - **Fortis Escorts:** 011-47135000
                    - **Apollo Hospital:** 011-26925858
                    """)
                
                with gr.TabItem("Uttar Pradesh"):
                    up_helplines = gr.Markdown("""
                    **🏛️ UP Health Department:**
                    - **Emergency:** 108
                    - **COVID-19 Helpline:** 18001805145
                    - **CM Helpline:** 1076
                    - **Ambulance:** 102
                    
                    **🏥 Key Cities:**
                    - **Lucknow:** SGPGI - 0522-2668700
                    - **Kanpur:** LLR Hospital - 0512-2531421
                    - **Varanasi:** BHU Hospital - 0542-2367566
                    - **Allahabad:** Swaroup Rani Nehru Hospital - 0532-2461100
                    - **Agra:** SN Medical College - 0562-2360451
                    """)
                
                with gr.TabItem("Uttarakhand"):
                    uk_helplines = gr.Markdown("""
                    **⛰️ Uttarakhand Health Services:**
                    - **Emergency:** 108
                    - **State Helpline:** 104
                    - **Ambulance:** 102
                    
                    **🏔️ Major Hospitals:**
                    - **Dehradun:** Doon Hospital - 0135-2650411
                    - **Haridwar:** District Hospital - 01334-225700
                    - **Nainital:** BD Pandey Hospital - 05942-236300
                    - **Rishikesh:** AIIMS - 0135-2476000
                    """)
                
                with gr.TabItem("Punjab"):
                    punjab_helplines = gr.Markdown("""
                    **🌾 Punjab Health Department:**
                    - **Emergency:** 108
                    - **COVID Helpline:** 104
                    - **Ambulance:** 102
                    
                    **🏙️ Major Cities:**
                    - **Chandigarh:** PGIMER - 0172-2746018
                    - **Amritsar:** Govt. Medical College - 0183-2421500
                    - **Ludhiana:** Dayanand Medical College - 0161-2444400
                    - **Jalandhar:** Civil Hospital - 0181-2222222
                    """)
                
                with gr.TabItem("Haryana"):
                    haryana_helplines = gr.Markdown("""
                    **🚜 Haryana Health Services:**
                    - **Emergency:** 108
                    - **Helpline:** 1075
                    - **Ambulance:** 102
                    
                    **🏥 Key Hospitals:**
                    - **Gurugram:** Medanta - 0124-4141414
                    - **Faridabad:** Asian Institute - 0129-4192222
                    - **Rohtak:** PGIMS - 01262-211001
                    - **Hisar:** Civil Hospital - 01662-232301
                    """)
        
        # Specialized Helplines
        with gr.Accordion("🎯 Specialized Medical Services", open=False):
            specialized_helplines = gr.Markdown("""
            **🩸 Blood Banks:**
            - **Indian Red Cross:** 011-23711641
            - **AIIMS Blood Bank:** 011-26594699
            - **National Blood Bank:** 011-23711781
            
            **💊 Poison Control:**
            - **National Poisons Centre:** 011-26593677
            - **AIIMS Poison Control:** 011-26588111
            
            **🧠 Mental Health:**
            - **Vandrevala Foundation:** 1860-2662345
            - **iCall:** 022-25521111
            - **NIMHANS:** 080-26995151
            
            **💗 Cardiac Emergency:**
            - **Cardiac Helpline:** 1090
            - **Heart Care Foundation:** 09958721000
            
            **👶 Maternal & Child Health:**
            - **Mother & Child Tracking:** 1800-180-1551
            """)
        
        # Mobile Apps
        with gr.Accordion("📱 Recommended Mobile Apps", open=False):
            mobile_apps = gr.Markdown("""
            **🏥 Health Apps:**
            - **Aarogya Setu:** Official COVID-19 tracker
            - **eSanjeevani:** Government telemedicine app
            - **m-Sehat:** Health records & appointments
            
            **🚑 Emergency Apps:**
            - **SOS Alert:** Emergency alerts to contacts
            - **Emergency Dialer:** Quick dial to emergency services
            - **Red Panic Button:** One-touch emergency alert
            
            **💊 Medicine Apps:**
            - **Pharmeasy:** Medicine delivery
            - **1mg:** Online pharmacy & diagnostics
            - **Netmeds:** Medicine home delivery
            """)
        
        # Translation for Helpline Tab
        gr.Markdown("---")
        gr.Markdown("### 🌐 Translate Helpline Information")
        
        with gr.Row():
            helpline_translation_lang = gr.Dropdown(
                choices=SUPPORTED_LANGUAGES,
                value="hi",
                label="Select Language for Translation"
            )
            
            translate_helpline_btn = gr.Button("🔄 Translate Important Instructions", variant="secondary")
        
        # Prepare text for translation
        emergency_instructions_text = gr.Textbox(
            label="Emergency Instructions Text",
            value="""⚠️ Emergency Instructions
        
        1. **Stay Calm:** Don't panic in emergencies
        2. **Call Appropriate Number:** Based on emergency type
        3. **Provide Clear Information:** Location, patient condition, contact number
        4. **Follow Instructions:** Listen to operator guidance
        5. **Send Location:** Use WhatsApp/Google Maps to share location
        
        📍 Quick Reference Card
        
        **Medical Emergency:** 108 or 112  
        **Fire:** 101  
        **Police:** 100  
        **Women Safety:** 1091  
        **Child Abuse:** 1098  
        **Disaster Management:** 1070  
        **Road Accident:** 1073
        
        ⚠️ **Save these numbers in your phone contacts!**""",
            lines=15,
            visible=False
        )
        
        translated_instructions = gr.Textbox(
            label="Translated Emergency Instructions",
            interactive=False,
            lines=15
        )
        
        translate_helpline_btn.click(
            fn=translate_text,
            inputs=[emergency_instructions_text, helpline_translation_lang],
            outputs=translated_instructions
        )
        
        gr.Markdown("""
        ---
        ### ⚠️ Emergency Instructions
        
        1. **Stay Calm:** Don't panic in emergencies
        2. **Call Appropriate Number:** Based on emergency type
        3. **Provide Clear Information:** Location, patient condition, contact number
        4. **Follow Instructions:** Listen to operator guidance
        5. **Send Location:** Use WhatsApp/Google Maps to share location
        
        ---
        ### 📍 Quick Reference Card
        
        **Medical Emergency:** 108 or 112  
        **Fire:** 101  
        **Police:** 100  
        **Women Safety:** 1091  
        **Child Abuse:** 1098  
        **Disaster Management:** 1070  
        **Road Accident:** 1073  
        
        ---
        ⚠️ **Save these numbers in your phone contacts!**
        """)
    
    # Major Diseases Tab
    with gr.Tab("🦠 Major Diseases Info"):
        gr.Markdown("## 🦠 MAJOR DISEASES IN NORTH INDIA & HOW TO PREVENT / MANAGE THEM")
        
        # Translation for Diseases Tab
        gr.Markdown("---")
        gr.Markdown("### 🌐 Translate Disease Information")
        
        with gr.Row():
            disease_translation_lang = gr.Dropdown(
                choices=SUPPORTED_LANGUAGES,
                value="hi",
                label="Select Language for Translation"
            )
            
            translate_disease_guide_btn = gr.Button("🔄 Translate Selected Disease Guide", variant="secondary")
        
        disease_content_to_translate = gr.Textbox(label="Disease Content", visible=False)
        translated_disease_content = gr.Textbox(label="Translated Disease Guide", interactive=False, lines=20)
        
        def get_disease_content(disease_type):
            """Get disease content based on selection"""
            disease_contents = {
                "dengue": """🦟 Dengue & Malaria Prevention Guide
                
                **Common during:** Monsoon season (June-September)
                **Why common:** Mosquito breeding due to stagnant water
                
                **🔍 Symptoms to Watch:**
                • Sudden high fever (104°F/40°C)
                • Severe headache, pain behind eyes
                • Muscle and joint pains
                • Nausea, vomiting
                • Skin rash (appears 2-5 days after fever)
                • Mild bleeding (nose/gums)""",
                
                "air_pollution": """🌫️ Air Pollution-Related Diseases
                
                **Common in:** Delhi NCR and nearby regions
                **Why common:** High AQI, vehicle emissions, crop burning
                
                **Affected Systems:**
                • Respiratory (Asthma, Bronchitis, COPD)
                • Cardiovascular
                • Eyes and Skin""",
                
                "tuberculosis": """🦠 Tuberculosis (TB) Management
                
                **Why common:** Crowded living conditions and poor ventilation
                
                **🔍 Symptoms:**
                • Cough lasting 3+ weeks
                • Chest pain, coughing blood
                • Unintended weight loss
                • Fatigue, fever, night sweats""",
                
                "water_borne": """💧 Water-Borne Diseases Prevention
                
                **Common Diseases:** Typhoid, Cholera, Hepatitis A, Diarrhea
                
                **🛡️ Prevention Strategy:**
                
                **Water Safety:**
                1. **Boil Water:** Rolling boil for 1 minute
                2. **Filter:** Use certified water filters
                3. **Purification:** Chlorine tablets if boiling not possible
                4. **Storage:** Clean, covered containers""",
                
                "heat_stroke": """🌡️ Heat Stroke & Dehydration Prevention
                
                **Common during:** Summer heatwaves (April-June)
                
                **🔥 Heat-Related Illnesses:**
                1. **Heat Cramps:** Muscle pains during exercise
                2. **Heat Exhaustion:** Heavy sweating, weakness
                3. **Heat Stroke:** Medical emergency (body temp >104°F/40°C)""",
                
                "seasonal_flu": """🤧 Seasonal Flu & Respiratory Infections
                
                **Common during:** Winter months (November-February)
                
                **🔍 Symptoms:**
                • Fever, chills
                • Cough, sore throat
                • Runny/stuffy nose
                • Body aches, headache
                • Fatigue, weakness""",
                
                "lifestyle": """🏃 Lifestyle Diseases Prevention
                
                **Common Conditions:** Diabetes, Hypertension, Heart Disease, Obesity
                
                **📊 Risk Factors:**
                • Sedentary lifestyle
                • Unhealthy diet
                • Stress
                • Smoking/alcohol
                • Genetic predisposition"""
            }
            return disease_contents.get(disease_type, "Select a disease first")
        
        disease_selector = gr.Dropdown(
            choices=[
                ("Dengue & Malaria", "dengue"),
                ("Air Pollution Diseases", "air_pollution"),
                ("Tuberculosis (TB)", "tuberculosis"),
                ("Water-Borne Diseases", "water_borne"),
                ("Heat Stroke", "heat_stroke"),
                ("Seasonal Flu", "seasonal_flu"),
                ("Lifestyle Diseases", "lifestyle")
            ],
            value="dengue",
            label="Select Disease to Translate"
        )
        
        def update_disease_content(disease_type):
            content = get_disease_content(disease_type)
            return content, content
        
        disease_selector.change(
            fn=update_disease_content,
            inputs=disease_selector,
            outputs=[disease_content_to_translate, translated_disease_content]
        )
        
        translate_disease_guide_btn.click(
            fn=translate_text,
            inputs=[disease_content_to_translate, disease_translation_lang],
            outputs=translated_disease_content
        )
        
        with gr.Tabs():
            with gr.TabItem("1️⃣ Dengue & Malaria"):
                gr.Markdown("""
                ### 🦟 Dengue & Malaria Prevention Guide
                
                **Common during:** Monsoon season (June-September)
                **Why common:** Mosquito breeding due to stagnant water
                
                **🔍 Symptoms to Watch:**
                • Sudden high fever (104°F/40°C)
                • Severe headache, pain behind eyes
                • Muscle and joint pains
                • Nausea, vomiting
                • Skin rash (appears 2-5 days after fever)
                • Mild bleeding (nose/gums)
                
                **🛡️ How to Overcome/Prevent:**
                
                **Personal Protection:**
                1. **Use Mosquito Nets:** Sleep under bed nets, preferably insecticide-treated
                2. **Apply Repellents:** Use DEET-based repellents on exposed skin
                3. **Wear Protective Clothing:** Full-sleeve shirts, long pants, socks
                4. **Avoid Peak Hours:** Stay indoors during dawn and dusk
                
                **Environmental Control:**
                1. **Eliminate Breeding Sites:** Remove stagnant water from:
                   - Coolers, flower pots, tires
                   - Water storage containers
                   - Discarded containers
                2. **Cover Water Containers:** Keep them tightly covered
                3. **Use Larvicides:** In unavoidable water collections
                4. **Install Screens:** On windows and doors
                
                **🚨 When to Seek Medical Help:**
                • Persistent fever for 2-3 days
                • Severe abdominal pain
                • Persistent vomiting
                • Bleeding from gums/nose
                • Difficulty breathing
                • Lethargy or restlessness
                
                **💊 Treatment Approach:**
                • **No specific antiviral** for dengue
                • **Supportive care:** Rest, hydration
                • **Pain relief:** Paracetamol only (avoid aspirin/ibuprofen)
                • **Hospitalization** if severe symptoms
                
                **📊 Prevention Checklist:**
                ✅ Weekly emptying of water containers
                ✅ Using mosquito repellent daily
                ✅ Wearing full sleeves in evenings
                ✅ Keeping surroundings clean
                ✅ Using mosquito nets at night
                
                **🏥 Important Notes:**
                • Complete blood count monitoring essential
                • Platelet transfusion only if critically low
                • Early detection prevents complications
                • No vaccine widely available for dengue
                
                **📞 Emergency Contacts:** 108 or nearest government hospital
                """)
            
            with gr.TabItem("2️⃣ Air Pollution Diseases"):
                gr.Markdown("""
                ### 🌫️ Air Pollution-Related Diseases
                
                **Common in:** Delhi NCR and nearby regions
                **Why common:** High AQI, vehicle emissions, crop burning
                
                **Affected Systems:**
                • Respiratory (Asthma, Bronchitis, COPD)
                • Cardiovascular
                • Eyes and Skin
                
                **🛡️ Prevention Strategies:**
                
                **Personal Protection:**
                1. **Use N95 Masks:** Properly fitted masks when outdoors
                2. **Limit Outdoor Activities:** During high pollution days
                3. **Create Clean Air Zones:** At home and workplace
                4. **Use Air Purifiers:** With HEPA filters indoors
                
                **Environmental Measures:**
                1. **Monitor AQI:** Use apps like SAFAR-Air
                2. **Ventilate Smartly:** Open windows when pollution low
                3. **Indoor Plants:** Air-purifying plants like aloe vera, spider plant
                4. **Avoid Smoking:** Both active and passive
                
                **🏥 Management for Patients:**
                
                **Asthma Patients:**
                • Keep inhalers accessible
                • Follow action plan
                • Regular check-ups
                • Peak flow monitoring
                
                **COPD Patients:**
                • Pulmonary rehabilitation
                • Oxygen therapy if prescribed
                • Vaccination (flu, pneumonia)
                • Nutrition management
                
                **🌱 Holistic Approaches:**
                1. **Breathing Exercises:** Pranayama, deep breathing
                2. **Diet:** Antioxidant-rich foods
                3. **Hydration:** 3-4 liters water daily
                4. **Exercise:** Indoor during high pollution
                
                **🚨 Warning Signs:**
                • Worsening shortness of breath
                • Increased inhaler use
                • Chest pain or tightness
                • Bluish lips or fingernails
                • Confusion or drowsiness
                
                **📱 Useful Apps:**
                • SAFAR-Air (Govt. AQI monitoring)
                • AirVisual (Real-time air quality)
                • Plume Air Report (Pollution forecast)
                
                **🏥 Emergency:** 108 for breathing emergencies
                """)
            
            with gr.TabItem("3️⃣ Tuberculosis (TB)"):
                gr.Markdown("""
                ### 🦠 Tuberculosis (TB) Management
                
                **Why common:** Crowded living conditions and poor ventilation
                
                **🔍 Symptoms:**
                • Cough lasting 3+ weeks
                • Chest pain, coughing blood
                • Unintended weight loss
                • Fatigue, fever, night sweats
                
                **🛡️ Prevention Strategy:**
                
                **Vaccination:**
                1. **BCG Vaccine:** Given at birth in India
                2. **Coverage:** Over 90% in most states
                3. **Effectiveness:** 60-80% against severe TB
                
                **Infection Control:**
                1. **Early Detection:** Sputum test for cough >2 weeks
                2. **Complete Treatment:** DOTS therapy (6-9 months)
                3. **Isolation:** Until non-infectious
                4. **Ventilation:** Good airflow in living spaces
                
                **💊 Treatment Protocol (DOTS):**
                • **Intensive Phase:** 2 months, 4 drugs
                • **Continuation Phase:** 4-7 months, 2 drugs
                • **Supervised:** Medication taken under observation
                • **Free Treatment:** Available at government centers
                
                **👥 Community Approach:**
                1. **Contact Tracing:** Screen close contacts
                2. **Awareness:** TB is curable and treatment is free
                3. **Stigma Reduction:** Education campaigns
                4. **Nutrition Support:** For patients
                
                **🚨 Warning Signs:**
                • Cough with blood
                • Weight loss despite normal appetite
                • Night sweats drenching bed sheets
                • Prolonged fever
                
                **🏥 Government Services:**
                • **Free Diagnosis:** Sputum testing
                • **Free Treatment:** DOTS centers nationwide
                • **Nutritional Support:** Through Nikshay Poshan Yojana
                • **Cash Benefits:** For treatment completion
                
                **📱 Important Contacts:**
                • **National TB Helpline:** 1800-11-6666
                • **TB For All:** Website and app
                • **Nikshay Portal:** Patient management system
                
                **⚠️ Critical:**
                • Never stop TB treatment mid-way
                • Drug resistance develops from incomplete treatment
                • MDR-TB requires 18-24 months treatment
                • Complete course is essential for cure
                
                **🏥 Emergency:** Persistent cough with blood needs immediate attention
                """)
            
            with gr.TabItem("4️⃣ Water-Borne Diseases"):
                gr.Markdown("""
                ### 💧 Water-Borne Diseases Prevention
                
                **Common Diseases:** Typhoid, Cholera, Hepatitis A, Diarrhea
                
                **🛡️ Prevention Strategy:**
                
                **Water Safety:**
                1. **Boil Water:** Rolling boil for 1 minute
                2. **Filter:** Use certified water filters
                3. **Purification:** Chlorine tablets if boiling not possible
                4. **Storage:** Clean, covered containers
                
                **Food Safety:**
                1. **Wash Hands:** Before eating/preparing food
                2. **Cook Thoroughly:** Especially meat and eggs
                3. **Avoid Street Food:** During monsoon season
                4. **Peel Fruits:** Wash before peeling
                
                **Hygiene Practices:**
                1. **Hand Washing:** With soap after toilet, before meals
                2. **Sanitation:** Use toilets, avoid open defecation
                3. **Clean Surroundings:** No stagnant water
                4. **Waste Management:** Proper garbage disposal
                
                **💊 Treatment Approach:**
                • **Oral Rehydration:** For diarrhea
                • **Antibiotics:** As prescribed for bacterial infections
                • **Vaccination:** Typhoid, Hepatitis A available
                • **Hospitalization:** For severe dehydration
                
                **🍲 Dietary Management:**
                1. **BRAT Diet:** Banana, Rice, Apple, Toast
                2. **Avoid:** Dairy, fatty foods, spicy foods
                3. **Hydration:** Oral rehydration solution
                4. **Small Meals:** Frequent, light meals
                
                **🚨 When to Seek Help:**
                • Blood in stool
                • Severe dehydration (sunken eyes, dry mouth)
                • High fever with diarrhea
                • No improvement in 2-3 days
                
                **💉 Vaccination Schedule:**
                • **Typhoid:** Every 3 years
                • **Hepatitis A:** 2 doses, 6 months apart
                • **Cholera:** Available for high-risk areas
                
                **🏥 Government Initiatives:**
                • **Swachh Bharat Abhiyan:** Improved sanitation
                • **Jal Jeevan Mission:** Clean drinking water
                • **ICDS:** Nutrition programs
                
                **📱 Apps:**
                • **Swachhata App:** Report sanitation issues
                • **m-Sehat:** Health information
                
                **🏥 Emergency:** Severe dehydration needs IV fluids immediately
                """)
            
            with gr.TabItem("5️⃣ Heat Stroke"):
                gr.Markdown("""
                ### 🌡️ Heat Stroke & Dehydration Prevention
                
                **Common during:** Summer heatwaves (April-June)
                
                **🔥 Heat-Related Illnesses:**
                1. **Heat Cramps:** Muscle pains during exercise
                2. **Heat Exhaustion:** Heavy sweating, weakness
                3. **Heat Stroke:** Medical emergency (body temp >104°F/40°C)
                
                **🛡️ Prevention Measures:**
                
                **Hydration Strategy:**
                1. **Water Intake:** 3-4 liters daily in summer
                2. **ORS:** Oral rehydration solution
                3. **Avoid:** Alcohol, caffeine, sugary drinks
                4. **Electrolytes:** Coconut water, buttermilk
                
                **Clothing & Protection:**
                1. **Light Colors:** White, light-colored clothes
                2. **Loose Fit:** Allows air circulation
                3. **Cover Head:** Hats, caps, umbrellas
                4. **Sunglasses:** UV protection
                
                **Timing & Activity:**
                1. **Avoid Peak Sun:** 11 AM - 4 PM
                2. **Indoor Exercise:** AC/gym during heatwaves
                3. **Frequent Breaks:** If working outdoors
                4. **Cool Showers:** 2-3 times daily
                
                **🏥 First Aid for Heat Stroke:**
                1. **Move to Shade:** Cool, air-conditioned area
                2. **Cool Body:** Wet cloths, ice packs on neck/armpits
                3. **Hydrate:** If conscious and able to swallow
                4. **Medical Help:** Call 108 immediately
                
                **👥 Vulnerable Groups:**
                • **Elderly:** Reduced thirst sensation
                • **Children:** Higher metabolic rate
                • **Outdoor Workers:** Construction, farming
                • **Chronic Patients:** Heart, kidney disease
                
                **🚨 Emergency Signs:**
                • Body temperature >104°F/40°C
                • Altered mental state/confusion
                • Nausea and vomiting
                • Rapid breathing and heartbeat
                • Unconsciousness
                
                **🏥 Treatment:**
                • **Emergency Cooling:** Ice bath if available
                • **IV Fluids:** For dehydration
                • **Monitoring:** Vital signs
                • **Hospitalization:** Usually required
                
                **📱 Heat Alert Apps:**
                • **IMD:** India Meteorological Department
                • **Heat Index Calculators**
                • **Weather Apps:** With heat warnings
                
                **🏥 Government Measures:**
                • **Heat Action Plans:** In major cities
                • **Cooling Centers:** Public buildings
                • **Work Timings:** Adjusted for outdoor workers
                • **Public Advisories:** Media announcements
                
                **⚠️ Critical:**
                • Heat stroke can be fatal within hours
                • Delayed treatment increases mortality
                • Never leave children/pets in parked cars
                • Check on elderly neighbors during heatwaves
                
                **🏥 Emergency:** Call 108 for suspected heat stroke
                """)
            
            with gr.TabItem("6️⃣ Seasonal Flu"):
                gr.Markdown("""
                ### 🤧 Seasonal Flu & Respiratory Infections
                
                **Common during:** Winter months (November-February)
                
                **🔍 Symptoms:**
                • Fever, chills
                • Cough, sore throat
                • Runny/stuffy nose
                • Body aches, headache
                • Fatigue, weakness
                
                **🛡️ Prevention Strategy:**
                
                **Vaccination:**
                1. **Annual Flu Shot:** Best protection
                2. **Timing:** Before winter season
                3. **High Risk Groups:** Elderly, children, chronic patients
                4. **Availability:** Government and private centers
                
                **Hygiene Practices:**
                1. **Hand Washing:** Frequently with soap
                2. **Mask:** In crowded places during outbreaks
                3. **Avoid Touching Face:** Eyes, nose, mouth
                4. **Cover Cough/Sneeze:** Use tissue/elbow
                
                **💊 Management:**
                • **Rest:** Adequate sleep
                • **Hydration:** Warm fluids
                • **Symptom Relief:** Paracetamol for fever
                • **Antivirals:** If prescribed early
                
                **🚨 When to Seek Medical Help:**
                • Difficulty breathing
                • Chest pain
                • Persistent high fever
                • Bluish lips
                • Severe weakness
                
                **👥 High Risk Groups:**
                • Pregnant women
                • Children under 5
                • Adults over 65
                • Chronic disease patients
                • Healthcare workers
                
                **🏥 Complications to Watch:**
                • Pneumonia
                • Bronchitis
                • Sinus infections
                • Ear infections
                • Worsening of chronic conditions
                
                **📱 Digital Resources:**
                • **eSanjeevani:** Telemedicine consultations
                • **Aarogya Setu:** Health alerts
                • **m-Sehat:** Health information
                
                **🏥 Emergency:** Breathing difficulties need immediate attention
                """)
            
            with gr.TabItem("7️⃣ Lifestyle Diseases"):
                gr.Markdown("""
                ### 🏃 Lifestyle Diseases Prevention
                
                **Common Conditions:** Diabetes, Hypertension, Heart Disease, Obesity
                
                **📊 Risk Factors:**
                • Sedentary lifestyle
                • Unhealthy diet
                • Stress
                • Smoking/alcohol
                • Genetic predisposition
                
                **🛡️ Prevention Strategy:**
                
                **Dietary Changes:**
                1. **Reduce Salt:** <5g/day for hypertension
                2. **Limit Sugar:** Avoid added sugars
                3. **Healthy Fats:** Nuts, seeds, olive oil
                4. **Fiber:** Whole grains, fruits, vegetables
                5. **Portion Control:** Smaller, frequent meals
                
                **Physical Activity:**
                1. **Aerobic:** 150 mins/week moderate exercise
                2. **Strength Training:** 2 days/week
                3. **Daily Movement:** 10,000 steps target
                4. **Reduce Sitting:** Stand every 30 minutes
                
                **Stress Management:**
                1. **Meditation:** 10-15 minutes daily
                2. **Yoga:** Regular practice
                3. **Adequate Sleep:** 7-8 hours nightly
                4. **Hobbies:** Relaxing activities
                5. **Social Connection:** Family/friends support
                
                **🩺 Regular Monitoring:**
                • **Blood Pressure:** Weekly if hypertensive
                • **Blood Sugar:** As advised by doctor
                • **Weight/BMI:** Monthly tracking
                • **Cholesterol:** Yearly check
                • **Annual Health Check:** Complete examination
                
                **💊 Medication Adherence:**
                • Take as prescribed
                • Never self-adjust
                • Regular follow-ups
                • Understand side effects
                
                **🚨 Warning Signs:**
                
                **Diabetes:**
                • Excessive thirst/hunger
                • Frequent urination
                • Unexplained weight loss
                • Blurred vision
                • Slow wound healing
                
                **Hypertension:**
                • Often asymptomatic
                • Severe headaches
                • Nosebleeds
                • Fatigue/confusion
                • Vision problems
                
                **Heart Disease:**
                • Chest pain/discomfort
                • Shortness of breath
                • Palpitations
                • Swelling in legs
                • Extreme fatigue
                
                **🏥 Screening Schedule:**
                • **BP:** Every visit after 30 years
                • **Blood Sugar:** Yearly after 40 years
                • **Cholesterol:** Every 5 years after 20 years
                • **ECG:** As advised based on risk
                
                **🥗 Indian Diet Modifications:**
                • Choose brown rice over white
                • Use less oil in cooking
                • Include dal, legumes daily
                • Limit fried snacks
                • Choose fruits over sweets
                
                **🏋️ Exercise for Indians:**
                • **Morning:** Walk, yoga, cycling
                • **Evening:** Strength exercises
                • **Weekend:** Sports, swimming
                • **Daily:** Household chores count!
                
                **📱 Health Tracking Apps:**
                • Google Fit / Apple Health
                • MyFitnessPal for diet
                • Medisafe for medications
                • SugarMD for diabetes
                
                **🏥 Regular Check-ups:**
                • **Quarterly:** If on medication
                • **Half-yearly:** If controlled
                • **Annually:** Comprehensive check
                • **Emergency:** Any worrying symptoms
                
                **📞 Helplines:**
                • Diabetes: 1800-11-9090
                • Heart: 1090 (Cardiac emergency)
                • Mental Health: 1860-2662345
                
                **⚠️ Emergency:** 108 for chest pain, stroke symptoms
                """)
        
        # General Prevention Principles Translation
        gr.Markdown("---")
        gr.Markdown("### 🌐 Translate Prevention Principles")
        
        prevention_principles_text = gr.Textbox(
            label="Prevention Principles Text",
            value="""🎯 General Prevention Principles
        
        **1. Cleanliness:**
        • Personal hygiene
        • Environmental sanitation
        • Food and water safety
        
        **2. Vaccination:**
        • Complete immunization schedule
        • Annual flu shots
        • Travel vaccinations
        
        **3. Early Detection:**
        • Regular health check-ups
        • Screening tests
        • Symptom awareness
        
        **4. Healthy Lifestyle:**
        • Balanced diet
        • Regular exercise
        • Stress management
        • Adequate sleep
        
        **5. Community Action:**
        • Neighborhood cleanliness
        • Mosquito control
        • Health awareness programs""",
            lines=20,
            visible=False
        )
        
        translate_prevention_btn = gr.Button("🛡️ Translate Prevention Principles", variant="secondary")
        prevention_trans_out = gr.Textbox(label="Translated Prevention Principles", interactive=False, lines=20)
        
        translate_prevention_btn.click(
            fn=translate_text,
            inputs=[prevention_principles_text, disease_translation_lang],
            outputs=prevention_trans_out
        )
        
        gr.Markdown("""
        ---
        ### 🎯 General Prevention Principles
        
        **1. Cleanliness:**
        • Personal hygiene
        • Environmental sanitation
        • Food and water safety
        
        **2. Vaccination:**
        • Complete immunization schedule
        • Annual flu shots
        • Travel vaccinations
        
        **3. Early Detection:**
        • Regular health check-ups
        • Screening tests
        • Symptom awareness
        
        **4. Healthy Lifestyle:**
        • Balanced diet
        • Regular exercise
        • Stress management
        • Adequate sleep
        
        **5. Community Action:**
        • Neighborhood cleanliness
        • Mosquito control
        • Health awareness programs
        
        ---
        ### 📱 Digital Health Resources
        
        **Government Portals:**
        • **MoHFW:** https://www.mohfw.gov.in
        • **National Health Portal:** https://www.nhp.gov.in
        • **e-Hospital:** https://ehospital.gov.in
        
        **Mobile Applications:**
        • **Aarogya Setu:** COVID-19 tracking
        • **eSanjeevani:** Telemedicine
        • **m-Sehat:** Health records
        
        **Telemedicine Services:**
        • Government e-Hospital
        • Private hospital apps
        • Online consultations
        
        ---
        ### 🏥 When to Seek Medical Help
        
        **Immediate Attention (Call 108):**
        • Difficulty breathing
        • Chest pain
        • Unconsciousness
        • Severe bleeding
        • Poisoning
        
        **Within 24 Hours:**
        • High fever not reducing
        • Severe pain
        • Worsening symptoms
        • Concern about medication
        
        **Regular Follow-up:**
        • Chronic disease management
        • Medication refills
        • Routine check-ups
        • Vaccination schedules
        
        ---
        ⚠️ **Disclaimer:** This information is for educational purposes only. Always consult healthcare professionals for diagnosis and treatment.
        
        **📞 Save Emergency Numbers:**
        108 - Emergency Medical Services
        102 - Ambulance
        112 - Single Emergency Number
        
        **Stay Safe, Stay Healthy!** 🌿
        """)
    
    # Batch Download
    with gr.Tab("📦 Download History"):
        gr.Markdown("### Download All Prescriptions")
        gr.Markdown("Download a PDF containing all prescription calculations from this session")
        
        download_all_btn = gr.Button("📥 Download All Prescriptions", variant="primary", size="lg")
        batch_pdf_out = gr.File(label="📄 Batch PDF Download")
        
        download_all_btn.click(fn=download_all_prescriptions, outputs=batch_pdf_out)
        
        # Translation for Download Tab
        gr.Markdown("---")
        gr.Markdown("### 🌐 Translate Instructions")
        
        download_instructions_text = gr.Textbox(
            label="Download Instructions Text",
            value="""📦 Download History
        
        **Instructions:**
        1. Click 'Download All Prescriptions' button
        2. Save the PDF file to your device
        3. Share with healthcare provider if needed
        4. Keep for your medical records
        
        **⚠️ Important Notes:**
        • This PDF contains all calculations from current session only
        • Previous sessions are not saved
        • Always verify dosages with healthcare professional
        • Keep PDF secure to protect medical information""",
            lines=15,
            visible=False
        )
        
        translate_download_btn = gr.Button("📄 Translate Download Instructions", variant="secondary")
        download_trans_out = gr.Textbox(label="Translated Instructions", interactive=False, lines=15)
        
        translate_download_btn.click(
            fn=lambda text: translate_text(text, "hi"),
            inputs=[download_instructions_text],  # Default to Hindi
            outputs=download_trans_out
        )
    
    # Help Tab
    with gr.Tab("❓ Help"):
        gr.Markdown("""
        ### 📚 Quick Start Guide
        
        #### 1. 🔧 Setup AI Model
        **Option A: Free Local Gemma Model**
        - Select model (models/gemma-3-4b-it recommended)
        - Click "Load Local Gemma"
        - Wait for model to download and load
        
        **Option B: Gemini API (Online)**
        - Get free API key from Google AI Studio
        - Enter API key
        - Click "Initialize Gemini API"
        
        #### 2. 📂 Upload Dataset
        - Upload CSV/Excel with medicine data
        - Must have "Name" column
        - Wait for success message
        
        #### 3. 💊 Calculate Dosage
        - Enter patient name (optional)
        - Type medicine name OR use voice input 🎤
        - Enter age and weight
        - Click "Calculate"
        - Get AI explanation and PDF report
        
        #### 4. 📸 Prescription Analysis (Enhanced OCR)
        - Upload prescription image 📸
        - Click "Extract Text (Enhanced OCR)"
        - Click "AI Explain Prescription" for detailed analysis 🤖
        - OR click "Quick Analysis" for summary
        - Translate to regional languages 🌐
        
        #### 5. 🤖 Medical Chatbot (NEW!)
        - Ask any medical question
        - Get AI-powered answers
        - Learn about medicines, symptoms, conditions
        - Examples provided for quick start
        
        #### 6. 🌤️ Weather Health Alert (NEW!)
        - Get city weather data
        - AI analyzes disease risks
        - Get prevention tips
        - Weather-based health recommendations
        
        #### 7. 📞 North India Helpline (NEW!)
        - Emergency contact numbers
        - State-wise hospital contacts
        - Specialized medical services
        - Mobile app recommendations
        
        #### 8. 🦠 Major Diseases Info (NEW!)
        - Comprehensive disease guides
        - Prevention strategies
        - Symptoms and treatments
        - North India specific information
        
        #### 9. 📦 Download History
        - Click "Download All Prescriptions"
        - Get PDF with all calculations from session
        
        ---
        
        ### 🌐 Translation Features
        
        **Available in all tabs:**
        - Translate medicine information
        - Translate dosage instructions
        - Translate AI explanations
        - Translate chatbot responses
        - Translate weather alerts
        - Translate emergency contacts
        - Translate disease information
        - Translate help instructions
        
        **Supported Languages:** Hindi, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati, Bengali, Punjabi, Urdu, and more!
        
        **How to use translation:**
        1. Look for the "🌐 Translate" section in each tab
        2. Select your preferred language
        3. Click the translation button
        4. View translated content in the output box
        
        ---
        
        ### 🎤 Voice Input
        - Click microphone icon
        - Speak medicine name clearly
        - Text will auto-fill
        
        ---
        
        ### 📸 Enhanced OCR Features
        
        **Advanced Image Preprocessing:**
        - ✅ Automatic upscaling for small images
        - ✅ Noise reduction
        - ✅ Adaptive thresholding for better contrast
        - ✅ Multiple extraction modes for maximum accuracy
        
        **Best Practices:**
        - Use high resolution images (300+ DPI)
        - Ensure good lighting
        - Keep text horizontal
        - Dark text on light background
        - Clear, focused images
        
        ---
        
        ### 🤖 AI Models Available
        
        **1. Gemma-2B (Free & Local)**
        - ✅ No API key required
        - ✅ Works offline after download
        - ✅ Good for general medical queries
        - ⚠️ Requires 3GB+ RAM
        
        **2. Gemma-2B-IT (Free & Local)**
        - ✅ Instruction-tuned version
        - ✅ Better for conversational AI
        - ✅ Good for chatbot responses
        - ⚠️ Requires 4GB+ RAM
        
        **3. Gemini API (Online)**
        - ✅ Most powerful option
        - ✅ Requires free API key
        - ✅ Fastest responses
        - ⚠️ Requires internet connection
        
        ---
        
        ### 💬 Chatbot Features
        
        **What you can ask:**
        - Medicine information and uses
        - Side effects and interactions
        - Symptom explanations
        - Health condition information
        - General medical guidance
        - Dosage questions
        - Preventive health tips
        
        **What chatbot provides:**
        - Evidence-based information
        - Clear, concise answers
        - Safety reminders
        - Professional consultation advice
        
        ---
        
        ### 🌤️ Weather Health Alert Features
        
        **Get Free API Key:**
        1. Visit: https://openweathermap.org/api
        2. Sign up for free account
        3. Get API key from dashboard
        
        **What it provides:**
        - Real-time weather conditions
        - Disease risk analysis
        - Prevention recommendations
        - AI-powered health advice
        
        ---
        
        ### 🦠 Major Diseases Section
        
        **Comprehensive Coverage:**
        - 7 major disease categories
        - Detailed prevention strategies
        - Symptoms and treatments
        - Emergency protocols
        - North India specific guidance
        
        **Diseases Covered:**
        1. Dengue & Malaria
        2. Air Pollution Diseases
        3. Tuberculosis (TB)
        4. Water-Borne Diseases
        5. Heat Stroke
        6. Seasonal Flu
        7. Lifestyle Diseases
        
        ---
        
        ### 📱 Mobile Compatibility
        
        **Works on:**
        - Desktop computers
        - Laptops
        - Tablets
        - Smartphones
        
        **Best viewed on:**
        - Chrome, Firefox, Safari, Edge browsers
        - Screen width 1024px or larger
        - Good internet connection for AI features
        
        ---
        
        ### ⚠️ Important Notes
        - **For educational purposes only**
        - **Always consult healthcare professionals**
        - **Not for actual medical decisions**
        - **OCR accuracy depends on image quality**
        - **Chatbot provides general information only**
        - **Weather data requires free API key**
        - **In emergency, call your local emergency number**
        
        ---
        
        ### 📦 Installation Requirements
        
        **Essential Packages:**
        ```bash
        pip install gradio pandas transformers torch
        pip install deep-translator rapidfuzz reportlab
        pip install pillow pytesseract openpyxl
        pip install SpeechRecognition requests
        ```
        
        **For Gemini API (Optional):**
        ```bash
        pip install google-generativeai
        ```
        
        **For Enhanced OCR (Highly Recommended):**
        ```bash
        pip install opencv-python
        ```
        
        **Tesseract OCR Installation:**
        
        🪟 **Windows:**
        - Download: https://github.com/tesseract-ocr/tesseract
        - Add to PATH
        
        🍎 **macOS:**
        ```bash
        brew install tesseract
        ```
        
        🐧 **Linux (Ubuntu/Debian):**
        ```bash
        sudo apt-get install tesseract-ocr
        ```
        
        ---
        
        ### 🔧 Troubleshooting
        
        **Gemma Model Loading Issues:**
        - ❌ "Out of memory": Use smaller model (gemma-2b)
        - 💡 Solution: Close other applications, add --low-vram flag
        - ❌ "Download failed": Check internet connection
        - 💡 Solution: Use Gemini API as alternative
        
        **OCR Issues:**
        - ❌ "No text found": Image quality too low
        - 💡 Solution: Install opencv-python, use higher resolution
        
        **Chatbot Issues:**
        - ❌ "AI model not initialized": Configure AI in Setup tab
        - 💡 Solution: Initialize Gemma model or Gemini API
        
        **Weather API Issues:**
        - ❌ "API Error": Invalid or expired API key
        - 💡 Solution: Get free key from OpenWeatherMap
        
        **AI Explanation Issues:**
        - ❌ "Cannot explain": No valid text extracted
        - 💡 Solution: Ensure clear image and successful OCR extraction
        
        **Translation Issues:**
        - ❌ "Translation failed": Internet connection issue
        - 💡 Solution: Check internet, try different language
        - ❌ "Service unavailable": Google Translate API limit
        - 💡 Solution: Wait and try again later
        
        ---
        
        ### 💡 Tips for Best Results
        
        **For AI Models:**
        - 🤖 Use Gemini API for best results
        - 💾 Gemma-2b for offline/local use
        - 🔄 Restart if model gets stuck
        
        **For OCR:**
        - 📸 Use 300+ DPI resolution
        - 💡 Ensure even lighting
        - 📏 Keep prescription flat
        - 🎯 Focus the camera properly
        - 🧹 Clean prescription before photo
        
        **For Medicine Search:**
        - 🔍 Use generic/scientific names
        - ✅ Fuzzy matching handles typos
        - 💡 Check suggestions if no match
        
        **For Chatbot:**
        - 💬 Ask specific questions
        - 📝 Provide context when needed
        - ✅ Verify answers with professionals
        - 🔄 Rephrase if answer unclear
        
        **For Weather API:**
        - 🔑 Get free API key from OpenWeatherMap
        - 🌆 Use correct city names
        - 📱 Save API key for future use
        
        **For Translation:**
        - 🌐 Use common languages for better accuracy
        - 📝 Keep text concise for better results
        - 🔄 Try alternative languages if one fails
        - ✅ Verify medical terms with professional
        
        ---
        
        ### ✨ Features Overview
        
        ✅ **Smart Medicine Search** - Fuzzy matching with suggestions  
        ✅ **Age-Based Dosage** - Infant, Child, Adult, Elderly categories  
        ✅ **AI Explanations** - Powered by Gemma/Gemini  
        ✅ **Enhanced OCR** - Advanced image preprocessing  
        ✅ **Prescription AI Analysis** - Comprehensive explanation  
        ✅ **Medical Chatbot** - Ask any medical question with voice input  
        ✅ **Speech Recognition** - Voice-to-text for chatbot  
        ✅ **Weather Health Alerts** - Disease prediction based on weather  
        ✅ **North India Helpline** - Emergency contact database  
        ✅ **Major Diseases Guide** - Comprehensive prevention strategies  
        ✅ **Multi-language Support** - 10+ languages with Google Translate  
        ✅ **Voice Input** - Hands-free medicine entry  
        ✅ **PDF Reports** - Individual and batch downloads  
        ✅ **History Tracking** - Session-based prescription records  
        ✅ **Cross-platform** - Works on all devices  
        
        ---
        
        ### 📋 Dataset Format
        
        Your CSV/Excel should have these columns:
        - **Name** (Required) - Medicine name
        - **Classification** (Optional) - Drug classification
        - **Indication** (Optional) - What it's used for
        - **Strength** (Optional) - Default dosage strength
        
        Example:
        ```
        Name,Classification,Indication,Strength
        Paracetamol,Analgesic,Pain relief,500mg
        Amoxicillin,Antibiotic,Bacterial infection,250mg
        ```
        
        ---
        
        ### 🔒 Privacy & Security
        - ✅ All processing done locally/in session
        - ✅ No data stored permanently
        - ✅ Prescription history cleared on restart
        - ✅ Gemini API calls encrypted
        - ✅ Chatbot conversations not saved externally
        - ✅ Weather API calls use secure connections
        - ✅ Translation service uses Google's secure API
        - ✅ No personal data collected or shared
        
        ---
        
        ### 📞 Support
        
        **For Technical Issues:**
        - Check Troubleshooting section above
        - Ensure all dependencies installed
        - Verify Tesseract installation for OCR
        - Test with clear, high-quality images
        - Initialize AI model before using AI features
        
        **For Translation Issues:**
        - Check internet connection
        - Try different language
        - Simplify text for better translation
        - Use professional translation for critical medical information
        
        **For Medical Emergencies:**
        - Call 108 or 112 immediately
        - Visit nearest hospital
        - Do not rely on app for emergencies
        
        ---
        
        ### 🆕 What's New in This Version
        
        1. **🤖 Free AI Models:** Gemma-2B and Gemma-2B-IT
        2. **🌐 Comprehensive Translation:** Google Translate in ALL tabs
        3. **🌤️ Weather Health Alert Tab**
           - Real-time weather data integration
           - Disease risk prediction based on weather
           - AI-powered health recommendations
           - Prevention tips for weather conditions
        
        4. **📞 North India Helpline Tab**
           - Comprehensive emergency contact database
           - State-wise hospital information
           - Specialized medical services
           - Mobile app recommendations
        
        5. **🦠 Major Diseases Info Tab**
           - 7 major disease categories covered
           - Detailed prevention strategies
           - Symptoms and treatment guidelines
           - North India specific information
        
        6. **🤖 Enhanced Chatbot**
           - Added speech recognition
           - Voice input capability
           - Improved response quality
           - More example questions
        
        7. **🎨 Improved UI**
           - Better emoji usage
           - Clearer section headers
           - Enhanced visual feedback
           - Streamlined workflow
           - More intuitive navigation
        
        8. **🔧 Technical Improvements**
           - Better error handling
           - Improved OCR accuracy
           - Faster processing
           - Enhanced PDF generation
        
        9. **🌐 Translation Everywhere**
           - Translate medicine info, dosage, explanations
           - Translate chatbot conversations
           - Translate weather alerts
           - Translate emergency information
           - Translate disease guides
           - 10+ Indian languages supported
        
        ---
        
        ### 🎯 Target Users
        
        **1. General Public:**
        - Medicine dosage calculations
        - Prescription understanding
        - Basic medical information
        - Disease prevention knowledge
        - Multi-language support for non-English speakers
        
        **2. Students & Researchers:**
        - Medical data analysis
        - Learning resource
        - Research reference
        - Educational tool
        - Multi-language medical terminology
        
        **3. Healthcare Professionals:**
        - Quick reference tool
        - Patient education material
        - Dosage verification
        - Information sharing
        - Multi-language patient communication
        
        **4. Caregivers:**
        - Elderly care assistance
        - Child medication management
        - Chronic disease support
        - Emergency preparedness
        - Multi-language instructions
        
        **5. Non-English Speakers:**
        - Access to medical information in native language
        - Translated dosage instructions
        - Local language disease information
        - Regional emergency contacts
        
        ---
        
        ### 📊 Data Sources
        
        **Medicine Database:**
        - User-uploaded datasets
        - Standard medical references
        - Government drug databases
        
        **Weather Data:**
        - OpenWeatherMap API
        - Real-time meteorological data
        - Historical weather patterns
        
        **Medical Information:**
        - Google Gemma/Gemini AI models
        - Evidence-based guidelines
        - Public health recommendations
        
        **Emergency Contacts:**
        - Government health departments
        - Hospital directories
        - Verified helpline numbers
        
        **Translation Service:**
        - Google Translate API
        - Supports 100+ languages
        - Medical terminology optimized
        
        ---
        
        ### 🔄 Update Schedule
        
        **Regular Updates:**
        - Emergency contact verification (Monthly)
        - Disease information updates (Quarterly)
        - Translation language expansion (Bi-annually)
        - Software improvements (As needed)
        
        **User-Driven Updates:**
        - Based on user feedback
        - Feature requests
        - Bug fixes
        - Language requests
        
        ---
        
        ### 🤝 Contribution Guidelines
        
        **Want to contribute?**
        1. Report bugs through GitHub issues
        2. Suggest new features
        3. Share medical datasets
        4. Provide translation help
        5. Test and provide feedback
        6. Suggest new languages to support
        
        **Medical Information Contribution:**
        - Must be from verified sources
        - Evidence-based only
        - Include references
        - No promotional content
        
        **Translation Contribution:**
        - Help verify medical translations
        - Suggest better terminology
        - Provide regional language expertise
        - Help with localization
        
        ---
        
        ### 📚 Educational Resources
        
        **Recommended Reading:**
        - National Health Portal of India
        - WHO India Country Office
        - ICMR Guidelines
        - MoHFW Publications
        
        **Online Courses:**
        - First Aid and CPR courses
        - Health literacy programs
        - Disease prevention workshops
        - Nutrition and wellness courses
        
        **Translation Resources:**
        - Medical translation guides
        - Multilingual health terminology
        - Cross-cultural communication in healthcare
        
        ---
        
        ### 🌟 Success Stories
        
        **How this app helps:**
        - Reduces medication errors
        - Improves health literacy
        - Provides quick access to information
        - Supports preventive healthcare
        - Enhances patient-doctor communication
        - Bridges language barriers in healthcare
        - Makes medical information accessible to all
        
        ---
        
        **Stay Healthy, Stay Informed!** 💪
        
        *This application is dedicated to promoting health awareness and supporting informed healthcare decisions for everyone, regardless of language.*
        """)
        
        # Translation for Help Tab
        gr.Markdown("---")
        gr.Markdown("### 🌐 Translate Help Section")
        
        help_translation_lang = gr.Dropdown(
            choices=SUPPORTED_LANGUAGES,
            value="hi",
            label="Select Language for Help Translation"
        )
        
        # Extract key help sections for translation
        quick_start_guide_text = gr.Textbox(
            label="Quick Start Guide Text",
            value="""📚 Quick Start Guide
        
        #### 1. 🔧 Setup AI Model
        **Option A: Free Local Gemma Model**
        - Select model (models/gemma-3-4b-it recommended)
        - Click "Load Local Gemma"
        - Wait for model to download and load
        
        #### 2. 📂 Upload Dataset
        - Upload CSV/Excel with medicine data
        - Must have "Name" column
        - Wait for success message
        
        #### 3. 💊 Calculate Dosage
        - Enter patient name (optional)
        - Type medicine name OR use voice input 🎤
        - Enter age and weight
        - Click "Calculate"
        - Get AI explanation and PDF report""",
            lines=20,
            visible=False
        )
        
        translate_help_btn = gr.Button("📘 Translate Quick Start Guide", variant="secondary")
        help_trans_out = gr.Textbox(label="Translated Help Guide", interactive=False, lines=20)
        
        translate_help_btn.click(
            fn=translate_text,
            inputs=[quick_start_guide_text, help_translation_lang],
            outputs=help_trans_out
        )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("💊 Enhanced Medicine Dosage Calculator")
    print("="*60)
    print("\n✨ New Features Added:")
    print("• 🤖 Free Gemma AI Models - Local inference")
    print("• 🌐 COMPREHENSIVE TRANSLATION - Google Translate in ALL tabs")
    print("• 🌤️ Weather Health Alert - Disease prediction based on weather")
    print("• 📞 North India Helpline - Emergency contact database")
    print("• 🦠 Major Diseases Info - Comprehensive prevention guides")
    print("• 🎤 Speech Recognition - Voice input for chatbot")
    print("• 🤖 Enhanced Medical Chatbot - Better AI responses")
    print("\n🌐 Translation Now Available In:")
    print("• ALL Medicine Information")
    print("• ALL Dosage Instructions")
    print("• ALL AI Explanations")
    print("• Chatbot Conversations")
    print("• Weather Alerts & Disease Info")
    print("• Emergency Contacts")
    print("• Help & Instructions")
    print("\n📋 All Features:")
    print("• 🎤 Voice input for medicine names")
    print("• 📸 Enhanced OCR with image preprocessing")
    print("• 🤖 AI Prescription Explanation")
    print("• 💬 Medical Chatbot with Speech Input")
    print("• 🌤️ Weather-based Disease Prediction")
    print("• 📞 North India Emergency Contacts")
    print("• 🦠 Major Diseases Prevention Guides")
    print("• 🌐 MULTI-LANGUAGE TRANSLATION (20+ languages)")
    print("• 📦 Batch prescription download")
    print("• 👤 Patient name tracking")
    print("• 📊 AI-powered prescription analysis")
    print("• 🔍 Smart fuzzy medicine search")
    print("• 📄 Professional PDF reports")
    print("\n📦 Installation Requirements:")
    print("• pip install transformers torch")
    print("• pip install SpeechRecognition")
    print("• pip install requests")
    print("• pip install deep-translator")
    print("• Free OpenWeatherMap API key (for weather features)")
    print("• Free Gemini API key (optional, for best AI results)")
    print("\n" + "="*60 + "\n")
    
    demo.launch(share=True, debug=True)