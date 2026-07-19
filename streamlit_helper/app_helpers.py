import os
import json
import base64
from datetime import datetime
from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

# Import centralized prompts
from .prompts import (
    VLM_DEV_PROMPT,
    VLM_USER_PROMPT,
    LLM_PATHOLOGIST_SYSTEM_PROMPT,
    get_llm_pathologist_user_prompt
)

# Default image size
IMG_SIZE = (128, 128)

# Paths configuration
_PROJECT_DIR = os.environ.get('PROJECT_DIR', '.')
BEST_MODEL_DIR = os.path.join(_PROJECT_DIR, 'best_model')
MODEL_INFO_FILE = os.path.join(BEST_MODEL_DIR, 'model_info.json')

@st.cache_resource
def load_best_model():
    """Load the best model from best_model folder"""
    model_path = os.path.join(BEST_MODEL_DIR, 'best_model.h5')
    if os.path.exists(model_path):
        try:
            return load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model: {model_path} : {str(e)}")
            return None
    return None

def load_model_info():
    """Load model information from JSON file"""
    if os.path.exists(MODEL_INFO_FILE):
        with open(MODEL_INFO_FILE, 'r') as f:
            return json.load(f)
    return None

def check_best_model_status():
    """Check if best model exists"""
    model_path = os.path.join(BEST_MODEL_DIR, 'best_model.h5')
    if os.path.exists(model_path):
        return True, "Best model loaded"
    return False, "Best model not found - Run notebook first"

def preprocess_image(uploaded_file, img_size=IMG_SIZE):
    """Preprocess uploaded image for model prediction"""
    uploaded_file.seek(0)
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize(img_size)
    # The loaded model contains a built-in input_rescaling layer that divides by 255.
    # We pass the raw [0, 255] values as float32 to avoid double normalization.
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

def predict_with_dl_model(model, img_array):
    """Make prediction using deep learning model"""
    prediction = model.predict(img_array, verbose=0)[0][0]
    label = 'Uninfected' if prediction > 0.5 else 'Parasitized'
    confidence = prediction if prediction > 0.5 else (1 - prediction)
    return prediction, label, confidence

def encode_image_to_base64(uploaded_file):
    """Encode image to base64 for OpenAI API"""
    uploaded_file.seek(0)
    image_bytes = uploaded_file.read()
    return base64.b64encode(image_bytes).decode('utf-8')

def analyze_with_openai_vlm(uploaded_file, OpenAIClient, azure_deployment):
    """
    Analyze blood cell image using OpenAI's GPT-4 Vision
    
    Args:
        uploaded_file: Uploaded image file
        OpenAIClient: The instantiated AzureOpenAI client
        azure_deployment: The deployment name for the model
    
    Returns:
        dict: Analysis results with infection_stage, insights, recommendations
    """
    try:        
        # Encode image to base64
        base64_image = encode_image_to_base64(uploaded_file)
        
        # VLM Prompt for malaria analysis
        dev_prompt = VLM_DEV_PROMPT
        user_prompt = VLM_USER_PROMPT

        response = OpenAIClient.chat.completions.create(
            model=azure_deployment,
            messages=[
                {
                    "role": "system",
                    "content": dev_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        # Parse response
        result_text = response.choices[0].message.content
        
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            import json as json_lib
            result = json_lib.loads(json_match.group())
            return {
                'success': True,
                'data': result,
                'raw_response': result_text
            }
        else:
            return {
                'success': True,
                'data': {
                    'infection_stage': 'Analysis Complete',
                    'confidence': 0.8,
                    'insights': result_text,
                    'recommendations': 'Review complete analysis above',
                    'severity_score': 5
                },
                'raw_response': result_text
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'data': None
        }

def generate_llm_report(prediction, confidence, model_info, OpenAIClient=None, azure_deployment=None):
    """
    Generate GenAI clinical report from model predictions using OpenAI Chat Completion
    """
    label = 'Parasitized' if prediction <= 0.5 else 'Uninfected'
    raw_score = prediction
    
    # If OpenAI is not configured, fallback to rule-based generation
    if OpenAIClient is None or azure_deployment is None:
        return _generate_local_fallback_report(label, confidence, raw_score, model_info)
        
    try:
        model_name = model_info.get('model_name', 'Best DL CNN Model') if model_info else 'Best DL CNN Model'
        accuracy = model_info.get('accuracy', 'N/A') if model_info else 'N/A'
        accuracy_str = accuracy if isinstance(accuracy, str) else f"{accuracy:.2%}"
        
        system_prompt = LLM_PATHOLOGIST_SYSTEM_PROMPT
        user_prompt = get_llm_pathologist_user_prompt(label, confidence, raw_score, model_name, accuracy_str)

        response = OpenAIClient.chat.completions.create(
            model=azure_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # Fallback if API fails
        fallback_prefix = f"⚠️ *Note: OpenAI API failed ({str(e)}). Displaying local rule-based report fallback.*\n\n"
        return fallback_prefix + _generate_local_fallback_report(label, confidence, raw_score, model_info)


def _generate_local_fallback_report(label, confidence, raw_score, model_info):
    model_name = model_info.get('model_name', 'Best Model') if model_info else 'Best Model'
    accuracy = model_info.get('accuracy', 'N/A') if model_info else 'N/A'
    
    if label == 'Parasitized':
        status_icon = "⚠️"
        status_color = "red"
        status_text = "POSITIVE FOR MALARIA"
        recommendations = """
        **Recommended Actions:**
        - 🏥 Seek immediate medical consultation
        - 🧪 Confirm with additional diagnostic tests (blood smear, PCR)
        - 💊 Antimalarial treatment may be required
        - 📋 Monitor for symptoms: fever, chills, headache, fatigue
        """
    else:
        status_icon = "✅"
        status_color = "green"
        status_text = "NEGATIVE FOR MALARIA"
        recommendations = """
        **Recommended Actions:**
        - ✅ No immediate treatment required
        - 📅 Continue routine monitoring if in endemic area
        - 🛡️ Maintain preventive measures (bed nets, repellents)
        - 📋 Consult doctor if symptoms develop
        """
    
    report = f"""
    ## 🩸 Malaria Detection Report
    
    | Field | Value |
    |-------|-------|
    | **Result** | **{status_icon} {status_text}** |
    | **Classification** | {label} |
    | **Confidence** | {confidence:.2%} |
    | **Raw Score** | {raw_score:.4f} |
    | **Model Used** | {model_name} |
    | **Model Accuracy** | {accuracy if isinstance(accuracy, str) else f'{accuracy:.4f} ({accuracy*100:.2f}%)'} |
    | **Analysis Time** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
    
    ---
    
    ### 🔬 Analysis Details
    
    The deep learning model has analyzed the uploaded blood cell image and classified it as **{label}**.
    
    {recommendations}
    
    ---
    
    *This is an AI-assisted diagnostic tool. Results should be confirmed by qualified medical professionals. Not intended for standalone clinical diagnosis.*
    """
    return report
