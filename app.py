"""
Streamlit GUI for Malaria Detection - Milestone 2
Complete Implementation with Best Model + LLM/VLM Integration
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import os
import json
import base64
from io import BytesIO
from datetime import datetime
from openai import AzureOpenAI


# Set MLflow environment variable to allow local file store backend
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import mlflow


# Configuration
IMG_SIZE = (128, 128)

# MLflow Configuration
MLFLOW_EXPERIMENT_NAME = "Malaria_Detection_App"
try:
    # Check if a remote tracking URI is defined in environment variables or Streamlit secrets
    remote_uri = None
    if "MLFLOW_TRACKING_URI" in os.environ:
        remote_uri = os.environ["MLFLOW_TRACKING_URI"]
    else:
        try:
            if "MLFLOW_TRACKING_URI" in st.secrets:
                remote_uri = st.secrets["MLFLOW_TRACKING_URI"]
        except Exception:
            pass
            
    if remote_uri:
        mlflow.set_tracking_uri(remote_uri)
        # Natively populate username/password credentials to environment variables for MLflow client authentication
        for cred_key in ["MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]:
            try:
                if cred_key in st.secrets:
                    os.environ[cred_key] = st.secrets[cred_key]
            except Exception:
                pass
    else:
        # Use absolute path for local mlruns fallback
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
        os.makedirs(MLRUNS_DIR, exist_ok=True)
        mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.replace(os.sep, '/')}")
        
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
except Exception as e:
    # Fallback if there is an issue with setting experiment
    print(f"MLflow set_experiment warning: {e}")



# Use environment variable for base path so the app works both locally and on Google Colab
# On Colab: set PROJECT_DIR env var to your Google Drive project path, e.g. /content/drive/MyDrive/ML2Base
# On local: defaults to relative path './best_model'
_PROJECT_DIR = os.environ.get('PROJECT_DIR', '.')
BEST_MODEL_DIR = os.path.join(_PROJECT_DIR, 'best_model')
MODEL_INFO_FILE = os.path.join(BEST_MODEL_DIR, 'model_info.json')

# Securely load credentials from Streamlit Secrets or Environment Variables
def get_secret(key_name, default_value=""):
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.getenv(key_name, default_value)

# Check if secrets/environment credentials are fully set
has_secrets = all([
    get_secret("AZURE_OPENAI_ENDPOINT"),
    get_secret("AZURE_OPENAI_API_KEY"),
    get_secret("AZURE_OPENAI_API_VERSION"),
    get_secret("AZURE_OPENAI_DEPLOYMENT_NAME"),
])

# Initialize session state for manual Azure OpenAI credentials
if "azure_endpoint" not in st.session_state:
    st.session_state.azure_endpoint = ""
if "azure_key" not in st.session_state:
    st.session_state.azure_key = ""
if "azure_apiversion" not in st.session_state:
    st.session_state.azure_apiversion = ""
if "azure_deployment" not in st.session_state:
    st.session_state.azure_deployment = ""

if has_secrets:
    azure_endpoint = get_secret("AZURE_OPENAI_ENDPOINT")
    azure_key = get_secret("AZURE_OPENAI_API_KEY")
    azure_apiversion = get_secret("AZURE_OPENAI_API_VERSION")
    azure_deployment = get_secret("AZURE_OPENAI_DEPLOYMENT_NAME")
    is_secure_mode = True
else:
    azure_endpoint = st.session_state.azure_endpoint
    azure_key = st.session_state.azure_key
    azure_apiversion = st.session_state.azure_apiversion
    azure_deployment = st.session_state.azure_deployment
    is_secure_mode = False

@st.cache_resource(show_spinner="Connecting to Azure OpenAI...")
def get_openai_client(endpoint, key, api_ver):
    """Instantiates client ONCE and caches it globally across reruns."""
    if not (endpoint and key and api_ver):
        return None
    try:
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version=api_ver
        )
    except Exception as e:
        print(f"Azure OpenAI Connection Error: {e}")
        return None


# Lock layout context to sidebar to keep spinner inside it
with st.sidebar:
    OpenAIClient = get_openai_client(
        azure_endpoint, azure_key, azure_apiversion
    )

client_ready = OpenAIClient is not None

# Render collapsible form or secure status in the sidebar
with st.sidebar:
    if is_secure_mode and client_ready:
        st.success("🔒 API credentials loaded securely from config.")
    else:
        expander_title = "✅ VLM connection is ready" if client_ready else "🔑 VLM API Configuration"
        with st.expander(expander_title, expanded=not client_ready):
            with st.form("credentials_form"):
                st.session_state.azure_endpoint = st.text_input("Azure end point", value=st.session_state.azure_endpoint, type="password", placeholder="https://")
                st.session_state.azure_key = st.text_input("Azure API Key", value=st.session_state.azure_key, type="password")
                st.session_state.azure_apiversion = st.text_input("API version", value=st.session_state.azure_apiversion, type="password", placeholder="2024-02-01")
                st.session_state.azure_deployment = st.text_input("Deployment name", value=st.session_state.azure_deployment, type="password", placeholder="e.g., prod-gpt4o-v1")
                submit_button = st.form_submit_button("Connect API Configuration")
                if submit_button:
                    st.rerun()


# Page config
st.set_page_config(
    page_title="Malaria Detection AI - Milestone 2",
    page_icon="🩸",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stAlert {
        border-radius: 10px;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .info-message {
        background-color: #e7f3ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


# Import helpers from streamlit_helper with hot-reload to avoid Streamlit module caching
# import importlib
# import streamlit_helper.app_helpers
# importlib.reload(streamlit_helper.app_helpers)
from streamlit_helper.app_helpers import (
    load_best_model,
    load_model_info,
    check_best_model_status,
    preprocess_image,
    predict_with_dl_model,
    analyze_with_openai_vlm,
    generate_llm_report
)


# Main app
def main():
    # Header
    st.title("🩸 Malaria Detection AI Platform")
    st.markdown("### A system developed using Deep Learning & Generative AI")
    st.markdown("*Milestone 1 & 2: Best Model + LLM/VLM Integration*")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ System Status")
    
    # Best Model Status
    st.sidebar.subheader("📦 Best Model Status")
    model_available, model_msg = check_best_model_status()
    if model_available:
        st.sidebar.markdown(
            f'<div class="status-box status-success">✅ {model_msg}</div>',
            unsafe_allow_html=True
        )
        # Load and display model info
        model_info = load_model_info()
        if model_info:
            st.sidebar.info(f"""
            **Model Details:**
            - Name: {model_info.get('model_name', 'N/A')}
            - Accuracy: {model_info.get('accuracy', 0)*100:.2f}%
            - Image Size: {model_info.get('image_size', [(128,128)])[0]}x{model_info.get('image_size', [(128,128)])[1]}
            """)
    else:
        st.sidebar.markdown(
            f'<div class="status-box status-error">❌ {model_msg}</div>',
            unsafe_allow_html=True
        )
        st.sidebar.warning("Please run the notebook to train and save the best model first.")
    
    st.sidebar.markdown("---")
    
    # OpenAI Key Status
    st.sidebar.subheader("🤖 OpenAI VLM Status")
    if OpenAIClient:
        st.sidebar.markdown(
            '<div class="status-box status-success">✅ OpenAI VLM Status: Ready</div>',
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            '<div class="status-box status-error">❌ OpenAI VLM Status: Offline</div>',
            unsafe_allow_html=True
        )
        st.sidebar.info("Set OpenAI details for VLM analysis")
    
    st.sidebar.markdown("---")
    
    # About section
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.markdown("""
    **AI-Powered Malaria Detection**
    
    This system combines:
    - 🧠 Best-performing DL model
    - 🤖 LLM for report generation  
    - 👁️ VLM for visual analysis
    
    **For research use only.**
    """)
    
    # Main content - Tabs
    tab1, tab2, tab3 = st.tabs(["🔬 Detection (Best Model + LLM/VLM)", "📁 Bulk VLM Analysis", "ℹ️ Model Information"])
    
    # ========== TAB 1: Detection with LLM/VLM ==========
    with tab1:
        st.header("Single Image Detection with AI Analysis")
        st.markdown("Upload a blood cell image for DL + VLM analysis")
        
        # Model selection dropdown - only 2 options
        analysis_mode = st.selectbox(
            "Select Analysis Mode:",
            ["Best Model Prediction ➜ LLM", "Cell Image ➜ VLM"],
            help="LLM: Generates report from DL prediction | VLM: Analyzes image directly using GPT-4 Vision"
        )
        
        # Initialize uploader key ID if not present
        if "uploader_id" not in st.session_state:
            st.session_state.uploader_id = 0
            
        uploader_key = f"single_{st.session_state.uploader_id}"
        
        # Sync widget value to custom session state cache if present
        if uploader_key in st.session_state and st.session_state[uploader_key] is not None:
            st.session_state.uploaded_file_cache = st.session_state[uploader_key]
            
        # Retrieve the cached image file
        uploaded_file = st.session_state.get("uploaded_file_cache", None)
        
        if uploaded_file is None:
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key=uploader_key)
            
            # Show placeholder message when no image selected
            st.markdown("""
            <div class="info-message">
                <h3>📷 No Image Selected</h3>
                <p>Please upload a blood cell image to proceed with analysis.</p>
                <p><strong>Supported formats:</strong> JPG, JPEG, PNG</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            ### How to Use
            
            1. **Select Analysis Mode:**
               - **Best Model + LLM**: Uses our trained DL model + generates detailed report
               - **Best Model + VLM**: Uses our DL model + OpenAI GPT-4 Vision for visual analysis
            
            2. **Upload Image:** Click the button above and select a blood cell image
            
            3. **View Results:** Analysis will appear below with predictions and recommendations
            """)
        else:
            # Image is selected, show it
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(uploaded_file, caption="Uploaded Image", width=180)
                if st.button("🔄 Change Image"):
                    st.session_state.uploader_id += 1
                    st.session_state.uploaded_file_cache = None
                    st.rerun()

            
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary"):
                # First, run DL model prediction
                dl_model = load_best_model()
                
                if dl_model:
                    model_info = load_model_info()
                    dl_model_name = "N/A"
                    if model_info:
                        base_name = model_info.get("model_name", "N/A")
                        version = model_info.get("version")
                        dl_model_name = f"{base_name}_v{version}" if version is not None else base_name
                    target_size = (128, 128)
                    if model_info and "image_size" in model_info:
                        size_list = model_info["image_size"]
                        if len(size_list) >= 2:
                            target_size = (size_list[0], size_list[1])
                            
                    img_array, _ = preprocess_image(uploaded_file, img_size=target_size)
                    dl_prediction, dl_label, dl_confidence = predict_with_dl_model(dl_model, img_array)
                    
                    # Start MLflow run
                    with mlflow.start_run(run_name=f"Single_Image_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                        # Log input params
                        mlflow.log_param("analysis_mode", analysis_mode)
                        mlflow.log_param("image_name", uploaded_file.name)
                        
                        # Log predictions and metrics
                        mlflow.log_metric("prediction", float(dl_prediction))
                        mlflow.log_metric("confidence", float(dl_confidence))
                        mlflow.log_param("label", dl_label.lower())

                            
                        if analysis_mode == "Best Model Prediction ➜ LLM":
                            # Generate LLM-style report
                            with col2:
                                st.subheader("🔬 DL Model Prediction")
                                
                                if dl_label == 'Parasitized':
                                    st.error(f"**{dl_label}** ⚠️")
                                else:
                                    st.success(f"**{dl_label}** ✅")
                                
                                st.metric("Confidence", f"{dl_confidence:.2%}")
                                st.metric("Raw Score", f"{dl_prediction:.4f}")
                            
                            st.markdown("---")
                            mlflow.log_param("model_name", dl_model_name)
                            report = generate_llm_report(dl_prediction, dl_confidence, model_info, OpenAIClient, azure_deployment)
                            st.markdown(report)
                            
                            # Log report as output artifact
                            temp_report_path = os.path.join(os.getcwd(), "report.md")
                            with open(temp_report_path, "w", encoding="utf-8") as f:
                                f.write(report)
                            mlflow.log_artifact(temp_report_path, artifact_path="outputs")
                            try:
                                os.remove(temp_report_path)
                            except Exception:
                                pass
                            
                        else:  # Best Model + VLM (Cell Image ➜ VLM)
                            # Show DL prediction
                            with col2:
                                st.subheader("🔬 DL Model Prediction")
                                
                                if dl_label == 'Parasitized':
                                    st.error(f"**{dl_label}** ⚠️")
                                else:
                                    st.success(f"**{dl_label}** ✅")
                                
                                st.metric("Confidence", f"{dl_confidence:.2%}")
                                st.metric("Raw Score", f"{dl_prediction:.4f}")
                            
                            st.markdown("---")
                            
                            # VLM Analysis
                            st.subheader("👁️ OpenAI VLM Analysis")
                            
                            if not OpenAIClient:
                                st.warning("⚠️ OpenAI not configured. Please set OpenAI details.")
                                mlflow.log_param("vlm_status", "OpenAI not configured")
                                mlflow.log_param("model_name", dl_model_name)
                            else:
                                with st.spinner("🤖 Analyzing image with GPT-4 Vision..."):
                                    try:
                                        uploaded_file.seek(0)
                                        vlm_result = analyze_with_openai_vlm(uploaded_file, OpenAIClient, azure_deployment)
                                        
                                        if vlm_result['success'] and vlm_result['data']:
                                            data = vlm_result['data']
                                            
                                            # Format insights dynamically
                                            raw_insights = data.get('insights', 'N/A')
                                            if isinstance(raw_insights, list):
                                                insights_display = "\n".join([f"- {item}" for item in raw_insights])
                                            else:
                                                insights_display = str(raw_insights)

                                            # Format recommendations dynamically
                                            raw_recs = data.get('recommendations', 'N/A')
                                            if isinstance(raw_recs, list):
                                                recs_display = "\n".join([f"- {item}" for item in raw_recs])
                                            else:
                                                recs_display = str(raw_recs)
                                            
                                            # Display VLM results
                                            col_v1, col_v2 = st.columns(2)
                                            
                                            with col_v1:
                                                st.markdown(f"""
                                                **Infection Stage:** {data.get('infection_stage', 'N/A')}
                                                
                                                **Severity Score:** {data.get('severity_score', 'N/A')}/10
                                                
                                                **Confidence:** {data.get('confidence', 0)*100:.1f}%
                                                """)
                                            
                                            with col_v2:
                                                st.markdown(f"""
                                                **Insights:**
                                                
                                                {insights_display}
                                                """)
                                            
                                            st.markdown(f"""
                                            **Recommendations:**
                                            
                                            {recs_display}
                                            """)
                                            
                                            # Log VLM results to MLflow
                                            mlflow.log_param("model_name", "vlm")
                                            
                                            # Log full VLM details as output JSON artifact
                                            temp_vlm_path = os.path.join(os.getcwd(), "vlm_results.json")
                                            with open(temp_vlm_path, "w", encoding="utf-8") as f:
                                                json.dump(data, f, indent=4)
                                            mlflow.log_artifact(temp_vlm_path, artifact_path="outputs")
                                            try:
                                                os.remove(temp_vlm_path)
                                            except Exception:
                                                pass
                                        else:
                                            err_msg = vlm_result.get('error', 'Unknown error')
                                            st.error(f"VLM Analysis failed: {err_msg}")
                                            mlflow.log_param("vlm_status", "Failed")
                                            mlflow.log_param("vlm_error", err_msg)
                                            mlflow.log_param("model_name", dl_model_name)
                                    except Exception as e:
                                        st.error(f"VLM Analysis encountered an exception: {e}")
                                        mlflow.log_param("vlm_status", "Exception")
                                        mlflow.log_param("vlm_error", str(e))
                                        mlflow.log_param("model_name", dl_model_name)
                else:
                    st.error("Best model not found. Please run the notebook to train and save the best model first.")
    
    # ========== TAB 2: Bulk VLM Analysis ==========
    with tab2:
        st.header("Bulk VLM Analysis")
        st.markdown("Analyze up to 10 images using OpenAI VLM")
        
        if not OpenAIClient:
            st.warning("⚠️ OpenAI not configured. Please set OpenAI details.")
        
        uploaded_files = st.file_uploader(
            "Choose up to 10 images...",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="bulk"
        )
        
        if uploaded_files:
            if len(uploaded_files) > 10:
                st.error("⚠️ Maximum 10 images allowed. Please remove some images.")
                uploaded_files = uploaded_files[:10]
            
            st.write(f"📁 {len(uploaded_files)} images selected")
            
            # Display thumbnails in a compact 5-column grid
            cols = st.columns(5)
            for idx, uploaded_file in enumerate(uploaded_files):
                col = cols[idx % 5]
                with col:
                    uploaded_file.seek(0)
                    img = Image.open(uploaded_file)
                    st.image(img, caption=uploaded_file.name, width=120)
                    uploaded_file.seek(0) # Reset stream pointer after reading

            
            if st.button("🚀 Analyze All Images", type="primary"):
                if not OpenAIClient:
                    st.error("Please configure OpenAI details first.")
                else:
                    if True:
                        results = []
                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        
                        for i, uploaded_file in enumerate(uploaded_files):
                            progress_text.text(f"Analyzing image {i+1}/{len(uploaded_files)}...")
                            
                            # Start an independent run for each image in the bulk batch
                            with mlflow.start_run(run_name=f"Bulk_Item_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                                mlflow.log_param("analysis_mode", "Bulk VLM Analysis Item")
                                mlflow.log_param("image_name", uploaded_file.name)
                                mlflow.log_param("model_name", "vlm")
                                
                                try:
                                    uploaded_file.seek(0)
                                    img_obj = Image.open(uploaded_file)
                                    uploaded_file.seek(0)
                                    vlm_result = analyze_with_openai_vlm(uploaded_file, OpenAIClient, azure_deployment)
                                    
                                    if vlm_result['success'] and vlm_result['data']:
                                        data = vlm_result['data']
                                        
                                        raw_insights = data.get('insights', 'N/A')
                                        insights_str = " ".join(raw_insights) if isinstance(raw_insights, list) else str(raw_insights)
                                        
                                        raw_recs = data.get('recommendations', 'N/A')
                                        recs_str = "; ".join(raw_recs) if isinstance(raw_recs, list) else str(raw_recs)

                                        results.append({
                                            'Filename': uploaded_file.name,
                                            'Image': img_obj,
                                            'Infection Stage': data.get('infection_stage', 'N/A'),
                                            'Severity (1-10)': data.get('severity_score', 'N/A'),
                                            'Confidence': f"{data.get('confidence', 0)*100:.1f}%",
                                            'Insights': insights_str,
                                            'Recommendations': recs_str
                                        })
                                        
                                        # Log VLM results to this item's MLflow run using unified keys
                                        mlflow.log_param("label", data.get('classification', 'uninfected').lower())
                                        try:
                                            vlm_conf = data.get('confidence', 0.0)
                                            mlflow.log_metric("confidence", float(vlm_conf))
                                        except Exception:
                                            pass
                                        try:
                                            severity = data.get('severity_score', 0.0)
                                            mlflow.log_metric("prediction", float(severity) / 10.0)
                                        except Exception:
                                            pass
                                        
                                        # Log VLM insights details as artifact under this item's run
                                        temp_vlm_path = os.path.join(os.getcwd(), f"vlm_results_{uploaded_file.name}.json")
                                        with open(temp_vlm_path, "w", encoding="utf-8") as f:
                                            json.dump(data, f, indent=4)
                                        mlflow.log_artifact(temp_vlm_path, artifact_path="outputs")
                                        try:
                                            os.remove(temp_vlm_path)
                                        except Exception:
                                            pass
                                    else:
                                        results.append({
                                            'Filename': uploaded_file.name,
                                            'Image': img_obj,
                                            'Infection Stage': 'Error',
                                            'Severity (1-10)': 'N/A',
                                            'Confidence': 'N/A',
                                            'Insights': vlm_result.get('error', 'Unknown error'),
                                            'Recommendations': 'N/A'
                                        })
                                        mlflow.log_param("vlm_status", "Failed")
                                except Exception as e:
                                    try:
                                        uploaded_file.seek(0)
                                        img_obj = Image.open(uploaded_file)
                                    except Exception:
                                        img_obj = None
                                    results.append({
                                        'Filename': uploaded_file.name,
                                        'Image': img_obj,
                                        'Infection Stage': 'Error',
                                        'Severity (1-10)': 'N/A',
                                        'Confidence': 'N/A',
                                        'Insights': str(e),
                                        'Recommendations': 'N/A'
                                    })
                                    mlflow.log_param("vlm_status", "Exception")
                                    mlflow.log_param("vlm_error", str(e))
                            
                            progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        progress_text.text("Analysis complete!")
                        
                        # Display results in custom layout with side-by-side images
                        st.subheader("📊 Analysis Results")
                        for item in results:
                            with st.container():
                                st.markdown(f"#### 📄 File: `{item['Filename']}`")
                                col_img, col_data = st.columns([1, 2.5])
                                with col_img:
                                    if item['Image'] is not None:
                                        st.image(item['Image'], width=150)
                                    else:
                                        st.warning("No image preview available")
                                with col_data:
                                    st.markdown(f"""
                                    **Infection Stage:** {item['Infection Stage']}
                                    
                                    **Severity Score:** {item['Severity (1-10)']}/10
                                    
                                    **Confidence:** {item['Confidence']}
                                    
                                    **Insights:** {item['Insights']}
                                    
                                    **Recommendations:** {item['Recommendations']}
                                    """)
                                st.markdown("---")

                        # Filter out the PIL Image objects for pandas DataFrame & export
                        clean_results = [{k: v for k, v in item.items() if k != 'Image'} for item in results]
                        results_df = pd.DataFrame(clean_results)

                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"malaria_vlm_bulk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        

        else:
            st.markdown("""
            <div class="info-message">
                <h3>📁 Select Images for Bulk Analysis</h3>
                <p>Upload up to 10 blood cell images for VLM analysis.</p>
                <p><strong>Features:</strong></p>
                <ul>
                    <li>Infection stage classification</li>
                    <li>Severity scoring (1-10)</li>
                    <li>Detailed insights and recommendations</li>
                    <li>Download results as CSV</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # ========== TAB 3: Model Information ==========
    with tab3:
        st.header("ℹ️ Model Information")
        
        # Best Model Info
        st.subheader("🏆 Best Model")
        model_info = load_model_info()
        model_available, _ = check_best_model_status()
        
        if model_available and model_info:
            col1, col2 = st.columns(2)
            
            with col1:
                # Format metrics nicely as percentages
                acc_val = model_info.get('accuracy')
                prec_val = model_info.get('precision')
                rec_val = model_info.get('recall')
                f1_val = model_info.get('f1-score') or model_info.get('f1_score')
                
                acc_str = f"{acc_val*100:.2f}%" if isinstance(acc_val, (int, float)) else str(acc_val)
                prec_str = f"{prec_val*100:.2f}%" if isinstance(prec_val, (int, float)) else str(prec_val)
                rec_str = f"{rec_val*100:.2f}%" if isinstance(rec_val, (int, float)) else str(rec_val)
                f1_str = f"{f1_val*100:.2f}%" if isinstance(f1_val, (int, float)) else str(f1_val)
                
                st.markdown(f"""
                **Model Architecture & Performance:**
                
                | Property | Value |
                |----------|-------|
                | Model Name | {model_info.get('model_name', 'N/A')} |
                | Version | {model_info.get('version', '1.0')} |
                | Accuracy | {acc_str} |
                | Precision | {prec_str} |
                | Recall | {rec_str} |
                | F1-Score | {f1_str} |
                """)
            
            with col2:
                # Handle image size extraction safely
                img_size_raw = model_info.get('image_size', [128, 128])
                if isinstance(img_size_raw, list) and len(img_size_raw) >= 2:
                    img_size_str = f"{img_size_raw[0]}x{img_size_raw[1]}"
                else:
                    img_size_str = "128x128"
                    
                st.markdown(f"""
                **Training & Deployment Info:**
                
                | Property | Value |
                |----------|-------|
                | Image Dimensions | {img_size_str} |
                | Saved At | {model_info.get('saved_at', 'N/A')} |
                | Model Path | `best_model/best_model.h5` |
                """)
        else:
            st.warning("Best model not found. Please run the notebook to train and save the best model.")
        
        st.markdown("---")
        
        # OpenAI VLM Info
        st.subheader("🤖 OpenAI VLM (GPT-4 Vision)")
        
        openai_available = OpenAIClient is not None
        
        st.markdown(f"""
        **VLM Configuration:**
        
        | Property | Value |
        |----------|-------|
        | Model | GPT-4 Vision (gpt-4o) |
        | API Key Status | {'✅ Configured' if openai_available else '❌ Not Configured'} |
        | Capabilities | Image analysis, infection staging, insights generation |
        | Max Images (Bulk) | 10 |
        """)
        
        st.markdown("""
        **VLM Analysis Features:**
        
        - 🔬 Infection stage classification (Uninfected, Early, Mid, Advanced)
        - 📊 Severity scoring (1-10 scale)
        - 💡 Detailed visual insights
        - 📋 Clinical recommendations
        - 📥 CSV export for bulk analysis
        """)
        
        st.markdown("---")
        
        # System Architecture
        st.subheader("🏗️ System Architecture")
        
        # Render Mermaid diagram using HTML component
        mermaid_diagram = """
        <div class="mermaid">
        flowchart LR
            A[Input Image<br/>128x128] --> B[Best DL Model<br/>Keras/TensorFlow]
            B --> C{Analysis Mode}
            C -->|LLM| D[Generate Report<br/>Classification + Insights]
            C -->|VLM| E[GPT-4 Vision<br/>Infection Staging]
            D --> F[Output:<br/>Classification + Confidence<br/>Insights + Recommendations]
            E --> F
            
            style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
            style B fill:#fff3e0,stroke:#f57c00,stroke-width:2px
            style C fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
            style D fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
            style E fill:#fce4ec,stroke:#c2185b,stroke-width:2px
            style F fill:#fff8e1,stroke:#fbc02d,stroke-width:2px
        </div>
        """
        
        st.components.v1.html(f"""
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        {mermaid_diagram}
        <script>
            function renderMermaid() {{
                if (typeof mermaid === 'undefined' || !mermaid.initialize) {{
                    setTimeout(renderMermaid, 100);
                    return;
                }}
                mermaid.initialize({{startOnLoad: false}});
                mermaid.run();
            }}
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', renderMermaid);
            }} else {{
                renderMermaid();
            }}
        </script>
        """, height=300)

    # Footer
    st.markdown("---")
    st.markdown("""
        ⚠️ **IMPORTANT NOTICE :**
        - This tool is for RESEARCH and EDUCATIONAL purposes only
        - NOT approved for clinical diagnosis or treatment decisions
        - Results may be inaccurate - always verify with medical professionals
        - Do not delay seeking professional medical advice based on this tool's output
        - The developers assume no liability for misuse or misinterpretation
        """)


if __name__ == "__main__":
    main()
