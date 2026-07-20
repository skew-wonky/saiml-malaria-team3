# AI-Powered Malaria Detection & Diagnostic Platform

An advanced research platform combining **Deep Learning (CNN)** and **Generative AI (LLM/VLM)** to detect and analyze malaria parasites in microscopic Giemsa-stained blood smear slides.

---

## Key Features

### 1. Single Cell Detection & Diagnosis
*   **DL Prediction + Pathologist LLM Report**: Employs the best-trained deep learning CNN model to classify the cell (`Parasitized` or `Uninfected`) with high confidence. The metrics (label, raw score, confidence) are sent to an **OpenAI Chat Completion API** (LLM) configured as a senior clinical pathologist to dynamically generate a detailed clinical interpretation, pathophysiological analysis, and clinical recommendations.
*   **Direct Visual VLM Staging**: Sends the raw image directly to the **Azure OpenAI GPT-4 Vision model** (VLM) for advanced morphological analysis, infection stage classification (Uninfected, Early, Mid, or Advanced Stage), severity scoring (1-10), visual insights, and patient recommendations.
*   **Dynamic Image Preprocessing**: Dynamically reads training metadata (`image_size`) to resize input images, ensuring they perfectly align with the dimensions expected by the trained model.

### 2. Bulk VLM Analysis
*   **Batch Image Processing**: Upload up to 10 microscopic images simultaneously for bulk visual analysis.
*   **Grid Visualization**: Responsive 5-column thumbnail grid with filename captions to preview uploaded cell slides.
*   **Independent MLflow Runs**: Logs each cell image as its own standalone MLflow tracking run (`Bulk_Item_<filename>_<timestamp>`), recording specific parameters, metrics (severity, confidence), and the raw VLM JSON response as an output artifact.
*   **Rich Results Layout**: Side-by-side visual card grid rendering the source slide image right next to its VLM clinical analysis.
*   **Summary Exports**: Clean CSV download containing full prediction summaries, insights, and recommendations for clinical archiving.

### 3. MLflow Tracking Integration
*   Organizes model runs under the `"Malaria_Detection_App"` experiment.
*   Logs parameters (such as `analysis_mode`, model metadata, classification labels) and metrics (such as model confidence, raw prediction values, VLM severity score).
*   Stores generated clinical markdown reports and raw VLM visual completions inside MLflow's local artifacts.

---

## Project Structure

```
saiml-malaria-team3-main/
├── app.py                              # Main Streamlit application entry point
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── best_model/
│   ├── best_model.h5                   # Trained Keras/TensorFlow CNN model weights
│   └── model_info.json                 # Model metadata (name, accuracy, image size, etc.)
└── streamlit_helper/
    ├── __init__.py                     # Package init
    ├── app_helpers.py                  # Core helper functions (model loading, prediction, VLM/LLM calls)
    └── prompts.py                      # Centralized GenAI prompts for VLM and LLM
```
### `app.py` — Main Application

The entry point for the Streamlit web app. It handles:
- MLflow experiment configuration (local file store or remote tracking server).
- Azure OpenAI client initialization and credential management (via Streamlit secrets, environment variables, or manual sidebar form).
- Sidebar system status (model status, VLM status, about section).
- The three main tabs (Detection, Bulk VLM Analysis, Model Information).
- MLflow run logging for single-image and bulk-image analyses.

### `streamlit_helper/*.py` — Helper functions for the main app
- Load the model and it's metadata
- runs inference with the model
- performs VLM analysis using structured prompt
- provides fallback response in case VLM response is not available

## Experiment Tracking and Model Behaviour

The platform uses **MLflow** to track every prediction and GenAI analysis as a structured experiment run under the `"Malaria_Detection_App"` experiment. Each run logs parameters (analysis mode, image name, model name, label), metrics (prediction score, confidence), and artifacts (LLM reports, VLM JSON responses) for full reproducibility and auditability.

The app supports both a local file-store backend (`mlruns/` directory) and remote MLflow tracking servers (via `MLFLOW_TRACKING_URI`).
Remote tracking on dagshub: https://dagshub.com/skew-wonky/saiml-malaria-team3.mlflow/#/experiments

## Setup & Installation

### 1. Prerequisites

Make sure you have **Python 3.9+** and `pip` installed.

### 2. Install Dependencies

Navigate to the root project directory and install the required libraries:
```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `streamlit` — Web application framework
- `tensorflow` — Deep learning model inference
- `Pillow` — Image processing
- `numpy` / `pandas` — Numerical and data operations
- `openai` — Azure OpenAI API client
- `mlflow` — Experiment tracking
- `httpx`, `protobuf`, `setuptools` — Pinned for compatibility

### 3. Best Model

Ensure that your best model weights (`best_model.h5`) and model metadata (`model_info.json`) are present in the `best_model/` folder.

---

## Running the Platform

Streamlit and the MLflow dashboard run as two separate local servers. Follow the sequence below to launch them:

### Step 1: Start the Streamlit Web Application

In your terminal, launch the Streamlit app:
```bash
streamlit run app.py
```
*   The web interface will automatically open at **`http://localhost:8501`**.
*   Upload single or batch slide images to execute evaluations, which will automatically initialize the local `mlruns/` tracking database.

### Step 2: Launch the MLflow UI Dashboard

Open a **second, separate terminal window** and run the following commands (depending on your shell) from the project directory:

*   **Windows PowerShell:**
    ```powershell
    $env:MLFLOW_ALLOW_FILE_STORE="true"
    mlflow ui
    ```
*   **Windows Command Prompt (cmd):**
    ```cmd
    set MLFLOW_ALLOW_FILE_STORE=true
    mlflow ui
    ```
*   **Linux / macOS:**
    ```bash
    export MLFLOW_ALLOW_FILE_STORE=true
    mlflow ui
    ```
*   Open your browser and navigate to **`http://localhost:5000`** to view tracked metrics, parameters, and download diagnostic artifact reports.

---

## GenAI Credentials Configuration

The app supports three methods for providing Azure OpenAI credentials (checked in priority order):

### Method 1: Streamlit Secrets (Recommended for deployment)

Create a `.streamlit/secrets.toml` file in the project root:
```toml
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
AZURE_OPENAI_API_KEY = "your-api-key"
AZURE_OPENAI_API_VERSION = "2024-02-01"
AZURE_OPENAI_DEPLOYMENT_NAME = "your-deployment-name"
```
When all four secrets are present, the app loads them securely and displays a `🔒 API credentials loaded securely from config.` status.

### Method 2: Environment Variables

Set the same four keys as environment variables:
```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_API_VERSION="2024-02-01"
export AZURE_OPENAI_DEPLOYMENT_NAME="your-deployment-name"
```

### Method 3: Manual Sidebar Form

1. Open the Streamlit sidebar (`http://localhost:8501`).
2. Expand the **🔑 VLM API Configuration** form.
3. Input your API configuration details (Endpoint, API Key, API Version, Deployment Name).
4. Click **Connect API Configuration**.
5. Once validated, the form collapses automatically and updates its status indicator to `✅ VLM connection is ready`.

> **Note:** The Azure OpenAI client is instantiated once and cached globally across Streamlit reruns for performance.

---

## Clinical Disclaimer
*This platform is developed for research and educational purposes. The AI-generated diagnostic summaries and pathophysiological insights do not represent absolute clinical diagnosis. Always correlate results with standard laboratory confirmation (blood smear microscopy, antigen RDTs, or PCR assays) by certified medical professionals.*
