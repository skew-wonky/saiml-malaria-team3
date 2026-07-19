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

## Setup & Installation

### 1. Prerequisites
Make sure you have **Python 3.9+** and `pip` installed.

### 2. Install Dependencies
Navigate to the root project directory and install the required libraries:
```bash
pip install -r requirements.txt
```

### 3. Train & Export the Best Model
Before starting the Streamlit app, ensure that your best model weights (`best_model.h5`) and model metadata (`model_info.json`) are exported in the `best_model/` folder. You can train and save this model by executing the provided notebooks: `malaria_milestone2-V7(128x128).ipynb`

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

To enable VLM Visual Staging and Pathologist LLM Reports:
1. Open the Streamlit sidebar (`http://localhost:8501`).
2. Expand the **🤖 Azure OpenAI Credentials** form.
3. Input your API configuration details (Endpoint, API Key, Deployment Name, API Version).
4. Click **Submit Connection**.
5. Once validated, the form collapses automatically and updates its status indicator to `✅ VLM connection is ready`.

---

## Clinical Disclaimer
*This platform is developed for research and educational purposes. The AI-generated diagnostic summaries and pathophysiological insights do not represent absolute clinical diagnosis. Always correlate results with standard laboratory confirmation (blood smear microscopy, antigen RDTs, or PCR assays) by certified medical professionals.*
