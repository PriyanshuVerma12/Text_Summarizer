# Text Summarizer

## 📌 Project Overview
This project presents an **end-to-end NLP pipeline** for **text summarization**, leveraging a **Hugging Face model**. This involves **fine-tuning** the pretrained model (**google/pegasus-cnn_dailymail**) using the **Samsum dataset**, enabling efficient summarization of any given text.

## 📂 Dataset Details
The project utilizes the **Samsum dataset** from Hugging Face, which consists of:
- **Dialogue**: A conversational text sample.
- **Summary**: A human-annotated summary corresponding to the dialogue.
- **ID**: A unique identifier for each sample.

📌 **Dataset Link**: [Samsum Dataset](https://huggingface.co/datasets/Samsung/samsum)

## 🔄 Pipeline Overview
The project follows a structured **pipeline** comprising the following stages:

### **1️⃣ Data Ingestion**
- Retrieved the dataset from Hugging Face.
- Extracted and stored the data in a local directory.

### **2️⃣ Data Transformation**
- Processed the dataset to generate input features compatible with the model tokenizer.
- Transformed columns:  
  **Original**: `[id, dialogue, summary]`  
  **Transformed**: `[id, dialogue, summary, input_ids, attention_mask, labels]`

### **3️⃣ Model Training**
- Fine-tuned the **google/pegasus-cnn_dailymail** model utilizing the **Hugging Face Transformers** library.
- Employed **TrainingArguments** and **Trainer** with optimized hyperparameters to enhance performance.

### **4️⃣ Model Evaluation**
- Assessed model performance using **ROUGE Scores** on the test dataset.

## 🚀 Training & Prediction Pipelines
- Developed a **training pipeline** integrating all stages, from **data ingestion** to **model evaluation**.
- Implemented a **prediction pipeline** for summarizing arbitrary input text.

## 🌐 API Development
Designed APIs using **FastAPI** to facilitate model interaction:
- **`/train`** → Initiates model training.
- **`/predict`** → Generates a summary for a given input text.



