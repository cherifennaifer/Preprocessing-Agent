# 🤖 AI Preprocessing Agent & Pipeline

A semi-automated web application built with Streamlit and CrewAI that replaces tedious manual data preprocessing with an intelligent, human-in-the-loop workflow. Upload your raw dataset, let the agents analyze and recommend a cleaning plan, review and override actions, and export a clean dataset instantly.

---

## ✨ Key Features

* **Multi-Step Wizard Interface:** Clean step-by-step UI tracking progress from upload to final export.
* **Smart Data Analysis:** Automatically profiles rows, columns, missing values, duplicates, and data types.
* **Human-in-the-Loop Validation (Step 4):** Review the agent's proposed filling methods (median, mode, datetime conversion, dropping) and override them to match domain knowledge before any code runs.
* **Resilient Multi-LLM Pool:** Dynamically rotates across free models (Groq, Gemini, OpenRouter, Cerebras, and local Ollama) with fallback handling to dodge rate limits.
* **Prompt-Injection Safeguards:** Protects backend execution by treating untrusted dataset values securely.
* **Comprehensive Reporting:** Generates a "Before vs. After" comparison report complete with metrics and missing-value distribution charts.

---

## 🛠️ Tech Stack

* **Frontend & UI:** [Streamlit](https://streamlit.io/)
* **Data Processing:** Pandas, NumPy, Scikit-learn, Openpyxl
* **Agents & LLMs:** CrewAI, LiteLLM, LangChain
* **Environment & Config:** Python-dotenv

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME