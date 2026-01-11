# CognitionFlow: Agentic Systems Lab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ishan-1010/CognitionFlow/blob/main/CognitionFlow.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![AI Framework](https://img.shields.io/badge/Framework-Microsoft_AutoGen-blueviolet)

**CognitionFlow** is an applied AI engineering project that explores the design and behavior of **multi-agent systems** with persistent vector memory. The project demonstrates how multiple AI agents can collaborate, retain context over time, and generate analytical artifacts through structured orchestration rather than single-prompt interactions.

This repository serves as a **systems-level prototype**, focusing on architecture, reasoning flow, and agent coordination using **Microsoft AutoGen**.

---

## Generated Artifacts & Analysis
Unlike standard chatbots, this system generates tangible outputs. Below is an example of the system health analysis generated autonomously by the *Analyst Agent* after interpreting synthetic log data:

![System Health Plot](server_health.png)
*(Figure: Multi-agent generated analysis of system metrics)*

---

## Project Overview

Modern LLM applications often rely on single-agent prompt chains, which limit scalability and collaborative reasoning. This project investigates an **agent orchestration** approach where:

* **Distinct Roles:** Agents assume specialized personas (e.g., *Coordinator, Executor, Analyst*).
* **Vector Memory:** Long-term context is stored and retrieved using vector embeddings.
* **Autonomous Loops:** Agents collaborate to simulate problem-solving workflows without human intervention.
* **Artifact Generation:** Outputs include both reasoning traces and generated files (plots, reports).

## Key Features

* **Multi-Agent Architecture:** Explicit role assignment using the AutoGen framework.
* **Persistent Vector Memory:** Implementation of a vector database (Chroma/FAISS) with `sentence-transformers` for semantic recall.
* **Simulation Engine:** Synthetic data generation to test agent reasoning under pressure.
* **Demonstration-Focused:** Designed as an interactive Jupyter Notebook for transparency and explainability.

---

## Technologies Used

* **Core:** Python 3.10+, Jupyter Notebook
* **Orchestration:** Microsoft AutoGen
* **Memory/Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`), ChromaDB/Vector Store
* **Visualization:** Matplotlib, Seaborn
* **Inference:** Groq LPU (compatible with OpenAI/GPT-4o logic)

---

## How to Run

### Option 1: Run in Cloud (Recommended)
Click the badge above to open the notebook directly in Google Colab. You will need your own API keys.

### Option 2: Run Locally
**Prerequisites:** Python 3.10+ and an API Key (Groq or OpenAI).

1. **Clone the repository**
```bash
git clone [https://github.com/ishan-1010/CognitionFlow.git](https://github.com/ishan-1010/CognitionFlow.git)
cd CognitionFlow
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
# Or install basics manually if requirements.txt is missing:
# pip install pyautogen sentence-transformers matplotlib seaborn

```


3. **Set Environment Variables**
```bash
export GROQ_API_KEY="your_key_here"
# or export OPENAI_API_KEY="your_key_here"

```


4. **Launch the Notebook**
```bash
jupyter notebook CognitionFlow.ipynb

```



---

## Architecture (High-Level)

1. **Initialization:** Environment setup and secure credential loading.
2. **Memory Layer:** Vector store initialization for semantic retrieval.
3. **Agent Orchestration:** Agents are instantiated with specific system messages and permitted interaction paths.
4. **Simulation:** Synthetic data is injected into the environment.
5. **Execution & Analysis:** Agents collaborate to process data and generate insights.
6. **Artifacts:** Final plots and reports are saved to disk.

---

## Design Decisions & Trade-offs

* **Notebook-first approach:** Chosen for immediate visual feedback and transparency of the "Chain of Thought" during development.
* **Lightweight embeddings:** `all-MiniLM-L6-v2` was selected to balance performance and latency costs.
* **Agent orchestration:** Prioritized collaborative loops over monolithic chains to study emergent behavior.

---

## Future Work

* Modularize notebook logic into a deployable FastAPI service.
* Add quantitative evaluation of memory retrieval accuracy.
* Experiment with hierarchical agent teams (Manager -> Worker topology).

---

## License

This project is released under the MIT License.
