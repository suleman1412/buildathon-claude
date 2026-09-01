# AI Research Paper Validator: Architecture & MVP Plan

This document outlines the architecture and execution strategy for building a scalable, autonomous AI agent capable of reading research papers, recreating their environments, and verifying their claimed metrics. 

## 1. System Architecture

To balance autonomy with security and cost, the system is divided into three core components:

*   **Ingestion & Extraction Engine:** Uses Vision-Language Models and LLMs to parse PDFs, preserving LaTeX/tables, and extracting structured JSON (code URLs, datasets, hyperparameters, claimed metrics).
*   **Execution Sandbox:** Serverless containerization using **Modal** (`modal.Sandbox`), allowing the agent to safely execute untrusted Python code, run terminal commands, and stream logs without local security risks.
*   **Multi-Agent Loop:** An orchestrated loop where the agent writes code, resolves dependencies, executes runs, and patches errors based on truncated stack traces.

## 2. Resource Optimization Strategy
**Budget Limits:** $100 Anthropic Credits & 2 Free-Tier Modal Accounts ($60/mo compute).

To avoid runaway costs from infinite agent loops and expensive GPU idle times:
*   **Model Routing:** Use **Claude 3 Haiku** for cheap JSON extraction. Reserve **Claude 3.5 Sonnet** strictly for complex debugging and code patching.
*   **Log Truncation:** Never feed full error logs to Claude. Use scripts to truncate stack traces to the last 30 lines to save input tokens.
*   **Strict Retries:** Cap the agent at 3 retries for dependency resolution and 5 for runtime errors. If it fails, mark the paper as "Failed to Replicate."
*   **Inference, Not Training:** Focus on evaluating pre-trained weights rather than full model training. 
*   **Hardware Caps:** Hardcode Modal to use cheap GPUs (`T4` or `A10G`), avoiding `H100`s unless strictly necessary for VRAM limits.

## 3. The MVP Pipeline (V1)

1.  **Extraction:** Haiku reads the PDF and outputs target metrics, GitHub repos, and HuggingFace assets.
2.  **Provisioning:** A Modal CPU instance clones the repository and builds the `requirements.txt`.
3.  **Execution & Debug Loop:** A Modal T4 Sandbox runs the evaluation script. If errors occur, the truncated stack trace is sent to Sonnet to generate a patch command (e.g., `pip install x`).
4.  **Verification:** The final extracted metric is compared against the paper's claimed metric, generating a Pass/Fail Markdown report.

## 4. Recommended Test Paper: *LoRA*

*   **Paper:** *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021)
*   **Target Task:** RoBERTa-large with LoRA adapters on the **SST-2** dataset.
*   **Target Metric:** **95.1% Accuracy**.
*   **Why it's ideal:**
    *   **Lightweight:** Fits easily in a 4GB VRAM T4 GPU.
    *   **Accessible Data:** SST-2 is natively hosted on Hugging Face; no auth required.
    *   **Accessible Code:** Clean implementation scripts are widely available.

## 5. Expected Pipeline Runtime & Costs

Under normal conditions, validating the LoRA paper will take **2 to 4 minutes** end-to-end.

| Pipeline Stage | Expected Runtime | Resource / Cost |
| :--- | :--- | :--- |
| **1. PDF Extraction** | 5 – 10 seconds | ~$0.005 (Claude 3 Haiku) |
| **2. Repo Setup & Deps** | 45 – 60 seconds | ~$0.001 (Modal CPU) |
| **3. Asset Download** | 10 – 15 seconds | High-bandwidth Modal network |
| **4. Inference & Eval** | 20 – 30 seconds | ~$0.005 (Modal T4 GPU) |
| **5. Debug Loop (if needed)** | 10 – 20s per retry | ~$0.03 per retry (Claude 3.5 Sonnet) |
| **6. Verification Report** | 2 – 5 seconds | Minimal Host CPU |

**Total Estimated Cost per Run:** ~$0.02 – $0.08 USD.