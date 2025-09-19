# Hackathon Project Plan – AI Scientist Team (Amazon Strands + Next.js)

## 🚀 Problem Statement
How can we build a multi-agent AI Scientist system that automates research, analysis, experimentation, and visualization using **Amazon Bedrock (Strands agents)** and a **Next.js frontend**?

---

## 🔹 High-Level Idea
We create an **AI Scientist Team** powered by Amazon Strands agents, each acting as a specialist (Researcher, Data Collector, Experimenter, Critic, and Visualizer).  
Each agent contributes step by step, with outputs shown to the user in the UI **before the next agent runs**, ensuring transparency and collaboration.  

---

## 🔹 Agents & Responsibilities

1. **Research Agent**
   - Formulates hypotheses based on the user’s query.  
   - Searches open-source datasets or relevant research via APIs.  
   - Writes findings into **Bedrock shared memory**.

2. **Data Agent**
   - Pulls datasets from open repositories (Kaggle, AWS Open Data, HuggingFace Datasets).  
   - Cleans, formats, and stores in **Amazon S3**.  
   - Updates shared context for next steps.

3. **Experiment Agent**
   - Runs analyses or small-scale simulations using **Amazon SageMaker**.  
   - Produces structured results (metrics, findings, comparisons).  
   - Stores results in memory for Critic Agent.

4. **Critic Agent**
   - Reviews outputs for accuracy, bias, and limitations.  
   - Flags missing data, errors, or weak correlations.  
   - Adds structured feedback into shared context.

5. **Visualization Agent**
   - Generates clear, insightful visualizations using **QuickSight or Plotly in Next.js**.  
   - Emphasizes:  
     - **Core Findings**  
     - **Supporting Visuals**  
     - **Confidence & Metrics**  
     - **Comparisons**  
     - **Limitations/Notes**  
   - Communicates results to the user in the Next.js UI.

---

## 🔹 Orchestration & Communication
- Agents are implemented as **Strands Agents** inside **Amazon Bedrock Agent Core**.  
- They **do not talk directly**. Instead:  
  - Each agent writes its output into **shared memory**.  
  - Bedrock reads this memory and decides the next agent to run.  
- Triggering flow:  
  **Research → Data → Experiment → Critic → Visualization**  
- If one step fails, Bedrock triggers fallback paths (e.g., request more input from user).

---

## 🔹 Role of Amazon Bedrock
Bedrock acts as the **orchestrator and conductor**:  
- Maintains **shared memory**.  
- Dynamically triggers the right agent or AWS service.  
- Ensures workflow is **coordinated, adaptive, and quality-controlled**.  

---

## 🔹 Tech Stack
- **Frontend (UI):** Next.js + TailwindCSS  
- **AI Agents:** Amazon Bedrock (Strands Agents + Agent Core)  
- **Storage:** Amazon S3 (datasets, experiment results)  
- **Computation:** Amazon SageMaker (model training, analysis)  
- **Visualization:** Amazon QuickSight / Plotly.js inside Next.js  

---

## 🔹 User Flow
1. User enters a research question in the Next.js UI.  
2. Research Agent proposes a hypothesis → displayed to user.  
3. Data Agent finds datasets → displayed to user.  
4. Experiment Agent runs analysis → displayed to user.  
5. Critic Agent evaluates results → displayed to user.  
6. Visualization Agent produces final insights + charts.  

---

## 🔹 Execution Steps

1. **Setup Infrastructure**
   - Provision AWS services (S3, SageMaker, Bedrock access).  
   - Configure Bedrock Strands Agents in Agent Core.  

2. **Develop Agents**
   - Implement Research, Data, Experiment, Critic, Visualization agents as Bedrock Strands workflows.  
   - Connect agents to shared memory context.  

3. **Integrate Services**
   - Data Agent → Pull datasets → S3.  
   - Experiment Agent → Run models/analysis in SageMaker.  
   - Visualization Agent → Generate charts with QuickSight/Plotly.  

4. **Frontend Development**
   - Build Next.js UI with step-by-step results view.  
   - Add Tailwind styling + chart embedding.  

5. **Testing & Demo**
   - Run a real-world query end-to-end.  
   - Ensure each agent’s result is displayed before moving on.  
   - Record metrics (speed, accuracy, user clarity).  

---

## 🔹 Deliverables
- Working **Next.js app** showing multi-step agent collaboration.  
- Live **AWS Bedrock orchestration demo** with Strands agents.  
- Clear **visualizations and insights**.  
- Architecture + Sequence diagram.  

---

## 🔹 Architecture Summary
**User → Next.js UI → Bedrock Orchestration → Agents (Research → Data → Experiment → Critic → Visualization) → AWS Services (S3, SageMaker, QuickSight) → Back to UI**
