# Advanced Portfolio Intelligence System
### Semantic Analysis for Low-Risk Capital Management

**Team:** Smeet Nalawade, Jelan Samatar, Hsin-Yu Shih

![System Architecture](images/system_architecture.png)

## 🚀 Project Overview
[cite_start]Equity research is inherently unscalable; analysts waste countless hours filtering through noisy, unstructured text to find relevant signals[cite: 8, 9]. 

This project is an **End-to-End Semantic Search Engine** that converts raw unstructured data into investment-grade intelligence. [cite_start]It ingests the Russell 1000 ETF holdings, enriches them with entity resolution (Wiki/Bing/yFinance), and uses LLM-powered summarization to build thematic investment baskets[cite: 3, 30].

## 🛠 System Architecture
The pipeline follows a robust **Extract-Transform-Load (ETL)** pattern:
1.  [cite_start]**Data Ingestion:** Automated scraping of Russell 1000 tickers with Selenium & yFinance fallbacks[cite: 30, 34].
2.  [cite_start]**Storage:** Live MongoDB Data Warehouse with a self-healing schema design[cite: 37, 81].
3.  [cite_start]**Processing:** LLM Map-Reduce summarization (Chunking → Extraction → Synthesis) to compress noise.
4.  [cite_start]**Vector Database:** Production embeddings generated using `nomic-embed-text` and `BAAI/bge-large`[cite: 139].
5.  [cite_start]**Hybrid Search:** Lucene-standard sparse search combined with dense vector retrieval[cite: 145].

## 📊 Key Results & Performance
We conducted a quantitative "Embedding Bake-Off" to select the optimal model for financial semantics.

* [cite_start]**Model Selection:** Nomic-based embeddings achieved the strongest sector-level separation (Silhouette Score: **0.064**) on LLM summaries[cite: 139, 141].
* [cite_start]**Search Precision:** The hybrid search engine achieved a **Mean Reciprocal Rank (MRR@10) of 0.4347**, confirming that relevant companies consistently rank at the top[cite: 148].
* [cite_start]**Investment Impact:** The "Cryptocurrency & Digital Assets" theme identified by the system materially outperformed the baseline over a 12-month backtest[cite: 205].

![Model Evaluation](images/model_evaluation.png)

## 🔧 Technical Highlights
* [cite_start]**Self-Healing Pipelines:** The system uses `$exists`-based flags and idempotent writes to detect missing data and only recompute what is absent[cite: 81, 85].
* [cite_start]**Map-Reduce Summarization:** Breaks down complex financial documents into material points before synthesis, reducing token costs and hallucination risk.
* [cite_start]**Human-in-the-Loop:** Final thematic classifications are reviewed to prevent misclassification errors[cite: 216].

## 📂 Repository Structure
```text
├── src/                 # Data Ingestion and Processing pipelines
├── models/              # Embedding generation and "Bake-Off" evaluation scripts
├── images/              # Project visualizations and architecture diagrams
├── tests/               # Unit tests for the search engine
└── README.md            # Project documentation