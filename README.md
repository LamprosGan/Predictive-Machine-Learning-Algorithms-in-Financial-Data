# Master's Thesis: Predictive Machine Learning Algorithms in Financial Data

This repository contains the code and experiments developed for my master's thesis: 
“Predictive Machine Learning Algorithms in Financial Data”. 
This project aims to explore and evaluate deep learning models for short-term stock price prediction using high-frequency Limit Order Book (LOB) data from stocks listed on the Athens Stock Exchange.

## 🔍 Project Overview

This project applies deep learning techniques to the problem of short-term price forecasting using Limit Order Book (LOB) data. It focuses on evaluating the effectiveness and generalizability of two models: **DeepLOB** and **TransLOB**. The work is organized into three main experiments:

### 📌 Experiment 1: Model Reproduction & Cross-Market Evaluation

1. **Reproduction of Published Models**  
   We re-implemented the DeepLOB and TransLOB architectures, based on their respective research papers, and validated their performance on the standard **FI-2010 benchmark dataset**.

2. **Generalization to the Athens Stock Exchange (Athex)**  
   After reproducing and confirming the reported results, we applied both models to **LOB data from five selected Athex-listed stocks** to investigate their robustness in a different market setting, characterized by lower liquidity and different trading behaviors.

### 📌 Experiment 2: Per-Stock Evaluation on Athex Data

In the second phase, we conducted a **per-stock evaluation** by training and testing the DeepLOB model individually on each of the five Athex stocks. This allowed us to:

- Assess model performance at the individual security level,
- Explore whether certain stocks yield more predictable LOB patterns,
- Compare performance variation across assets within the same exchange.

### 📌 Experiment 3: Deep Reinforcement Learning for Automated Trading

Building on the predictive models, we implemented a **Deep Reinforcement Learning (DRL)** approach inspired by the paper _“Deep Reinforcement Learning for Active High Frequency Trading”_ by Wei et al. (2019). The goal was to:

- Train a trading agent that learns an optimal policy for submitting buy/sell/hold orders based on the LOB state,
- Simulate a real-time trading environment with realistic constraints (latency, transaction costs, market impact),
- Evaluate the agent's profitability and stability on Athex data.

Although results were preliminary, this experiment demonstrated the potential and challenges of applying RL in realistic market scenarios.

## 📦 Datasets

The project uses two main datasets:

### 1. FI-2010 Benchmark Dataset

This is the standard dataset used in DeepLOB and many related papers. It contains high-frequency Limit Order Book data for five stocks from Nasdaq Stock Market.

- 📥 **Download link**: [FI-2010 Dataset](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649)


### 2. Athex LOB Data (Example Subset)

A subset of messages as they arrived from the Athens Stock Exchange (Athex). The full dataset used in this thesis is not publicly available due to size and licensing restrictions. However, a small sample is provided for demonstration purposes.

- 📥 **Download example data (stocks_sample.pkl)**: [Google Drive Link](https://drive.google.com/drive/folders/1XxX74Jau7vuTtma6S4bcDYC1NbFxDI60?usp=drive_link)

## 🧭 Project Workflow

The implementation of this thesis is structured into four main stages:

---

### 1️⃣ Data Mining (Only for Athex Data)

Raw Messages from the Athens Stock Exchange (Athex) was collected from a market maker's feed. This stage includes:

- Parsing raw message and order book files.
- Saving cleaned datasets for further processing.

---

### 2️⃣ Data Preparation (Only for Athex Data)

The raw Athex data was transformed into a format compatible with DeepLOB and TransLOB models. This includes:

- Windowing LOB data into sequences.
- Creating train/val/test splits.
- Standardizing features.
- Saving Pickle-ready files.

---

### 3️⃣ Experiments

#### 📌 Experiment 1 – Reproduce Published Results

- ✅ **Data**: FI-2010 dataset + Processed Athex data
- ✅ **Models**: DeepLOB & TransLOB
- ✅ **Goal**: Confirm performance on FI-2010 and test generalization on Athex
- ✅ **Provided**: Pre-trained weights + Evaluation metrics

---

#### 📌 Experiment 2 – Per-Stock Evaluation on Athex

- ✅ **Data**: Processed Athex data
- ✅ **Model**: DeepLOB
- ✅ **Goal**: Evaluate model performance separately on each Athex stock
- ✅ **Provided**: Trained weights per stock + Metrics + Visualizations

---

### 4️⃣ DRL Agent for Automated Trading

- ✅ **Data Source**: `data_collect.ipynb` (samples from Athex LOB stream)
- ✅ **Goal**: Train a Deep Reinforcement Learning agent to issue buy/sell/hold actions
- ✅ **Provided**: Training logs + evaluation results + saved models

📁 Output: `/runs/`, `/runs_results/`

---

### 🎯 Summary

- You **do not need to retrain** any model to view results.
- All **model weights, configs, and outputs** are provided.
- Every stage is modular, and you can run only the part you're interested in.



## 📚 References & Resources

- **DeepLOB Paper**: [Zhang, Zohren, & Roberts, 2019 – DeepLOB: Deep Convolutional Neural Networks for Limit Order Books](https://arxiv.org/pdf/1808.03668)  
  Original implementation: [https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books)

- **TransLOB Paper**: [Wallbridge, 2020 – Transformers for limit order books](https://arxiv.org/pdf/2003.00130)  
  Original implementation: [https://github.com/jwallbridge/translob)

- **DRL for LOBs**: [Antonio, Turiel, Marcaccioli, Cauderan, & Aste , 2021 – Deep Reinforcement Learning for Active High Frequency Trading](https://arxiv.org/abs/2101.07107)

- **My Thesis Repository**: [GitHub – LamprosGanias/Predictive-Machine-Learning-Algorithms-in-Financial-Data](https://github.com/LamprosGan/Predictive-Machine-Learning-Algorithms-in-Financial-Data)
- **My Thesis Paper**: []
