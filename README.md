# HepaTox-AI: Predicting Drug-Induced Liver Injury

An AI-powered toolkit for predicting Drug-Induced Liver Injury (DILI) using machine learning and graph neural networks.

---

## Introduction: The Silent Threat in Drug Development

One of the most significant challenges in pharmacology is predicting adverse drug reactions before a new medication reaches patients. Among the most severe and common reasons for a drug to fail in clinical trials or be withdrawn from the market is **Drug-Induced Liver Injury (DILI)**. The liver, as the body's primary metabolic engine, is uniquely vulnerable to damage from new chemical compounds. Predicting DILI early is a critical mission that can save development costs, accelerate the creation of safer medicines, and, most importantly, protect patient health.

This project, **HepaTox-AI**, aims to address this challenge by building a machine learning model that can predict a drug's potential to cause liver injury based solely on its molecular structure. We will move beyond traditional methods by implementing advanced Graph Neural Network (GNN) architectures and focusing on creating an **explainable AI (XAI)** that not only predicts risk but also helps us understand the underlying chemical reasons for that risk.

---

## The DILIrank Dataset

The foundation of our model is the **DILIrank Dataset**, a publicly available, expert-curated resource provided by the U.S. Food and Drug Administration (FDA). This dataset is considered a gold standard for DILI research.

### Background

The DILIrank dataset provides a comprehensive list of FDA-approved drugs, each assigned a DILI severity and concern classification based on a thorough evaluation of clinical data and published literature. This expert labeling is what makes it possible to train a supervised machine learning model. The dataset contains 1,036 drug records, which, due to their high quality and information density, are sufficient for building a robust predictive model.

### Dataset Reference

* **Source:** U.S. Food and Drug Administration (FDA)
* **Homepage:** [Drug-Induced Liver Injury Rank (DILIrank) Dataset](https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-rank-dilirank-dataset)
* **Citation:** Chen, M., Suzuki, A., Thakkar, S. et al. DILIrank: the FDA-approved drug database for ranking drug-induced liver injury severity. *Hepatology* 64, 1579–1582 (2016).

### Dataset Contents

The dataset includes the following key columns that we will be using:

| Column Name     | Description                                                                                             |
| :-------------- | :------------------------------------------------------------------------------------------------------ |
| `LTKBID`        | A unique identifier for the compound within the Liver Toxicity Knowledge Base.                          |
| `Compound Name` | The common name of the drug.                                                                            |
| `Severity Class`| The drug's DILI classification (e.g., "No-DILI," "Less-DILI," "Most-DILI"). **This is our target label.** |
| `vDILIConcern`  | A simplified, three-level concern classification: "No," "Less," or "Most" concern for DILI.               |
