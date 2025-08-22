# HepaTox: Predicting Drug-Induced Liver Injury

An AI-powered toolkit for predicting Drug-Induced Liver Injury (DILI) using machine learning and graph neural networks.

---

## Introduction: The Silent Threat in Drug Development

One of the most significant challenges in pharmacology is predicting adverse drug reactions before a new medication reaches patients. Among the most severe and common reasons for a drug to fail in clinical trials or be withdrawn from the market is **Drug-Induced Liver Injury (DILI)**. The liver, as the body's primary metabolic engine, is uniquely vulnerable to damage from new chemical compounds. Predicting DILI early is a critical mission that can save development costs, accelerate the creation of safer medicines, and, most importantly, protect patient health.

This project, **HepaTox**, was undertaken to address this challenge by building and comparing machine learning models that can predict a drug's potential to cause liver injury based solely on its molecular structure.

---

## The DILIrank Dataset

The foundation of our model is the **DILIrank Dataset**, a publicly available, expert-curated resource provided by the U.S. Food and Drug Administration (FDA). This dataset is considered a gold standard for DILI research.

### Background

The DILIrank dataset provides a comprehensive list of FDA-approved drugs, each assigned a DILI severity and concern classification based on a thorough evaluation of clinical data and published literature. This expert labeling is what makes it possible to train a supervised machine learning model. The dataset contains 1,036 drug records, which, due to their high quality and information density, are sufficient for building a robust predictive model.

### Dataset Reference

* **Source:** U.S. Food and Drug Administration (FDA)
* **Homepage:** [Drug-Induced Liver Injury Rank (DILIrank) Dataset](https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-rank-dilirank-dataset)
* **Citation:** Chen, M., Suzuki, A., Thakkar, S. et al. DILIrank: the FDA-approved drug database for ranking drug-induced liver injury severity. *Hepatology* 64, 1579–1582 (2016).

---

## Project Execution & Findings

The development followed a rigorous, iterative process to find the best possible model.

### Phase 1: Baseline Model (RandomForest)

We began by establishing a strong baseline. We converted the drug molecules into **molecular fingerprints** (numerical vectors representing chemical features) and trained a **RandomForest Classifier**. This simple but powerful model proved to be highly effective, achieving an ROC AUC score of **0.761**.

### Phase 2: Graph Neural Network (GNN) Exploration

Next, we explored a more advanced architecture, a **Graph Attention Network (GAT)**, to see if learning directly from the 2D molecular graph structure could improve performance. Through several rounds of fine-tuning (improving atom/bond features, adjusting the model, and weighting the loss function), the GNN learned successfully but did not surpass the baseline, achieving a peak ROC AUC of **0.609**.

### Phase 3: Hybrid Model

Our final experiment involved creating a **hybrid model**. We used the trained GNN as a sophisticated feature extractor to create "graph embeddings" and fed these, along with the original fingerprints, into the RandomForest classifier. While this approach showed promise, it also did not outperform our initial baseline.

---

## Final Results & Conclusion

After a thorough and systematic comparison, the conclusion is clear:

| Metric      | RandomForest (Baseline) | GNN-Only | Tuned Hybrid Model |
| :---------- | :---------------------- | :------- | :----------------- |
| **ROC AUC** | **0.761** | 0.609    | 0.677              |

The **RandomForest Classifier trained on molecular fingerprints is the winning model**. This project demonstrates a classic machine learning principle: the most complex model is not always the best. For this dataset, a robust, feature-based approach was the most effective strategy.

---

## How to Use the Final Model

The project has been structured into a reusable command-line tool.

**1. Install Dependencies:**
Ensure all required libraries are installed by running:

pip install -r requirements.txt

**2. Process Raw Data:**
Run the data processing script to generate the clean data.

python3 src/data_processing.py

**3. Train the Final Model:**
Run the training script to train the winning RandomForest model on the full dataset and save it.

python3 src/train.py

**4. Launch the GUI Predictor:**
Run the gui.py script to open the application. Enter a drug name and click "Predict Risk" to see the results.

python3 gui.py

