**UCL Match Predictor: Neural Network vs. XGBoost**
This project implements a robust machine learning pipeline to predict UEFA Champions League match outcomes (Home Win, Draw, Away Win). It leverages historical match statistics, advanced Expected Goals (xG) data, and compares the performance of a deep learning approach against a gradient-boosted tree model.

🚀 **Key Features**
Data Integration: Merges standard match statistics (shots, saves, possession) with scraped Expected Goals (xG) and Delta_xG metrics.

Fuzzy Name Matching: Uses difflib to synchronize team names between different data sources (e.g., "Man City" vs. "Manchester City").

Advanced Preprocessing: Handles complex string-based stats, implements time-series chronological sorting, and uses MinMaxScaler for feature normalization.

Three-Way Data Split: Implements a strict 70/15/15 (Train/Validation/Test) split to ensure unbiased evaluation and effective early stopping.

Model Comparison: Direct head-to-head comparison between a TensorFlow/Keras Neural Network and an XGBoost Classifier.

🏗️ **Pipeline Architecture**
1. Data Engineering & Cleaning

The script processes raw match data by:

Converting date formats for multi-year seasonal analysis.

Parsing fractional statistics (e.g., "5 of 10 saves") into meaningful floats.

Normalizing possession percentages.

Filtering out incomplete rows to ensure numerical stability.

2. Model Architectures

Neural Network (TensorFlow)

Input Layer: Dynamic size based on feature count.

Hidden Layers: 16-node ReLU layer → 20% Dropout (for regularization) → 8-node ReLU layer.

Output Layer: 3-node Softmax (representing Away Win, Draw, Home Win).

Optimization: Adam optimizer with SparseCategoricalCrossentropy loss.

Training Protection: Implements Early Stopping monitoring val_loss with a patience of 15 epochs.

XGBoost Classifier

Type: Gradient Boosted Decision Trees.

Objective: multi:softmax for multi-class classification.

Reliability: Uses 5-Fold Cross-Validation during training to assess model stability across different data subsets.

📊 **Evaluation & Visualization**
The project generates several key visualizations to diagnose model performance:

Training History: Plots accuracy and loss curves for the Neural Network to detect overfitting.

Confusion Matrices: Provides a side-by-side comparison of how each model handles specific outcomes (crucial for identifying if models are struggling with "Draws").

Feature Importance: A gain-based plot showing which statistics (e.g., Delta_xG) most influence XGBoost's decisions.

🛠️ **Setup and Installation**
Environment: Designed for Python 3.10+ (compatible with Google Colab).

**Dependencies:**

Bash
pip install pandas numpy tensorflow xgboost matplotlib seaborn scikit-learn
Data Requirements: Requires champions_league_matches.csv and all_scraped_xg_stats.csv in the local directory.

📈 **Next Steps**
Market Value Integration: Incorporate team squad values to provide a financial "quality baseline."

Rolling Averages: Implement expanding window rolling stats to capture team form.

Odds Comparison: Compare model probabilities against bookmaker odds to calculate potential ROI (Backtesting).
