# Project: Sentiment Analysis with Pretrained Transformers

## Overview

This project implements a sentiment analysis pipeline using pretrained transformer models. The repository contains three major components:

1. **Dataset:** A CSV file containing review data.
2. **EDA File:** A notebook that performs Exploratory Data Analysis (EDA) and preprocessing on the dataset.
3. **Main Code File:** A notebook that trains and evaluates three different transformer models (DistilBERT, XLNet, and RoBERTa) for sentiment classification.

---

## Repository Structure

```
- 20191226-reviews.csv          # The dataset file
- EDA_Preprocessing_Notebook.ipynb # Notebook for EDA and preprocessing
- combined-file.ipynb           # Notebook for training and evaluating models
```

---

## Dataset

The `20191226-reviews.csv` file contains the following columns:

- **title**: Title of the review.
- **body**: Body of the review.
- **rating**: Numerical rating (1-5).

The dataset is preprocessed by:

- Concatenating the `title` and `body` columns.
- Mapping `rating` values to sentiments:
  - **1-2**: Negative
  - **3**: Neutral
  - **4-5**: Positive
- Removing null rows and duplicates.

---

## EDA

The `EDA_Preprocessing_Notebook.ipynb` file performs the following:

- Statistical summary of the dataset.
- Visualization of data distributions (e.g., ratings, word clouds).
- Preprocessing operations like:
  - Removing stopwords.
  - Tokenization.
  - Lemmatization and stemming.

---

## Main Code File

The `combined-file.ipynb` notebook includes:

1. **Model Training:**
   - Three transformer models (DistilBERT, XLNet, RoBERTa) are used.
   - The models are fine-tuned on the dataset.
2. **Evaluation Metrics:**
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Confusion Matrix
3. **Visualization:**
   - Generates confusion matrices.
   - Saves trained models and tokenizers to the specified directory.
4. **Model Saving:**
   - Models are saved to Google Drive for easy access.

---

## How to Run

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- Required libraries: pandas, numpy, matplotlib, seaborn, nltk, torch, transformers, wordcloud

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/Shaista7866/sentiment-analysis
   ```
2. Navigate to the repository directory:
   ```bash
   cd sentiment-analysis
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the EDA notebook:
   ```bash
   jupyter notebook EDA_Preprocessing_Notebook.ipynb
   ```
5. Execute the main notebook to train and evaluate models:
   ```bash
   jupyter notebook combined-file.ipynb
   ```

---

## Results

The trained models achieve the following:

- High accuracy and F1-scores for sentiment classification.
- Detailed confusion matrices for error analysis.

---

## Contribution

Feel free to submit issues or pull requests for improvements or bug fixes.

---

## License

This project is licensed under the MIT License.


