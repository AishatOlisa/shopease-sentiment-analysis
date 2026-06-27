 # ShopEase Sentiment Analysis

## Overview
An end-to-end multilingual sentiment analysis system for ShopEase Europe,
processing customer feedback across e-commerce platforms, social media,
and support channels.

## Project Phases

This project was executed in two phases due to a data quality discovery 
made during initial modelling.

### Phase 1 - Initial Dataset ( Synthetic Data)
The project began with a synthetic dataset of 120,000 customer reviews. 
Data quality assessment, cleaning, preprocessing, EDA, and classical NLP 
modelling were completed on this dataset. During model evaluation, both 
Naive Bayes and Logistic Regression achieved a perfect weighted F1-score 
of 1.0000, an immediate red flag. Investigation revealed the dataset 
was generated from only 444 unique template phrases with zero overlap 
between sentiment classes, meaning models were memorising templates 
rather than learning genuine sentiment patterns.

To confirm this finding, 80 genuine Amazon customer reviews were sourced 
from a public Hugging Face dataset and used to validate the trained 
models. Performance collapsed to approximately 0.48 weighted F1, 
confirming the synthetic dataset's limitation and the models' failure 
to generalise to real-world text.

**Phase 1 notebooks:** `notebooks/phase1_initial_dataset/`

### Phase 2 - Production Dataset (Genuine Amazon Reviews)
Following this discovery, a real dataset of 21,055 genuine Amazon 
customer reviews was provided for final model development. This phase 
repeats data quality assessment, cleaning, EDA, and classical modelling 
on authentic data, followed by fine-tuning DistilBERT - the model 
ultimately selected for production given the dataset's confirmed 
monolingual English composition.

## Workflow

### Phase 1 — Initial Dataset
| Stage | Notebook | Status |
|---|---|---|
| Data Quality Assessment | 01_data_quality_assessment.ipynb | ✅ Complete |
| Data Cleaning Pipeline | 02_data_cleaning_pipeline.ipynb | ✅ Complete |
| Text Preprocessing | 03_text_preprocessing_pipeline.ipynb | ✅ Complete |
| EDA — Sentiment & Category | 04_eda_sentiment_category.ipynb | ✅ Complete |
| EDA — Geographic Sentiment | 05_eda_geographic_sentiment.ipynb | ✅ Complete |
| EDA — Sentiment Drivers | 06_eda_sentiment_drivers.ipynb | ✅ Complete |
| Classical NLP Modelling | 07_classical_nlp_modelling.ipynb | ✅ Complete |
| Real-World Validation | 08_real_world_validation.ipynb | ✅ Complete |

### Phase 2 — Production Dataset
| Stage | Notebook | Status |
|---|---|---|
| Data Quality Assessment | 09_data_quality_assessment.ipynb | 🔄 In Progress |
| Text Preprocessing | 10_text_preprocessing.ipynb | ⬜ Pending |
| EDA — Sentiment & Category | 11_eda_sentiment_category.ipynb | ⬜ Pending |
| EDA — Geographic Sentiment | 12_eda_geographic_sentiment.ipynb | ⬜ Pending |
| EDA — Sentiment Drivers | 13_eda_sentiment_drivers.ipynb | ⬜ Pending |
| Classical Modelling | 14_classical_modelling.ipynb | ⬜ Pending |
| DistilBERT Fine-Tuning | 15_distilbert_modelling.ipynb | ⬜ Pending |

### Deployment
| Deliverable | Tool | Status |
|---|---|---|
| Interactive Dashboard | Power BI / Tableau | ⬜ Pending |
| Live Sentiment Classifier | Streamlit | ⬜ Pending |


**Phase 2 notebooks:** `notebooks/phase2_production_dataset/`

### Why This Structure Matters
Rather than discarding Phase 1 work, it is retained as evidence of 
methodological rigour, demonstrating data quality investigation, 
critical evaluation of suspiciously perfect results, and validation 
against external real-world data before drawing conclusions.

## Business Problem
ShopEase Europe generates thousands of multilingual customer reviews daily.
This system automates sentiment classification and topic extraction to
deliver actionable insights to marketing, product, and operations teams.

## Project Structure
```
shopease-sentiment-analysis/
├── data/
│   ├── raw/              # Original, immutable data (both datasets)
│   ├── processed/        # Cleaned and transformed data
│   └── external/         # Third-party validation data
├── notebooks/
│   ├── phase1_initial_dataset/    # Synthetic data - discovery & validation
│   └── phase2_production_dataset/ # Real data - production modelling
├── src/                  # Reusable source modules
├── reports/figures/      # Generated visualisations
├── models/                # Saved model artifacts
├── tests/                # Unit tests
├── config.yaml           # Project configuration
└── requirements.txt      # Dependencies
```


