# AI-Based Fake News Detection

## Project Description

AI-Based Fake News Detection is a machine learning project
that predicts whether a news article is Fake or Real.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Streamlit
- TF-IDF
- Logistic Regression

## Project Structure

AI-Fake-News-Detection/

├── app.py
├── train_model.py
├── preprocess.py
├── requirements.txt
├── .env
├── README.md
├── data/
│   └── news.csv
├── model/
│   └── fake_news_model.pkl
└── assets/
    └── style.css

## How to Run

### Install dependencies

pip install -r requirements.txt

### Train the model

python train_model.py

### Start Streamlit

streamlit run app.py

## Machine Learning Algorithm

TF-IDF is used for converting text into numerical features.

Logistic Regression is used for classifying news as Fake or Real.