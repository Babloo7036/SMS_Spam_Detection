# SMS Spam Detection

## Overview
This project is an SMS Spam Detection system that classifies messages as spam or ham (not spam). The system utilizes extensive text preprocessing techniques and machine learning models to achieve high accuracy. A Streamlit-based web application is developed for user-friendly interaction, and the model is deployed on Render for accessibility.

webapp Link - [https://sms-spam-detection-1hjo.onrender.com]

## Features
- Extensive data cleaning and preprocessing.
- Machine learning model training and evaluation.
- Achieved 98% accuracy in classification.
- Web application built using Streamlit.
- Deployed on Render for public access.

## Dataset
Dateset Link - [https://github.com/Babloo7036/SMS_Spam_Detection/blob/main/spam.csv]

The dataset used in this project consists of SMS messages labeled as spam or ham. It was preprocessed to remove noise, standardize text, and convert messages into a suitable format for machine learning.

## Text Preprocessing Steps
1. Lowercasing the text.
2. Removing special characters, numbers, and punctuation.
3. Tokenization.
4. Removing stopwords.
5. Lemmatization.
6. Converting text into numerical form using TF-IDF or CountVectorizer.

## Machine Learning Model
Several models were trained and evaluated, with the best-performing model achieving 98% accuracy. The models tested include:
 - Support vector classifier
 - KNeighborsClassifier
 - MultinomialNB
 - DecisionTreeClassifier
 - LogisticRegression
 - RandomForestClassifier
 - AdaBoostClassifier
 - BaggingClassifier
 - ExtraTreesClassifier
 - GradientBoostingClassifier
 - XGBClassifier

The final model was chosen based on precision, recall, F1-score, and overall accuracy.

## Web Application (Streamlit)
A Streamlit web application was developed to provide an interactive interface for users to classify SMS messages. The application allows users to:
- Input an SMS message.
- Receive a real-time prediction of whether the message is spam or ham.

## Deployment
The application was deployed on Render, making it accessible to anyone with an internet connection. The deployment process included:
- Packaging the model and dependencies.
- Creating a Python script for Streamlit.
- Hosting the app on Render with an appropriate environment configuration.

## Installation & Usage
To run the application locally:
1. Clone the repository:
   ```sh
   git clone https://github.com/babloo7036/SMS-Spam-Detection.git
   ```
2. Navigate to the project directory:
   ```sh
   cd SMS_Spam_Detection
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```
5. Open the provided local URL in a browser to interact with the application.

## Technologies Used
- Python
- Pandas, NumPy (Data Processing)
- Scikit-learn (Machine Learning)
- NLTK (Natural Language Processing)
- Streamlit (Web Application)
- Render (Deployment)

## Contributing
If you'd like to contribute, feel free to fork the repository and submit a pull request with enhancements or bug fixes.

## Contact
For any queries or suggestions, feel free to reach out:
- GitHub: [Babloo](https://github.com/babloo7036)
- Email: babloo77018@gmail.com

---
This project showcases the power of machine learning in text classification and demonstrates deployment using modern web technologies. Hope you find it useful!

