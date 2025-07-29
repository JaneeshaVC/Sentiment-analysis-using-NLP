"Aim- to classify airline customer reviews as Positive, Neutral, or Negative"

# Import necessary libraries
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Load and clean dataset
data = pd.read_csv("D:/Masters/Semester 2/NLP/text classification system/airlines_reviews.csv")
#selected key columns only
data = data[['Airline', 'Reviews', 'Overall Rating', 'Verified']]
data.dropna(inplace=True)
data = data[data['Verified'].astype(str).str.upper() == 'TRUE']

# Sentiment labeling - numericals to sentiment labels
def sentiment_rating(rating):
    try:
        rating = float(rating)
        if rating <= 3:
            return "Negative"
        elif rating <= 7:
            return "Neutral"
        else:
            return 'Positive'
    except:
        return None

data["Sentiment"] = data['Overall Rating'].apply(sentiment_rating)
data.dropna(subset=['Sentiment'], inplace=True)
print(data['Sentiment'].value_counts())
#count plots were used for class distribution
sns.countplot(data=data, x='Sentiment')
plt.title("Sentiment Class Distribution")
plt.show()

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower() 
    text = re.sub(r'[^a-z\s]', '', text) 
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

data["Processed_reviews"] = data['Reviews'].apply(preprocess_text)

# Train-test split
X = data["Processed_reviews"]
y = data["Sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF Vectorization - to convert the text into numeric features.  For 5000 features. 
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Oversampling - As neutral were less, to balance the sentiments
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_vec, y_train)
print("Resampled class distribution:", Counter(y_train_resampled))

# Helper functions
model_results = []
classification_reports=[]

#storing the metrics in a list
def collect_metrics(name, y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    model_results.append({
        'Model': name,
        'Accuracy': round(accuracy, 3),
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'F1-Score': round(f1, 3)
    })

#create the confusion matrix 
def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=["Negative", "Neutral", "Positive"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Neutral", "Positive"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

#to take the classification report 
def store_classification_report(y_true, y_pred, model_name):
    report = classification_report(y_true, y_pred, target_names=["Negative", "Neutral", "Positive"])
    classification_reports.append(f"=== Classification Report: {model_name} ===\n{report}")

# Logistic Regression (Base)
lr_base = LogisticRegression(max_iter=1000)
lr_base.fit(X_train_resampled, y_train_resampled)
lr_pred = lr_base.predict(X_test_vec)
collect_metrics("LogReg (Base)", y_test, lr_pred)
plot_cm(y_test, lr_pred, "Logistic Regression (Base)")
store_classification_report(y_test, lr_pred, "Logistic Regression (Base)")

# Logistic Regression (Tuned)
lr_params = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['liblinear']
}
lr_grid = GridSearchCV(LogisticRegression(max_iter=1000), lr_params, cv=5)
lr_grid.fit(X_train_resampled, y_train_resampled)
lr_best = lr_grid.best_estimator_
lr_pred_tuned = lr_best.predict(X_test_vec)
print("Best Params (LR):", lr_grid.best_params_)
collect_metrics("LogReg (Tuned)", y_test, lr_pred_tuned)
plot_cm(y_test, lr_pred_tuned, "Logistic Regression (Tuned)")
store_classification_report(y_test, lr_pred_tuned, "Logistic Regression (Tuned)")

# SVM (Base)
svm_base = SVC()
svm_base.fit(X_train_resampled, y_train_resampled)
svm_pred = svm_base.predict(X_test_vec)
collect_metrics("SVM (Base)", y_test, svm_pred)
plot_cm(y_test, svm_pred, "SVM (Base)")
store_classification_report(y_test, svm_pred, "Support Vector Machine (Base)")

# SVM (Tuned)
svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svm_grid = GridSearchCV(SVC(), svm_params, cv=3)
svm_grid.fit(X_train_resampled, y_train_resampled)
svm_best = svm_grid.best_estimator_
svm_pred_tuned = svm_best.predict(X_test_vec)
print("Best Params (SVM):", svm_grid.best_params_)
collect_metrics("SVM (Tuned)", y_test, svm_pred_tuned)
plot_cm(y_test, svm_pred_tuned, "SVM (Tuned)")
store_classification_report(y_test, svm_pred_tuned, "Support Vector Machine (Tuned)")

# Random Forest (Base)
rf_base = RandomForestClassifier()
rf_base.fit(X_train_resampled, y_train_resampled)
rf_pred = rf_base.predict(X_test_vec)
collect_metrics("RF (Base)", y_test, rf_pred)
plot_cm(y_test, rf_pred, "Random Forest (Base)")
store_classification_report(y_test, rf_pred, "Random Forest (Base)")

# Random Forest (Tuned)
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=3)
rf_grid.fit(X_train_resampled, y_train_resampled)
rf_best = rf_grid.best_estimator_
rf_pred_tuned = rf_best.predict(X_test_vec)
print("Best Params (RF):", rf_grid.best_params_)
collect_metrics("RF (Tuned)", y_test, rf_pred_tuned)
plot_cm(y_test, rf_pred_tuned, "Random Forest (Tuned)")
store_classification_report(y_test, rf_pred_tuned, "Random Forest (Tuned)")

# Display summary
results_df = pd.DataFrame(model_results)
print("\n=== Model Performance Summary ===")
print(results_df.to_string(index=False))

print("\n\n=== All Classification Reports ===\n")
for report in classification_reports:
    print(report)
    print("-" * 60)

"""Support Vector Machine (Base) had the highest overall performance:"
#       - Accuracy: 77.1%
#       - F1-score: 0.76 (Weighted)
#       - Especially strong on Positive (0.85) and Negative (0.83) reviews

=== Model Performance Summary ===
         Model  Accuracy  Precision  Recall  F1-Score
 LogReg (Base)     0.748      0.750   0.748     0.749
LogReg (Tuned)     0.712      0.713   0.712     0.713
    SVM (Base)     0.771      0.753   0.771     0.758
   SVM (Tuned)     0.770      0.752   0.770     0.756
     RF (Base)     0.734      0.701   0.734     0.694
    RF (Tuned)     0.752      0.731   0.752     0.716"""