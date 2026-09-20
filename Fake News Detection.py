import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm, metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pickle
import nltk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Define stopwords
stop_words = set(stopwords.words('english'))


# Load the datasets
print("Loading datasets...")
fake_df = pd.read_csv("E:\Vs py\Fake news\Fake.csv")
true_df = pd.read_csv("E:\Vs py\Fake news\True.csv")

# Add labels to the datasets
fake_df['label'] = 'fake'
true_df['label'] = 'real'



# Combine datasets
news_dataset = pd.concat([fake_df, true_df], ignore_index=True)
print("Data loaded successfully.")
print(f"Dataset shape: {news_dataset.shape}")

# Define a function to clean text
def cleanup(text):
    if not isinstance(text, str):
        return ''
    text = re.sub('[^A-Za-z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

# Apply cleanup function to the text column
news_dataset['text'] = news_dataset['text'].fillna('').apply(cleanup)



# Drop rows with missing values in essential columns
news_dataset = news_dataset.dropna(subset=['text', 'label'])

# Check for class imbalance
print(news_dataset['label'].value_counts())

# Balance the dataset manually without SMOTE
fake_news = news_dataset[news_dataset['label'] == 'fake']
real_news = news_dataset[news_dataset['label'] == 'real']

min_count = min(len(fake_news), len(real_news))
balanced_fake_news = fake_news.sample(min_count, random_state=42)
balanced_real_news = real_news.sample(min_count, random_state=42)
news_dataset_balanced = pd.concat([balanced_fake_news, balanced_real_news])

# Split data into training and testing sets
X = news_dataset_balanced['text']
y = news_dataset_balanced['label']
count_vectorizer = CountVectorizer(stop_words='english')
X_vectorized = count_vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.33, random_state=53)
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")



# Train and evaluate SVM model
clf = svm.SVC(kernel='linear', class_weight='balanced')
clf.fit(X_train, y_train)
svm_pred = clf.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, svm_pred))

# Train and evaluate Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_train_acc = accuracy_score(y_train, lr_model.predict(X_train))
lr_test_acc = accuracy_score(y_test, lr_model.predict(X_test))
print(f"Logistic Regression Train Accuracy: {lr_train_acc}")
print(f"Logistic Regression Test Accuracy: {lr_test_acc}")

# Train and evaluate Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_train_acc = accuracy_score(y_train, dt_model.predict(X_train))
dt_test_acc = accuracy_score(y_test, dt_model.predict(X_test))
print(f"Decision Tree Train Accuracy: {dt_train_acc}")
print(f"Decision Tree Test Accuracy: {dt_test_acc}")

# Plot the confusion matrix for SVM
svm_cm = metrics.confusion_matrix(y_test, svm_pred, labels=['fake', 'real'])
def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plt.figure()
plot_confusion_matrix(svm_cm, classes=['fake', 'real'])
plt.title("SVM Confusion Matrix")
plt.tight_layout()
plt.show()

# Plot the confusion matrix for Logistic Regression
lr_pred = lr_model.predict(X_test)
lr_cm = metrics.confusion_matrix(y_test, lr_pred, labels=['fake', 'real'])
plt.figure()
plot_confusion_matrix(lr_cm, classes=['fake', 'real'])
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.show()

# Plot the confusion matrix for Decision Tree
dt_pred = dt_model.predict(X_test)
dt_cm = metrics.confusion_matrix(y_test, dt_pred, labels=['fake', 'real'])
plt.figure()
plot_confusion_matrix(dt_cm, classes=['fake', 'real'])
plt.title("Decision Tree Confusion Matrix")
plt.tight_layout()
plt.show()

# Save the Logistic Regression model and vectorizer
pickle.dump(count_vectorizer, open('count_vectorizer.pickle', "wb"))
pickle.dump(lr_model, open('finalized_model_LogReg.pkl', 'wb'))