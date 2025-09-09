import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
# Load dataset
df = pd.read_csv("Default_Fin.csv")

# Drop Index column if it exists
if "Index" in df.columns:
    df = df.drop("Index", axis=1)

# Fix missing values (if any)
df['Defaulted?'] = df['Defaulted?'].fillna(df['Defaulted?'].mode()[0])
df['Bank Balance'] = df['Bank Balance'].fillna(df['Bank Balance'].mode()[0])
df['Annual Salary'] = df['Annual Salary'].fillna(df['Annual Salary'].mode()[0])

# Features & Target
x = df.drop('Defaulted?', axis=1)
y = df['Defaulted?']

# Standardize features
scaler = StandardScaler()
x_scaled= scaler.fit_transform(x)


# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x_test)

# Evaluation
"""print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))"""

# Decision Tree Classifier
dtmodel = DecisionTreeClassifier(random_state=42)
dtmodel.fit(x_train,y_train)

y_pred_dt= dtmodel.predict(x_test)

# Evaluation
"""print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))"""

# Random Forest Classifier
ranfor = RandomForestClassifier(random_state=42)
ranfor.fit(x_train, y_train)

y_pred_rf= ranfor.predict(x_test)

# Evaluation
"""print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))"""

# Gradient Boosting Classifier
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(x_train,y_train)

t_pred_gbc= gbc.predict(x_test)

# Evaluation
"""print("Gradient Boosting Accuracy:", accuracy_score(y_test, t_pred_gbc))
print("Confusion Matrix:\n", confusion_matrix(y_test, t_pred_gbc))
print("Classification Report:\n", classification_report(y_test, t_pred_gbc))"""

# Voting Classifier

vote = VotingClassifier(estimators= [('lr', model), ('dt', dtmodel), ('rf', ranfor), ('gbc', gbc)], voting= 'soft')

vote.fit(x_train,y_train)

y_pred_vote = vote.predict(x_test)

#evaluation
#print("Voting Classifier Accuracy:", accuracy_score(y_test, y_pred_vote))


# Bagging Classifier
bag = BaggingClassifier(estimator=DecisionTreeClassifier(),
                        n_estimators=50, random_state=42)

bag.fit(x_train,y_train)

y_pred_bag= bag.predict(x_test)

# Evaluation
#print("Bagging Classifier Accuracy:", accuracy_score(y_test, y_pred_bag))

# AdaBoost Classifier
ada = AdaBoostClassifier(n_estimators=100, random_state=42)

ada.fit(x_train,y_train)
y_pred_ada= ada.predict(x_test)

# Evaluation
#print("AdaBoost Classifier Accuracy:", accuracy_score(y_test, y_pred_ada))

# Stacking Classifier
stack = StackingClassifier(estimators=[('lr', model), ('dt', dtmodel), ('rf', ranfor), ('gbc', gbc)],
                           final_estimator=LogisticRegression())

stack.fit(x_train,y_train)
y_pred_stack= stack.predict(x_test)
# Evaluation
#print("Stacking Classifier Accuracy:", accuracy_score(y_test, y_pred_stack))


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

def evaluate_model(name, model, x_test, y_test):
    y_pred = model.predict(x_test)
    print(f"\n{name}")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Evaluate all models
evaluate_model("Logistic Regression", model, x_test, y_test)
evaluate_model("Decision Tree", dtmodel, x_test, y_test)
evaluate_model("Random Forest", ranfor, x_test, y_test)
evaluate_model("Gradient Boosting", gbc, x_test, y_test)
evaluate_model("Voting Classifier", vote, x_test, y_test)
evaluate_model("Bagging Classifier", bag, x_test, y_test)
evaluate_model("AdaBoost Classifier", ada, x_test, y_test)
evaluate_model("Stacking Classifier", stack, x_test, y_test)
