'''
Problem Number      :   Chapter 4 - Applied 16 from ITSL book

Problem statement : 

Using the Boston data set, fit classification models in order to predict whether a given suburb has 
a crime rate above or below the median. Explore logistic regression, LDA, naive Bayes, and KNN models 
using various subsets of the predictors. 
Describe your findings.

Hint: You will have to create the response variable yourself, using the variables that are contained in the Boston data set.

'''

from ISLP import load_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


def plot_roc(model_name, y_test, y_pred, y_selected_test, y_selected_pred):
   
   # Calculate false positive rate (fpr) and true positive rate (tpr)
   fpr, tpr, _ = roc_curve(y_test, y_pred)
   fpr_selected, tpr_selected, _ = roc_curve(y_selected_test, y_selected_pred)
   
   # Calculate Area Under the ROC Curve (AUC)
   roc_auc = auc(fpr, tpr)
   roc_auc_selected = auc(fpr_selected, tpr_selected)
   
   # Define color definitions for models
   color_dict = {'Model with selected features': 'blue', 'Model with all features': 'darkorange'}
   # Create custom legend handles with colors
   legend_handles = [Line2D([0], [0], color=color_dict[model], lw=2) for model in color_dict.keys()]
   # Create custom legend labels with colors and AUC scores
   legend_labels = [f'{model} (AUC = {roc_auc:.2f})' for model, roc_auc in zip(color_dict.keys(), [roc_auc_selected, roc_auc])]
   
    # Plot ROC curve
   plt.figure()
   plt.plot(fpr, tpr, color='darkorange', label='ROC curve (AUC = %0.2f)' % roc_auc)
   plt.plot(fpr_selected, tpr_selected, color='blue', label='ROC curve (AUC = %0.2f)' % roc_auc_selected)
   plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
   
   plt.xlim([0.0, 1.0])
   plt.ylim([0.0, 1.05])
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title(f'ROC Curve of {model_name}')
   
   plt.legend(legend_handles, legend_labels, loc="lower right")
   plt.show()


def model_evaluation(model_name, model, X_train, X_test, y_train, y_test):
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)

   print(f"{model_name} Performance:")
   print(classification_report(y_test, y_pred))
   print("-"*30)

   return y_pred

def correlation_plot(dataset, feature):
    corr_matrix = dataset.corr()
    crime_corr = corr_matrix[feature]

    # Plot correlation values
    plt.figure(figsize=(8, 6))
    sns.barplot(x=crime_corr.index, y=crime_corr.values)
    plt.xticks(rotation=90)
    plt.title(f'Correlation between {feature} and Other Features')
    plt.xlabel('Features')
    plt.ylabel('Correlation')
    plt.show()



if __name__ == "__main__":

    # Define Models
    models = {
        "logistic_regression": LogisticRegression(),
        "LDA": LinearDiscriminantAnalysis(),
        "naive_bayes": GaussianNB(),
        "KNN": KNeighborsClassifier()
    }

    Boston = load_data('Boston')
    
    # Create binary response variable - high crime 
    # If the crime rate is greater than the median, the label value is 1 (high crime).
    # If the crime rate is less than or equal to the median, the label value is 0 (low crime).
    y_column = "high_crime" 
    Boston[y_column] = (Boston["crim"] > Boston["crim"].median()).astype(int)

    # show correlation plot between y_column and other features to get the idea of the datset
    correlation_plot(Boston, y_column)

    X = Boston.drop(y_column, axis=1)

    # drop some features which has low correlation with high crime
    X_selected = Boston.drop(["chas", "rm", "ptratio", "medv", "high_crime"], axis=1)
    
    y = Boston[y_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split data into training and testing sets with selected columns
    X_selected_train, X_selected_test, y_selected_train, y_selected_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    for name, model in models.items():
        #  model_evaluation(model_name, model, X_train, X_test, y_train, y_test)
        prediction = model_evaluation(name, model, X_train, X_test, y_train, y_test)
        selected_prediction = model_evaluation(name, model, X_selected_train, X_selected_test, y_selected_train, y_selected_test)
        plot_roc(name, y_test, prediction, y_selected_test, selected_prediction)
