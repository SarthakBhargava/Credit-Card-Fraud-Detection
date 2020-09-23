import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from time import sleep

# -------------------------------------Read File and locate rows----------------------------- #

df = pd.read_csv('creditcard.csv')

x = df.iloc[:, 1:30].values  # To get all rows of column 1-29
y = df.iloc[:, 30].values  # To Get all Values of Column 30

print('Read File Complete.............................................')
sleep(1)

# --------------------------------End of Read File and locate rows----------------------------- #


# -------------------------------Get Total Frauds and Normal----------------------------------- #

frauds = df[df['Class'] == 1]
normal = df[df['Class'] == 0]

# ----------------------------End of Getting Frauds and Normal----------------------------------- #


# -----------------------------Split into Training set and test set----------------------- #

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.25, random_state=1)

# -------------------------End Of Split into Training set and test set--------------------- #


# ----------------------------------standardisation---------------------------------------- #

stdsc = StandardScaler()
xtrain = stdsc.fit_transform(xtrain)
xtest = stdsc.fit_transform(xtest)
print('Data Standardized...................................................')
sleep(1)

# ----------------------------------End of standardisation---------------------------------------- #


# --------------------------------------Decision Tree ------------------------------------------ #

print("DecisionTreeClassifier Initialized..................................")

dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt.fit(xtrain, ytrain)
dt_y_pred = dt.predict(xtest)
dt_con_matrix = confusion_matrix(ytest, dt_y_pred)
dt_acc = accuracy_score(ytest, dt_y_pred)
dt_prec = precision_score(ytest, dt_y_pred)
dt_rec = recall_score(ytest, dt_y_pred)
dt_f1 = f1_score(ytest, dt_y_pred)
dt_MCC = matthews_corrcoef(ytest, dt_y_pred)

print("DecisionTreeClassifier Work Done....................................")
sleep(1)

# --------------------------------------End Decision Tree ------------------------------------------ #


# ------------------------------------------Naive Bays------------------------------------ #

print("Naive Bayes Classifier Initialized..................................")
gnb = GaussianNB()
gnb.fit(xtrain, ytrain)
gnb_y_pred = gnb.predict(xtest)
gnb_con_matrix = confusion_matrix(ytest, gnb_y_pred)
gnb_acc = accuracy_score(ytest, gnb_y_pred)
gnb_prec = precision_score(ytest, gnb_y_pred)
gnb_rec = recall_score(ytest, gnb_y_pred)
gnb_f1 = f1_score(ytest, gnb_y_pred)
gnb_MCC = matthews_corrcoef(ytest, gnb_y_pred)
print("Naive Bayes Classifier Work Done....................................")
sleep(2)

# ------------------------------------------End Naive Bays------------------------------------ #


# -----------------------------------------Control Code---------------------------------------- #

while 1:

    sleep(3)
    print('\n\nEnter What You want to see\n')
    print('1 To show top five rows')
    print('2 To show Basic Info of Data Set')
    print('3 To Get Shape of Normal')
    print('4 To Get Shape of Frauds')
    print('5 To Get Shape of data set and Target')
    print('6 Check Whether There is Null Value')
    print('7 To Plotting Graph ')
    print('8 To Print Training set shape')
    print('9 To Print Test set shape')
    print('10 Confusion Matrix of Division Tree Model')
    print('11 Value of Accuracy of Division Tree Model')
    print('12 Confusion Matrix of Naive Bayes Classifiers Model')
    print('13 Value of Accuracy of Naive Bayes Classifiers Model')
    print('14 Print Values of RandomForest Classifier Modal')

    command = input()

    if command == '1':
        print(df.head())

    if command == '2':
        print(df.info())

    if command == '3':
        print('Normal : ', normal.shape)

    if command == '4':
        print('Fraud : ', frauds.shape)

    if command == '5':
        print('Dataset : ', x.shape, '\nTarget : ', y.shape)

    if command == '6':
        print(df.isnull().values.any())

    if command == '7':

        filtered_data = df[['Time', 'Amount', 'Class']]
        creditCard_genuine = filtered_data.loc[filtered_data["Class"] == 0]
        creditCard_fraud = filtered_data.loc[filtered_data["Class"] == 1]

        print("    Enter Type of Graph to show Normal And Fraud Transctions Distributions")
        print('    1. Bar Graph')
        print('    2. Heat Graph')
        print('    3. Scatter Graph')
        print('    4. Graph')
        print('    5. Spread Graph (axis = time)')
        print('    6. Spread Graph (axis = amount)')
        print('    7. DistPlot')
        print('    8. Box Plot (class, time)')
        print('    9. Box Plot (class, amount)')

        g_type = input('    ')

        if g_type == '1':
            graph = pd.value_counts(df['Class'], sort=True)  # To set Data for graph
            graph.plot(kind='bar', rot=0)  # To set kind of data
            plt.title('Class Distribution of Transaction')  # Title of Data
            plt.xticks(range(2), ["Normal", "Fraud"])  # Naming of Bars
            plt.xlabel('Class Values')
            plt.ylabel('Number of Occurences')

        if g_type == '2':
            cor = df.corr()
            top_cor = cor.index
            plt.figure(figsize=(12, 9))
            heat_map = sns.heatmap(df[top_cor].corr(), vmax=.8, square=True)

        if g_type == '3':
            sns.set_style("whitegrid")
            sns.FacetGrid(df, hue="Class").map(plt.scatter, "Time", "Amount").add_legend()

        if g_type == '4':
            sns.set_style("whitegrid")
            sns.pairplot(filtered_data, hue="Class")

        if g_type == '5':

            plt.plot(creditCard_genuine["Time"], np.zeros_like(creditCard_genuine["Time"]), "o")
            plt.plot(creditCard_fraud["Time"], np.zeros_like(creditCard_fraud["Time"]), "o")

        if g_type == '6':

            plt.plot(creditCard_genuine["Amount"], np.zeros_like(creditCard_genuine["Amount"]), "o")
            plt.plot(creditCard_fraud["Amount"], np.zeros_like(creditCard_fraud["Amount"]), "o")

        if g_type == '7':
            sns.FacetGrid(filtered_data, hue="Class").map(sns.distplot, "Time").add_legend()

        if g_type == '8':
            sns.boxplot(x="Class", y="Time", data=df)

        if g_type == '9':
            sns.boxplot(x="Class", y="Amount", data=df)

        plt.show()


    if command == '8':
        print('Training Set', xtrain.shape, ", ", ytrain.shape)

    if command == '9':
        print('Test Set', xtest.shape, ", ", ytest.shape)

    if command == '10':
        print('Confusion Matrix of DecisionTree Model\n', dt_con_matrix)

    if command == '11':
        print("The accuracy of DecisionTree is ", str(dt_acc * 100))
        print("The precision of Decision Tree is ", str(dt_prec))
        print("The recall of Decision Tree is", str(dt_rec))
        print("The Matthews correlation coefficient of Decision tree is ", str(dt_MCC*100))

    if command == '12':
        print('Confusion Matrix of Naive Bays Model\n', gnb_con_matrix)

    if command == '13':
        print("The accuracy of Naive Bays is Model", str(gnb_acc * 100))
        print("The precision of Naive Bays Model is ", str(gnb_prec*100))
        print("The recall of Naive Bays Model is", str(gnb_rec*100))
        print("The Matthews correlation coefficient of Naive Bays Model is ", str(gnb_MCC*100))

    if command == '14':
        rfc = RandomForestClassifier()
        rfc.fit(xtrain, ytrain)
        rfc_y_pred = rfc.predict(xtest)
        rfc_con_matrix = confusion_matrix(ytest, rfc_y_pred)
        rfc_acc = accuracy_score(ytest, rfc_y_pred)
        rfc_prec = precision_score(ytest, rfc_y_pred)
        rfc_rec = recall_score(ytest, rfc_y_pred)
        rfc_MCC = matthews_corrcoef(ytest, rfc_y_pred)

        print("The accuracy of Naive Bays is Model", str(rfc_acc * 100))
        print("The precision of Naive Bays Model is ", str(rfc_prec * 100))
        print("The recall of Naive Bays Model is", str(rfc_rec * 100))
        print("The Matthews correlation coefficient of Naive Bays Model is ",str(rfc_MCC * 100))

        print('Confusion Matrix \n', rfc_con_matrix)

# ----------------------------------End Of Control Code---------------------------------------- #


# ------------------------------------End Of Program----------------------------------------------#
