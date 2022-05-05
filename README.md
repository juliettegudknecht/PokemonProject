# Python for Data Science - Pokemon Project
Class project analyzing Pokemon legendary status

# Introduction 

The data set I chose is easily available online at kaggle.com if you search "Pokémon dataset", or is available here: https://www.kaggle.com/datasets/abcsds/pokemon

The reason I chose this data set is because I have a personal interest in Pokémon. Pokémon is a role-playing game in which players must assemble a small team of monsters in order to battle other monsters in a journey to become the "very best". Pokémon are classified into distinct categories, such as psychic or fairy, each having its own set of abilities. The game of chess might be compared to their battles. Pokémon develop strength and new skills as they acquire experience. There isn't necessarily a problem to be solved, but I thought it would be interesting to predict if a Pokémon is legendary or not based on their stats. Legendary Pokémon are a special type of Pokémon that are very powerful that are often associated with legends of creation and/or destruction within their endemic regions. Some legendary pokemon are based off real cultures and historic mythical creatures. I went through the processes of Data/Data Preprocessing, Exploratory Analyses, Methods, Detailed Results, Discussion, and References. The models I used are Logistic Regression, K-Nearest Neighbors, Decision Trees, and XG Boost. They are explained in detail in the methods section. The data set includes the following variables: Name of Pokemon, Type 1 of Pokemon, Type 2 of Pokemon, Total Score, HP, Attack Score, Defense Score, Special Attack, Special Defense, Speed, Generation and Legendary Status.

# Methods

I used the following models because I believe they work best for my data set that is mostly categorical. I chose a test/train split of .25 and .75. The reason for this is to prevent overfitting and to accurately evaluate the model.

A classification report contains the following:
Precision — What percent of your predictions were correct?
Precision refers to a classifier's ability to avoid labeling a negative instance as positive. It is calculated as the ratio of true positives to the sum of true positives and false positives for each class.

Precision:- Accuracy of positive predictions.

Precision = TP/(TP + FP)

Recall — What percent of the positive cases did you catch?
The capacity of a classifier to discover all positive cases is known as recall. It is calculated as the ratio of true positives to the sum of true positives and false negatives for each class.

Recall:- Fraction of positives that were correctly identified.

Recall = TP/(TP+FN)

F1 score — What percent of positive predictions were correct?
The F1 score is a weighted harmonic mean of precision and recall, with 1.0 being the highest and 0.0 being the lowest. F1 scores are lower than accuracy measurements because they factor on precision and recall.

F1 Score = 2*(Recall * Precision) / (Recall + Precision)

Support
The number of actual occurrences of the class in the provided dataset is known as support. Imbalanced support in the training data could reveal fundamental problems in the classifier's reported scores, necessitating stratified sampling or rebalancing. Support does not alter depending on the model; instead, it diagnoses the evaluation process.

I got most of these explanations from a website listed in my references, since they were not explained in detail in class.

## Logistic Regression
A logistic regression model is a model that has a certain fixed number of parameters that depend on the number of input features, and it outputs a categorical prediction. The model is very similar to linear regression, but for categorial outcome variables. The assumption of this model is that the outcome variable is binary, and it is. I'll be using the package sklearn for all of my models.

## K-Nearest Neighbors
The supervised learning technique K-nearest neighbors (KNN) is used for both regression and classification. By computing the distance between the test data and all of the training points, KNN tries to predict the proper class for the test data. Then choose the K number of points that are the most similar to the test data. The KNN algorithm analyzes the likelihood of test data belonging to each of the 'K' training data classes, and the class with the highest probability is chosen.

## Decision Trees
A decision tree is a flowchart-like tree structure in which each internal node represents an attribute test, each branch reflects the test's conclusion, and each leaf node (terminal node) stores a class label. By separating the source set into subgroups based on an attribute value test, a tree can be "trained." Recursive partitioning is the process of repeating this method on each derived subset. When all of the subsets at a node have the same value of the target variable, or when splitting no longer adds value to the predictions, the recursion is complete.

## XG Boost
XGBoost stands for Extreme Gradient Boosting, a popular algorithm in modern machine learning. Boosting is an ensemble modeling strategy that aims to create a strong classifier out of a large number of weak ones. It is accomplished by constructing a model from a sequence of weak models. To begin, a model is created using the training data. The second model is then created, which attempts to correct the faults in the first model. This approach is repeated until either the entire training data set is properly predicted or the maximum number of models has been added. In XGBoost, Decision trees are constructed sequentially. Weights are very significant. All of the independent variables are given weights, which are subsequently fed into the decision tree, which predicts outcomes. The weight of factors that the tree predicted incorrectly is increased, and these variables are fed into the second decision tree. These various classifiers/predictors are then combined to create a more powerful and precise model. It can be used to solve problems including regression, classification, ranking, and user-defined prediction.

# Results 
Here is a summary of all of the accuracy rates for the different models (when I ran them):

Logistic Regression: 96.5%

K-Nearest Neighbors: 98.0%

Decision Trees: 98.0%

XGBoost: 98.0%

The most accurate algorithm for this data set is decision trees / xgboost / knn. Meaning, all three of those classifiers correctly classified pokemon 98.0% of the time compared to other model that had a slightly lower accuracy rate (logistic regression).

# Discussion 
My goal for this project was to investigate legendary pokemon and create algorithms to predict legendary status. XGBoost, Decision Trees and KNN performed the best at a tie at 98.0% accuracy, out of the four models used. Logistic regression still performed very well at 96.5% accuracy. All of the models have good precision, recall, F1 score and support scores. The weighted average F1 score can be used to compare accuracy, the XGBoost, Decision Tree and KNN have a .98 score and the Logistic Regression has a .96 score, similar to the accuracy rates. Showing again, that the three models tie for the best accuracy rates and performance of the models in my case. This is interesting because usually XGBoost dominates in model performance, but there isn't a significant difference here in the models. XGBoost is created to outdo decision trees, so it is interesting that here they perform the same. I created multiple graphs and descriptive tables that showed legendary pokemon have higher statistics than the normal pokemon. The graphs were described throughout the notebook. Ways to improve include using more types of analyses, potentially incorportating hypothesis testing, and editing the data set to include more pokemon to be the most up to date. There are limitations to this study, including that the sample size was a little small at 800 pokemon. Another way I could improve is to test all of the assumptions of say, the logistic model like collinearity. Overall, I feel my analyses were appropriate for the dataset used.

# Thank you for reading my project!
