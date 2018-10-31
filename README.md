# Detecting Fraud System (97% Recall)
Handling Imbalanced Datasets with Credit Card Fraud Case

This project is to apply machine learning to help detect credit card fraud, we will use the Credit Card Fraud dataset posted on Kaggle. You can download it here: https://www.kaggle.com/mlg-ulb/creditcardfraud. The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days. Our goal in this project is to construct models to predict whether a credit card transaction is fraudulent.

The main challenge of this dataset is being extremely imbalanced, which means out of nearly 284,807 transactions only 492 (0.172%) were labelled as fraudulent. This happens because machine learning algorithms are usually designed to improve accuracy by reducing the error. Thus, they do not take into account the class distribution / proportion or balance of classes.

As the problem description on Kaggle, usual confusion matrix techniques for measuring accuracy are biased and inaccurate, even the “null model” will have very high accuracy. To handle this situation we will need another way to measure the model’s success. In the analysis of highly imbalanced dataset, recall score & Cohen’s kappa coefficient are some good metrics.

The first thing any data scientist would do with this problem, after checking a few basic plots, is to run a logistic regression model as it is the standard method for a binary classifier with multiple features. The output from this run should look like this:

[[85288 9] [ 52 94]] Accuracy: 99.93% Recall: 64.38% Cohen Kappa: 0.755

You might initially think the model did a good job. After all, it got 99.92% of its predictions correct. That is true, but accuracy is not the reliable measure of a model’s effectiveness, as the model misclassified more 35% of fraudulent transactions.

A random forest is another model which perfectly suits this problem. Trying this classifier will get you results similar to the following:

[[85294    28]
 [    3   118]]
Accuracy: 99.96%
Recall: 97.52%
Cohen Kappa: 0.884

That's quite a bit better. The random forest model actual performs very well on this dataset. Note the accuracy went up slightly, but the other scores showed significant improvements as well, this means there are a few features which control almost all of the behaviour here.
