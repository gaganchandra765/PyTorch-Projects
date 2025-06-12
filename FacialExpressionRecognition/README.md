Facial Expression Recognition with Limited Labeled Data

Project Overview

This project was developed as part of a challenge by "Emotion AI" — a startup focused on building real-time facial expression recognition systems. The catch? We were only allowed a small labeled dataset but had access to a much larger collection of unlabeled facial images. Our task was to leverage both to build a robust classifier using classical machine learning tools.

Problem Statement

Emotion AI wants to classify facial expressions into categories such as:

Happy

Sad

Neutral

Angry

Surprise

Disgust

Fear

Due to ethical, privacy, and cost constraints, only a small labeled dataset (CK+ Extended) was available. In contrast, a large unlabeled dataset (LFW) was provided to learn generic facial features.

Solution Strategy

Our approach is based on combining unsupervised representation learning with supervised classification, structured in three parts:

1. Data Preprocessing

CK+ images are parsed from 48×48 grayscale pixel strings, resized to 100×100.

LFW images are grayscale and resized similarly.

Both datasets are flattened into 10,000-dimensional vectors.

2. Dimensionality Reduction with PCA

We performed Singular Value Decomposition (SVD) on the LFW dataset to explore variance structure.

A Principal Component Analysis (PCA) model was trained on the unlabeled LFW data (n=13,233) to extract a global "face subspace."

CK+ labeled images were projected into this 150-dimensional PCA space.

3. Expression Classification with SVM

Trained a Support Vector Machine (SVM) classifier on PCA-transformed CK+ data.

Used GridSearchCV to tune hyperparameters (kernel, C, gamma).

Stratified 70/30 train-test split was used.

Evaluation Results

Accuracy: >85% (varies depending on kernel and # components)

Precision / Recall / F1-score reported per class

Confusion Matrix visualized

Visualizations

Sample CK+ and LFW face grids

Top singular values (SVD)

Cumulative explained variance (PCA)

2D PCA scatter plot of CK+ data

Confusion matrix heatmap

Optional: SVM decision boundary in 2D PCA space

Dataset Sources

CK+ Extended (Labeled): https://www.kaggle.com/datasets/davilsena/ckdataset

LFW (Unlabeled): https://www.kaggle.com/datasets/jessicali9530/lfw-dataset or sklearn.datasets.fetch_lfw_people

Insights

SVD revealed that most variance in facial structure is captured in the top ~100 components.

PCA trained on the large unlabeled dataset offered much more generalizable features than if we had used only the small labeled CK+ set.

SVM worked surprisingly well in the PCA space even with limited supervision.

Limitations

The CK+ dataset has posed expressions — may not generalize to spontaneous expressions.

Faces are aligned; the system might struggle with large pose variations.

Real-time performance not tested.

Next Steps

Use semi-supervised learning (e.g., pseudo-labeling) to enhance the CK+ set.

Integrate a lightweight CNN trained on PCA features.

Benchmark vs deep learning models with few-shot learning.
