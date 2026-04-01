# Ai_vs_REAL
A custom, from-scratch Convolutional Neural Network (CNN) built in PyTorch to classify images as either "AI-generated" or "Real".  
It uses a dataset from kaggle consisting of AI and Real images, it also has more types of classifications but since the dataset is small, I trained my model only for binary classification.
The code does not use a pre trained model instead I chose to use the basic torch.nn library for this project since it was solely for learning.
I have used 5 fold cross validation, since the dataset includes only 995 samples, I chose this to eliminate irregularity in my results.

