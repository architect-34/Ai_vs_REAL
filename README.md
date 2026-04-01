AI vs. REAL: Image Classification CNN

A custom Convolutional Neural Network (CNN) built from scratch using PyTorch. This project is designed to distinguish between AI-generated images and Real photographs.
Project Overview

The primary goal of this project was to explore the fundamentals of deep learning and computer vision without relying on pre-trained architectures (Transfer Learning). By building the model using the base torch.nn library, I gained a deeper understanding of feature extraction and the mechanics of gradient descent.

Dataset

    Source: [Kaggle - AI vs Real Image Dataset]

    Size: 995 samples.

    Scope: While the original dataset contains multiple sub-classifications, this project focuses on Binary Classification (AI vs. Real) to maintain high model reliability given the limited sample size.

Architecture

The model is a sequential CNN designed to balance depth with the risk of overfitting:

    3 Convolutional Layers: Carefully selected to extract hierarchical features (edges, textures, and shapes) without creating an overly complex parameter space.

    Max Pooling: Applied after each convolution to reduce spatial dimensions and focus on the most prominent features.

    Fully Connected Layer: A final linear layer that maps the extracted features to a single binary output (logit).

Technical Implementation

Because the dataset is relatively small (995 images), I implemented specific strategies to ensure the results are statistically significant:

    5-Fold Cross-Validation: Instead of a single train/test split, the data is divided into 5 folds. The model is trained 5 times, with each fold serving as the validation set once. This eliminates "luck of the draw" irregularities in accuracy reporting.

    From-Scratch Development: No pre-trained weights (like ResNet or VGG) were used. All kernels and weights were initialized and learned specifically for this task.

    Binary Logits: The model outputs raw logits, intended for use with BCEWithLogitsLoss for numerical stability during training.

 Key Learnings

    Managing the bias-variance tradeoff in small datasets.

    Implementing manual data pipelines using torch.utils.data.

    The mathematical impact of kernel sizes and strides on feature map dimensions.
