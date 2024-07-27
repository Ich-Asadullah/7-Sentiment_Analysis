# Sentiment Analysis with LSTM

## Overview

This repository contains a Jupyter notebook and a pre-trained LSTM model for sentiment analysis. The model is trained on a subset of the IMDB reviews dataset, specifically 50% of the full dataset. This project demonstrates the use of Long Short-Term Memory (LSTM) networks to classify sentiment in text data.

## Files

- **`LSTM_Sentiment_Analysis.ipynb`**: A Jupyter notebook that contains the code for training and evaluating the LSTM model on the sentiment analysis task. The notebook includes data preprocessing, model training, and evaluation steps.
- **`LSTM_model.pth`**: A PyTorch model file containing the weights of the trained LSTM model. This file can be used to load the pre-trained model for inference or further fine-tuning.

## Installation

To run the notebook and use the model, you need to have the following packages installed:

- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `torch`
- `tensorflow`

You can install these packages using pip:

```bash
pip install torch numpy pandas scikit-learn torch tensorflow
```

## Usage

1. **Load the Model**: You can load the pre-trained model using PyTorch:
   code for loading and testing the model is available in notebook

2. **Run the Notebook**: Open the `LSTM_Sentiment_Analysis.ipynb` notebook in Jupyter and follow the steps to preprocess data, train the model, and evaluate its performance.

## Dataset

The model is trained on a portion of the IMDB reviews dataset. Only 50% of the dataset was used for training purposes. For more comprehensive analysis, consider using the full IMDB dataset.

## Contributing

Feel free to fork the repository and contribute. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
