---
title: Spam Guard
emoji: 🦀
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 3.40.1
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Spam Guard - a Spam Detection Project

Welcome to the Spam Detection project! This document provides a comprehensive guide to building a robust spam detection model using the DistilBERT model. This project covers the entire pipeline, from data collection and preprocessing to model building, training, evaluation, and the creation of an interactive Gradio interface for users to classify messages as spam or not.

# Table of Contents
- [The Project](#the-project)
  - [Project Dependencies](#project-dependencies)
  - [Datasets](#datasets)
  - [Project Structure](#project-structure)
    - [Data Collection and Preprocessing](#data-collection-and-preprocessing)
    - [Model Building and Training](#model-building-and-training)
    - [Model Evaluation and Inference](#model-evaluation-and-inference)
    - [Interactive Gradio Interface](#interactive-gradio-interface)
- [The Application](#the-application)
  - [App Dependencies](#app-dependencies)
  - [Usage](#usage)
  - [Overview](#overview)
  - [Deployment](#deployment)
---

# The Project [Training and Fine tuning on Pre Trained LLM with Transformers] - [(Jupyter Notebook)](./Spam_detection.ipynb)

## Project Dependencies

To run this project, you'll need the following libraries installed(anyway they are mentioned in the jupyter notebook):

- `datasets` (preferred version: 2.14.4)
- `transformers` (preferred version: 4.31.0)
- `gradio` (preferred version: 3.40.1)
- `pandas` (preferred version: 1.5.3)
- `scikit-learn` (preferred version: 1.2.2)
- `tensorflow` (preferred version: 2.12.0)

If needed, You can create a virtual environment(Kernel):
- In Windows powershell,
```bash
python -m venv myenv
.\myenv\Scripts\Activate
```

- In Linux,
```bash
python3 -m venv myenv
source myenv/bin/activate
```

You can install these libraries using the following command:

```bash
pip install datasets transformers gradio pandas scikit-learn tensorflow
```

## Datasets

1. **Downloaded SMS Spam Collection Dataset:** A dataset comprising 5572 rows was obtained from Kaggle. This dataset can be found [here](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

2. **Utilized the `datasets` Module:** Using the `datasets` module, the "Deysi/spam-detection-dataset" was loaded, resulting in a spam detection dataset with 10900 rows. This dataset is available [here](https://huggingface.co/datasets/Deysi/spam-detection-dataset) in HuggingFace.

3. **Downloaded and Uploaded Spam Detector Dataset:** The Spam Detector dataset, downloadable from [here(Hugging Face)](https://huggingface.co/datasets/0x7194633/spam_detector), was acquired. Subsequently, the CSV dataset was uploaded to a drive. By mounting the drive, access to the dataset, containing 10900 rows, was established.

**Ultimately, these datasets were merged:** After converting them into pandas Dataframes, the datasets were combined into a unified DataFrame.

## Project Structure

The Spam Detection project is organized into several key phases:

### Data Collection and Preprocessing

1. **Mounting Google Drive:** We begin by mounting Google Drive to access files and data stored in your drive.

2. **Installing Required Libraries:** We install essential libraries `datasets`, `transformers`, and `gradio` using pip.

3. **Loading and Preparing Data:** Data is sourced from various repositories, including Kaggle, Hugging Face datasets, and CSV(from hugging face) files. The data is then combined into a single DataFrame to facilitate further processing.

4. **Preparing Data Labels:** Data labels are converted into binary values (`0` or `1`) using one-hot encoding.

5. **Splitting Data:** The dataset is divided into training and testing sets using the `train_test_split` function from scikit-learn.

### Model Building and Training

6. **Initializing Tokenizer:** The DistilBERT tokenizer from Hugging Face is initialized to tokenize the text data.

7. **Tokenizing Data:** Both training and testing data are tokenized using the previously initialized tokenizer.

8. **Creating TensorFlow Datasets:** TensorFlow datasets are created using the tokenized encodings and labels.

9. **Defining Training Arguments:** Training arguments for the `TFTrainer` from Hugging Face are defined. These include the number of epochs, batch sizes, evaluation strategy, and more.

10. **Initializing and Training Model:** Within the TensorFlow strategy scope, the DistilBERT model is created and initialized. The model is then trained using the `TFTrainer`, and a Progress Bar callback is added for monitoring.

### Model Evaluation and Inference

11. **Evaluating Model:** The trained model's performance is evaluated using the testing dataset.

12. **Saving Trained Model:** The trained model is saved in .h5 format as [spam_detection_model](./spam_detection_model/)

### Interactive Gradio Interface

13. **Inference on Sample Text:** The trained model is used to perform inference on a sample text.

14. **Creating Gradio Interface:** An interactive Gradio interface is created using the trained model. Users can input messages, and the model predicts whether the message is *spam* or not using the `process_input` core function.

---

# The Application

## App Dependencies

To run this application, you'll need the following libraries installed:

- `transformers` (preferred version: 4.31.0)
- `gradio` (preferred version: 3.40.1)
- `tensorflow` (preferred version: 2.12.0)

If needed, You can create a virtual environment(Kernel):
- In Windows powershell,
```bash
python -m venv spam_detection_virtual_environment
.\spam_detection_virtual_environment\Scripts\Activate
```

- In Linux,
```bash
python3 -m venv spam_detection_virtual_environment
source spam_detection_virtual_environment/bin/activate
```

You can install these libraries using the following command:

```bash
pip install transformers gradio tensorflow
```

## Usage

To utilize the Spam Detection Application:

1. Make sure that specified dependencies are installed
2. Run 
```bash
python app.py
```
3. Navigate to the URL of the Gradio application(will be printed in terminal)
4. Optional: append `'/?__theme=dark'` to the URL for dark theme
5. Input a SMS/Email/Message(you can also enter example by clicking on that) and submit it, You will get the corresponding output in the application
6. After completion, Interrupt the terminal by clicking 'Ctrc + C' to terminate the process from the particular port or close the application.


## Overview

The code follows a structured flow, starting with loading the model and tokenizer, defining the process_input function, creating the Gradio interface, and finally launching the interface for interactive spam detection.

1. **Import Libraries:**
   - Import necessary libraries for building the application, including TensorFlow, Transformers, and Gradio.

2. **Load Model and Tokenizer:**
   - Load the pre-trained DistilBERT model for sequence classification (`TFDistilBertForSequenceClassification`) from Hugging Face's Transformers library.
   - Load the tokenizer (`DistilBertTokenizerFast`) to preprocess input text.

3. **Define `process_input` Function:**
   - Define a function named `process_input` that takes an input text and performs the following steps:
     - Preprocess the text using the tokenizer.
     - Perform inference using the loaded model to obtain logits.
     - Convert logits to probabilities using softmax.
     - Determine the predicted class (spam or not spam) and map it to a label.
     - Return the predicted label and probabilities.

4. **Define Gradio Interface:**
   - Define the Gradio interface:
     - Set the title of the interface.
     - Create a list of example messages for testing the model.
     - Create Gradio components for input (`input_text`) and output (`probabilities_text`, `output_text`).
     - Shuffle the examples to show random samples.

5. **Initialize and Launch Interface:**
   - Initialize the Gradio interface using the `gr.Interface` class:
     - Provide the process_input function (`fn` parameter).
     - Specify the input and output components.
     - Set the title, examples, interpretation mode, theme, CSS, and examples per page.
   - Launch the Gradio interface using the `launch` method.
   - Optionally, you can print a message with instructions to access the dark theme.

## Deployment

- Deployed the Gradio Application in hugging face platform.
- You can interact with my application [here](https://huggingface.co/spaces/lostUchiha/spam-detector)

# Unique Challenges faced

- Searched a lot on google for resources gathered from different platforms(Kaggle, Hugging face) and merged all the datasets to gether using the `read_data()` function.
- Trained the pre-trained model using the this large dataset took upto 1hr.
- 

## Acknowledgments

This project leverages the capabilities of the DistilBERT model from Hugging Face, combined with the user-friendly Gradio library for creating an interactive interface.

For more information or inquiries, please feel free to reach out to the project author, Ravindra Mohith, at ravindramohith@gmail.com.
