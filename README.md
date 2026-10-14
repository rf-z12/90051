# COMP90051 – Facial Expression Recognition Robustness

## About the Project
In this project, we are looking at how different image perturbations affect facial expression recognition models. 
We want to see whether simpler machine learning models and a deep learning model perform differently when the images are changed.

Our research question is:
**How do different image perturbations affect facial expression recognition across models with different levels of complexity?**

We are comparing three models:
- Logistic Regression
- RBF Support Vector Machine (SVM)
- EfficientNet-B0

We will first evaluate the models on clean facial images, then test them with Gaussian blur, Gaussian noise, and facial occlusion to see how their performance changes.

## Dataset
We are using the Expression in-the-Wild (ExpW) dataset, which contains face images with seven emotion labels:
- 0: Angry
- 1: Disgust
- 2: Fear
- 3: Happy
- 4: Sad
- 5: Surprise
- 6: Neutral
The original dataset needs to be downloaded separately by each team member from the official source. We are not uploading the original images, annotation files, or dataset archives to this repository because of the dataset usage conditions.

## Data Preparation
The data preparation code is in `data_preparation.ipynb`.

The notebook reads the original `label.lst` file and checks whether the images are available. It then adjusts the face bounding boxes so they stay within the image boundaries and removes faces smaller than 100 × 100 pixels.

If an image has more than one detected face, we keep the one with the highest detection confidence. We also use class-based subsampling to reduce the imbalance between emotion classes.

The selected images and their information are saved in a metadata CSV file.

For this step, we use:
- Random seed: `90051`
- Maximum of `2600` samples per class
- Minimum face size: `100 × 100` pixels

### How to Run
1. Download and extract the ExpW dataset.
2. Make sure `label.lst` is available on your computer.
3. Change the file paths in `data_preparation.ipynb` to match where you saved the dataset.
4. Run the notebook from the first cell to the last.
You will need Python, pandas, and Pillow.

## Model Evaluation
We plan to use nested cross-validation to evaluate the models.
- Outer CV: 10 folds
- Inner CV: 3 folds
- Random seed: `90051`

We will compare the results on the original images and the perturbed images to see which models are more affected by blur, noise, and occlusion.

## Team Members
- **Hinata:** Data preparation and Logistic Regression
- **Fanrui:** RBF-SVM
- **Jonathan:** EfficientNet-B0
We will work together on the evaluation, combining the results, and writing the final report.

## Reproducibility
This repository contains our code and instructions for running the experiments. The original ExpW images and annotation files are not included, so anyone running the code will need to obtain the dataset separately and follow its usage conditions.

We will use the same data preparation settings and evaluation splits across the models so that we can compare their results fairly.
