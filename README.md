# ANN Classification + Salary Regression Project

This folder contains a small, end-to-end ANN workflow used in the GenAI course to teach tabular preprocessing, model training, experiment tracking, and inference. The materials are written for students who are learning how to translate raw data into a working neural-network pipeline.

Special acknowledgement: Krish Niak is the supervisor of the overall project and learning flow.

## What each file does (tutor-style walkthrough)

### [app.py](app.py)
This is the lightweight application script that demonstrates how to use the trained model for inference. It typically loads the saved preprocessing artifacts (label encoder, one-hot encoder, scaler) and the trained neural network, then applies the same transformations to new user inputs before producing a prediction. As a student, treat this file as the "serving" or "deployment" example that mirrors how real products wrap ML models for end users.

Key learning points:
- How to load serialized preprocessing objects safely and consistently.
- How to reproduce the exact training-time feature pipeline at prediction time.
- How to call the model to generate a final output.

### [Churn_Modelling.csv](Churn_Modelling.csv)
This is the tabular dataset used for training and evaluation. It contains demographic and account features that are transformed into numeric inputs for the ANN. Even though the project focuses on salary regression, the same raw dataset can be used for other experiments like churn prediction or classification baselines.

Key learning points:
- Inspecting columns to decide which ones to drop (IDs, names) vs keep (features).
- Understanding categorical vs numeric feature handling.
- Splitting into features and target.

### [experiments.ipynb](experiments.ipynb)
This notebook is the playground where feature engineering, baseline models, and quick tests are run. It is meant for trying ideas without breaking the main training pipeline. As a learner, use it to compare different preprocessing decisions or model shapes.

Key learning points:
- Iterative experimentation workflow.
- Capturing results and observations during model tuning.
- Knowing when to move a stable experiment into the main training notebook.

### [hyperparametertuningann.ipynb](hyperparametertuningann.ipynb)
This notebook focuses on improving model performance by tuning ANN hyperparameters. You will typically see different layer sizes, activations, learning rates, or training epochs explored here.

Key learning points:
- Why hyperparameters matter for convergence and generalization.
- How to structure experiments to compare results fairly.
- Recognizing signs of overfitting and underfitting.

### [label_encoder_gender.pkl](label_encoder_gender.pkl)
This is a serialized `LabelEncoder` for the `Gender` column. It converts text labels (e.g., "Male", "Female") into numeric values so the ANN can process them.

Key learning points:
- Why categorical values must be encoded for neural networks.
- Why you must reuse the exact same encoder at inference time.

### [model.h5](model.h5)
This is a saved Keras model from the ANN training process. It contains the model architecture and learned weights.

Key learning points:
- How trained models are persisted for reuse.
- The difference between training a model and serving a model.

### [onehot_encoder_geo.pkl](onehot_encoder_geo.pkl)
This is a serialized `OneHotEncoder` for the `Geography` feature. It expands each geography into separate binary columns.

Key learning points:
- How one-hot encoding prevents the model from assuming an ordinal relationship between categories.
- Why `handle_unknown='ignore'` matters for production robustness.

### [prediction.ipynb](prediction.ipynb)
This notebook demonstrates how to load the model and preprocessing artifacts to make predictions on new data. It is the notebook version of what [app.py](app.py) automates.

Key learning points:
- Building a prediction pipeline step by step.
- Reproducing training-time preprocessing in a controlled environment.
- Interpreting model outputs in a regression task.

### [requirements.txt](requirements.txt)
This file lists Python package dependencies needed to run the notebooks and scripts in this folder. Use it to recreate a consistent environment.

Key learning points:
- Why dependency pinning matters for reproducibility.
- How to share a project so others can run it without guesswork.

### [salaryregression.ipynb](salaryregression.ipynb)
This is the main tutorial notebook that trains an ANN for salary regression. It covers data loading, preprocessing (label encoding, one-hot encoding, scaling), train-test split, model building, training with callbacks, evaluation, and saving artifacts.

Key learning points:
- End-to-end ANN workflow on structured data.
- Proper use of `StandardScaler` for numeric inputs.
- Early stopping and TensorBoard for monitoring training.
- Saving artifacts for reproducible inference.

### [scaler.pkl](scaler.pkl)
This is the serialized `StandardScaler` fitted on the training data. It ensures that new inputs are scaled in exactly the same way as the data used during training.

Key learning points:
- Why consistent scaling is critical for neural networks.
- Why you must never refit scalers on inference data.

## Suggested learning path
1. Start with [salaryregression.ipynb](salaryregression.ipynb) to understand the full pipeline.
2. Review [prediction.ipynb](prediction.ipynb) to see inference in a notebook.
3. Read [app.py](app.py) to understand the script-based version of inference.
4. Explore [experiments.ipynb](experiments.ipynb) and [hyperparametertuningann.ipynb](hyperparametertuningann.ipynb) for improvements and variations.

## Notes for students
- Always keep preprocessing artifacts together with the model; they are part of the model.
- If your results differ from expected values, verify the random seed and scaling steps.
- Use TensorBoard logs to diagnose training quality and overfitting.
