# Experiments Notebook Flow (experiments.ipynb)

## Goal
Train an ANN (binary classifier) to predict customer churn from the Churn_Modelling dataset, while saving preprocessing artifacts for reuse.

## Data Pipeline Overview
1. **Load data**
   - Read `Churn_Modelling.csv` into a DataFrame.

2. **Drop irrelevant columns**
   - Remove `RowNumber`, `CustomerId`, `Surname`.
   - These are identifiers and do not help model learning.

3. **Encode categorical features**
   - `Gender`: label-encoded to 0/1 using `LabelEncoder`.
   - `Geography`: one-hot encoded using `OneHotEncoder`.
   - The one-hot columns are appended back to the main DataFrame.

4. **Persist encoders**
   - Save `label_encoder_gender.pkl` and `onehotEncoder_geo.pkl` with `pickle`.
   - This ensures consistent preprocessing at inference time.

5. **Split features and target**
   - `X = data.drop('Exited')`
   - `y = data['Exited']`
   - Train/test split with `test_size=0.2` and `random_state=42`.

6. **Scale numeric features**
   - Use `StandardScaler` fitted on `X_train`.
   - Transform both `X_train` and `X_test`.

7. **Persist scaler**
   - Save to `scaler.pkl` for later inference.

## Model Pipeline Overview
1. **Model definition (Sequential)**
   - Input: number of features in `X_train`.
   - Hidden layers: 64 (ReLU), 32 (ReLU).
   - Output: 1 neuron with sigmoid for binary classification.

2. **Compile**
   - Optimizer: Adam with learning rate 0.01.
   - Loss: binary cross-entropy.
   - Metric: accuracy.

3. **Callbacks**
   - TensorBoard logging under `logs/fit/<timestamp>`.
   - EarlyStopping on `val_loss` with `patience=10` and `restore_best_weights=True`.

4. **Training**
   - Fit for up to 100 epochs with validation on the test split.
   - EarlyStopping prevents overfitting and keeps best weights.

5. **Save model**
   - Save to `model.h5`.

## TensorBoard
- `%load_ext tensorboard`
- `%tensorboard --logdir logs/fit`
- Visualizes loss/accuracy curves and histograms logged during training.

## Files Produced
- `label_encoder_gender.pkl`
- `onehotEncoder_geo.pkl`
- `scaler.pkl`
- `model.h5`
- `logs/fit/...` (TensorBoard logs)

## Quick Notes
- Preprocessing must be repeated in the exact same order at inference time.
- One-hot encoder returns a sparse matrix; it is converted to dense for DataFrame creation.
- Early stopping restores the best validation-loss weights, not the final epoch.
