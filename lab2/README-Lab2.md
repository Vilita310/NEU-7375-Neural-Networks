### Multilayer Neural Network for Heart Disease Classification

## Project Overview

This project focuses on developing and training a multilayer (deep) neural network to classify the presence of heart disease using the UCI Heart Disease dataset. The project was implemented as part of Lab 1 and Lab 2 tasks.

### Model Architecture

The model follows this architecture:
- Layer 1: 10 neurons with ReLU activation
- Layer 2: 8 neurons with ReLU activation
- Layer 3: 8 neurons with ReLU activation
- Layer 4: 4 neurons with ReLU activation
- Layer 5: 1 neuron with Sigmoid activation (for binary classification)

### Steps Implemented
1. **Data Preprocessing**:
   - Handled missing values by filling with mean values.
   - One-hot encoded categorical variables.
   - Standardized the features using `StandardScaler`.

2. **Model Compilation**:
   - Optimizer: Adam
   - Loss function: Binary cross-entropy
   - Metrics: Accuracy

3. **Training**:
   - Epochs: 50
   - Batch size: 32
   - Validation split: 20%

4. **Evaluation**:
   - The model was evaluated on a separate test set to determine its accuracy and loss.

## Experiment Results
- **Test Accuracy**: Achieved approximately 76-77% accuracy.
- **Test Loss**: Observed consistent reduction in loss during training and validation.

### Screenshots and Logs

- Training and validation logs are attached below.

- Sample prediction outputs:****

  Epoch 1/50 7/7 - 1s - 85ms/step - accuracy: 0.7202 - loss: 0.6394 - val_accuracy: 0.7347 - val_loss: 0.6041 

  Epoch 2/50 7/7 - 0s - 4ms/step - accuracy: 0.7409 - loss: 0.6254 - val_accuracy: 0.7551 - val_loss: 0.5882 

  Epoch 3/50 7/7 - 0s - 4ms/step - accuracy: 0.7565 - loss: 0.6129 - val_accuracy: 0.7551 - val_loss: 0.5731 

  Epoch 4/50 7/7 - 0s - 4ms/step - accuracy: 0.7668 - loss: 0.6005 - val_accuracy: 0.7755 - val_loss: 0.5585 

  Epoch 5/50 7/7 - 0s - 4ms/step - accuracy: 0.7668 - loss: 0.5888 - val_accuracy: 0.7959 - val_loss: 0.5474 

  

  ......

  Epoch 45/50 7/7 - 0s - 3ms/step - accuracy: 0.8756 - loss: 0.3324 - val_accuracy: 0.8163 - val_loss: 0.3729 

  Epoch 46/50 7/7 - 0s - 3ms/step - accuracy: 0.8808 - loss: 0.3295 - val_accuracy: 0.8163 - val_loss: 0.3728 

  Epoch 47/50 7/7 - 0s - 3ms/step - accuracy: 0.8756 - loss: 0.3276 - val_accuracy: 0.8163 - val_loss: 0.3714 

  Epoch 48/50 7/7 - 0s - 4ms/step - accuracy: 0.8756 - loss: 0.3256 - val_accuracy: 0.8163 - val_loss: 0.3722 

  Epoch 49/50 7/7 - 0s - 3ms/step - accuracy: 0.8756 - loss: 0.3237 - val_accuracy: 0.8163 - val_loss: 0.3756 

  Epoch 50/50 7/7 - 0s - 4ms/step - accuracy: 0.8756 - loss: 0.3218 - val_accuracy: 0.8163 - val_loss: 0.3771 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8058 - loss: 0.5052  2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step 

  Test Loss: 0.5059189796447754, Test Accuracy: 0.7868852615356445 

  Predictions: [[0.47337997] [0.8293041 ] [0.8788618 ] [0.5105831 ] [0.94085425]] 

  True Labels: [[0] [1] [1] [1] [1]]

## Files Included

- `train_model.py`: Contains the complete source code for training the neural network.
- `README.md`: This documentation file.

## Running the Project
1. Ensure all dependencies (`pandas`, `numpy`, `scikit-learn`, `tensorflow`) are installed.
2. Run the `train_model.py` script to train and evaluate the model.

