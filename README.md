# Diabetes Prediction with Transformers

This project explores the application of transformer-based architectures for predicting diabetes from tabular health data. We compare a custom TabTransformer against a traditional Deep Neural Network (DNN) baseline and an Ensemble Model to evaluate the effectiveness of self-attention mechanisms in capturing complex inter-feature relationships within structured data.

The TabTransformer, trained with a Focal Loss criterion, demonstrated superior performance, achieving an F1-score of 0.7907, AUC of 0.9779, and an accuracy of 96.57% on the validation set. This highlights the potential of transformer architectures to generalize well on complex, class-imbalanced medical datasets, as is consistent with the real world.

## Setup

1. Install the required packages:
```bash
pip install torch pandas numpy sklearn matplotlib imblearn
```

2. Place your diabetes dataset (CSV file) in the project directory, data should have the following columns in the same order:

```
gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, diabetes
```

Example row:

```
Female, 80.0, 0, 1, never, 25.19, 6.6, 140, 0
```

## Usage

Train a model using the following command:

```bash
python main.py --model [transformer|dnn] --epochs N --batch_size B --data_path path/to/data.csv
```

### Arguments:
- `--model`: Choose between 'transformer' or 'dnn' (default: transformer)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size for training (default: 64)
- `--data_path`: Path to the diabetes dataset CSV file (default: 'diabetes.csv')
- `--augment`: Use SMOTE data augmentation to handle class imbalance (optional)

### Example:
```bash
# Train TabTransformer for 30 epochs
python main.py --model transformer --epochs 30

# Train baseline DNN with data augmentation
python main.py --model dnn --augment
```

## Output

The training process will:
1. Save the best model checkpoint based on validation AUC
2. Generate training history plots showing:
   - Training loss over epochs
   - Validation metrics (AUC, F1, Accuracy) over epochs
3. Display final test performance metrics

## Project Structure

- `main.py`: Main training script with command-line interface
- `model.py`: Model architectures (TabTransformer and BaselineDNN)
- `dataloader.py`: Data preprocessing and loading utilities
- `utils.py`: Plotting and utility functions

### Conclusion

Take a look at the report.pdf file in this repository. Any useful findings from our side will be in there.
