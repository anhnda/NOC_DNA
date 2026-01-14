# DNA Forensics: Number of Contributors Prediction

This project implements machine learning models to predict the number of contributors (NOC) in DNA forensic samples using the PROVEDIt dataset.

## Dataset

**PROVEDIt_1-5-Person CSVs Filtered (5sec subset)**

The dataset contains DNA mixture samples with 1-5 contributors, processed with a 5-second analysis window. Features are extracted from STR (Short Tandem Repeat) markers commonly used in forensic DNA analysis.

### STR Markers Included:
- AMEL (Amelogenin)
- CSF1PO, D10S1248, D12S391, D13S317, D16S539, D18S51, D19S433
- D1S1656, D21S11, D22S1045, D2S1338, D2S441, D3S1358
- D5S818, D7S820, D8S1179
- DYS391 (Y-chromosome)
- FGA, SE33, TH01, TPOX, vWA
- Yindel (Y-chromosome insertion/deletion)

## Models

Three machine learning models were evaluated for NOC prediction:

1. **CNN (1D Convolutional Neural Network)**
   - 3 convolutional blocks (64 → 128 → 256 channels)
   - Batch normalization and max pooling
   - 3 fully connected layers with dropout
   - 150 training epochs
   - Automatic GPU/CPU detection

2. **XGBoost**
   - Gradient boosting classifier
   - 100 estimators
   - Maximum depth: 6
   - Learning rate: 0.1

3. **TabPFN**
   - Tabular Prior-Fitted Network
   - Pre-trained transformer for tabular data
   - Zero hyperparameter tuning required

## Performance Results

All models were evaluated using 5-fold stratified cross-validation.

### CNN
| Metric | Micro | Macro |
|--------|-------|-------|
| Precision | 0.9455 ± 0.0083 | 0.8800 ± 0.0225 |
| Recall | 0.9455 ± 0.0083 | 0.7962 ± 0.0382 |
| F1-Score | 0.9455 ± 0.0083 | 0.8329 ± 0.0300 |

### XGBoost
| Metric | Micro | Macro |
|--------|-------|-------|
| Precision | 0.9414 ± 0.0100 | 0.8846 ± 0.0278 |
| Recall | 0.9414 ± 0.0100 | 0.7672 ± 0.0421 |
| F1-Score | 0.9414 ± 0.0100 | 0.8170 ± 0.0346 |

### TabPFN (Best Performance)
| Metric | Micro | Macro |
|--------|-------|-------|
| Precision | 0.9704 ± 0.0096 | 0.9442 ± 0.0301 |
| Recall | 0.9704 ± 0.0096 | 0.8859 ± 0.0365 |
| F1-Score | 0.9704 ± 0.0096 | 0.9122 ± 0.0312 |

**TabPFN achieves the best overall performance** with micro F1-score of 0.9704 and macro F1-score of 0.9122.

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost tabpfn torch
```

## Usage

### 1. XGBoost Model
```bash
python XGB.py
```

### 2. TabPFN Model
```bash
python Tab.py
```

### 3. CNN Model
```bash
python CNN.py
```

## File Structure

```
.
├── config.py                           # Configuration paths
├── data/
│   ├── combined_preprocessed_data.csv  # Data for XGBoost and TabPFN
│   └── combined_preprocessed_cnn.csv   # Data for CNN
├── XGB.py                              # XGBoost implementation
├── Tab.py                              # TabPFN implementation
├── CNN.py                              # CNN implementation
└── README.md                           # This file
```

## Configuration

Data paths are defined in `config.py`:
- `COMBINE_PREPROCESSED_PATH`: Data for XGBoost and TabPFN
- `COMBINE_PREPROCESSED_CNN_PATH`: Data for CNN model

## Evaluation Metrics

- **Micro-averaged**: Aggregates contributions of all classes, giving equal weight to each sample
- **Macro-averaged**: Calculates metric for each class independently and takes the average, giving equal weight to each class

## Notes

- All models use the same 5-fold stratified cross-validation split (random_state=42)
- Labels are converted from 1-5 to 0-4 for model compatibility
- CNN model uses StandardScaler for feature normalization
- TabPFN automatically handles GPU/CPU selection
- Results show mean ± standard deviation across 5 folds
