# CB513 Protein Secondary Structure Prediction

A lightweight neural network implementation for predicting protein secondary structures using the CB513 dataset. This project demonstrates practical bioinformatics machine learning with TensorFlow/Keras.

## 🧬 Project Overview

This project implements a sequence-to-sequence LSTM model to predict protein secondary structures (Helix, Sheet, Coil) from amino acid sequences. The model is optimized for laptop-friendly training while maintaining scientific validity.

## 📊 Dataset

- **CB513 Dataset**: 511 protein sequences with secondary structure annotations
- **Input**: Amino acid sequences (20 standard + 3 non-standard amino acids)
- **Output**: 3-state secondary structure classification (H/E/C)
- **Sequence lengths**: 20-874 residues (filtered to ≤256 for efficiency)

## 🏗️ Model Architecture

- **Embedding Layer**: 32-dimensional amino acid representations
- **LSTM Layer**: 64 units with dropout regularization
- **Dense Layers**: 32 → 3 units with softmax activation
- **Total Parameters**: ~27K (laptop-optimized)

## 📈 Results

- **Q3 Accuracy**: 55.33% (66% improvement over random baseline)
- **Per-class Performance**:
  - Helix (H): 64.0%
  - Sheet (E): 52.7% 
  - Coil (C): 50.2%

## 🛠️ Technical Features

- **Class imbalance handling** with computed sample weights
- **Variable sequence length** support with padding/masking
- **Early stopping** and learning rate reduction
- **Memory-efficient** preprocessing for laptop compatibility
- **Proper train/test splitting** with sequence length stratification

## 📁 Project Structure

```
CB513_Protein_ML/
├── data/
│   └── CB513.csv                    # Dataset
├── models/
│   └── lightweight_protein_model.h5 # Trained model
├── src/
│   ├── main.py                      # Data exploration script
│   ├── neuralnet.py                 # Original full model
│   └── lightweight_model.py         # Laptop-optimized model
├── results/
│   └── training_plots.png           # Training visualization
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
python 3.8+
tensorflow 2.x
pandas
numpy
matplotlib
scikit-learn
```

### Installation

```bash
git clone https://github.com/yourusername/CB513-Protein-ML.git
cd CB513-Protein-ML
pip install -r requirements.txt
```

### Usage

1. **Data Exploration**:
```bash
python src/main.py
```

2. **Train Model**:
```bash
python src/lightweight_model.py
```

3. **Make Predictions**:
```bash
python src/predict.py --sequence "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL"
```

## 📚 Scientific Context

Protein secondary structure prediction is a fundamental problem in bioinformatics. This implementation focuses on:

- **Practical ML application** to biological sequence data
- **Handling class imbalance** in biological datasets
- **Sequence-to-sequence learning** for variable-length proteins
- **Evaluation using domain-standard metrics** (Q3 accuracy)

## 🔬 Methodology

1. **Data Preprocessing**: Amino acid encoding, sequence padding, structure label encoding
2. **Class Balancing**: Computed sample weights to handle 44% coil vs 22% sheet imbalance
3. **Model Training**: LSTM with early stopping and learning rate scheduling
4. **Evaluation**: Q3 accuracy with per-residue and per-class metrics

## 🎯 Future Improvements

- [ ] Implement 8-state secondary structure prediction
- [ ] Add convolutional layers for local pattern recognition
- [ ] Ensemble multiple models for improved accuracy
- [ ] Web interface for interactive predictions
- [ ] Comparison with state-of-the-art methods

## 📖 References

- CB513 Dataset: [Cuff & Barton, 1999](https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1097-0134(19990901)36:4%3C400::AID-PROT5%3E3.0.CO;2-3)
- DSSP Secondary Structure: [Kabsch & Sander, 1983](https://onlinelibrary.wiley.com/doi/abs/10.1002/bip.360221211)

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

---

**Note**: This project prioritizes educational value and laptop compatibility over state-of-the-art performance. For production applications, consider larger models and datasets.