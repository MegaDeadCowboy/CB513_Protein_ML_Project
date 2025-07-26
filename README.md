# CB513 Protein Secondary Structure Prediction
A neural network implementation for predicting protein secondary structures using the CB513 dataset. This project demonstrates practical bioinformatics machine learning with TensorFlow/Keras, featuring both lightweight and advanced model architectures.

## 🧬 Project Overview
This project implements sequence-to-sequence models to predict protein secondary structures (Helix, Sheet, Coil) from amino acid sequences. Two model architectures are provided: a lightweight version optimized for resource-constrained environments and a high-performance model with advanced deep learning techniques.

## 📊 Dataset
- **CB513 Dataset**: 511 protein sequences with secondary structure annotations
- **Input**: Amino acid sequences (20 standard + 3 non-standard amino acids)
- **Output**: 3-state secondary structure classification (H/E/C)
- **Sequence lengths**: 20-874 residues (lightweight model filters to ≤256 for efficiency)

## 🏗️ Model Architectures

### Lightweight Model (`lightmodel.py`)
**Optimized for laptop-friendly training**
- **Embedding Layer**: 32-dimensional amino acid representations
- **LSTM Layer**: 64 units with dropout regularization
- **Dense Layers**: 32 → 3 units with softmax activation
- **Parameters**: 27,683 (108KB)
- **Training Time**: ~20 seconds on modern hardware
- **Q3 Accuracy**: 55.55%

### Advanced Model (`bigmodel.py`)
**High-performance architecture with attention mechanisms**
- **Embedding Layer**: 128-dimensional amino acid representations
- **Bidirectional LSTM Layers**: 256 + 128 units (stacked)
- **Multi-Head Attention**: Captures long-range amino acid dependencies
- **Layer Normalization**: Residual connections for stable training
- **Dense Layers**: 256 → 128 → 3 units with dropout
- **Parameters**: 2,073,731 (7.91MB)
- **Training Time**: ~21 minutes on AMD Ryzen 7
- **Q3 Accuracy**: 64.93%

## 📈 Performance Results

### Model Comparison
| Model | Parameters | Training Time | Q3 Accuracy | Best Use Case |
|-------|------------|---------------|-------------|---------------|
| Lightweight | 27K | 20 seconds | 55.55% | Quick experiments, limited hardware |
| Advanced | 2.07M | 21 minutes | 64.93% | Research-grade predictions, modern hardware |

### Advanced Model Detailed Results
```
Per-class Performance:
- Helix (H): 72% precision, 69% F1-score
- Coil (C):  73% precision, 65% F1-score  
- Sheet (E): 67% recall, 59% F1-score

Overall Q3 Accuracy: 64.93%
```

### Scientific Context
- **Random Baseline**: 33.3% (3-class prediction)
- **Our Lightweight Model**: 55.55% ✅
- **Our Advanced Model**: 64.93% ✅
- **State-of-the-art**: 70-80% (with massive computational resources)

**Achievement**: Professional-grade results on consumer hardware! 🏆

## 🚀 Quick Start

### Prerequisites
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
```

### Running the Models
```bash
# Clone the repository
git clone https://github.com/MegaDeadCowboy/CB513_Protein_ML_Project.git
cd CB513_Protein_ML_Project/src

# Run lightweight model (fast, laptop-friendly)
python lightmodel.py

# Run advanced model (high accuracy, requires modern hardware)
python bigmodel.py

# Interactive prediction
python predict.py
```

### Helper Script
```bash
# From project root
./run_analysis.sh
```

## 📁 Project Structure
```
CB513_Protein_ML_Project/
├── README.md                    # This documentation
├── requirements.txt             # Python dependencies
├── run_analysis.sh             # Helper execution script
├── data/
│   └── CB513.csv               # Dataset (511 protein sequences)
├── src/
│   ├── lightmodel.py           # Lightweight LSTM model
│   ├── bigmodel.py             # Advanced model with attention
│   ├── predict.py              # Interactive prediction interface
│   └── CB513_quickview.py      # Data exploration script
├── models/                     # Trained model files
└── results/                    # Training outputs and analysis
```

## 🔬 Technical Features

### Advanced Model Innovations
- **Bidirectional LSTM**: Processes sequences in both directions for context
- **Multi-Head Attention**: Captures complex amino acid interaction patterns
- **Residual Connections**: Enables stable training of deeper networks
- **Layer Normalization**: Improves convergence and generalization
- **Class Weight Balancing**: Handles imbalanced secondary structure distribution

### Preprocessing Pipeline
- **Amino acid vocabulary**: Standard 20 + 3 non-standard (U→C, X→A, Z→E)
- **Sequence filtering**: Configurable length limits for memory efficiency
- **Class weight computation**: Automatic balancing for protein structure distribution
- **Masking system**: Proper handling of padded positions during training

## 💻 Hardware Requirements

### Lightweight Model
- **RAM**: 4GB minimum
- **CPU**: Any modern processor
- **Training Time**: <1 minute
- **Use Case**: Rapid prototyping, educational purposes

### Advanced Model
- **RAM**: 8GB+ recommended (16GB ideal)
- **CPU**: Multi-core processor (AMD Ryzen 7 / Intel i7+)
- **Training Time**: 15-30 minutes
- **Use Case**: Research, production applications

## 🧪 Interactive Prediction
```bash
# Run interactive prediction system
python src/predict.py

# Example usage:
Enter protein sequence: MKLLVLSLSLVLVAPMAAQTPFQQ
Predicted structure:   CCCCCCCCCCCCCCCCCCCCCCC
```

## 📊 Biological Relevance

### Secondary Structure Types
- **Helix (H)**: α-helical regions - highly structured, easier to predict
- **Sheet (E)**: β-sheet regions - structured but more variable
- **Coil (C)**: Random coil regions - unstructured, inherently difficult

### Model Performance Insights
- **Helix Prediction Excellence**: 72% precision demonstrates strong α-helix pattern recognition
- **Balanced Performance**: All three structure types predicted above random baseline
- **Attention Benefits**: Multi-head attention captures long-range amino acid dependencies crucial for structure formation

## 🔄 Future Enhancements
- [ ] 8-state secondary structure prediction (vs current 3-state)
- [ ] Convolutional layers for local amino acid pattern recognition
- [ ] Ensemble methods combining multiple model architectures
- [ ] Transfer learning from larger protein structure datasets
- [ ] Web interface for broader accessibility
- [ ] Integration with protein databases (PDB, UniProt)

## 📄 Citation
If you use this work in your research, please cite:
```
CB513 Protein Secondary Structure Prediction
GitHub: https://github.com/MegaDeadCowboy/CB513_Protein_ML_Project
```

## 🤝 Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📝 License
This project is open source and available under the MIT License.

---

**Built with**: TensorFlow, Python