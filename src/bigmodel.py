#!/usr/bin/env python3
"""
CB513 Protein Secondary Structure Prediction
Data preprocessing and neural network model implementation
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class CB513Preprocessor:
    """Handles all data preprocessing for CB513 dataset"""
    
    def __init__(self, max_length=512):
        self.max_length = max_length
        self.amino_acid_vocab = {}
        self.structure_encoder = LabelEncoder()
        self.class_weights = None
        
    def create_amino_acid_vocabulary(self, sequences):
        """Create vocabulary mapping for amino acids"""
        all_amino_acids = set(''.join(sequences))
        # Standard 20 amino acids + common variations
        standard_aa = 'ACDEFGHIKLMNPQRSTVWY'
        
        # Add padding token and unknown token
        vocab = {'<PAD>': 0, '<UNK>': 1}
        
        # Add standard amino acids first
        for i, aa in enumerate(standard_aa, 2):
            vocab[aa] = i
            
        # Add any non-standard amino acids found in dataset
        next_idx = len(vocab)
        for aa in sorted(all_amino_acids):
            if aa not in vocab:
                vocab[aa] = next_idx
                next_idx += 1
                
        self.amino_acid_vocab = vocab
        print(f"Created amino acid vocabulary with {len(vocab)} tokens:")
        print(f"Standard: {dict(list(vocab.items())[2:22])}")
        if len(vocab) > 22:
            print(f"Non-standard: {dict(list(vocab.items())[22:])}")
        
        return vocab
    
    def encode_sequences(self, sequences):
        """Convert amino acid sequences to numerical arrays"""
        encoded_sequences = []
        
        for seq in sequences:
            encoded_seq = [self.amino_acid_vocab.get(aa, 1) for aa in seq]  # 1 is <UNK>
            encoded_sequences.append(encoded_seq)
            
        return encoded_sequences
    
    def pad_sequences(self, sequences):
        """Pad sequences to uniform length"""
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, 
            maxlen=self.max_length, 
            padding='post',
            truncating='post',
            value=0  # <PAD> token
        )
        return padded_sequences
    
    def encode_structures(self, structures):
        """Encode secondary structures to numerical labels"""
        # Flatten all structure sequences to fit LabelEncoder
        flat_structures = ''.join(structures)
        
        # Fit encoder on all possible structure types
        unique_structures = sorted(set(flat_structures))
        self.structure_encoder.fit(unique_structures)
        
        print(f"Secondary structure classes: {dict(enumerate(self.structure_encoder.classes_))}")
        
        # Encode each sequence
        encoded_structures = []
        for struct_seq in structures:
            encoded_seq = self.structure_encoder.transform(list(struct_seq))
            encoded_structures.append(encoded_seq)
            
        return encoded_structures
    
    def compute_class_weights(self, y_flat):
        """Compute class weights to handle imbalance"""
        unique_classes = np.unique(y_flat)
        weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y_flat
        )
        
        self.class_weights = dict(zip(unique_classes, weights))
        print(f"Class weights for imbalanced data: {self.class_weights}")
        return self.class_weights
    
    def preprocess_dataset(self, df):
        """Complete preprocessing pipeline"""
        print("🔄 Starting preprocessing pipeline...")
        
        # Extract sequences and structures
        sequences = df['input'].tolist()
        structures = df['dssp3'].tolist()
        
        # Create vocabularies and encoders
        self.create_amino_acid_vocabulary(sequences)
        
        # Encode sequences
        print("Encoding amino acid sequences...")
        X_encoded = self.encode_sequences(sequences)
        X_padded = self.pad_sequences(X_encoded)
        
        # Encode structures
        print("Encoding secondary structures...")
        y_encoded = self.encode_structures(structures)
        y_padded = self.pad_sequences(y_encoded)
        
        # Compute class weights
        y_flat = np.concatenate([seq[:len(orig)] for seq, orig in zip(y_encoded, structures)])
        self.compute_class_weights(y_flat)
        
        # Create masks to ignore padded positions during training
        masks = []
        for i, orig_seq in enumerate(sequences):
            mask = [1] * len(orig_seq) + [0] * (self.max_length - len(orig_seq))
            masks.append(mask[:self.max_length])
        
        masks = np.array(masks)
        
        print(f"Preprocessed data shapes:")
        print(f"  X (sequences): {X_padded.shape}")
        print(f"  y (structures): {y_padded.shape}")
        print(f"  masks: {masks.shape}")
        
        return X_padded, y_padded, masks

class ProteinStructurePredictor:
    """Neural network model for protein secondary structure prediction"""
    
    def __init__(self, vocab_size, num_classes, max_length=512):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.max_length = max_length
        self.model = None
        
    def build_model(self):
        """Build LSTM-based model for sequence-to-sequence prediction"""
        
        # Input layer
        inputs = layers.Input(shape=(self.max_length,), name='sequence_input')
        
        # Embedding layer for amino acids
        embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=128,
            mask_zero=True,  # Automatically handle padding
            name='amino_acid_embedding'
        )(inputs)
        
        # Bidirectional LSTM layers
        lstm1 = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            name='bidirectional_lstm_1'
        )(embedding)
        
        lstm2 = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            name='bidirectional_lstm_2'
        )(lstm1)
        
        # Attention mechanism (simplified)
        attention = layers.MultiHeadAttention(
            num_heads=8,
            key_dim=64,
            dropout=0.1,
            name='multi_head_attention'
        )(lstm2, lstm2)
        
        # Add & Normalize
        add_norm = layers.Add()([lstm2, attention])
        add_norm = layers.LayerNormalization()(add_norm)
        
        # Dense layers for classification
        dense1 = layers.Dense(256, activation='relu', name='dense_1')(add_norm)
        dropout1 = layers.Dropout(0.4)(dense1)
        
        dense2 = layers.Dense(128, activation='relu', name='dense_2')(dropout1)
        dropout2 = layers.Dropout(0.4)(dense2)
        
        # Output layer - one prediction per residue
        outputs = layers.Dense(
            self.num_classes, 
            activation='softmax',
            name='structure_prediction'
        )(dropout2)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='ProteinStructurePredictor')
        
        return self.model
    
    def compile_model(self, class_weights=None):
        """Compile model with appropriate loss and metrics"""
        
        # Use sparse categorical crossentropy for integer labels
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
            # Note: sample_weight_mode removed - handled automatically in newer Keras
        )
        
        print("Model compiled successfully!")
        return self.model
    
    def print_model_summary(self):
        """Print detailed model architecture"""
        print("\n" + "="*60)
        print("🧠 NEURAL NETWORK ARCHITECTURE")
        print("="*60)
        self.model.summary()
        
        # Count parameters
        total_params = self.model.count_params()
        print(f"\nTotal trainable parameters: {total_params:,}")

def create_sample_weight_matrix(y_true, masks, class_weights):
    """Create sample weights for masked sequence training"""
    sample_weights = np.zeros_like(y_true, dtype=np.float32)
    
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            if masks[i][j] == 1:  # Only weight non-padded positions
                class_idx = y_true[i][j]
                sample_weights[i][j] = class_weights.get(class_idx, 1.0)
            else:
                sample_weights[i][j] = 0.0  # Zero weight for padded positions
                
    return sample_weights

def plot_training_history(history):
    """Plot training metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test, masks_test, preprocessor):
    """Evaluate model performance with detailed metrics"""
    print("\n" + "="*50)
    print("📊 MODEL EVALUATION")
    print("="*50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=-1)
    
    # Flatten predictions and true labels, considering only non-padded positions
    y_true_flat = []
    y_pred_flat = []
    
    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            if masks_test[i][j] == 1:  # Only consider non-padded positions
                y_true_flat.append(y_test[i][j])
                y_pred_flat.append(y_pred_classes[i][j])
    
    y_true_flat = np.array(y_true_flat)
    y_pred_flat = np.array(y_pred_flat)
    
    # Classification report
    target_names = preprocessor.structure_encoder.classes_
    print("Classification Report:")
    print(classification_report(y_true_flat, y_pred_flat, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Calculate Q3 accuracy (standard metric in protein structure prediction)
    q3_accuracy = np.mean(y_true_flat == y_pred_flat)
    print(f"\nQ3 Accuracy (Overall): {q3_accuracy:.4f}")
    
    return q3_accuracy, y_pred_classes

def main():
    """Main training pipeline"""
    print("🧬 CB513 Protein Secondary Structure Prediction Pipeline")
    print("="*60)
    
    # Load data
    data_path = "../data/CB513.csv"
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} protein sequences")
    
    # Initialize preprocessor
    max_length = 512  # Reasonable max length based on your data analysis
    preprocessor = CB513Preprocessor(max_length=max_length)
    
    # Preprocess data
    X, y, masks = preprocessor.preprocess_dataset(df)
    
    # Train-test split (stratified by length to ensure diverse sequences in both sets)
    sequence_lengths = [len(seq) for seq in df['input']]
    length_bins = pd.cut(sequence_lengths, bins=5, labels=False)
    
    X_train, X_test, y_train, y_test, masks_train, masks_test = train_test_split(
        X, y, masks, test_size=0.2, random_state=42, stratify=length_bins
    )
    
    print(f"\nData split:")
    print(f"  Training: {X_train.shape[0]} sequences")
    print(f"  Testing: {X_test.shape[0]} sequences")
    
    # Build model
    vocab_size = len(preprocessor.amino_acid_vocab)
    num_classes = len(preprocessor.structure_encoder.classes_)
    
    model_builder = ProteinStructurePredictor(
        vocab_size=vocab_size,
        num_classes=num_classes,
        max_length=max_length
    )
    
    model = model_builder.build_model()
    model = model_builder.compile_model(class_weights=preprocessor.class_weights)
    model_builder.print_model_summary()
    
    # Create sample weights for training
    sample_weights_train = create_sample_weight_matrix(y_train, masks_train, preprocessor.class_weights)
    sample_weights_test = create_sample_weight_matrix(y_test, masks_test, preprocessor.class_weights)
    
    # Training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
        keras.callbacks.ModelCheckpoint('../models/best_protein_model.h5', save_best_only=True)
    ]
    
    # Train model
    print("\n🚀 Starting model training...")
    history = model.fit(
        X_train, y_train,
        sample_weight=sample_weights_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test, sample_weights_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    q3_accuracy, predictions = evaluate_model(model, X_test, y_test, masks_test, preprocessor)
    
    print(f"\n🎉 Training complete! Final Q3 accuracy: {q3_accuracy:.4f}")
    
    # Save preprocessor info for future use
    import pickle
    with open('../models/preprocessor_info.pkl', 'wb') as f:
        pickle.dump({
            'amino_acid_vocab': preprocessor.amino_acid_vocab,
            'structure_encoder': preprocessor.structure_encoder,
            'class_weights': preprocessor.class_weights,
            'max_length': max_length
        }, f)
    
    print("Model and preprocessor saved!")
    
    return model, preprocessor, history

if __name__ == "__main__":
    model, preprocessor, history = main()