#!/usr/bin/env python3
"""
CB513 Protein Secondary Structure Prediction - LIGHTWEIGHT VERSION
Designed to run smoothly on laptops without causing thermal issues!
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Limit TensorFlow memory growth to prevent crashes
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Set CPU thread limit to prevent overheating
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

class LightweightPreprocessor:
    """Simplified, memory-efficient preprocessing"""
    
    def __init__(self, max_length=256):  # Reduced from 512
        self.max_length = max_length
        self.amino_acid_vocab = {}
        self.structure_encoder = LabelEncoder()
        self.class_weights = None
        
    def create_amino_acid_vocabulary(self, sequences):
        """Create compact vocabulary mapping"""
        # Standard 20 amino acids only for simplicity
        standard_aa = 'ACDEFGHIKLMNPQRSTVWY'
        vocab = {'<PAD>': 0}
        
        for i, aa in enumerate(standard_aa, 1):
            vocab[aa] = i
            
        # Map any non-standard amino acids to most similar standard ones
        non_standard_mapping = {'U': 3, 'X': 2, 'Z': 5}  # U->C, X->A, Z->E
        vocab.update(non_standard_mapping)
        
        self.amino_acid_vocab = vocab
        print(f"Compact amino acid vocabulary: {len(set(vocab.values()))} unique tokens")
        return vocab
    
    def preprocess_lightweight(self, df):
        """Fast, memory-efficient preprocessing"""
        print("🔄 Lightweight preprocessing...")
        
        sequences = df['input'].tolist()
        structures = df['dssp3'].tolist()
        
        # Filter out very long sequences to save memory
        filtered_data = []
        for seq, struct in zip(sequences, structures):
            if len(seq) <= self.max_length:
                filtered_data.append((seq, struct))
        
        print(f"Using {len(filtered_data)} sequences (≤{self.max_length} residues)")
        sequences, structures = zip(*filtered_data)
        
        # Create vocabularies
        self.create_amino_acid_vocabulary(sequences)
        
        # Encode sequences efficiently
        X = np.zeros((len(sequences), self.max_length), dtype=np.int32)
        y = np.zeros((len(sequences), self.max_length), dtype=np.int32)
        masks = np.zeros((len(sequences), self.max_length), dtype=np.float32)
        
        # Structure mapping
        struct_map = {'C': 0, 'E': 1, 'H': 2}
        
        for i, (seq, struct) in enumerate(zip(sequences, structures)):
            # Encode sequence
            for j, aa in enumerate(seq):
                X[i, j] = self.amino_acid_vocab.get(aa, 1)  # Default to A
                
            # Encode structure
            for j, ss in enumerate(struct):
                y[i, j] = struct_map[ss]
                masks[i, j] = 1.0
        
        # Compute simple class weights
        y_flat = y[masks == 1].astype(int)
        unique, counts = np.unique(y_flat, return_counts=True)
        total = len(y_flat)
        
        self.class_weights = {}
        for cls, count in zip(unique, counts):
            self.class_weights[cls] = total / (len(unique) * count)
            
        print(f"Class weights: {self.class_weights}")
        print(f"Data shapes: X{X.shape}, y{y.shape}, masks{masks.shape}")
        
        return X, y, masks

class SimpleLSTM:
    """Lightweight LSTM model that won't crash your laptop"""
    
    def __init__(self, vocab_size=21, num_classes=3, max_length=256):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.max_length = max_length
        self.model = None
        
    def build_simple_model(self):
        """Build a simple but effective model"""
        
        inputs = layers.Input(shape=(self.max_length,))
        
        # Small embedding
        embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=32,  # Much smaller than before
            mask_zero=True
        )(inputs)
        
        # Single LSTM layer
        lstm = layers.LSTM(
            64,  # Reduced from 256
            return_sequences=True,
            dropout=0.2
        )(embedding)
        
        # Simple dense layer
        dense = layers.Dense(32, activation='relu')(lstm)
        dropout = layers.Dropout(0.3)(dense)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(dropout)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def compile_model(self):
        """Compile with conservative settings"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),  # Higher learning rate for faster training
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Lightweight model compiled!")
        
    def print_summary(self):
        """Show model info"""
        print("\n🧠 LIGHTWEIGHT MODEL SUMMARY")
        print("="*40)
        self.model.summary()
        print(f"Total parameters: {self.model.count_params():,}")

def create_sample_weights(y, masks, class_weights):
    """Efficient sample weight creation"""
    weights = np.zeros_like(y, dtype=np.float32)
    for i in range(len(y)):
        for j in range(len(y[i])):
            if masks[i, j] == 1:
                weights[i, j] = class_weights.get(y[i, j], 1.0)
    return weights

def evaluate_simple(model, X_test, y_test, masks_test):
    """Simple evaluation metrics"""
    print("\n📊 MODEL EVALUATION")
    print("="*30)
    
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=-1)
    
    # Extract only valid (non-padded) predictions
    y_true_valid = []
    y_pred_valid = []
    
    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            if masks_test[i, j] == 1:
                y_true_valid.append(y_test[i, j])
                y_pred_valid.append(y_pred[i, j])
    
    # Calculate Q3 accuracy
    q3_accuracy = accuracy_score(y_true_valid, y_pred_valid)
    
    # Class-wise accuracy
    structure_names = ['Coil (C)', 'Sheet (E)', 'Helix (H)']
    print("Per-class accuracy:")
    for i, name in enumerate(structure_names):
        mask = np.array(y_true_valid) == i
        if np.sum(mask) > 0:
            class_acc = accuracy_score(
                np.array(y_true_valid)[mask], 
                np.array(y_pred_valid)[mask]
            )
            print(f"  {name}: {class_acc:.3f}")
    
    print(f"\nOverall Q3 Accuracy: {q3_accuracy:.4f}")
    return q3_accuracy

def main():
    """Laptop-friendly main function"""
    print("🧬 CB513 Lightweight Protein Structure Prediction")
    print("="*50)
    print("💻 Laptop-safe mode: Reduced model size and memory usage")
    
    # Load data
    try:
        df = pd.read_csv("../data/CB513.csv")
        print(f"Loaded {len(df)} sequences")
    except FileNotFoundError:
        print("❌ CB513.csv not found. Please check the path.")
        print("Expected location: ../data/CB513.csv")
        print("Make sure you're running from the src/ directory")
        return None
    
    # Preprocess with lightweight settings
    preprocessor = LightweightPreprocessor(max_length=256)
    X, y, masks = preprocessor.preprocess_lightweight(df)
    
    # Small train-test split
    test_size = min(0.2, 50 / len(X))  # Use max 50 sequences for testing
    X_train, X_test, y_train, y_test, masks_train, masks_test = train_test_split(
        X, y, masks, test_size=test_size, random_state=42
    )
    
    print(f"Training: {len(X_train)} sequences, Testing: {len(X_test)} sequences")
    
    # Build lightweight model
    model_builder = SimpleLSTM(vocab_size=21, num_classes=3, max_length=256)
    model = model_builder.build_simple_model()
    model_builder.compile_model()
    model_builder.print_summary()
    
    # Create sample weights
    sample_weights_train = create_sample_weights(y_train, masks_train, preprocessor.class_weights)
    
    # Conservative training settings
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1)
    ]
    
    print("\n🚀 Starting training (laptop-friendly)...")
    print("This should complete in 2-5 minutes without overheating!")
    
    # Train with small batches and few epochs
    history = model.fit(
        X_train, y_train,
        sample_weight=sample_weights_train,
        epochs=20,  # Reduced from 50
        batch_size=16,  # Small batches
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    
    # Simple training plot
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Evaluate
    q3_accuracy = evaluate_simple(model, X_test, y_test, masks_test)
    
    print(f"\n🎉 Training complete!")
    print(f"Final Q3 accuracy: {q3_accuracy:.4f}")
    print("Model trained successfully without melting your laptop! 💻❄️")
    
    # Save model
    model.save('../models/lightweight_protein_model.h5')
    print("Model saved as '../models/lightweight_protein_model.h5'")
    
    return model, preprocessor, history

if __name__ == "__main__":
    model, preprocessor, history = main()