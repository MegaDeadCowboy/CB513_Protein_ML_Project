#!/usr/bin/env python3
"""
Interactive Protein Secondary Structure Prediction
Uses the trained CB513 model to predict structures for new sequences
"""

import numpy as np
import tensorflow as tf
import argparse
import sys

class ProteinPredictor:
    """Load trained model and make predictions on new sequences"""
    
    def __init__(self, model_path='../models/lightweight_protein_model.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.max_length = 256
        
        # Amino acid vocabulary (matches training)
        standard_aa = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_vocab = {'<PAD>': 0}
        for i, aa in enumerate(standard_aa, 1):
            self.aa_vocab[aa] = i
            
        # Non-standard mappings
        self.aa_vocab.update({'U': 3, 'X': 2, 'Z': 5})  # U->C, X->A, Z->E
        
        # Structure mapping
        self.structure_names = {0: 'C (Coil)', 1: 'E (Sheet)', 2: 'H (Helix)'}
        self.structure_chars = {0: 'C', 1: 'E', 2: 'H'}
        
    def encode_sequence(self, sequence):
        """Encode amino acid sequence for model input"""
        # Convert to uppercase and filter valid amino acids
        sequence = sequence.upper()
        valid_sequence = ''.join([aa for aa in sequence if aa in 'ACDEFGHIKLMNPQRSTVWYUXZ'])
        
        if len(valid_sequence) != len(sequence):
            print(f"Warning: Filtered sequence from {len(sequence)} to {len(valid_sequence)} residues")
            
        if len(valid_sequence) > self.max_length:
            print(f"Warning: Sequence truncated from {len(valid_sequence)} to {self.max_length} residues")
            valid_sequence = valid_sequence[:self.max_length]
            
        # Encode sequence
        encoded = [self.aa_vocab.get(aa, 2) for aa in valid_sequence]  # Default to A
        
        # Pad to max length
        padded = encoded + [0] * (self.max_length - len(encoded))
        
        return np.array([padded]), len(valid_sequence), valid_sequence
    
    def predict_structure(self, sequence):
        """Predict secondary structure for a given sequence"""
        encoded_seq, seq_length, clean_sequence = self.encode_sequence(sequence)
        
        # Make prediction
        predictions = self.model.predict(encoded_seq, verbose=0)
        predicted_classes = np.argmax(predictions[0], axis=-1)
        confidence_scores = np.max(predictions[0], axis=-1)
        
        # Extract only the valid (non-padded) predictions
        valid_predictions = predicted_classes[:seq_length]
        valid_confidences = confidence_scores[:seq_length]
        
        return clean_sequence, valid_predictions, valid_confidences
    
    def format_prediction(self, sequence, predictions, confidences):
        """Format prediction results nicely"""
        structure_string = ''.join([self.structure_chars[pred] for pred in predictions])
        
        print(f"\n{'='*60}")
        print("🧬 PROTEIN SECONDARY STRUCTURE PREDICTION")
        print(f"{'='*60}")
        print(f"Sequence Length: {len(sequence)} residues")
        print(f"Average Confidence: {np.mean(confidences):.3f}")
        
        # Structure distribution
        unique, counts = np.unique(predictions, return_counts=True)
        print(f"\nStructure Composition:")
        for struct_idx, count in zip(unique, counts):
            percentage = (count / len(predictions)) * 100
            print(f"  {self.structure_names[struct_idx]}: {count} ({percentage:.1f}%)")
        
        print(f"\nSequence and Prediction:")
        print(f"Amino Acids: {sequence}")
        print(f"Structure  : {structure_string}")
        
        # Show regions of high confidence
        high_conf_mask = confidences > 0.8
        if np.any(high_conf_mask):
            print(f"Confidence : {''.join(['*' if conf > 0.8 else ' ' for conf in confidences])}")
            print("             (* = high confidence > 0.8)")
        
        return structure_string
    
    def analyze_sequence_features(self, sequence, predictions):
        """Provide biological insights about the predicted structure"""
        print(f"\n📊 STRUCTURAL ANALYSIS:")
        
        # Hydrophobic amino acids
        hydrophobic = 'AILVFPWMY'
        hydrophobic_count = sum(1 for aa in sequence if aa in hydrophobic)
        hydrophobic_pct = (hydrophobic_count / len(sequence)) * 100
        
        # Charged amino acids
        charged = 'DEKR'
        charged_count = sum(1 for aa in sequence if aa in charged)
        charged_pct = (charged_count / len(sequence)) * 100
        
        print(f"  Hydrophobic residues: {hydrophobic_count} ({hydrophobic_pct:.1f}%)")
        print(f"  Charged residues: {charged_count} ({charged_pct:.1f}%)")
        
        # Predicted secondary structure regions
        structure_regions = []
        current_structure = predictions[0]
        start_pos = 0
        
        for i in range(1, len(predictions)):
            if predictions[i] != current_structure:
                structure_regions.append((start_pos, i-1, current_structure))
                current_structure = predictions[i]
                start_pos = i
        structure_regions.append((start_pos, len(predictions)-1, current_structure))
        
        print(f"\n  Predicted structural regions:")
        for start, end, struct in structure_regions:
            if end - start + 1 >= 3:  # Only show regions of 3+ residues
                struct_name = self.structure_chars[struct]
                length = end - start + 1
                print(f"    {struct_name}: positions {start+1}-{end+1} ({length} residues)")

def main():
    parser = argparse.ArgumentParser(description='Predict protein secondary structure')
    parser.add_argument('--sequence', type=str, help='Amino acid sequence to predict')
    parser.add_argument('--file', type=str, help='File containing sequences (FASTA format)')
    parser.add_argument('--model', type=str, default='../models/lightweight_protein_model.h5', 
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    try:
        predictor = ProteinPredictor(args.model)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure '../models/lightweight_protein_model.h5' exists")
        print("Train the model first by running: python lightweight_model.py")
        return
    
    if args.sequence:
        # Single sequence prediction
        sequence, predictions, confidences = predictor.predict_structure(args.sequence)
        structure = predictor.format_prediction(sequence, predictions, confidences)
        predictor.analyze_sequence_features(sequence, predictions)
        
    elif args.file:
        # File-based predictions (simplified FASTA support)
        try:
            with open(args.file, 'r') as f:
                content = f.read().strip()
                # Simple FASTA parsing
                if content.startswith('>'):
                    lines = content.split('\n')
                    sequence = ''.join([line for line in lines if not line.startswith('>')])
                else:
                    sequence = content
                    
            sequence, predictions, confidences = predictor.predict_structure(sequence)
            structure = predictor.format_prediction(sequence, predictions, confidences)
            predictor.analyze_sequence_features(sequence, predictions)
            
        except FileNotFoundError:
            print(f"❌ File not found: {args.file}")
            return
            
    else:
        # Interactive mode
        print("🧬 Interactive Protein Structure Predictor")
        print("Enter 'quit' to exit\n")
        
        while True:
            sequence = input("Enter amino acid sequence: ").strip()
            
            if sequence.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
                
            if not sequence:
                continue
                
            try:
                seq, predictions, confidences = predictor.predict_structure(sequence)
                structure = predictor.format_prediction(seq, predictions, confidences)
                predictor.analyze_sequence_features(seq, predictions)
                print("\n" + "-"*60 + "\n")
                
            except Exception as e:
                print(f"❌ Error processing sequence: {e}")

if __name__ == "__main__":
    # Example sequences for testing
    if len(sys.argv) == 1:
        print("🧬 CB513 Protein Structure Predictor")
        print("\nUsage examples:")
        print("  python predict.py --sequence MKWVTFISLLFLFSSAYS")
        print("  python predict.py --file protein.fasta")
        print("  python predict.py  # Interactive mode")
        print("\nTry this example sequence:")
        print("MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCD")
        print()
    
    main()