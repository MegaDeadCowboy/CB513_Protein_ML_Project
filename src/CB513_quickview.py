#!/usr/bin/env python3
"""
CB513 Protein Dataset Exploration
Comprehensive analysis of protein sequences and secondary structures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_and_inspect_data(file_path):
    """Load CB513 dataset and perform initial inspection"""
    print("🧬 Loading CB513 Protein Dataset...")
    df = pd.read_csv(file_path)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    return df

def analyze_sequences(df):
    """Analyze amino acid sequences"""
    print("\n" + "="*50)
    print("🔬 AMINO ACID SEQUENCE ANALYSIS")
    print("="*50)
    
    # Get sequence lengths
    sequence_lengths = df['input'].str.len()
    
    print(f"Sequence Length Statistics:")
    print(f"  Mean: {sequence_lengths.mean():.1f}")
    print(f"  Median: {sequence_lengths.median():.1f}")
    print(f"  Min: {sequence_lengths.min()}")
    print(f"  Max: {sequence_lengths.max()}")
    print(f"  Std: {sequence_lengths.std():.1f}")
    
    # Plot sequence length distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(sequence_lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Protein Sequence Lengths')
    plt.grid(True, alpha=0.3)
    
    # Amino acid frequency analysis
    all_amino_acids = ''.join(df['input'].tolist())
    aa_counts = Counter(all_amino_acids)
    
    print(f"\nAmino Acid Frequencies (top 10):")
    for aa, count in aa_counts.most_common(10):
        percentage = (count / len(all_amino_acids)) * 100
        print(f"  {aa}: {count:,} ({percentage:.1f}%)")
    
    # Plot amino acid frequencies
    plt.subplot(1, 2, 2)
    aa_letters = list(aa_counts.keys())
    aa_frequencies = list(aa_counts.values())
    
    plt.bar(aa_letters, aa_frequencies, color='lightcoral', alpha=0.7)
    plt.xlabel('Amino Acid')
    plt.ylabel('Frequency')
    plt.title('Amino Acid Distribution Across All Sequences')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return sequence_lengths, aa_counts

def analyze_secondary_structures(df):
    """Analyze secondary structure labels"""
    print("\n" + "="*50)
    print("🏗️  SECONDARY STRUCTURE ANALYSIS")
    print("="*50)
    
    # 3-state analysis (H/E/C)
    print("3-State Secondary Structure (dssp3):")
    all_dssp3 = ''.join(df['dssp3'].tolist())
    dssp3_counts = Counter(all_dssp3)
    
    structure_names = {'H': 'Helix', 'E': 'Sheet', 'C': 'Coil'}
    
    for structure, count in dssp3_counts.most_common():
        percentage = (count / len(all_dssp3)) * 100
        name = structure_names.get(structure, structure)
        print(f"  {structure} ({name}): {count:,} ({percentage:.1f}%)")
    
    # 8-state analysis
    if 'dssp8' in df.columns:
        print(f"\n8-State Secondary Structure (dssp8):")
        all_dssp8 = ''.join(df['dssp8'].tolist())
        dssp8_counts = Counter(all_dssp8)
        
        for structure, count in dssp8_counts.most_common(8):
            percentage = (count / len(all_dssp8)) * 100
            print(f"  {structure}: {count:,} ({percentage:.1f}%)")
    
    # Plot secondary structure distributions
    plt.figure(figsize=(12, 4))
    
    # 3-state plot
    plt.subplot(1, 2, 1)
    labels_3 = [structure_names.get(k, k) for k in dssp3_counts.keys()]
    values_3 = list(dssp3_counts.values())
    colors_3 = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    plt.pie(values_3, labels=labels_3, autopct='%1.1f%%', colors=colors_3, startangle=90)
    plt.title('3-State Secondary Structure Distribution')
    
    # 8-state plot (if available)
    if 'dssp8' in df.columns:
        plt.subplot(1, 2, 2)
        dssp8_items = list(dssp8_counts.most_common(8))
        labels_8 = [item[0] for item in dssp8_items]
        values_8 = [item[1] for item in dssp8_items]
        
        plt.bar(labels_8, values_8, color='lightgreen', alpha=0.7)
        plt.xlabel('Secondary Structure Type')
        plt.ylabel('Frequency')
        plt.title('8-State Secondary Structure Distribution')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return dssp3_counts, dssp8_counts if 'dssp8' in df.columns else None

def analyze_masking(df):
    """Analyze sequence masking if available"""
    if 'cb513_mask' in df.columns:
        print("\n" + "="*50)
        print("🎭 SEQUENCE MASKING ANALYSIS")
        print("="*50)
        
        # Count mask patterns
        mask_lengths = df['cb513_mask'].str.len()
        print(f"Mask Length Statistics:")
        print(f"  Mean: {mask_lengths.mean():.1f}")
        print(f"  Min: {mask_lengths.min()}")
        print(f"  Max: {mask_lengths.max()}")
        
        # Analyze mask values (1s and 0s)
        all_masks = ''.join(df['cb513_mask'].tolist())
        mask_counts = Counter(all_masks)
        print(f"\nMask Value Distribution:")
        for value, count in mask_counts.items():
            percentage = (count / len(all_masks)) * 100
            meaning = "Valid residue" if value == '1' else "Padding"
            print(f"  {value} ({meaning}): {count:,} ({percentage:.1f}%)")

def sequence_structure_alignment_check(df):
    """Verify that sequences and structures are properly aligned"""
    print("\n" + "="*50)
    print("🔍 SEQUENCE-STRUCTURE ALIGNMENT CHECK")
    print("="*50)
    
    misaligned = 0
    for idx, row in df.iterrows():
        seq_len = len(row['input'])
        dssp3_len = len(row['dssp3'])
        
        if seq_len != dssp3_len:
            misaligned += 1
            if misaligned <= 3:  # Show first 3 examples
                print(f"  Row {idx}: Sequence={seq_len}, DSSP3={dssp3_len}")
    
    if misaligned == 0:
        print("✅ All sequences and secondary structures are properly aligned!")
    else:
        print(f"⚠️  Found {misaligned} misaligned sequences")

def generate_summary_stats(df, sequence_lengths, aa_counts, dssp3_counts):
    """Generate summary statistics for the dataset"""
    print("\n" + "="*60)
    print("📊 DATASET SUMMARY")
    print("="*60)
    
    print(f"Total Proteins: {len(df):,}")
    print(f"Total Amino Acid Residues: {sequence_lengths.sum():,}")
    print(f"Average Sequence Length: {sequence_lengths.mean():.1f} ± {sequence_lengths.std():.1f}")
    print(f"Unique Amino Acids: {len(aa_counts)}")
    print(f"Secondary Structure Classes: {len(dssp3_counts)}")
    
    # Dataset size estimation
    total_residues = sequence_lengths.sum()
    print(f"\nDataset Size Estimation:")
    print(f"  Training examples (residue-level): ~{total_residues:,}")
    print(f"  Sequence-level examples: {len(df):,}")
    
    # Class balance
    print(f"\nClass Balance (3-state):")
    total_structures = sum(dssp3_counts.values())
    for structure, count in dssp3_counts.most_common():
        percentage = (count / total_structures) * 100
        balance_status = "Balanced" if 25 <= percentage <= 40 else "Imbalanced"
        print(f"  {structure}: {percentage:.1f}% ({balance_status})")

def main():
    """Main execution function"""
    # Update this path to your actual data location
    data_path = "../data/CB513.csv"
    
    try:
        # Load and inspect data
        df = load_and_inspect_data(data_path)
        
        # Perform comprehensive analysis
        sequence_lengths, aa_counts = analyze_sequences(df)
        dssp3_counts, dssp8_counts = analyze_secondary_structures(df)
        analyze_masking(df)
        sequence_structure_alignment_check(df)
        generate_summary_stats(df, sequence_lengths, aa_counts, dssp3_counts)
        
        print("\n🎉 Data exploration complete! Ready for preprocessing and model development.")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find CB513.csv at {data_path}")
        print("Please update the data_path variable with the correct location.")
        return None
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return None

if __name__ == "__main__":
    df = main()