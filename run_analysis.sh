#!/bin/bash
echo "🧬 CB513 Protein Secondary Structure Prediction Pipeline"
echo "======================================================="

if [ ! -d "src" ] || [ ! -d "data" ]; then
    echo "❌ Please run from CB513_Protein_ML root directory"
    exit 1
fi

if [ ! -f "data/CB513.csv" ]; then
    echo "❌ CB513.csv not found in data/ directory"
    exit 1
fi

mkdir -p models results
cd src

echo "Choose an option:"
echo "1. Run data exploration"
echo "2. Train model"  
echo "3. Make predictions"

read -p "Enter choice (1-3): " choice

case $choice in
    1) python main.py ;;
    2) python lightweight_model.py ;;
    3) python predict.py ;;
    *) echo "Invalid choice" ;;
esac
