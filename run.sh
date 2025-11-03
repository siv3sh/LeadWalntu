#!/bin/bash

# SEO Content Quality & Duplicate Detector - Quick Start Script

echo "🔍 SEO Content Quality Analyzer"
echo "================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
if [ ! -f "venv/.installed" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    python3 -c "import nltk; nltk.download('punkt')"
    touch venv/.installed
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

# Check if data has been processed
if [ ! -f "data/extracted_content.csv" ]; then
    echo ""
    echo "⚠️  No processed data found!"
    echo "📓 Please run the Jupyter notebook first:"
    echo "   jupyter notebook notebooks/seo_pipeline.ipynb"
    echo ""
    read -p "Do you want to open the notebook now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        jupyter notebook notebooks/seo_pipeline.ipynb &
        echo "📓 Notebook opened. Process your data first, then run this script again."
        exit 0
    fi
fi

# Launch Streamlit
echo ""
echo "🚀 Launching Streamlit dashboard..."
echo "📊 Dashboard will open at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
