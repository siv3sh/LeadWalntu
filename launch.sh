#!/bin/bash

# SEO Content Analyzer - Standalone Launcher

echo "🔍 SEO Content Quality Analyzer"
echo "================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python 3 found"

# Check/create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
if [ ! -f "venv/.installed" ]; then
    echo "📥 Installing dependencies..."
    pip install --upgrade pip > /dev/null 2>&1
    pip install -r requirements.txt
    python3 -c "import nltk; nltk.download('punkt', quiet=True)"
    touch venv/.installed
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

# Launch Streamlit
echo ""
echo "🚀 Launching Streamlit Dashboard..."
echo "📊 Dashboard will open at: http://localhost:8501"
echo ""
echo "💡 Tips:"
echo "   • No Jupyter notebook needed!"
echo "   • Use 🔎 Live Analysis to analyze any URL or HTML"
echo "   • Use 🏠 Dashboard to process your data.csv"
echo ""
echo "Press Ctrl+C to stop"
echo ""

streamlit run app.py
