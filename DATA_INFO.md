# Dataset Information

## Source

**Dataset Name**: dataset_for_assignment  
**Platform**: Kaggle  
**Author**: naveen1729  
**URL**: https://www.kaggle.com/datasets/naveen1729/dataset-for-assignment  
**License**: Check Kaggle dataset page for license details

## Dataset Description

This dataset contains web pages collected for SEO content analysis and quality assessment purposes.

### Statistics

- **Total Records**: 264,716 web pages
- **Format**: CSV (Comma-Separated Values)
- **Encoding**: UTF-8
- **File Size**: ~25 MB (compressed)

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `url` | string | The webpage URL |
| `html_content` | string | Complete HTML source code of the page |

## Dataset Structure

```
data.csv
├── Column 1: url (264,716 entries)
│   └── Format: https://example.com/page
│   └── Examples: cybersecurity blogs, tech articles, business content
│
└── Column 2: html_content (264,716 entries)
    └── Format: HTML string with full page source
    └── Includes: <!DOCTYPE>, <html>, <head>, <body> tags
    └── Contains: text, scripts, styles, metadata
```

## Download Instructions

### Method 1: Manual Download (Recommended)

1. Visit: https://www.kaggle.com/datasets/naveen1729/dataset-for-assignment
2. Click "Download" button (requires Kaggle account)
3. Extract the ZIP file
4. Rename to `data.csv` and place in the `data/` directory

### Method 2: Kaggle API

```bash
# Install Kaggle API
pip install kaggle

# Configure API credentials
# (Place kaggle.json in ~/.kaggle/)

# Download dataset
kaggle datasets download -d naveen1729/dataset-for-assignment

# Extract and move
unzip dataset-for-assignment.zip -d data/
mv data/dataset-for-assignment.csv data/data.csv
```

## Data Processing Pipeline

### Step 1: HTML Parsing
- Remove scripts and styles
- Extract text content
- Preserve structure (headings, paragraphs)
- Clean whitespace

### Step 2: Feature Extraction
- **Word Count**: Total words in cleaned text
- **Sentence Count**: Number of sentences
- **Readability**: Flesch Reading Ease score
- **Title**: Page title from <title> tag

### Step 3: Quality Classification
Based on extracted features:
- **High**: >1500 words, readability 50-70
- **Medium**: 500-1500 words, readability 30-70
- **Low**: <500 words or readability <30

### Step 4: Duplicate Detection
- Generate embeddings using Sentence Transformers
- Compute cosine similarity
- Flag pairs with >80% similarity

## Output Files

After processing, the following files are generated:

### 1. extracted_content.csv
```csv
title,clean_text,word_count,url
"Page Title","Cleaned text...",1500,"https://example.com"
```

### 2. features.csv
```csv
url,word_count,sentence_count,readability,quality_label
"https://example.com",1500,75,60.5,"High"
```

### 3. duplicates.csv
```csv
url_1,url_2,similarity
"https://example.com/page1","https://example.com/page2",0.85
```

## Data Quality Notes

### Strengths
- Large volume (264k+ pages)
- Diverse content types
- Real-world HTML structures
- Suitable for NLP and ML tasks

### Considerations
- Some pages may have minimal content
- HTML quality varies (malformed tags, etc.)
- Some URLs may be duplicates
- Encoding issues in some entries

### Preprocessing Recommendations
1. **Filter empty content**: Remove pages with `null` or empty HTML
2. **Handle encoding**: Use UTF-8 with error handling
3. **Limit sample size**: For testing, use first 1000-10000 rows
4. **Validate URLs**: Check for malformed or duplicate URLs

## Usage Examples

### Load Dataset
```python
import pandas as pd

# Load full dataset
df = pd.read_csv('data/data.csv')

# Load sample for testing
df_sample = pd.read_csv('data/data.csv', nrows=1000)

# Check for missing values
print(f"Missing URLs: {df['url'].isna().sum()}")
print(f"Missing HTML: {df['html_content'].isna().sum()}")
```

### Basic Statistics
```python
# Dataset info
print(f"Total pages: {len(df)}")
print(f"Unique URLs: {df['url'].nunique()}")

# HTML content length distribution
df['html_length'] = df['html_content'].str.len()
print(df['html_length'].describe())
```

## Performance Considerations

### Processing Time Estimates
- **HTML Parsing**: ~10 pages/second (26 hours for full dataset)
- **Feature Extraction**: ~15 pages/second (18 hours)
- **Embeddings**: ~50 pages/second (5 hours)
- **Total Pipeline**: ~30-35 hours for full dataset

### Memory Requirements
- **Full Dataset Load**: ~3-4 GB RAM
- **With Embeddings**: ~6-8 GB RAM
- **Recommended**: 16 GB RAM for comfortable processing

### Optimization Tips
1. Process in batches (10,000 rows at a time)
2. Use multiprocessing for HTML parsing
3. Cache intermediate results
4. Use GPU for embeddings if available

## Citation

If you use this dataset in your research or project, please credit:

```
Dataset: dataset_for_assignment
Author: naveen1729
Source: Kaggle (https://www.kaggle.com/datasets/naveen1729/dataset-for-assignment)
Accessed: November 2025
```

## Support

For dataset-specific issues:
- Visit the Kaggle dataset page
- Check the dataset discussion section
- Contact the dataset author through Kaggle

For processing pipeline issues:
- Check our project README.md
- Review CHECKLIST.md
- Open an issue on GitHub

## Related Datasets

Other similar datasets for SEO/content analysis:
- Common Crawl web corpus
- ClueWeb datasets
- Web scraped content collections on Kaggle

---

**Last Updated**: November 3, 2025  
**Dataset Version**: Check Kaggle for latest version  
**Processing Pipeline Version**: 1.0.0
