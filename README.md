# Real Estate Listing Scraper

Scrapes Redfin property listings from a given URL and displays them in a Streamlit dashboard.

Demo https://www.loom.com/share/95793999527646bbaf23494a3073fc4b

## Quick Start

1. **Setup**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Run**:
   ```bash
   source s1-env/bin/activate
   streamlit run scrape_redfin_Copy1.py
   ```

3. **Open** your browser to `http://localhost:8501`

## What it does

- Scrapes property listings from Redfin
- Shows interactive dashboard with charts and filters
- Saves data to CSV files

## Requirements

- Python 3.8+
- Internet connection

## Files

- `scrape_redfin_Copy1.py` - Main app
- `requirements.txt` - Dependencies
- `setup.sh` - Auto-setup script
- `kensington_redfin_listings.csv` - Raw data
- `kensington_redfin_listings_cleaned.csv` - Cleaned data
