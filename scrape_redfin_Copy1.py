#!/usr/bin/env python
# coding: utf-8

"""
Redfin Property Scraper and Dashboard
ABB #5 - Homework 1
Code authored by: Michael Trang
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import streamlit as st
from typing import List, Dict, Optional
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configuration
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}
BASE_URL = "https://www.redfin.com/neighborhood/156651/PA/Philadelphia/Kensington/filter/viewport=40.01614:39.96379:-75.10399:-75.14235"
NUM_PAGES = 7
CSV_FILENAME = "kensington_redfin_listings.csv"
CLEANED_CSV_FILENAME = "kensington_redfin_listings_cleaned.csv"

def scrape_redfin_listings() -> List[Dict]:
    """Scrape property listings from Redfin"""
    all_listings = []
    
    for page in range(1, NUM_PAGES + 1):
        url = BASE_URL if page == 1 else f"{BASE_URL}/page-{page}"
        print(f"Scraping page {page}: {url}")
        
        html_content = fetch_page_with_retry(url)
        if not html_content:
            continue
            
        page_listings = parse_listings_from_html(html_content)
        all_listings.extend(page_listings)
        time.sleep(2)  # Be polite to the server
    
    return all_listings

def fetch_page_with_retry(url: str, max_attempts: int = 5) -> Optional[str]:
    """Fetch page content with retry logic for rate limiting"""
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 429:
                print(f"Rate limited. Waiting 30 seconds before retry {attempt + 1}/{max_attempts}")
                time.sleep(30)
            else:
                print(f"Failed to retrieve page. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching page: {e}")
            return None
    return None

def parse_listings_from_html(html_content: str) -> List[Dict]:
    """Parse property listings from HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    listings = soup.find_all('div', class_='HomeCardContainer')
    
    parsed_listings = []
    for card in listings:
        # Extract property type from JSON-LD script
        property_type = extract_property_type(card)
        
        listing = {
            "address": extract_text(card, 'div', 'bp-Homecard__Address'),
            "price": extract_text(card, 'span', 'bp-Homecard__Price--value'),
            "beds": extract_text(card, 'span', 'bp-Homecard__Stats--beds'),
            "baths": extract_text(card, 'span', 'bp-Homecard__Stats--baths'),
            "sqft": extract_text(card, 'span', 'bp-Homecard__LockedStat--value'),
            "property_type": property_type,
        }
        parsed_listings.append(listing)
    
    return parsed_listings

def extract_property_type(card) -> Optional[str]:
    """Extract property type from JSON-LD structured data"""
    try:
        # Find the script tag with JSON-LD data
        script_tag = card.find('script', type='application/ld+json')
        if script_tag:
            import json
            json_data = json.loads(script_tag.string)
            
            # Look for the property type in the structured data
            for item in json_data:
                if isinstance(item, dict) and '@type' in item:
                    property_type = item['@type']
                    # Convert schema.org types to more readable format
                    type_mapping = {
                        'SingleFamilyResidence': 'Single Family',
                        'Apartment': 'Apartment',
                        'Condo': 'Condo',
                        'Townhouse': 'Townhouse',
                        'MultiFamilyResidence': 'Multi Family'
                    }
                    return type_mapping.get(property_type, property_type)
    except (json.JSONDecodeError, AttributeError, KeyError):
        pass
    
    return None

def extract_text(element, tag: str, class_name: str) -> Optional[str]:
    """Extract text from HTML element safely"""
    found_element = element.find(tag, class_=class_name)
    return found_element.get_text(strip=True) if found_element else None

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataframe by removing empty rows"""
    # Remove rows where all columns are empty
    df_cleaned = df.dropna(how='all')
    
    # Remove rows where important columns are empty
    df_cleaned = df_cleaned.dropna(subset=['address', 'price'])
    
    return df_cleaned

def prepare_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare price data for filtering and visualization"""
    df_copy = df.copy()
    
    # Clean price column for filtering
    df_copy['price_int'] = df_copy['price'].str.replace(r'[\$,]', '', regex=True).astype(float)
    
    # Calculate price per square foot
    df_copy['price_per_sqft'] = calculate_price_per_sqft(df_copy)
    
    return df_copy

def calculate_price_per_sqft(df: pd.DataFrame) -> pd.Series:
    """Calculate price per square foot for each listing"""
    # Clean square footage data
    df_copy = df.copy()
    
    # Convert sqft to numeric, handling various formats
    df_copy['sqft_clean'] = df_copy['sqft'].astype(str).str.replace(',', '').str.replace('sq ft', '').str.strip()
    
    # Convert to numeric, with errors='coerce' to handle non-numeric values
    df_copy['sqft_numeric'] = pd.to_numeric(df_copy['sqft_clean'], errors='coerce')
    
    # Calculate price per square foot
    price_per_sqft = df_copy['price_int'] / df_copy['sqft_numeric']
    
    # Handle division by zero or missing values
    price_per_sqft = price_per_sqft.replace([float('inf'), -float('inf')], None)
    
    return price_per_sqft

def create_streamlit_dashboard(df: pd.DataFrame):
    """Create the Streamlit dashboard"""
    st.title("Kensington Redfin Listings Dashboard")
    
    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Listings", len(df))
    with col2:
        avg_price = df['price_int'].mean()
        st.metric("Average Price", f"${avg_price:,.0f}")
    with col3:
        median_price = df['price_int'].median()
        st.metric("Median Price", f"${median_price:,.0f}")
    with col4:
        # Show average price per sqft
        if 'price_per_sqft' in df.columns:
            avg_price_per_sqft = df['price_per_sqft'].dropna().mean()
            st.metric("Avg Price/sqft", f"${avg_price_per_sqft:.0f}")
    
    # Show the data table with price per sqft
    st.subheader("All Listings")
    
    # Format the display dataframe
    display_df = df.copy()
    if 'price_per_sqft' in display_df.columns:
        display_df['price_per_sqft_formatted'] = display_df['price_per_sqft'].apply(
            lambda x: f"${x:.0f}" if pd.notna(x) else "N/A"
        )
        # Reorder columns to show price per sqft prominently
        column_order = ['address', 'price', 'beds', 'baths', 'sqft', 'price_per_sqft_formatted']
        if 'property_type' in display_df.columns:
            column_order.insert(1, 'property_type')
        
        # Only show columns that exist
        existing_columns = [col for col in column_order if col in display_df.columns]
        display_df = display_df[existing_columns]
    
    st.dataframe(display_df, use_container_width=True)
    
    # Property type filter (if available)
    if 'property_type' in df.columns:
        st.subheader("Filter by Property Type")
        property_types = ['All'] + df['property_type'].dropna().unique().tolist()
        selected_type = st.selectbox("Select property type:", property_types)
        
        if selected_type != 'All':
            df = df[df['property_type'] == selected_type]
    
    # Price filter
    st.subheader("Filter by Price")
    min_price = int(df['price_int'].min())
    max_price = int(df['price_int'].max())
    
    price_range = st.slider(
        "Select price range ($)", 
        min_price, 
        max_price, 
        (min_price, max_price),
        step=1000
    )
    
    # Price per sqft filter
    if 'price_per_sqft' in df.columns:
        st.subheader("Filter by Price per Square Foot")
        valid_price_per_sqft = df['price_per_sqft'].dropna()
        if len(valid_price_per_sqft) > 0:
            min_price_per_sqft = int(valid_price_per_sqft.min())
            max_price_per_sqft = int(valid_price_per_sqft.max())
            
            price_per_sqft_range = st.slider(
                "Select price per sqft range ($)", 
                min_price_per_sqft, 
                max_price_per_sqft, 
                (min_price_per_sqft, max_price_per_sqft),
                step=10
            )
    
    # Filter data
    filtered_df = df[(df['price_int'] >= price_range[0]) & (df['price_int'] <= price_range[1])]
    
    # Apply price per sqft filter if available
    if 'price_per_sqft' in df.columns and len(valid_price_per_sqft) > 0:
        filtered_df = filtered_df[
            (filtered_df['price_per_sqft'] >= price_per_sqft_range[0]) & 
            (filtered_df['price_per_sqft'] <= price_per_sqft_range[1])
        ]
    
    st.write(f"Showing **{len(filtered_df)}** listings in selected ranges:")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Improved Price Distribution Chart
    st.subheader("Price Distribution")
    create_price_histogram(filtered_df, 'price_int', 'Price ($)', 'Number of Listings')
    
    # Improved Price per Square Foot Distribution Chart
    if 'price_per_sqft' in df.columns:
        st.subheader("Price per Square Foot Distribution")
        valid_price_per_sqft_filtered = filtered_df['price_per_sqft'].dropna()
        if len(valid_price_per_sqft_filtered) > 0:
            create_price_histogram(filtered_df, 'price_per_sqft', 'Price per Square Foot ($/sqft)', 'Number of Listings')
    
    # Property type distribution chart (if available)
    if 'property_type' in df.columns:
        st.subheader("Property Type Distribution")
        type_counts = df['property_type'].value_counts()
        st.bar_chart(type_counts)

def create_price_histogram(df: pd.DataFrame, column: str, x_label: str, y_label: str):
    """Create a proper histogram for price data"""
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Get valid data
    valid_data = df[column].dropna()
    
    if len(valid_data) == 0:
        st.write("No valid data for this chart.")
        return
    
    # Calculate optimal number of bins using Sturges' formula
    n_bins = int(1 + 3.322 * np.log10(len(valid_data)))
    n_bins = max(5, min(20, n_bins))  # Keep between 5 and 20 bins
    
    # Create histogram using plotly
    fig = px.histogram(
        valid_data,
        nbins=n_bins,
        title=f"{x_label} Distribution",
        labels={'value': x_label, 'count': y_label},
        opacity=0.7,
        color_discrete_sequence=['#1f77b4']
    )
    
    # Customize the layout
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=False,
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Add hover information
    fig.update_traces(
        hovertemplate=f"<b>{x_label}:</b> %{{x}}<br><b>{y_label}:</b> %{{y}}<extra></extra>"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"Mean {x_label}", f"${valid_data.mean():,.0f}")
    with col2:
        st.metric(f"Median {x_label}", f"${valid_data.median():,.0f}")
    with col3:
        st.metric(f"Std Dev {x_label}", f"${valid_data.std():,.0f}")

def main():
    """Main function to run the scraper and dashboard"""
    # Check if we should scrape new data or use existing
    if st.sidebar.checkbox("Scrape new data (takes time)"):
        st.info("Scraping new data from Redfin... This may take a few minutes.")
        listings = scrape_redfin_listings()
        df = pd.DataFrame(listings)
        df.to_csv(CSV_FILENAME, index=False)
        st.success("Scraping completed!")
    else:
        # Load existing data
        try:
            df = pd.read_csv(CSV_FILENAME)
        except FileNotFoundError:
            st.error("No data file found. Please check 'Scrape new data' to get started.")
            return
    
    # Clean the data
    df_cleaned = clean_dataframe(df)
    df_cleaned.to_csv(CLEANED_CSV_FILENAME, index=False)
    
    # Prepare data for dashboard
    df_prepared = prepare_price_data(df_cleaned)
    
    # Create dashboard
    create_streamlit_dashboard(df_prepared)
    
    # Display data info
    with st.expander("📈 Data Information"):
        st.write(f"**Original rows:** {len(df)}")
        st.write(f"**Cleaned rows:** {len(df_cleaned)}")
        st.write(f"**Data shape:** {df_cleaned.shape}")

if __name__ == "__main__":
    main()



