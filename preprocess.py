import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def preprocess_caida_data(input_file='codered-august.table.txt', output_file='codered_processed.csv', freq='30T'):
    """
    Preprocess CAIDA Code Red dataset for malware dynamics modeling.
    
    Parameters:
    - input_file: Path to the CAIDA table file
    - output_file: Path for the processed CSV output  
    - freq: Time binning frequency ('30T' for 30-minute intervals)
    
    Returns:
    - DataFrame with processed intensity data
    """
    
    print(f"Processing CAIDA Code Red data from {input_file}...")
    
    # Read the data with proper column names
    column_names = ['start_time', 'end_time', 'tld', 'country', 'latitude', 'longitude', 'as_number']
    
    try:
        df = pd.read_csv(input_file, sep='\t', comment='#', names=column_names)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        print("Please download the CAIDA Code Red dataset from:")
        print("https://www.caida.org/catalog/datasets/codered/")
        return None
    
    # Convert Unix timestamps to datetime
    df['start_datetime'] = pd.to_datetime(df['start_time'], unit='s')
    df['end_datetime'] = pd.to_datetime(df['end_time'], unit='s')
    
    # Sort by start time
    df = df.sort_values('start_time')
    
    # Extract the date range
    start_date = df['start_datetime'].min()
    end_date = df['start_datetime'].max()
    print(f"Data spans from {start_date} to {end_date}")
    
    # Create time bins
    time_bins = pd.date_range(start=start_date, end=end_date, freq=freq)
    df['time_bin'] = pd.cut(df['start_datetime'].astype('int64') // 10**9, 
                            bins=time_bins.astype('int64') // 10**9, 
                            labels=time_bins[:-1])
    
    # Count events per time bin
    intensity_series = df.groupby('time_bin').size()
    
    # Fill missing values with 0
    full_range = pd.DataFrame(index=time_bins[:-1])
    intensity_series = intensity_series.reindex(full_range.index, fill_value=0)
    
    # Convert to DataFrame for easier manipulation
    intensity_df = intensity_series.reset_index()
    intensity_df.columns = ['timestamp', 'intensity']
    
    # Calculate cumulative infections
    intensity_df['cumulative'] = intensity_df['intensity'].cumsum()
    
    # Save processed data
    intensity_df.to_csv(output_file, index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(intensity_df['timestamp'], intensity_df['intensity'])
    plt.title('Code Red Infection Intensity (per 30-minute interval)')
    plt.ylabel('Number of Infections')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 1, 2)
    plt.plot(intensity_df['timestamp'], intensity_df['cumulative'])
    plt.title('Code Red Cumulative Infections')
    plt.ylabel('Total Infections')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('codered_intensity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Processed data saved to '{output_file}'")
    print(f"Visualization saved to 'codered_intensity.png'")
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print(intensity_df['intensity'].describe())
    print(f"Total data points: {len(intensity_df)}")
    print(f"Time resolution: {freq}")
    
    return intensity_df

if __name__ == "__main__":
    # Default processing
    result = preprocess_caida_data()
    
    if result is not None:
        print("\nData preprocessing complete!")
        print("You can now run the modeling scripts:")
        print("1. julia parameter_estimator.jl codered_processed.csv")
        print("2. julia ode.jl codered_processed.csv")
        print("3. julia ude.jl codered_processed.csv") 
        print("4. julia neural_ode.jl codered_processed.csv")
        print("5. julia analysis.jl")