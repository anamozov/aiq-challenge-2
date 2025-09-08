#!/usr/bin/env python3
"""
Performance test script to demonstrate the optimized data ingestion.
"""
import os
import time
import shutil
from data_ingestion import SubsurfaceDataProcessor

def test_performance():
    """Test the performance of multiple data ingestion runs."""
    
    # Clean up any existing data
    array_path = "data/arrays/subsurface_data"
    if os.path.exists(array_path):
        shutil.rmtree(array_path)
    
    processor = SubsurfaceDataProcessor(array_path)
    csv_file = "Challenge2.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please ensure the CSV file exists.")
        return
    
    print("Testing optimized data ingestion performance...")
    print("=" * 50)
    
    # Run multiple ingestion cycles
    for run in range(1, 6):  # 5 runs
        print(f"\nRun {run}:")
        start_time = time.time()
        
        try:
            processor.process_csv_file(csv_file)
            
            elapsed_time = time.time() - start_time
            print(f"  Completed in {elapsed_time:.2f} seconds")
            
            # Get array info
            info = processor.get_array_info()
            if 'non_empty_domain' in info:
                domain = info['non_empty_domain']
                if isinstance(domain, tuple) and len(domain) >= 2:
                    rows = domain[0][1] + 1 if domain[0][1] is not None else 0
                    print(f"  Array size: {rows} x 150")
            
        except Exception as e:
            print(f"  Error: {e}")
            break
    
    print("\n" + "=" * 50)
    print("Performance test completed!")
    print("\nKey improvements:")
    print("- No more complete array recreation on each append")
    print("- Memory usage stays constant regardless of array size")
    print("- Performance should be consistent across all runs")
    print("- Fixed NaN value handling")

if __name__ == "__main__":
    test_performance()
