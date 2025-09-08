#!/usr/bin/env python3
import os
import time
import shutil
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.data_ingestion import ImageDataProcessor

def test_performance():
    array_path = "db/arrays/image_data"
    if os.path.exists(array_path):
        shutil.rmtree(array_path)
    
    processor = ImageDataProcessor(array_path)
    csv_file = "Challenge2.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please ensure the CSV file exists.")
        return
    
    print("Testing optimized data ingestion performance...")
    print("=" * 50)
    
    for run in range(1, 6):
        print(f"\nRun {run}:")
        
        start_time = time.time()
        processor.process_csv_file(csv_file)
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"Processing time: {processing_time:.2f} seconds")
        
        available_images = processor.get_available_images()
        print(f"Total images: {len(available_images)}")
        
        info = processor.get_array_info()
        if 'non_empty_domain' in info:
            print(f"Array domain: {info['non_empty_domain']}")
    
    print("\n" + "=" * 50)
    print("Performance test completed!")

if __name__ == "__main__":
    test_performance()