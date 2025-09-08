#!/usr/bin/env python3
import os
import time
import shutil
import requests
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_automated_pipeline():
    print("🚀 Testing Automated Data Pipeline")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    print("1. Checking pipeline status...")
    try:
        response = requests.get(f"{base_url}/pipeline/status")
        if response.status_code == 200:
            status = response.json()
            print(f"   Pipeline status: {status['status']}")
            print(f"   Watch directory: {status['watch_directory']}")
            print(f"   Total processed: {status['total_processed']}")
        else:
            print(f"   Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"   Error connecting to API: {e}")
        return False
    
    print("\n2. Starting pipeline...")
    try:
        response = requests.post(f"{base_url}/pipeline/start")
        if response.status_code == 200:
            result = response.json()
            print(f"   {result['message']}")
        else:
            print(f"   Error: {response.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n3. Setting up test scenario...")
    incoming_dir = Path("data")
    incoming_dir.mkdir(parents=True, exist_ok=True)
    
    if os.path.exists("Challenge2.csv"):
        test_file = incoming_dir / "test_survey.csv"
        print(f"   Copying Challenge2.csv to {test_file}")
        shutil.copy2("Challenge2.csv", test_file)
        
        print("   Waiting for automatic processing...")
        time.sleep(5)
        
        print("\n4. Checking processing results...")
        response = requests.get(f"{base_url}/pipeline/status")
        if response.status_code == 200:
            status = response.json()
            print(f"   Total processed: {status['total_processed']}")
            print(f"   Recent history: {len(status['recent_history'])} entries")
            
            if status['recent_history']:
                latest = status['recent_history'][-1]
                print(f"   Latest processing: {latest['file']} - {'✅ Success' if latest['success'] else '❌ Failed'}")
                if latest.get('processing_time'):
                    print(f"   Processing time: {latest['processing_time']:.2f}s")
    else:
        print("   No Challenge2.csv found for testing")
    
    print("\n5. Testing manual scan trigger...")
    try:
        response = requests.post(f"{base_url}/pipeline/scan")
        if response.status_code == 200:
            result = response.json()
            print(f"   {result['message']}")
        else:
            print(f"   Error: {response.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n6. Checking images after pipeline processing...")
    try:
        response = requests.get(f"{base_url}/images")
        if response.status_code == 200:
            images = response.json()
            print(f"   Total images available: {images['total_images']}")
            for img in images['images'][:3]:
                print(f"   - Image {img['image_id']}: depths {img['depth_min']:.1f}-{img['depth_max']:.1f}")
        else:
            print(f"   Error: {response.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n7. Getting pipeline history...")
    try:
        response = requests.get(f"{base_url}/pipeline/history")
        if response.status_code == 200:
            history = response.json()
            print(f"   Processing history: {len(history['processing_history'])} entries")
            print(f"   Total processed: {history['total_processed']}")
            print(f"   Failed files: {len(history['failed_files'])}")
        else:
            print(f"   Error: {response.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Pipeline test completed!")
    print("\n📁 Directory structure:")
    for directory in ["data", "data/processed", "data/failed", "db"]:
        if os.path.exists(directory):
            if directory == "db":
                print(f"   {directory}: TileDB database")
            else:
                files = list(Path(directory).glob("*.csv"))
                print(f"   {directory}: {len(files)} CSV files")
    
    return True

def cleanup_test_data():
    print("\n🧹 Cleaning up test data...")
    test_dirs = ["data/incoming", "data/processed", "data/failed"]
    for directory in test_dirs:
        if os.path.exists(directory):
            for file in Path(directory).glob("test_*.csv"):
                file.unlink()
                print(f"   Removed {file}")

if __name__ == "__main__":
    try:
        test_automated_pipeline()
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted")
    finally:
        cleanup_test_data()
