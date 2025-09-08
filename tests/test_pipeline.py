#!/usr/bin/env python3
import os
import time
import shutil
import requests
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.logging_config import get_test_logger

logger = get_test_logger(__name__)

def test_automated_pipeline():
    """Test the complete automated pipeline system"""
    logger.info("🚀 Testing Automated Data Pipeline")
    logger.info("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Test pipeline status endpoint
    logger.info("1. Checking pipeline status...")
    try:
        response = requests.get(f"{base_url}/pipeline/status")
        if response.status_code == 200:
            status = response.json()
            logger.info(f"   Pipeline status: {status['status']}")
            logger.info(f"   Watch directory: {status['watch_directory']}")
            logger.info(f"   Total processed: {status['total_processed']}")
        else:
            logger.error(f"   Error: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"   Error connecting to API: {e}")
        return False
    
    # Start the pipeline
    logger.info("\n2. Starting pipeline...")
    try:
        response = requests.post(f"{base_url}/pipeline/start")
        if response.status_code == 200:
            result = response.json()
            logger.info(f"   {result['message']}")
        else:
            logger.error(f"   Error: {response.status_code}")
    except Exception as e:
        logger.error(f"   Error: {e}")
    
    # Test file processing by copying CSV
    logger.info("\n3. Setting up test scenario...")
    incoming_dir = Path("data")
    incoming_dir.mkdir(parents=True, exist_ok=True)
    
    if os.path.exists("Challenge2.csv"):
        test_file = incoming_dir / "test_survey.csv"
        logger.info(f"   Copying Challenge2.csv to {test_file}")
        shutil.copy2("Challenge2.csv", test_file)
        
        # Wait for pipeline to detect and process file
        logger.info("   Waiting for automatic processing...")
        time.sleep(5)
        
        logger.info("\n4. Checking processing results...")
        response = requests.get(f"{base_url}/pipeline/status")
        if response.status_code == 200:
            status = response.json()
            logger.info(f"   Total processed: {status['total_processed']}")
            logger.info(f"   Recent history: {len(status['recent_history'])} entries")
            
            if status['recent_history']:
                latest = status['recent_history'][-1]
                logger.info(f"   Latest processing: {latest['file']} - {'✅ Success' if latest['success'] else '❌ Failed'}")
                if latest.get('processing_time'):
                    logger.info(f"   Processing time: {latest['processing_time']:.2f}s")
    else:
        logger.warning("   No Challenge2.csv found for testing")
    
    logger.info("\n5. Testing manual scan trigger...")
    try:
        response = requests.post(f"{base_url}/pipeline/scan")
        if response.status_code == 200:
            result = response.json()
            logger.info(f"   {result['message']}")
        else:
            logger.error(f"   Error: {response.status_code}")
    except Exception as e:
        logger.error(f"   Error: {e}")
    
    logger.info("\n6. Checking images after pipeline processing...")
    try:
        response = requests.get(f"{base_url}/images")
        if response.status_code == 200:
            images = response.json()
            logger.info(f"   Total images available: {images['total_images']}")
            for img in images['images'][:3]:
                logger.info(f"   - Image {img['image_id']}: depths {img['depth_min']:.1f}-{img['depth_max']:.1f}")
        else:
            logger.error(f"   Error: {response.status_code}")
    except Exception as e:
        logger.error(f"   Error: {e}")
    
    logger.info("\n7. Getting pipeline history...")
    try:
        response = requests.get(f"{base_url}/pipeline/history")
        if response.status_code == 200:
            history = response.json()
            logger.info(f"   Processing history: {len(history['processing_history'])} entries")
            logger.info(f"   Total processed: {history['total_processed']}")
            logger.info(f"   Failed files: {len(history['failed_files'])}")
        else:
            logger.error(f"   Error: {response.status_code}")
    except Exception as e:
        logger.error(f"   Error: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info("✅ Pipeline test completed!")
    logger.info("\n📁 Directory structure:")
    for directory in ["data", "data/processed", "data/failed", "db"]:
        if os.path.exists(directory):
            if directory == "db":
                logger.info(f"   {directory}: TileDB database")
            else:
                files = list(Path(directory).glob("*.csv"))
                logger.info(f"   {directory}: {len(files)} CSV files")
    
    return True

def cleanup_test_data():
    """Remove test files after testing"""
    logger.info("\n🧹 Cleaning up test data...")
    test_dirs = ["data/incoming", "data/processed", "data/failed"]
    for directory in test_dirs:
        if os.path.exists(directory):
            for file in Path(directory).glob("test_*.csv"):
                file.unlink()
                logger.info(f"   Removed {file}")

if __name__ == "__main__":
    try:
        test_automated_pipeline()
    except KeyboardInterrupt:
        logger.warning("\n⏹️  Test interrupted")
    finally:
        cleanup_test_data()
