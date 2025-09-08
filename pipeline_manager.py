import os
import time
import json
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import schedule
import pandas as pd
from data_ingestion import ImageDataProcessor
from typing import Dict, List, Optional
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CSVValidator:
    def __init__(self):
        self.required_columns = 201
        self.expected_headers = ['depth'] + [f'pixel_{i}' for i in range(200)]
    
    def validate_csv_structure(self, file_path):
        try:
            df = pd.read_csv(file_path, nrows=1)
            
            if len(df.columns) != self.required_columns:
                return False, f"Expected {self.required_columns} columns, got {len(df.columns)}"
            
            try:
                depth_col = df.iloc[:, 0]
                float(depth_col.iloc[0])
            except (ValueError, IndexError):
                return False, "First column (depth) must be numeric"
            
            for i in range(1, min(10, len(df.columns))):
                try:
                    float(df.iloc[0, i])
                except ValueError:
                    return False, f"Pixel column {i} contains non-numeric data"
            
            return True, "Valid CSV structure"
            
        except Exception as e:
            return False, f"CSV validation error: {str(e)}"
    
    def validate_data_quality(self, file_path):
        try:
            df = pd.read_csv(file_path)
            
            if len(df) == 0:
                return False, "CSV file is empty"
            
            depth_col = df.iloc[:, 0]
            if depth_col.isnull().sum() > 0:
                return False, "Depth column contains null values"
            
            pixel_cols = df.iloc[:, 1:]
            null_percentage = (pixel_cols.isnull().sum().sum() / pixel_cols.size) * 100
            
            if null_percentage > 10:
                return False, f"Too many null values in pixel data: {null_percentage:.1f}%"
            
            return True, f"Data quality check passed. Null values: {null_percentage:.1f}%"
            
        except Exception as e:
            return False, f"Data quality validation error: {str(e)}"

class ProcessingStatus:
    def __init__(self, status_file="db/pipeline_status.json"):
        self.status_file = status_file
        self.ensure_status_file()
    
    def ensure_status_file(self):
        os.makedirs(os.path.dirname(self.status_file), exist_ok=True)
        if not os.path.exists(self.status_file):
            self.save_status({
                "last_run": None,
                "total_processed": 0,
                "failed_files": [],
                "processing_history": []
            })
    
    def load_status(self):
        try:
            with open(self.status_file, 'r') as f:
                return json.load(f)
        except:
            return {
                "last_run": None,
                "total_processed": 0,
                "failed_files": [],
                "processing_history": []
            }
    
    def save_status(self, status):
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2, default=str)
    
    def update_processing_result(self, file_path, success, error_message=None, processing_time=None):
        status = self.load_status()
        
        result = {
            "file": str(file_path),
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "processing_time": processing_time,
            "error": error_message
        }
        
        status["processing_history"].append(result)
        status["last_run"] = datetime.now().isoformat()
        
        if success:
            status["total_processed"] += 1
            if str(file_path) in status["failed_files"]:
                status["failed_files"].remove(str(file_path))
        else:
            if str(file_path) not in status["failed_files"]:
                status["failed_files"].append(str(file_path))
        
        status["processing_history"] = status["processing_history"][-50:]
        
        self.save_status(status)

class FileWatcher(FileSystemEventHandler):
    def __init__(self, pipeline_manager):
        self.pipeline_manager = pipeline_manager
        self.processed_files = set()
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.csv'):
            filename = os.path.basename(event.src_path)
            if filename not in self.processed_files:
                logger.info(f"New CSV file detected: {event.src_path}")
                time.sleep(1)
                self.processed_files.add(filename)
                self.pipeline_manager.process_file(event.src_path)
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.csv'):
            filename = os.path.basename(event.src_path)
            if filename not in self.processed_files:
                logger.info(f"Modified CSV file detected: {event.src_path}")
                time.sleep(1)
                self.processed_files.add(filename)
                self.pipeline_manager.process_file(event.src_path)

class DataPipelineManager:
    def __init__(self, 
                 watch_directory="data",
                 processed_directory="data/processed",
                 failed_directory="data/failed"):
        
        self.watch_directory = Path(watch_directory)
        self.processed_directory = Path(processed_directory)
        self.failed_directory = Path(failed_directory)
        
        self.watch_directory.mkdir(parents=True, exist_ok=True)
        self.processed_directory.mkdir(parents=True, exist_ok=True)
        self.failed_directory.mkdir(parents=True, exist_ok=True)
        
        self.validator = CSVValidator()
        self.status = ProcessingStatus()
        self.processor = ImageDataProcessor(array_path="db/arrays/image_data")
        
        self.observer = None
        self.running = False
    
    def process_file(self, file_path):
        file_path = Path(file_path)
        start_time = time.time()
        
        logger.info(f"Processing file: {file_path}")
        
        try:
            valid_structure, structure_msg = self.validator.validate_csv_structure(file_path)
            if not valid_structure:
                raise ValueError(f"Structure validation failed: {structure_msg}")
            
            valid_quality, quality_msg = self.validator.validate_data_quality(file_path)
            if not valid_quality:
                raise ValueError(f"Quality validation failed: {quality_msg}")
            
            logger.info(f"Validation passed: {quality_msg}")
            
            self.processor.process_csv_file(str(file_path))
            
            processing_time = time.time() - start_time
            logger.info(f"Successfully processed {file_path} in {processing_time:.2f}s")
            
            self._move_file(file_path, self.processed_directory)
            self.status.update_processing_result(file_path, True, processing_time=processing_time)
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Failed to process {file_path}: {error_msg}")
            
            self._move_file(file_path, self.failed_directory)
            self.status.update_processing_result(file_path, False, error_msg, processing_time)
    
    def _move_file(self, source, destination_dir):
        try:
            destination = destination_dir / source.name
            counter = 1
            while destination.exists():
                name_parts = source.stem, counter, source.suffix
                destination = destination_dir / f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
                counter += 1
            
            source.rename(destination)
            logger.info(f"Moved {source} to {destination}")
        except Exception as e:
            logger.error(f"Failed to move file {source}: {e}")
    
    def scan_existing_files(self):
        logger.info(f"Scanning for existing files in {self.watch_directory}")
        
        csv_files = list(self.watch_directory.glob("*.csv"))
        if csv_files:
            logger.info(f"Found {len(csv_files)} existing CSV files")
            for csv_file in csv_files:
                self.process_file(csv_file)
        else:
            logger.info("No existing CSV files found")
    
    def start_file_watcher(self):
        if self.observer is not None:
            return
        
        logger.info(f"Starting file watcher on {self.watch_directory}")
        
        event_handler = FileWatcher(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.watch_directory), recursive=False)
        self.observer.start()
    
    def stop_file_watcher(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("File watcher stopped")
    
    def start_scheduler(self):
        schedule.every(30).minutes.do(self.scan_existing_files)
        schedule.every().day.at("03:00").do(self.scan_existing_files)
        
        logger.info("Scheduler started: scanning every 30 minutes and daily at 3 AM")
        
        def run_scheduler():
            while self.running:
                schedule.run_pending()
                time.sleep(60)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
    
    def start(self):
        logger.info("Starting Data Pipeline Manager")
        self.running = True
        
        self.scan_existing_files()
        self.start_file_watcher()
        self.start_scheduler()
        
        logger.info("Data Pipeline Manager started successfully")
    
    def stop(self):
        logger.info("Stopping Data Pipeline Manager")
        self.running = False
        self.stop_file_watcher()
        logger.info("Data Pipeline Manager stopped")
    
    def get_status(self):
        return self.status.load_status()

def main():
    pipeline = DataPipelineManager()
    
    try:
        pipeline.start()
        
        logger.info("Pipeline is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(10)
            
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()
