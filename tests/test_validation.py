import requests
import json
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.logging_config import get_test_logger

logger = get_test_logger(__name__)

class ValidationTester:
    """Test query parameter validation"""
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_valid_queries(self):
        """Test valid query parameters"""
        logger.info("Testing valid queries...")
        
        test_cases = [
            {
                "name": "Valid image 1, small range",
                "params": {"depth_min": 9000.1, "depth_max": 9005.0, "image_id": 1}
            },
            {
                "name": "Valid image 5, medium range", 
                "params": {"depth_min": 9200.0, "depth_max": 9250.0, "image_id": 5}
            },
            {
                "name": "Valid image 8, large range",
                "params": {"depth_min": 9500.0, "depth_max": 9546.0, "image_id": 8}
            },
        ]
        
        for case in test_cases:
            try:
                response = self.session.get(f"{self.base_url}/frames", params=case["params"])
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"  ✓ {case['name']}: {data['total_frames']} frames")
                else:
                    logger.error(f"  ✗ {case['name']}: {response.status_code}")
            except Exception as e:
                logger.error(f"  ✗ {case['name']}: {e}")
    
    def test_invalid_image_ids(self):
        """Test invalid image ID handling"""
        logger.info("\nTesting invalid image IDs...")
        
        invalid_ids = [0, 999, -1, 100]
        
        for image_id in invalid_ids:
            try:
                params = {"depth_min": 9000, "depth_max": 9010, "image_id": image_id}
                response = self.session.get(f"{self.base_url}/frames", params=params)
                
                if response.status_code == 404:
                    logger.info(f"  ✓ Image ID {image_id}: Correctly rejected (404)")
                else:
                    logger.error(f"  ✗ Image ID {image_id}: Expected 404, got {response.status_code}")
            except Exception as e:
                logger.error(f"  ✗ Image ID {image_id}: {e}")
    
    def test_invalid_depth_ranges(self):
        logger.info("\nTesting invalid depth ranges...")
        
        test_cases = [
            {"name": "Inverted range", "depth_min": 9010, "depth_max": 9000},
            {"name": "Out of bounds low", "depth_min": 8000, "depth_max": 8010},
            {"name": "Out of bounds high", "depth_min": 10000, "depth_max": 10010},
            {"name": "Partially out of bounds", "depth_min": 8999, "depth_max": 9001},
        ]
        
        for case in test_cases:
            try:
                params = {
                    "depth_min": case["depth_min"], 
                    "depth_max": case["depth_max"], 
                    "image_id": 1
                }
                response = self.session.get(f"{self.base_url}/frames", params=params)
                
                if response.status_code == 400:
                    logger.info(f"  ✓ {case['name']}: Correctly rejected (400)")
                else:
                    logger.error(f"  ✗ {case['name']}: Expected 400, got {response.status_code}")
            except Exception as e:
                logger.error(f"  ✗ {case['name']}: {e}")
    
    def test_invalid_parameters(self):
        logger.info("\nTesting invalid parameters...")
        
        test_cases = [
            {"name": "Invalid format", "params": {"depth_min": 9000, "depth_max": 9010, "image_id": 1, "format": "xml"}},
            {"name": "Missing depth_min", "params": {"depth_max": 9010, "image_id": 1}},
            {"name": "Missing depth_max", "params": {"depth_min": 9000, "image_id": 1}},
        ]
        
        for case in test_cases:
            try:
                response = self.session.get(f"{self.base_url}/frames", params=case["params"])
                
                if response.status_code in [400, 422]:
                    logger.info(f"  ✓ {case['name']}: Correctly rejected ({response.status_code})")
                else:
                    logger.error(f"  ✗ {case['name']}: Expected 400/422, got {response.status_code}")
            except Exception as e:
                logger.error(f"  ✗ {case['name']}: {e}")
    
    def test_edge_cases(self):
        logger.info("\nTesting edge cases...")
        
        test_cases = [
            {"name": "Single depth point", "depth_min": 9000.1, "depth_max": 9000.1},
            {"name": "Very small range", "depth_min": 9000.1, "depth_max": 9000.2},
            {"name": "Exact boundary match", "depth_min": 9000.1, "depth_max": 9546.0},
        ]
        
        for case in test_cases:
            try:
                params = {
                    "depth_min": case["depth_min"], 
                    "depth_max": case["depth_max"], 
                    "image_id": 1
                }
                response = self.session.get(f"{self.base_url}/frames", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"  ✓ {case['name']}: {data['total_frames']} frames")
                elif response.status_code == 400:
                    logger.info(f"  ⚠ {case['name']}: No data in range (400)")
                else:
                    logger.error(f"  ✗ {case['name']}: {response.status_code}")
            except Exception as e:
                logger.error(f"  ✗ {case['name']}: {e}")
    
    def test_format_options(self):
        logger.info("\nTesting format options...")
        
        formats = ["json", "base64"]
        
        for fmt in formats:
            try:
                params = {"depth_min": 9000, "depth_max": 9010, "image_id": 1, "format": fmt}
                response = self.session.get(f"{self.base_url}/frames", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    if fmt == "json" and data['frames'] and 'rgb_data' in data['frames'][0]:
                        logger.info(f"  ✓ Format {fmt}: RGB data structure")
                    elif fmt == "base64" and data['frames'] and 'image_data' in data['frames'][0]:
                        logger.info(f"  ✓ Format {fmt}: Base64 image data")
                    else:
                        logger.info(f"  ⚠ Format {fmt}: Unexpected response structure")
                else:
                    logger.error(f"  ✗ Format {fmt}: {response.status_code}")
            except Exception as e:
                logger.error(f"  ✗ Format {fmt}: {e}")
    
    def test_colormap_options(self):
        logger.info("\nTesting colormap options...")
        
        colormap_options = [True, False]
        
        for colormap in colormap_options:
            try:
                params = {"depth_min": 9000, "depth_max": 9010, "image_id": 1, "colormap": colormap}
                response = self.session.get(f"{self.base_url}/frames", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    if data['frames']:
                        frame = data['frames'][0]
                        if colormap and 'rgb_data' in frame:
                            logger.info(f"  ✓ Colormap {colormap}: RGB data provided")
                        elif not colormap and 'grayscale_data' in frame:
                            logger.info(f"  ✓ Colormap {colormap}: Grayscale data provided")
                        else:
                            logger.info(f"  ⚠ Colormap {colormap}: Unexpected data format")
                    else:
                        logger.info(f"  ⚠ Colormap {colormap}: No frames returned")
                else:
                    logger.error(f"  ✗ Colormap {colormap}: {response.status_code}")
            except Exception as e:
                logger.error(f"  ✗ Colormap {colormap}: {e}")
    
    def test_performance_validation(self):
        logger.info("\nTesting query performance...")
        
        params = {"depth_min": 9000, "depth_max": 9050, "image_id": 1}
        
        times = []
        for i in range(5):
            start_time = time.time()
            try:
                response = self.session.get(f"{self.base_url}/frames", params=params)
                end_time = time.time()
                
                if response.status_code == 200:
                    query_time = (end_time - start_time) * 1000
                    times.append(query_time)
                    logger.info(f"  Query {i+1}: {query_time:.1f}ms")
                else:
                    logger.info(f"  Query {i+1}: Failed ({response.status_code})")
            except Exception as e:
                logger.info(f"  Query {i+1}: Error ({e})")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            logger.info(f"  Performance summary: avg={avg_time:.1f}ms, min={min_time:.1f}ms, max={max_time:.1f}ms")
    
    def run_all_validation_tests(self):
        logger.info("🔍 Starting comprehensive validation tests...")
        logger.info("=" * 60)
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code != 200:
                logger.error("❌ API health check failed - cannot run validation tests")
                return False
        except Exception as e:
            logger.info(f"❌ Cannot connect to API: {e}")
            return False
        
        self.test_valid_queries()
        self.test_invalid_image_ids()
        self.test_invalid_depth_ranges()
        self.test_invalid_parameters()
        self.test_edge_cases()
        self.test_format_options()
        self.test_colormap_options()
        self.test_performance_validation()
        
        logger.info("=" * 60)
        logger.info("✅ Validation tests completed!")
        return True

def main():
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8000"
    
    logger.info(f"Running validation tests against: {base_url}")
    
    tester = ValidationTester(base_url)
    success = tester.run_all_validation_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()