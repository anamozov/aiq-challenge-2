"""
Test script for the Subsurface Imaging API.
"""
import requests
import json
import time
import base64
from PIL import Image
import io
import sys

class APITester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health(self):
        print("Testing health endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"Health check passed: {data['status']}")
                if data.get('array_accessible'):
                    print(f"Array accessible with {data.get('total_surveys', 0)} surveys")
                return True
            else:
                print(f"Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"Health check error: {e}")
            return False
    
    def test_stats(self):
        print("Testing statistics endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/stats")
            if response.status_code == 200:
                data = response.json()
                print("Statistics retrieved successfully:")
                print(f"Array path: {data['array_path']}")
                print(f"Total surveys: {data['total_surveys']}")
                print(f"Depth range: {data['depth_range']['min']:.1f} - {data['depth_range']['max']:.1f}")
                print(f"Compression: {data['compression_info']}")
                return True
            else:
                print(f"Statistics failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"Statistics error: {e}")
            return False
    
    def test_frames_json(self):
        print("Testing frames endpoint (JSON format)...")
        try:
            params = {
                "depth_min": 9100.0,
                "depth_max": 9110.0,
                "survey_id": 1,
                "format": "json",
                "colormap": True
            }
            
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/frames", params=params)
            request_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                print("JSON frames retrieved successfully:")
                print(f"Request time: {request_time:.1f}ms")
                print(f"Total frames: {data['total_frames']}")
                print(f"Processing time: {data['processing_time_ms']:.1f}ms")
                
                if data['frames']:
                    first_frame = data['frames'][0]
                    print(f"Frame dimensions: {first_frame['width']}x{first_frame['height']}")
                    print(f"Format: {first_frame['format']}")
                    print(f"Depth: {first_frame['depth']}")
                
                return True
            else:
                print(f"JSON frames failed: {response.status_code}")
                print(f"Error: {response.text}")
                return False
        except Exception as e:
            print(f"JSON frames error: {e}")
            return False
    
    def test_frames_base64(self):
        print("Testing frames endpoint (Base64 format)...")
        try:
            params = {
                "depth_min": 9100.0,
                "depth_max": 9105.0,
                "survey_id": 1,
                "format": "base64",
                "colormap": True
            }
            
            response = self.session.get(f"{self.base_url}/frames", params=params)
            
            if response.status_code == 200:
                data = response.json()
                print("Base64 frames retrieved successfully:")
                print(f"Total frames: {data['total_frames']}")
                
                if data['frames'] and 'image_data' in data['frames'][0]:
                    img_data = data['frames'][0]['image_data']
                    img_bytes = base64.b64decode(img_data)
                    img = Image.open(io.BytesIO(img_bytes))
                    print(f"First image size: {img.size}")
                    print(f"Image mode: {img.mode}")
                    
                    img.save("test_frame.png")
                    print(f"Saved test image: test_frame.png")
                
                return True
            else:
                print(f"Base64 frames failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"Base64 frames error: {e}")
            return False
    
    def test_frames_image(self):
        print("Testing direct image endpoint...")
        try:
            params = {
                "depth_min": 9100.0,
                "depth_max": 9120.0,
                "survey_id": 1,
                "colormap": True
            }
            
            response = self.session.get(f"{self.base_url}/frames/image", params=params)
            
            if response.status_code == 200:
                print("Direct image retrieved successfully:")
                print(f"Content type: {response.headers.get('content-type')}")
                print(f"Content length: {len(response.content)} bytes")
                
                with open("test_combined_image.png", "wb") as f:
                    f.write(response.content)
                print(f"Saved combined image: test_combined_image.png")
                
                return True
            else:
                print(f"Direct image failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"Direct image error: {e}")
            return False
    
    def test_error_handling(self):
        print("Testing error handling...")
        
        try:
            params = {
                "depth_min": 9200.0,
                "depth_max": 9100.0,
                "survey_id": 1
            }
            response = self.session.get(f"{self.base_url}/frames", params=params)
            if response.status_code == 400:
                print("Invalid depth range properly rejected")
            else:
                print(f"Expected 400, got {response.status_code}")
        except Exception as e:
            print(f"Error handling test failed: {e}")
    
    def test_performance(self):
        print("Testing performance...")
        
        test_cases = [
            {"name": "Small range (10 depths)", "depth_min": 9100.0, "depth_max": 9110.0},
            {"name": "Medium range (50 depths)", "depth_min": 9100.0, "depth_max": 9150.0},
            {"name": "Large range (100 depths)", "depth_min": 9100.0, "depth_max": 9200.0},
        ]
        
        for case in test_cases:
            try:
                params = {
                    "depth_min": case["depth_min"],
                    "depth_max": case["depth_max"],
                    "survey_id": 1,
                    "format": "json"
                }
                
                start_time = time.time()
                response = self.session.get(f"{self.base_url}/frames", params=params)
                request_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"{case['name']}: {request_time:.1f}ms ({data['total_frames']} frames)")
                else:
                    print(f"{case['name']}: Failed ({response.status_code})")
            except Exception as e:
                print(f"{case['name']}: Error - {e}")
    
    def run_all_tests(self):
        print("Starting API Tests")
        print("----------------------------------------")
        
        tests = [
            self.test_health,
            self.test_stats,
            self.test_frames_json,
            self.test_frames_base64,
            self.test_frames_image,
            self.test_error_handling,
            self.test_performance
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f"Test failed with exception: {e}")
        
        print("----------------------------------------")
        print(f"Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("All tests passed!")
            return True
        else:
            print("Some tests failed. Check the output above.")
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Subsurface Imaging API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--wait", type=int, default=5, help="Seconds to wait for API startup")
    
    args = parser.parse_args()
    
    print(f"Waiting {args.wait} seconds for API to start...")
    time.sleep(args.wait)
    
    tester = APITester(args.url)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
