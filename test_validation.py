"""
Comprehensive validation tests for the Subsurface Imaging API.
"""
import requests
import json
import time
import sys

class ValidationTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_valid_queries(self):
        """Test that valid queries work correctly."""
        print("Testing valid queries...")
        
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
            {
                "name": "Valid with base64 format",
                "params": {"depth_min": 9100.0, "depth_max": 9110.0, "image_id": 2, "format": "base64"}
            }
        ]
        
        passed = 0
        for case in test_cases:
            try:
                response = self.session.get(f"{self.base_url}/frames", params=case["params"])
                if response.status_code == 200:
                    data = response.json()
                    print(f"✓ {case['name']}: {data['total_frames']} frames")
                    passed += 1
                else:
                    print(f"✗ {case['name']}: Failed with {response.status_code}")
                    print(f"  Error: {response.text}")
            except Exception as e:
                print(f"✗ {case['name']}: Exception - {e}")
        
        print(f"Valid queries: {passed}/{len(test_cases)} passed\n")
        return passed == len(test_cases)
    
    def test_invalid_image_ids(self):
        """Test validation of invalid image IDs."""
        print("Testing invalid image IDs...")
        
        test_cases = [
            {"image_id": 0, "expected_status": 404},
            {"image_id": 9, "expected_status": 404},
            {"image_id": 10, "expected_status": 404},
            {"image_id": -1, "expected_status": 404},
            {"image_id": 999, "expected_status": 404}
        ]
        
        passed = 0
        for case in test_cases:
            try:
                params = {
                    "depth_min": 9000.1,
                    "depth_max": 9005.0,
                    "image_id": case["image_id"]
                }
                response = self.session.get(f"{self.base_url}/frames", params=params)
                
                if response.status_code == case["expected_status"]:
                    error_data = response.json()
                    if "available_images" in error_data:
                        print(f"✓ Image ID {case['image_id']}: Correctly rejected with available images list")
                    else:
                        print(f"✓ Image ID {case['image_id']}: Correctly rejected")
                    passed += 1
                else:
                    print(f"✗ Image ID {case['image_id']}: Expected {case['expected_status']}, got {response.status_code}")
                    print(f"  Response: {response.text}")
            except Exception as e:
                print(f"✗ Image ID {case['image_id']}: Exception - {e}")
        
        print(f"Invalid image IDs: {passed}/{len(test_cases)} passed\n")
        return passed == len(test_cases)
    
    def test_invalid_depth_ranges(self):
        """Test validation of invalid depth ranges."""
        print("Testing invalid depth ranges...")
        
        test_cases = [
            {
                "name": "Depth min >= depth max",
                "params": {"depth_min": 9005.0, "depth_max": 9000.0, "image_id": 1},
                "expected_status": 400
            },
            {
                "name": "Depth range too low",
                "params": {"depth_min": 8000.0, "depth_max": 8500.0, "image_id": 1},
                "expected_status": 400
            },
            {
                "name": "Depth range too high",
                "params": {"depth_min": 9600.0, "depth_max": 9700.0, "image_id": 1},
                "expected_status": 400
            },
            {
                "name": "Depth range partially out of bounds (low)",
                "params": {"depth_min": 8999.0, "depth_max": 9005.0, "image_id": 1},
                "expected_status": 400
            },
            {
                "name": "Depth range partially out of bounds (high)",
                "params": {"depth_min": 9540.0, "depth_max": 9550.0, "image_id": 1},
                "expected_status": 400
            },
            {
                "name": "Depth range too large",
                "params": {"depth_min": 9000.0, "depth_max": 10000.0, "image_id": 1},
                "expected_status": 400
            }
        ]
        
        passed = 0
        for case in test_cases:
            try:
                response = self.session.get(f"{self.base_url}/frames", params=case["params"])
                
                if response.status_code == case["expected_status"]:
                    error_data = response.json()
                    print(f"✓ {case['name']}: Correctly rejected")
                    if "available_range" in error_data:
                        print(f"  Available range: {error_data['available_range']['min']:.1f} - {error_data['available_range']['max']:.1f}")
                    passed += 1
                else:
                    print(f"✗ {case['name']}: Expected {case['expected_status']}, got {response.status_code}")
                    print(f"  Response: {response.text}")
            except Exception as e:
                print(f"✗ {case['name']}: Exception - {e}")
        
        print(f"Invalid depth ranges: {passed}/{len(test_cases)} passed\n")
        return passed == len(test_cases)
    
    def test_invalid_parameters(self):
        """Test validation of invalid parameters."""
        print("Testing invalid parameters...")
        
        test_cases = [
            {
                "name": "Invalid format",
                "params": {"depth_min": 9000.1, "depth_max": 9005.0, "image_id": 1, "format": "xml"},
                "expected_status": 400
            },
            {
                "name": "Invalid format (empty)",
                "params": {"depth_min": 9000.1, "depth_max": 9005.0, "image_id": 1, "format": ""},
                "expected_status": 400
            }
        ]
        
        passed = 0
        for case in test_cases:
            try:
                response = self.session.get(f"{self.base_url}/frames", params=case["params"])
                
                if response.status_code == case["expected_status"]:
                    error_data = response.json()
                    print(f"✓ {case['name']}: Correctly rejected")
                    if "Valid formats" in error_data.get("detail", ""):
                        print(f"  Error message includes valid formats")
                    passed += 1
                else:
                    print(f"✗ {case['name']}: Expected {case['expected_status']}, got {response.status_code}")
                    print(f"  Response: {response.text}")
            except Exception as e:
                print(f"✗ {case['name']}: Exception - {e}")
        
        print(f"Invalid parameters: {passed}/{len(test_cases)} passed\n")
        return passed == len(test_cases)
    
    def test_image_endpoint_validation(self):
        """Test validation on the /frames/image endpoint."""
        print("Testing /frames/image endpoint validation...")
        
        test_cases = [
            {
                "name": "Valid query",
                "params": {"depth_min": 9000.1, "depth_max": 9005.0, "image_id": 1},
                "expected_status": 200
            },
            {
                "name": "Invalid image ID",
                "params": {"depth_min": 9000.1, "depth_max": 9005.0, "image_id": 99},
                "expected_status": 404
            },
            {
                "name": "Invalid depth range",
                "params": {"depth_min": 8000.0, "depth_max": 8500.0, "image_id": 1},
                "expected_status": 400
            }
        ]
        
        passed = 0
        for case in test_cases:
            try:
                response = self.session.get(f"{self.base_url}/frames/image", params=case["params"])
                
                if response.status_code == case["expected_status"]:
                    print(f"✓ {case['name']}: Status {response.status_code}")
                    if case["expected_status"] == 200:
                        print(f"  Content-Type: {response.headers.get('content-type')}")
                        print(f"  Content-Length: {len(response.content)} bytes")
                    passed += 1
                else:
                    print(f"✗ {case['name']}: Expected {case['expected_status']}, got {response.status_code}")
                    print(f"  Response: {response.text}")
            except Exception as e:
                print(f"✗ {case['name']}: Exception - {e}")
        
        print(f"Image endpoint validation: {passed}/{len(test_cases)} passed\n")
        return passed == len(test_cases)
    
    def test_error_message_quality(self):
        """Test that error messages are helpful and informative."""
        print("Testing error message quality...")
        
        test_cases = [
            {
                "name": "Image not found error",
                "params": {"depth_min": 9000.1, "depth_max": 9005.0, "image_id": 99},
                "check_fields": ["error", "message", "available_images"]
            },
            {
                "name": "Invalid depth range error",
                "params": {"depth_min": 8000.0, "depth_max": 8500.0, "image_id": 1},
                "check_fields": ["error", "message", "available_range", "requested_range"]
            }
        ]
        
        passed = 0
        for case in test_cases:
            try:
                response = self.session.get(f"{self.base_url}/frames", params=case["params"])
                
                if response.status_code in [400, 404]:
                    error_data = response.json()
                    missing_fields = []
                    
                    # Check if the error data is nested under 'detail' (FastAPI format)
                    actual_data = error_data.get('detail', error_data)
                    
                    for field in case["check_fields"]:
                        if field not in actual_data:
                            missing_fields.append(field)
                    
                    if not missing_fields:
                        print(f"✓ {case['name']}: All required fields present")
                        print(f"  Error: {actual_data.get('error', 'N/A')}")
                        print(f"  Message: {actual_data.get('message', 'N/A')}")
                        passed += 1
                    else:
                        print(f"✗ {case['name']}: Missing fields: {missing_fields}")
                        print(f"  Response: {error_data}")
                else:
                    print(f"✗ {case['name']}: Expected 400/404, got {response.status_code}")
            except Exception as e:
                print(f"✗ {case['name']}: Exception - {e}")
        
        print(f"Error message quality: {passed}/{len(test_cases)} passed\n")
        return passed == len(test_cases)
    
    def run_all_validation_tests(self):
        """Run all validation tests."""
        print("Starting Validation Tests")
        print("=" * 50)
        
        tests = [
            self.test_valid_queries,
            self.test_invalid_image_ids,
            self.test_invalid_depth_ranges,
            self.test_invalid_parameters,
            self.test_image_endpoint_validation,
            self.test_error_message_quality
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f"Test failed with exception: {e}")
        
        print("=" * 50)
        print(f"Validation Test Results: {passed}/{total} test suites passed")
        
        if passed == total:
            print("🎉 All validation tests passed!")
            return True
        else:
            print("❌ Some validation tests failed. Check the output above.")
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test validation in the Subsurface Imaging API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--wait", type=int, default=5, help="Seconds to wait for API startup")
    
    args = parser.parse_args()
    
    print(f"Waiting {args.wait} seconds for API to start...")
    time.sleep(args.wait)
    
    tester = ValidationTester(args.url)
    success = tester.run_all_validation_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
