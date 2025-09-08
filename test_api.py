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
    
    def test_images_endpoint(self):
        print("\nTesting /images endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/images")
            if response.status_code == 200:
                data = response.json()
                print(f"Found {data['total_images']} images")
                for img in data['images']:
                    print(f"  Image {img['image_id']}: depths {img['depth_min']:.1f}-{img['depth_max']:.1f} ({img['depth_count']} levels)")
                return data['images']
            else:
                print(f"Images endpoint failed: {response.status_code}")
                return []
        except Exception as e:
            print(f"Images endpoint error: {e}")
            return []
    
    def test_frames_endpoint(self, image_id=1, depth_min=9000, depth_max=9010):
        print(f"\nTesting /frames endpoint (image {image_id}, depth {depth_min}-{depth_max})...")
        try:
            params = {
                "image_id": image_id,
                "depth_min": depth_min,
                "depth_max": depth_max,
                "format": "json",
                "colormap": True
            }
            
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/frames", params=params)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                query_time = (end_time - start_time) * 1000
                print(f"Query successful: {data['total_frames']} frames in {query_time:.1f}ms")
                print(f"Processing time: {data['processing_time_ms']:.1f}ms")
                
                if data['frames']:
                    frame = data['frames'][0]
                    print(f"Sample frame: depth={frame['depth']}, format={frame['format']}")
                    if 'rgb_data' in frame:
                        rgb = frame['rgb_data']
                        print(f"RGB data available: R={len(rgb['red'])}, G={len(rgb['green'])}, B={len(rgb['blue'])} pixels")
                
                return True
            else:
                print(f"Frames endpoint failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"Frames endpoint error: {e}")
            return False
    
    def test_image_generation(self, image_id=1, depth_min=9000, depth_max=9020):
        print(f"\nTesting /frames/image endpoint (image {image_id}, depth {depth_min}-{depth_max})...")
        try:
            params = {
                "image_id": image_id,
                "depth_min": depth_min,
                "depth_max": depth_max,
                "colormap": True
            }
            
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/frames/image", params=params)
            end_time = time.time()
            
            if response.status_code == 200:
                query_time = (end_time - start_time) * 1000
                print(f"Image generation successful in {query_time:.1f}ms")
                print(f"Content type: {response.headers.get('content-type')}")
                print(f"Content length: {len(response.content)} bytes")
                
                try:
                    img = Image.open(io.BytesIO(response.content))
                    print(f"Image dimensions: {img.size[0]}x{img.size[1]}")
                    print(f"Image mode: {img.mode}")
                    
                    output_file = f"test_output_image_{image_id}_{depth_min}_{depth_max}.png"
                    img.save(output_file)
                    print(f"Saved test image as: {output_file}")
                except Exception as e:
                    print(f"Could not process image: {e}")
                
                return True
            else:
                print(f"Image generation failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"Image generation error: {e}")
            return False
    
    def test_stats_endpoint(self):
        print("\nTesting /stats endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/stats")
            if response.status_code == 200:
                data = response.json()
                print(f"Array path: {data['array_path']}")
                print(f"Total surveys: {data['total_surveys']}")
                print(f"Depth range: {data['depth_range']['min']:.1f} to {data['depth_range']['max']:.1f}")
                print(f"Depth span: {data['depth_range']['span']:.1f}")
                print(f"Dimensions: depth_index={data['dimensions']['depth_index']}, pixel_index={data['dimensions']['pixel_index']}")
                print(f"Attributes: {', '.join(data['attributes'])}")
                return True
            else:
                print(f"Stats endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"Stats endpoint error: {e}")
            return False
    
    def test_error_handling(self):
        print("\nTesting error handling...")
        
        print("  Testing invalid image ID...")
        try:
            params = {"image_id": 999, "depth_min": 9000, "depth_max": 9010}
            response = self.session.get(f"{self.base_url}/frames", params=params)
            if response.status_code == 404:
                print("    ✓ Invalid image ID properly rejected")
            else:
                print(f"    ✗ Expected 404, got {response.status_code}")
        except Exception as e:
            print(f"    ✗ Error testing invalid image ID: {e}")
        
        print("  Testing invalid depth range...")
        try:
            params = {"image_id": 1, "depth_min": 9010, "depth_max": 9000}
            response = self.session.get(f"{self.base_url}/frames", params=params)
            if response.status_code == 400:
                print("    ✓ Invalid depth range properly rejected")
            else:
                print(f"    ✗ Expected 400, got {response.status_code}")
        except Exception as e:
            print(f"    ✗ Error testing invalid depth range: {e}")
    
    def run_comprehensive_test(self):
        print("🚀 Starting comprehensive API tests...")
        print("=" * 50)
        
        if not self.test_health():
            print("❌ Health check failed - API may not be running")
            return False
        
        images = self.test_images_endpoint()
        if not images:
            print("❌ No images available for testing")
            return False
        
        test_image = images[0]
        image_id = test_image['image_id']
        depth_min = test_image['depth_min'] + 1
        depth_max = min(test_image['depth_min'] + 20, test_image['depth_max'])
        
        success = True
        success &= self.test_frames_endpoint(image_id, depth_min, depth_max)
        success &= self.test_image_generation(image_id, depth_min, depth_max)
        success &= self.test_stats_endpoint()
        
        self.test_error_handling()
        
        print("=" * 50)
        if success:
            print("✅ All core tests passed!")
        else:
            print("❌ Some tests failed")
        
        return success

def main():
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8000"
    
    print(f"Testing API at: {base_url}")
    
    tester = APITester(base_url)
    success = tester.run_comprehensive_test()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()