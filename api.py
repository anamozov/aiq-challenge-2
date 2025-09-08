"""
Subsurface imaging API for depth range queries.
"""
import os
import io
import base64
from pathlib import Path
import numpy as np
import tiledb
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import cv2
from PIL import Image
import logging
import time
from datetime import datetime
from typing import List, Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Subsurface Imaging API",
    description="API for querying subsurface imaging data by depth ranges",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

ARRAY_PATH = "data/arrays/subsurface_data"
CACHE_DIR = "data/cache"
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

class FrameInfo(BaseModel):
    depth: float
    image_id: int
    width: int = 150
    height: int = 1
    timestamp: str = None

class QueryInfo(BaseModel):
    depth_min: float
    depth_max: float
    image_id: int
    format: str
    colormap: bool

class FrameData(BaseModel):
    depth: float
    width: int
    height: int
    format: str
    image_data: str = None
    rgb_data: dict = None
    grayscale_data: list = None

class DepthRangeResponse(BaseModel):
    query_info: QueryInfo
    frames: list
    total_frames: int
    processing_time_ms: float

class DepthRange(BaseModel):
    min: float
    max: float
    span: float

class Dimensions(BaseModel):
    depth_index: str
    pixel_index: str

class ImageInfo(BaseModel):
    image_id: int
    depth_count: int
    depth_min: float
    depth_max: float
    processed_at: int

class ImagesResponse(BaseModel):
    images: list
    total_images: int

class ArrayStats(BaseModel):
    array_path: str
    total_surveys: int
    depth_range: DepthRange
    dimensions: Dimensions
    attributes: list
    compression_info: str

class ValidationError(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None

class ImageNotFoundError(ValidationError):
    error: str = "Image not found"
    available_images: List[int]

class InvalidDepthRangeError(ValidationError):
    error: str = "Invalid depth range"
    available_range: DepthRange
    requested_range: DepthRange

class QueryValidator:
    def __init__(self, api_instance):
        self.api = api_instance
        self._cached_images = None
        self._cache_timestamp = None
        self._cache_ttl = 60  # Cache for 60 seconds
    
    def _get_cached_images(self):
        """Get cached images or refresh cache if needed."""
        current_time = time.time()
        
        if (self._cached_images is None or 
            self._cache_timestamp is None or 
            current_time - self._cache_timestamp > self._cache_ttl):
            
            try:
                images_response = self.api.get_images()
                self._cached_images = {img.image_id: img for img in images_response.images}
                self._cache_timestamp = current_time
                logger.info(f"Cached {len(self._cached_images)} images")
            except Exception as e:
                logger.error(f"Failed to cache images: {e}")
                self._cached_images = {}
        
        return self._cached_images
    
    def validate_image_id(self, image_id: int) -> ImageInfo:
        """Validate that the image ID exists and return image info."""
        images = self._get_cached_images()
        
        if not images:
            raise HTTPException(
                status_code=503, 
                detail="Unable to retrieve available images. Please try again later."
            )
        
        if image_id not in images:
            available_ids = sorted(images.keys())
            error_detail = ImageNotFoundError(
                message=f"Image ID {image_id} not found. Available images: {available_ids}",
                available_images=available_ids
            )
            raise HTTPException(status_code=404, detail=error_detail.dict())
        
        return images[image_id]
    
    def validate_depth_range(self, image_info: ImageInfo, depth_min: float, depth_max: float):
        """Validate that the depth range is within the image's available range."""
        # Check if depth_min is less than depth_max
        if depth_min >= depth_max:
            raise HTTPException(
                status_code=400, 
                detail="depth_min must be less than depth_max"
            )
        
        # Check if depth range is within available range
        if depth_min < image_info.depth_min or depth_max > image_info.depth_max:
            error_detail = InvalidDepthRangeError(
                message=f"Depth range {depth_min}-{depth_max} is outside available range for image {image_info.image_id}",
                available_range=DepthRange(
                    min=image_info.depth_min,
                    max=image_info.depth_max,
                    span=image_info.depth_max - image_info.depth_min
                ),
                requested_range=DepthRange(
                    min=depth_min,
                    max=depth_max,
                    span=depth_max - depth_min
                )
            )
            raise HTTPException(status_code=400, detail=error_detail.dict())
        
        # Check if depth range is too large (optional business rule)
        max_range_size = 1000.0
        if depth_max - depth_min > max_range_size:
            raise HTTPException(
                status_code=400,
                detail=f"Depth range too large (max {max_range_size} units). Requested range: {depth_max - depth_min:.1f} units"
            )
    
    def validate_query_parameters(self, image_id: int, depth_min: float, depth_max: float, 
                                format: str = "json", colormap: bool = True):
        """Comprehensive validation of all query parameters."""
        # Validate image ID and get image info
        image_info = self.validate_image_id(image_id)
        
        # Validate depth range
        self.validate_depth_range(image_info, depth_min, depth_max)
        
        # Validate format parameter
        valid_formats = ["json", "base64"]
        if format not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format '{format}'. Valid formats: {valid_formats}"
            )
        
        # Validate colormap parameter (should be boolean, but FastAPI handles this)
        if not isinstance(colormap, bool):
            raise HTTPException(
                status_code=400,
                detail="colormap parameter must be a boolean value"
            )
        
        return image_info

class SubsurfaceAPI:
    def __init__(self, array_path=ARRAY_PATH):
        self.array_path = array_path
    
    def array_exists(self):
        try:
            return tiledb.object_type(self.array_path) == "array"
        except Exception:
            return False
    
    def _validate_array(self):
        if not tiledb.object_type(self.array_path) == "array":
            raise RuntimeError(f"TileDB array not found at {self.array_path}. Run data_ingestion.py first.")
    
    def get_array_stats(self):
        try:
            if not self.array_exists():
                raise RuntimeError(f"TileDB array not found at {self.array_path}.")
            with tiledb.open(self.array_path, mode='r') as array:
                schema = array.schema
                try:
                    non_empty_domain = array.nonempty_domain()
                except AttributeError:
                    non_empty_domain = array.non_empty_domain()
                
                # Get array dimensions efficiently
                total_rows = non_empty_domain[0][1] + 1 if non_empty_domain[0][1] is not None else 0
                
                # Sample data to get statistics without loading entire array
                sample_size = min(1000, total_rows)
                if sample_size > 0:
                    sample_data = array[0:sample_size, 0:1]
                    depth_values = sample_data['depth_value']
                    image_ids = sample_data['image_id']
                    
                    unique_images = np.unique(image_ids)
                    unique_depths = np.unique(depth_values)
                    
                    # Estimate total images from sample
                    estimated_images = len(unique_images)
                else:
                    estimated_surveys = 0
                    unique_depths = np.array([0.0])
                
                return ArrayStats(
                    array_path=self.array_path,
                    total_surveys=estimated_images,
                    depth_range=DepthRange(
                        min=float(np.min(unique_depths)),
                        max=float(np.max(unique_depths)),
                        span=float(np.max(unique_depths) - np.min(unique_depths))
                    ),
                    dimensions=Dimensions(
                        depth_index="depth_index",
                        pixel_index="pixel_index"
                    ),
                    attributes=[schema.attr(i).name for i in range(schema.nattr)],
                    compression_info="ZSTD level 1"
                )
        except Exception as e:
            logger.error(f"Error getting array stats: {e}")
            raise HTTPException(status_code=500, detail=f"Error accessing array: {str(e)}")
    
    def get_images(self):
        try:
            if not self.array_exists():
                raise RuntimeError(f"TileDB array not found at {self.array_path}.")
            
            with tiledb.open(self.array_path, mode='r') as array:
                # Get array dimensions efficiently
                non_empty_domain = array.nonempty_domain()
                total_rows = non_empty_domain[0][1] + 1 if non_empty_domain[0][1] is not None else 0
                
                # Sample data from the beginning of each image (every 5461 rows)
                # Each image is 5461 rows, so sample from start of each image
                image_size = 5461
                sample_size = 5461  # Sample the full image to get complete depth range
                unique_images = set()
                all_depth_values = []
                all_image_ids = []
                all_processed_at = []
                
                if total_rows > 0:
                    # For 3D array: sample by image_id directly
                    non_empty_domain = array.nonempty_domain()
                    if non_empty_domain[0][0] is not None:
                        # Get all existing image_ids
                        for img_id in range(non_empty_domain[0][0], non_empty_domain[0][1] + 1):
                            try:
                                # Get all depths for this image to find actual range
                                sample = array[img_id, :, 0:1]  # All depths
                                depth_values = sample['depth_value'][:, 0]
                                
                                # Find actual data range (skip NaN values)
                                valid_depths = depth_values[~np.isnan(depth_values)]
                                if len(valid_depths) > 0:
                                    unique_images.add(img_id)
                                    all_depth_values.extend([valid_depths[0], valid_depths[-1]])
                                    all_image_ids.extend([img_id, img_id])
                                    all_processed_at.extend([sample['processed_at'][0, 0], sample['processed_at'][len(valid_depths)-1, 0]])
                            except:
                                pass
                    
                    # Convert to numpy arrays for processing
                    depth_values = np.array(all_depth_values)
                    image_ids = np.array(all_image_ids)
                    processed_at = np.array(all_processed_at)
                    
                    images = []
                    
                    for image_id in unique_images:
                        image_mask = image_ids == image_id
                        
                        if np.any(image_mask):
                            image_depths = depth_values[image_mask]
                            
                            if hasattr(processed_at, '__getitem__'):
                                image_timestamps = processed_at[image_mask]
                            else:
                                image_timestamps = np.array([processed_at] * np.sum(image_mask))
                            
                            unique_depths = np.unique(image_depths)
                        else:
                            continue
                        
                        finite_depths = unique_depths[np.isfinite(unique_depths)]
                        
                        if len(finite_depths) == 0:
                            logger.warning(f"No valid depth values for image {image_id}")
                            continue
                        
                        depth_min = float(np.min(finite_depths))
                        depth_max = float(np.max(finite_depths))
                        processed_at_val = int(np.min(image_timestamps))
                        
                        if not (np.isfinite(depth_min) and np.isfinite(depth_max)):
                            logger.warning(f"Invalid depth values for image {image_id}: min={depth_min}, max={depth_max}")
                            continue
                        
                        images.append(ImageInfo(
                            image_id=int(image_id),
                            depth_count=len(unique_depths),
                            depth_min=depth_min,
                            depth_max=depth_max,
                            processed_at=processed_at_val
                        ))
                    
                    images.sort(key=lambda x: x.image_id)
                    
                    return ImagesResponse(
                        images=images,
                        total_images=len(images)
                    )
                else:
                    return ImagesResponse(images=[], total_images=0)
                
        except Exception as e:
            logger.error(f"Error getting images: {e}")
            raise HTTPException(status_code=500, detail=f"Error accessing images: {str(e)}")
    
    def query_depth_range(self, image_id, depth_min, depth_max, return_format="json", include_colormap=True):
        start_time = time.time()
        
        try:
            if not self.array_exists():
                raise RuntimeError(f"TileDB array not found at {self.array_path}.")
            
            with tiledb.open(self.array_path, mode='r') as array:
                logger.info(f"Querying depth range {depth_min} to {depth_max} for image {image_id}")
                
                # Get array dimensions efficiently
                non_empty_domain = array.nonempty_domain()
                # For 3D array: [image_id, depth_index, pixel_index]
                image_domain = non_empty_domain[0]
                depth_domain = non_empty_domain[1]
                
                if image_domain[1] is None or depth_domain[1] is None:
                    return DepthRangeResponse(
                        query_info=QueryInfo(
                            image_id=image_id,
                            depth_min=depth_min,
                            depth_max=depth_max,
                            format=return_format,
                            colormap=include_colormap
                        ),
                        frames=[],
                        total_frames=0,
                        processing_time_ms=(time.time() - start_time) * 1000
                    )
                
                # OPTIMIZED: Use 3D range query instead of loading entire array
                # Query only the specific image and depth range
                query_data = array[image_id, :, :]  # Get all depths for this image
                
                if len(query_data) == 0:
                    return DepthRangeResponse(
                        query_info=QueryInfo(
                            image_id=image_id,
                            depth_min=depth_min,
                            depth_max=depth_max,
                            format=return_format,
                            colormap=include_colormap
                        ),
                        frames=[],
                        total_frames=0,
                        processing_time_ms=(time.time() - start_time) * 1000
                    )
                
                # For 3D array, data structure is [depth_index, pixel_index]
                depth_values = query_data['depth_value']
                intensity_values = query_data['intensity_value']
                
                # Get depth values from first pixel column (all pixels have same depth)
                unique_depths = depth_values[:, 0]
                
                # Filter by depth range
                depth_mask = (unique_depths >= depth_min) & (unique_depths <= depth_max)
                final_mask = depth_mask
                
                if not np.any(final_mask):
                    logger.warning(f"No data found for image {image_id} in depth range {depth_min}-{depth_max}")
                    return DepthRangeResponse(
                        query_info=QueryInfo(
                            image_id=image_id,
                            depth_min=depth_min,
                            depth_max=depth_max,
                            format=return_format,
                            colormap=include_colormap
                        ),
                        frames=[],
                        total_frames=0,
                        processing_time_ms=(time.time() - start_time) * 1000
                    )
                
                filtered_depths = unique_depths[final_mask]
                filtered_intensities = intensity_values[final_mask]
                
                unique_depths = np.unique(filtered_depths)
                frames = []
                
                if include_colormap and 'red_value' in query_data:
                    red_values = query_data['red_value'][final_mask]
                    green_values = query_data['green_value'][final_mask]
                    blue_values = query_data['blue_value'][final_mask]
                    
                    for i, depth in enumerate(unique_depths):
                        frame_data = {
                            "depth": float(depth),
                            "width": 150,
                            "height": 1,
                            "format": "rgb"
                        }
                        
                        red_pixels = red_values[i, :]
                        green_pixels = green_values[i, :]
                        blue_pixels = blue_values[i, :]
                        
                        if return_format == "base64":
                            rgb_array = np.stack([red_pixels, green_pixels, blue_pixels], axis=-1)
                            rgb_array = rgb_array.reshape(1, 150, 3)
                            
                            img = Image.fromarray(rgb_array.astype(np.uint8))
                            
                            buffer = io.BytesIO()
                            img.save(buffer, format='PNG')
                            img_base64 = base64.b64encode(buffer.getvalue()).decode()
                            
                            frame_data["image_data"] = img_base64
                        else:
                            frame_data["rgb_data"] = {
                                "red": red_pixels.tolist(),
                                "green": green_pixels.tolist(),
                                "blue": blue_pixels.tolist()
                            }
                        
                        frames.append(frame_data)
                else:
                    for i, depth in enumerate(unique_depths):
                        frame_data = {
                            "depth": float(depth),
                            "width": 150,
                            "height": 1,
                            "format": "grayscale"
                        }
                        
                        intensity_pixels = filtered_intensities[i, :]
                        
                        if len(intensity_pixels) != 150:
                            logger.warning(f"Expected 150 pixels, got {len(intensity_pixels)} for depth {depth}")
                            if len(intensity_pixels) > 150:
                                intensity_pixels = intensity_pixels[:150]
                            else:
                                padded = np.zeros(150, dtype=intensity_pixels.dtype)
                                padded[:len(intensity_pixels)] = intensity_pixels
                                intensity_pixels = padded
                        
                        if return_format == "base64":
                            gray_array = intensity_pixels.reshape(1, 150)
                            img = Image.fromarray(gray_array.astype(np.uint8), mode='L')
                            
                            buffer = io.BytesIO()
                            img.save(buffer, format='PNG')
                            img_base64 = base64.b64encode(buffer.getvalue()).decode()
                            
                            frame_data["image_data"] = img_base64
                        else:
                            frame_data["grayscale_data"] = intensity_pixels.tolist()
                        
                        frames.append(frame_data)
                
                processing_time = (time.time() - start_time) * 1000
                
                return DepthRangeResponse(
                    query_info=QueryInfo(
                        image_id=image_id,
                        depth_min=depth_min,
                        depth_max=depth_max,
                        format=return_format,
                        colormap=include_colormap
                    ),
                    frames=frames,
                    total_frames=len(frames),
                    processing_time_ms=processing_time
                )
                
        except Exception as e:
            logger.error(f"Error querying depth range: {e}")
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

api = SubsurfaceAPI()
validator = QueryValidator(api)

@app.get("/")
async def root():
    return {
        "message": "Subsurface Imaging API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    try:
        if not api.array_exists():
            return {
                "status": "degraded",
                "array_accessible": False,
                "timestamp": datetime.now().isoformat()
            }
        stats = api.get_array_stats()
        return {
            "status": "healthy",
            "array_accessible": True,
            "total_surveys": stats.total_surveys,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/stats", response_model=ArrayStats)
async def get_array_statistics():
    return api.get_array_stats()

@app.get("/images", response_model=ImagesResponse)
async def get_images():
    return api.get_images()

@app.get("/frames", response_model=DepthRangeResponse)
async def get_frames_by_depth_range(
    depth_min: float = Query(..., description="Minimum depth value"),
    depth_max: float = Query(..., description="Maximum depth value"),
    image_id: int = Query(1, description="Image ID"),
    format: str = Query("json", description="Response format: 'json' or 'base64'"),
    colormap: bool = Query(True, description="Include color-mapped data")
):
    # Debug logging
    logging.info(f"Starting validation for image_id={image_id}, depth_min={depth_min}, depth_max={depth_max}")
    
    # Validate all query parameters
    try:
        image_info = validator.validate_query_parameters(
            image_id=image_id,
            depth_min=depth_min,
            depth_max=depth_max,
            format=format,
            colormap=colormap
        )
        logging.info(f"Validation passed for image_id={image_id}")
    except Exception as e:
        logging.error(f"Validation failed for image_id={image_id}: {e}")
        raise
    
    result = api.query_depth_range(
        image_id=image_id,
        depth_min=depth_min,
        depth_max=depth_max,
        return_format=format,
        include_colormap=colormap
    )
    
    return result

@app.get("/frames/image")
async def get_frames_as_image(
    depth_min: float = Query(..., description="Minimum depth value"),
    depth_max: float = Query(..., description="Maximum depth value"),
    image_id: int = Query(1, description="Image ID"),
    colormap: bool = Query(True, description="Use color mapping")
):
    # Validate all query parameters
    image_info = validator.validate_query_parameters(
        image_id=image_id,
        depth_min=depth_min,
        depth_max=depth_max,
        format="json",  # Always JSON for this endpoint
        colormap=colormap
    )
    
    try:
        result = api.query_depth_range(
            image_id=image_id,
            depth_min=depth_min,
            depth_max=depth_max,
            return_format="json",
            include_colormap=colormap
        )
        
        frames = result.frames
        if not frames:
            raise HTTPException(status_code=404, detail="No frames found in specified depth range")
        
        height = len(frames)
        width = 150
        
        if colormap and "rgb_data" in frames[0]:
            combined_image = np.zeros((height, width, 3), dtype=np.uint8)
            for i, frame in enumerate(frames):
                rgb_data = frame["rgb_data"]
                combined_image[i, :, 0] = rgb_data["red"]
                combined_image[i, :, 1] = rgb_data["green"]  
                combined_image[i, :, 2] = rgb_data["blue"]
        else:
            combined_image = np.zeros((height, width), dtype=np.uint8)
            for i, frame in enumerate(frames):
                combined_image[i, :] = frame["grayscale_data"]
        
        if len(combined_image.shape) == 3:
            img = Image.fromarray(combined_image, mode='RGB')
        else:
            img = Image.fromarray(combined_image, mode='L')
        
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(buffer.read()),
            media_type="image/png",
            headers={
                "Content-Disposition": f"inline; filename=frames_{depth_min}_{depth_max}.png"
            }
        )
        
    except Exception as e:
        logger.error(f"Error creating combined image: {e}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
