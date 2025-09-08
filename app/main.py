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
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from app.pipeline_manager import DataPipelineManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Image Processing API",
    description="API for querying image data by depth ranges with real-time colormap application",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

ARRAY_PATH = "db/arrays/image_data"
CACHE_DIR = "db/cache"
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

def create_custom_colormap():
    colors = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000']
    n_bins = 256
    return LinearSegmentedColormap.from_list('geological', colors, N=n_bins)

CUSTOM_COLORMAP = create_custom_colormap()

def apply_colormap_to_intensities(intensity_values):
    if intensity_values.max() > intensity_values.min():
        normalized = (intensity_values - intensity_values.min()) / (intensity_values.max() - intensity_values.min())
    else:
        normalized = np.zeros_like(intensity_values, dtype=np.float32)
    
    colored = CUSTOM_COLORMAP(normalized)
    
    red = (colored[:, :, 0] * 255).astype(np.uint8)
    green = (colored[:, :, 1] * 255).astype(np.uint8) 
    blue = (colored[:, :, 2] * 255).astype(np.uint8)
    
    return red, green, blue

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
    rgb_data: Optional[Dict] = None
    grayscale_data: Optional[List] = None
    image_data: Optional[str] = None

class DepthRangeResponse(BaseModel):
    query_info: QueryInfo
    frames: List[FrameData]
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
    images: List[ImageInfo]
    total_images: int

class ArrayStats(BaseModel):
    array_path: str
    total_surveys: int
    depth_range: DepthRange
    dimensions: Dimensions
    attributes: List[str]

class ImageAPI:
    def __init__(self, array_path=ARRAY_PATH):
        self.array_path = array_path
    
    def array_exists(self):
        return tiledb.object_type(self.array_path) == "array"
    
    def get_available_images(self):
        if not self.array_exists():
            return []
        
        try:
            with tiledb.open(self.array_path, mode='r') as array:
                non_empty_domain = array.nonempty_domain()
                
                if non_empty_domain[0][0] is None or non_empty_domain[0][1] is None:
                    return []
                
                min_image_id = non_empty_domain[0][0]
                max_image_id = non_empty_domain[0][1]
                
                images = []
                for image_id in range(min_image_id, max_image_id + 1):
                    try:
                        query_data = array[image_id, :, :]
                        
                        if len(query_data) == 0:
                            continue
                        
                        depth_values = query_data['depth_value']
                        processed_at = query_data['processed_at']
                        
                        unique_depths = depth_values[:, 0]
                        valid_depths = unique_depths[~np.isnan(unique_depths)]
                        
                        if len(valid_depths) > 0:
                            images.append({
                                "image_id": int(image_id),
                                "depth_count": len(valid_depths),
                                "depth_min": float(valid_depths.min()),
                                "depth_max": float(valid_depths.max()),
                                "processed_at": int(processed_at[0, 0])
                            })
                    
                    except Exception as e:
                        logger.warning(f"Could not process image {image_id}: {e}")
                        continue
                
                return images
        
        except Exception as e:
            logger.error(f"Error getting available images: {e}")
            return []
    
    def get_array_stats(self):
        if not self.array_exists():
            raise RuntimeError(f"TileDB array not found at {self.array_path}.")
        
        try:
            with tiledb.open(self.array_path, mode='r') as array:
                non_empty_domain = array.nonempty_domain()
                schema = array.schema
                
                sample_data = array[non_empty_domain[0][0], :10, :10]
                depth_values = sample_data['depth_value']
                
                valid_depths = depth_values[~np.isnan(depth_values)]
                
                total_images = non_empty_domain[0][1] - non_empty_domain[0][0] + 1 if non_empty_domain[0][1] is not None else 0
                
                return ArrayStats(
                    array_path=self.array_path,
                    total_surveys=total_images,
                    depth_range=DepthRange(
                        min=float(valid_depths.min()) if len(valid_depths) > 0 else 0.0,
                        max=float(valid_depths.max()) if len(valid_depths) > 0 else 0.0,
                        span=float(valid_depths.max() - valid_depths.min()) if len(valid_depths) > 0 else 0.0
                    ),
                    dimensions=Dimensions(
                        depth_index=f"0 to {non_empty_domain[1][1]}" if non_empty_domain[1][1] is not None else "empty",
                        pixel_index=f"0 to {non_empty_domain[2][1]}" if non_empty_domain[2][1] is not None else "empty"
                    ),
                    attributes=[attr.name for attr in schema]
                )
        
        except Exception as e:
            logger.error(f"Error getting array stats: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get array stats: {str(e)}")
    
    def query_depth_range(self, image_id, depth_min, depth_max, return_format="json", include_colormap=True):
        start_time = time.time()
        
        try:
            if not self.array_exists():
                raise RuntimeError(f"TileDB array not found at {self.array_path}.")
            
            with tiledb.open(self.array_path, mode='r') as array:
                logger.info(f"Querying depth range {depth_min} to {depth_max} for image {image_id}")
                
                non_empty_domain = array.nonempty_domain()
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
                
                query_data = array[image_id, :, :]
                
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
                
                depth_values = query_data['depth_value']
                intensity_values = query_data['intensity_value']
                
                unique_depths = depth_values[:, 0]
                
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
                
                if include_colormap:
                    for i, depth in enumerate(unique_depths):
                        frame_data = {
                            "depth": float(depth),
                            "width": 150,
                            "height": 1,
                            "format": "rgb"
                        }
                        
                        intensity_row = filtered_intensities[i, :].reshape(1, 150)
                        
                        red_pixels, green_pixels, blue_pixels = apply_colormap_to_intensities(intensity_row)
                        red_pixels = red_pixels.flatten()
                        green_pixels = green_pixels.flatten()
                        blue_pixels = blue_pixels.flatten()
                        
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
                    processing_time_ms=(time.time() - start_time) * 1000
                )
                
        except Exception as e:
            logger.error(f"Error querying depth range: {e}")
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

api = ImageAPI()
pipeline_manager = DataPipelineManager()

class QueryValidator:
    def __init__(self, api_instance):
        self.api = api_instance
        self.cache_timeout = 60
        self._image_cache = None
        self._cache_timestamp = 0
    
    def _get_cached_images(self):
        current_time = time.time()
        if (self._image_cache is None or 
            current_time - self._cache_timestamp > self.cache_timeout):
            self._image_cache = self.api.get_available_images()
            self._cache_timestamp = current_time
        return self._image_cache
    
    def validate_query_parameters(self, image_id, depth_min, depth_max, format, colormap):
        available_images = self._get_cached_images()
        
        if not available_images:
            raise HTTPException(
                status_code=404, 
                detail="No images available in the system"
            )
        
        image_ids = [img["image_id"] for img in available_images]
        if image_id not in image_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Image ID {image_id} not found. Available images: {image_ids}"
            )
        
        target_image = next(img for img in available_images if img["image_id"] == image_id)
        
        if depth_min > depth_max:
            raise HTTPException(
                status_code=400,
                detail=f"depth_min ({depth_min}) cannot be greater than depth_max ({depth_max})"
            )
        
        image_depth_min = target_image["depth_min"]
        image_depth_max = target_image["depth_max"]
        
        if depth_max < image_depth_min or depth_min > image_depth_max:
            raise HTTPException(
                status_code=400,
                detail=f"Requested depth range ({depth_min}-{depth_max}) does not overlap with image {image_id} range ({image_depth_min:.1f}-{image_depth_max:.1f})"
            )
        
        if format not in ["json", "base64"]:
            raise HTTPException(
                status_code=400,
                detail="format parameter must be either 'json' or 'base64'"
            )
        
        if not isinstance(colormap, bool):
            raise HTTPException(
                status_code=400,
                detail="colormap parameter must be a boolean value"
            )
        
        return target_image

validator = QueryValidator(api)

@app.get("/")
async def root():
    return {
        "message": "Image Processing API with Automated Pipeline",
        "version": "2.0.0",
        "endpoints": {
            "images": "/images",
            "frames": "/frames", 
            "frames_image": "/frames/image",
            "stats": "/stats",
            "health": "/health",
            "pipeline_status": "/pipeline/status",
            "pipeline_start": "/pipeline/start",
            "pipeline_stop": "/pipeline/stop",
            "pipeline_scan": "/pipeline/scan",
            "pipeline_history": "/pipeline/history"
        },
        "pipeline": {
            "watch_directory": "data",
            "processed_directory": "data/processed", 
            "failed_directory": "data/failed",
            "database_directory": "db"
        }
    }

@app.get("/health")
async def health_check():
    try:
        array_exists = api.array_exists()
        if array_exists:
            images = api.get_available_images()
            return {
                "status": "healthy",
                "array_exists": True,
                "total_images": len(images),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "degraded",
                "array_exists": False,
                "message": "TileDB array not found",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/images", response_model=ImagesResponse)
async def get_images():
    try:
        images = api.get_available_images()
        return ImagesResponse(
            images=[ImageInfo(**img) for img in images],
            total_images=len(images)
        )
    except Exception as e:
        logger.error(f"Error getting images: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get images: {str(e)}")

@app.get("/stats", response_model=ArrayStats)
async def get_stats():
    try:
        return api.get_array_stats()
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.get("/frames", response_model=DepthRangeResponse)
async def get_frames(
    depth_min: float = Query(..., description="Minimum depth value"),
    depth_max: float = Query(..., description="Maximum depth value"),
    image_id: int = Query(1, description="Image ID"),
    format: str = Query("json", description="Response format: 'json' or 'base64'"),
    colormap: bool = Query(True, description="Include color-mapped data")
):
    image_info = validator.validate_query_parameters(
        image_id=image_id,
        depth_min=depth_min,
        depth_max=depth_max,
        format=format,
        colormap=colormap
    )
    
    try:
        return api.query_depth_range(
            image_id=image_id,
            depth_min=depth_min,
            depth_max=depth_max,
            return_format=format,
            include_colormap=colormap
        )
    except Exception as e:
        logger.error(f"Error querying frames: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/frames/image")
async def get_frames_as_image(
    depth_min: float = Query(..., description="Minimum depth value"),
    depth_max: float = Query(..., description="Maximum depth value"),
    image_id: int = Query(1, description="Image ID"),
    colormap: bool = Query(True, description="Use color mapping")
):
    image_info = validator.validate_query_parameters(
        image_id=image_id,
        depth_min=depth_min,
        depth_max=depth_max,
        format="json",
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
        
        if colormap and frames[0].rgb_data:
            combined_image = np.zeros((height, width, 3), dtype=np.uint8)
            for i, frame in enumerate(frames):
                rgb_data = frame.rgb_data
                combined_image[i, :, 0] = rgb_data["red"]
                combined_image[i, :, 1] = rgb_data["green"]  
                combined_image[i, :, 2] = rgb_data["blue"]
        else:
            combined_image = np.zeros((height, width), dtype=np.uint8)
            for i, frame in enumerate(frames):
                combined_image[i, :] = frame.grayscale_data
        
        if len(combined_image.shape) == 3:
            img = Image.fromarray(combined_image, mode='RGB')
        else:
            img = Image.fromarray(combined_image, mode='L')
        
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="image/png",
            headers={
                "Content-Disposition": f"inline; filename=depth_{depth_min}_{depth_max}_image_{image_id}.png"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@app.get("/pipeline/status")
async def get_pipeline_status():
    try:
        status = pipeline_manager.get_status()
        return {
            "status": "active" if pipeline_manager.running else "stopped",
            "last_run": status.get("last_run"),
            "total_processed": status.get("total_processed", 0),
            "failed_files_count": len(status.get("failed_files", [])),
            "recent_history": status.get("processing_history", [])[-10:],
            "watch_directory": str(pipeline_manager.watch_directory),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline status: {str(e)}")

@app.post("/pipeline/start")
async def start_pipeline():
    try:
        if pipeline_manager.running:
            return {"message": "Pipeline is already running", "status": "active"}
        
        pipeline_manager.start()
        return {
            "message": "Pipeline started successfully",
            "status": "active",
            "watch_directory": str(pipeline_manager.watch_directory),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start pipeline: {str(e)}")

@app.post("/pipeline/stop")
async def stop_pipeline():
    try:
        if not pipeline_manager.running:
            return {"message": "Pipeline is already stopped", "status": "stopped"}
        
        pipeline_manager.stop()
        return {
            "message": "Pipeline stopped successfully",
            "status": "stopped",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error stopping pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop pipeline: {str(e)}")

@app.post("/pipeline/scan")
async def trigger_scan():
    try:
        pipeline_manager.scan_existing_files()
        return {
            "message": "File scan triggered successfully",
            "watch_directory": str(pipeline_manager.watch_directory),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error triggering scan: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger scan: {str(e)}")

@app.get("/pipeline/history")
async def get_processing_history():
    try:
        status = pipeline_manager.get_status()
        return {
            "processing_history": status.get("processing_history", []),
            "total_processed": status.get("total_processed", 0),
            "failed_files": status.get("failed_files", []),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting processing history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get processing history: {str(e)}")