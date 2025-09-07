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
    survey_id: int
    width: int = 150
    height: int = 1
    timestamp: str = None

class QueryInfo(BaseModel):
    depth_min: float
    depth_max: float
    survey_id: int
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

class SurveyInfo(BaseModel):
    survey_id: int
    depth_count: int
    depth_min: float
    depth_max: float
    processed_at: int

class SurveysResponse(BaseModel):
    surveys: list
    total_surveys: int

class ArrayStats(BaseModel):
    array_path: str
    total_surveys: int
    depth_range: DepthRange
    dimensions: Dimensions
    attributes: list
    compression_info: str

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
                
                dim_names = [d.name for d in schema.domain]
                def get_range(dim_name):
                    if isinstance(non_empty_domain, dict):
                        return non_empty_domain.get(dim_name, (0, 0))
                    idx = dim_names.index(dim_name)
                    return non_empty_domain[idx]

                data = array[:]
                depth_values = data['depth_value']
                survey_ids = data['survey_id']
                
                unique_surveys = np.unique(survey_ids)
                unique_depths = np.unique(depth_values)
                
                return ArrayStats(
                    array_path=self.array_path,
                    total_surveys=len(unique_surveys),
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
                    compression_info="ZSTD level 3"
                )
        except Exception as e:
            logger.error(f"Error getting array stats: {e}")
            raise HTTPException(status_code=500, detail=f"Error accessing array: {str(e)}")
    
    def get_surveys(self):
        try:
            if not self.array_exists():
                raise RuntimeError(f"TileDB array not found at {self.array_path}.")
            
            with tiledb.open(self.array_path, mode='r') as array:
                data = array[:]
                depth_values = data['depth_value']
                survey_ids = data['survey_id']
                processed_at = data['processed_at']
                
                unique_surveys = np.unique(survey_ids[:, 0])
                surveys = []
                
                for survey_id in unique_surveys:
                    survey_mask = survey_ids[:, 0] == survey_id
                    
                    if np.any(survey_mask):
                        survey_depths = depth_values[survey_mask]
                        
                        if hasattr(processed_at, '__getitem__'):
                            survey_timestamps = processed_at[survey_mask]
                        else:
                            survey_timestamps = np.array([processed_at] * np.sum(survey_mask))
                        
                        unique_depths = np.unique(survey_depths)
                    else:
                        continue
                    
                    finite_depths = unique_depths[np.isfinite(unique_depths)]
                    
                    if len(finite_depths) == 0:
                        logger.warning(f"No valid depth values for survey {survey_id}")
                        continue
                    
                    depth_min = float(np.min(finite_depths))
                    depth_max = float(np.max(finite_depths))
                    processed_at = int(np.min(survey_timestamps))
                    
                    if not (np.isfinite(depth_min) and np.isfinite(depth_max)):
                        logger.warning(f"Invalid depth values for survey {survey_id}: min={depth_min}, max={depth_max}")
                        continue
                    
                    surveys.append(SurveyInfo(
                        survey_id=int(survey_id),
                        depth_count=len(unique_depths),
                        depth_min=depth_min,
                        depth_max=depth_max,
                        processed_at=processed_at
                    ))
                
                surveys.sort(key=lambda x: x.survey_id)
                
                return SurveysResponse(
                    surveys=surveys,
                    total_surveys=len(surveys)
                )
                
        except Exception as e:
            logger.error(f"Error getting surveys: {e}")
            raise HTTPException(status_code=500, detail=f"Error accessing surveys: {str(e)}")
    
    def query_depth_range(self, survey_id, depth_min, depth_max, return_format="json", include_colormap=True):
        start_time = time.time()
        
        try:
            if not self.array_exists():
                raise RuntimeError(f"TileDB array not found at {self.array_path}.")
            with tiledb.open(self.array_path, mode='r') as array:
                depth_min_int = int(depth_min * 10)
                depth_max_int = int(depth_max * 10)
                
                logger.info(f"Querying depth range {depth_min} to {depth_max} (int: {depth_min_int} to {depth_max_int}) for survey {survey_id}")
                
                data = array[:]
                depth_values = data['depth_value']
                intensity_values = data['intensity_value']
                survey_ids = data['survey_id']
                
                unique_depths = depth_values[:, 0]
                unique_surveys = survey_ids[:, 0]
                
                depth_mask = (unique_depths >= depth_min) & (unique_depths <= depth_max)
                survey_mask = unique_surveys == survey_id
                final_mask = depth_mask & survey_mask
                
                if not np.any(final_mask):
                    logger.warning(f"No data found for survey {survey_id} in depth range {depth_min}-{depth_max}")
                    return DepthRangeResponse(
                        query_info=QueryInfo(
                            survey_id=survey_id,
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
                filtered_surveys = unique_surveys[final_mask]
                
                unique_depths = np.unique(filtered_depths)
                frames = []
                
                if include_colormap and 'red_value' in data:
                    red_values = data['red_value'][final_mask]
                    green_values = data['green_value'][final_mask]
                    blue_values = data['blue_value'][final_mask]
                    
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
                        survey_id=survey_id,
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

@app.get("/surveys", response_model=SurveysResponse)
async def get_surveys():
    return api.get_surveys()

@app.get("/frames", response_model=DepthRangeResponse)
async def get_frames_by_depth_range(
    depth_min: float = Query(..., description="Minimum depth value", ge=9000.0),
    depth_max: float = Query(..., description="Maximum depth value", le=10000.0),
    survey_id: int = Query(1, description="Survey ID", ge=1),
    format: str = Query("json", description="Response format: 'json' or 'base64'"),
    colormap: bool = Query(True, description="Include color-mapped data")
):
    if depth_min >= depth_max:
        raise HTTPException(status_code=400, detail="depth_min must be less than depth_max")
    
    if depth_max - depth_min > 1000:
        raise HTTPException(status_code=400, detail="Depth range too large (max 1000 units)")
    
    result = api.query_depth_range(
        survey_id=survey_id,
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
    survey_id: int = Query(1, description="Survey ID"),
    colormap: bool = Query(True, description="Use color mapping")
):
    if depth_min >= depth_max:
        raise HTTPException(status_code=400, detail="depth_min must be less than depth_max")
    
    try:
        result = api.query_depth_range(
            survey_id=survey_id,
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
