# AIQ Challenge 2 - Image Processing API

Solution that processes CSV image data, resizes images from 200px to 150px width, stores them in database, and provides REST API for querying image frames by depth ranges with custom colormap.

## Core Requirements

- **Image Resizing**: Resize images from 200px to 150px width
- **Database Storage**: Store resized images in database
- **Depth-based API**: Request image frames based on depth_min and depth_max
- **Custom Colormap**: Apply blue-to-red colormap to generated frames
- **Python Solution**: Built with FastAPI and Python
- **Containerized**: Full Docker deployment with docker-compose

## Technical Implementation

- **Image Processing**: SciPy resizing with variance preservation
- **Database**: TileDB 3D arrays with ZSTD compression
- **API Framework**: FastAPI with automatic documentation
- **File Pipeline**: Automated CSV processing with error handling
- **Storage**: Organized data management with processed/failed directories

## Quick Start

```bash
# Start the application
docker-compose up -d

# Start automated file monitoring
curl -X POST http://localhost:8000/pipeline/start

# Add CSV files to data/ directory
cp your_survey.csv data/

# Check processing status
curl http://localhost:8000/pipeline/status
```

## API Endpoints

- `GET /images` - List available images with depth ranges
- `GET /frames` - Query frames by depth range (JSON/base64 response)
- `GET /frames/image` - Generate PNG image frames with colormap
- `GET /stats` - Array statistics
- `GET /health` - System health check
- `GET /pipeline/status` - Pipeline status
- `POST /pipeline/start` - Start automated file monitoring
- `POST /pipeline/stop` - Stop pipeline operations

## Sample Output

The solution generates colorized PNG image frames from depth ranges:

```bash
# Generate colorized frame with custom colormap
curl -H "Accept: image/png" "http://localhost:8000/frames/image?image_id=1&depth_min=9201.9&depth_max=9210.3" -o frame.png

# Generate grayscale frame
curl -H "Accept: image/png" "http://localhost:8000/frames/image?image_id=1&depth_min=9201.9&depth_max=9210.3&colormap=false" -o frame_gray.png

# List available images
curl "http://localhost:8000/images"
```

![Sample Frame](test_colormap.png)

## Project Structure

```
aiq-challenge-2/
├── app/                   # Main application
│   ├── main.py           # FastAPI application
│   ├── data_ingestion.py # Data processing pipeline
│   └── pipeline_manager.py # Automated pipeline
├── tests/                # Test suite
├── data/                 # Data directories
│   ├── processed/        # Processed CSV files
│   └── failed/          # Failed files with logs
├── db/                   # Database storage
│   ├── arrays/          # TileDB 3D arrays
│   └── cache/           # API response cache
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container definition
└── docker-compose.yml    # Container orchestration
```

## Access Points

- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health


## Requirements

- Python 3.8+
- Docker & Docker Compose
- 2GB RAM minimum
- 1GB disk space


## Disclaimer

GenAI assistance has been used for docstrings, comments, frontend components, documentation, and testing implementation.

