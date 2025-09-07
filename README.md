# Subsurface Imaging API

A complete solution for processing and querying subsurface imaging data with efficient storage and API access.

## Features

- **Image Processing**: Resize images from 200px to 150px width
- **Efficient Storage**: TileDB arrays with ZSTD compression
- **REST API**: Query images by depth range with multiple response formats
- **Custom Colormap**: Geological visualization with blue-to-red colormap
- **Docker Support**: Containerized deployment ready
- **Multi-Survey Support**: Handle multiple survey datasets

## Quick Start

### 1. Process Data
```bash
python data_ingestion.py
```

### 2. Start API
```bash
python -m uvicorn api:app --reload
```

### 3. Test API
```bash
python test_api.py
```

## Docker Deployment

```bash
# Complete pipeline
docker-compose up --build

# Individual services
./run_ingestion.sh  # Data processing
./run_api.sh        # API service
```

## API Endpoints

- `GET /health` - Health check
- `GET /stats` - Array statistics
- `GET /surveys` - List available surveys
- `GET /frames` - Query frames by depth range
- `GET /frames/image` - Get frames as PNG image

## Example Usage

Query frames from depth 9100 to 9200:
```bash
curl "http://localhost:8000/frames?depth_min=9100&depth_max=9200&survey_id=1"
```

## Requirements

- Python 3.8+
- TileDB
- FastAPI
- OpenCV
- NumPy
- Pandas

## Project Structure

```
├── api.py                 # Main FastAPI application
├── data_ingestion.py      # Data processing pipeline
├── test_api.py           # Comprehensive test suite
├── requirements.txt      # Python dependencies
├── Dockerfile.*          # Container definitions
├── docker-compose.yml    # Service orchestration
├── run_*.sh             # Execution scripts
└── README.md            # Project documentation
```