# Subsurface Imaging API

Simple API for processing and querying subsurface imaging data.

## Setup

1. Process the CSV data:
```bash
python data_ingestion.py
```

2. Start the API:
```bash
python -m uvicorn api:app --reload
```

3. Test the API:
```bash
python test_api.py
```

## API Endpoints

- `/health` - Check if API is running
- `/stats` - Get data statistics
- `/surveys` - List available surveys
- `/frames` - Query frames by depth range
- `/frames/image` - Get frames as PNG image

## Example Usage

Query frames from depth 9100 to 9200:
```bash
curl "http://localhost:8000/frames?depth_min=9100&depth_max=9200&survey_id=1"
```

## Docker

Run with Docker:
```bash
docker-compose up --build
```
