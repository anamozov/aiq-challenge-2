#!/bin/bash

echo "Starting API Service..."

if [ ! -d "data/arrays/subsurface_data" ]; then
    echo "Error: No data found. Run data ingestion first:"
    echo "   ./run_ingestion.sh"
    exit 1
fi

echo "Starting FastAPI server..."
docker-compose up --build subsurface-api

echo "API service started!"
echo "API: http://localhost:8000"
echo "Docs: http://localhost:8000/docs"
