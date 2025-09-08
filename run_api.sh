#!/bin/bash

echo "Starting API Service..."

if [ ! -d "db/arrays/image_data" ]; then
    echo "Error: No data found. Run data ingestion first:"
    echo "   ./run_ingestion.sh"
    exit 1
fi

echo "Starting FastAPI server..."
docker-compose up --build image-api

echo "API service started!"
echo "API: http://localhost:8000"
echo "Docs: http://localhost:8000/docs"
