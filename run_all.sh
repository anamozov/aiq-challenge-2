#!/bin/bash

echo "Starting Complete Pipeline..."

echo ""
echo "STEP 1: Data Ingestion"
echo "----------------------"
if [ ! -f "Challenge2.csv" ]; then
    echo "Error: Challenge2.csv not found in current directory"
    exit 1
fi

echo "Running data ingestion..."
docker-compose --profile ingestion up --build data-ingestion

if [ $? -ne 0 ]; then
    echo "Data ingestion failed!"
    exit 1
fi

echo "Data ingestion completed!"

echo ""
echo "STEP 2: API Service"
echo "------------------"
echo "Starting API server..."
echo "Press Ctrl+C to stop the API service"

docker-compose up --build subsurface-api
