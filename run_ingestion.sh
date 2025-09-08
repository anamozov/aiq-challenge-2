#!/bin/bash
# Run only data ingestion

echo "Starting Data Ingestion..."

# Check if CSV file exists
if [ ! -f "Challenge2.csv" ]; then
    echo "Error: Challenge2.csv not found in current directory"
    exit 1
fi

echo "Processing CSV data..."
docker-compose --profile ingestion up --build data-ingestion

echo "Data ingestion completed!"
echo "Check ./db/arrays/ for TileDB arrays"
