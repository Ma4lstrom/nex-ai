# Docker Setup for Food Vision API

## Prerequisites
- Docker installed on your system
- Docker Compose installed (usually comes with Docker Desktop)

## Quick Start

### 1. Set up environment variables
Copy the example environment file and add your API key:
```bash
cp .env.example .env
```

Edit `.env` and add your Anthropic API key:
```
ANTHROPIC_API_KEY=your_actual_api_key_here
```

### 2. Build and run with Docker Compose
```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`

### 3. Access the API documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Docker Commands

### Build the image
```bash
docker-compose build
```

### Start the container
```bash
docker-compose up
```

### Start in detached mode (background)
```bash
docker-compose up -d
```

### Stop the container
```bash
docker-compose down
```

### View logs
```bash
docker-compose logs -f
```

### Rebuild and restart
```bash
docker-compose up --build
```

## Using Docker without Docker Compose

### Build the image
```bash
docker build -t food-vision-api .
```

### Run the container
```bash
docker run -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/storage:/app/storage \
  -v $(pwd)/models:/app/models \
  food-vision-api
```

## Data Persistence

The following directories are mounted as volumes to persist data:
- `./storage` - Reference and temporary images
- `./models` - ML models

These directories will be created automatically if they don't exist.

## Troubleshooting

### Port already in use
If port 8000 is already in use, modify the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Change 8001 to any available port
```

### Permission issues
If you encounter permission issues with volumes, ensure the directories exist and have proper permissions:
```bash
mkdir -p storage/references storage/temp models
chmod -R 755 storage models
```

### Check container health
```bash
docker-compose ps
```

### Access container shell
```bash
docker-compose exec food-vision-api /bin/bash
```
