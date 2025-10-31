# Docker Deployment Guide

This guide covers deploying Scout Data Discovery Platform using Docker and Docker Compose.

## Prerequisites

- Docker 20.10 or higher
- Docker Compose 2.0 or higher
- 2GB+ free disk space
- 4GB+ RAM recommended

## Quick Start with Docker Compose

### 1. Build and Start Services

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### 2. Access the Application

- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs

### 3. Stop Services

```bash
# Stop services
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove containers + volumes
docker-compose down -v
```

## Building Individual Services

### Backend Only

```bash
docker build -f Dockerfile.backend -t scout-backend .
docker run -d -p 8080:8080 --name scout-backend scout-backend
```

### Frontend Only

```bash
docker build -f Dockerfile.frontend -t scout-frontend .
docker run -d -p 8501:8501 --name scout-frontend scout-frontend
```

## Configuration

### Environment Variables

Backend:
- `LOG_LEVEL`: Logging level (default: INFO)
- `CACHE_TTL`: Cache time-to-live in seconds
- `MAX_WORKERS`: Number of worker threads

Frontend:
- `BACKEND_API_URL`: Backend API endpoint (default: http://backend:8080)
- `STREAMLIT_SERVER_PORT`: Frontend port (default: 8501)

### Custom Configuration

Create a `.env` file:

```env
# Backend
LOG_LEVEL=DEBUG
CACHE_TTL=600

# Frontend
STREAMLIT_THEME_PRIMARY_COLOR=#1f77b4
```

Then use it:

```bash
docker-compose --env-file .env up -d
```

## Data Persistence

### Cache Persistence

The docker-compose configuration mounts the cache directory:

```yaml
volumes:
  - ./backend/cache:/app/backend/cache
```

This ensures cached data persists between container restarts.

### Log Persistence

Logs are stored in the `./logs` directory:

```yaml
volumes:
  - ./logs:/app/logs
```

## Health Checks

Both services include health checks:

### Backend Health Check
```bash
curl http://localhost:8080/api/health
```

### Frontend Health Check
```bash
curl http://localhost:8501/_stcore/health
```

### Docker Health Status
```bash
docker-compose ps
```

## Troubleshooting

### Check Logs

```bash
# All services
docker-compose logs

# Specific service
docker-compose logs backend
docker-compose logs frontend

# Follow logs
docker-compose logs -f
```

### Restart Services

```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart backend
```

### Rebuild After Changes

```bash
# Rebuild and restart
docker-compose up -d --build

# Force rebuild
docker-compose build --no-cache
docker-compose up -d
```

### Common Issues

#### Port Already in Use

```bash
# Check what's using the port
lsof -i :8080
lsof -i :8501

# Stop conflicting services or change ports in docker-compose.yml
```

#### Out of Memory

```bash
# Check container resource usage
docker stats

# Limit memory in docker-compose.yml
services:
  backend:
    mem_limit: 1g
```

#### Permission Issues

```bash
# Fix cache directory permissions
chmod -R 755 backend/cache

# Or run with user context
docker-compose run --user $(id -u):$(id -g) backend
```

## Production Deployment

### Security Considerations

1. **Use secrets for sensitive data**:
```yaml
secrets:
  api_key:
    file: ./secrets/api_key.txt
```

2. **Enable HTTPS** with a reverse proxy (nginx/traefik)

3. **Limit resource usage**:
```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
```

### Scaling

Scale services horizontally:

```bash
# Scale backend to 3 instances
docker-compose up -d --scale backend=3

# Use load balancer in front
```

### Monitoring

Add monitoring with Prometheus and Grafana:

```yaml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
```

## Advanced Configuration

### Using External Database

Modify docker-compose.yml to add database service:

```yaml
services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: scout
      POSTGRES_USER: scout
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  postgres-data:
```

### Custom Networks

```yaml
networks:
  scout-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16
```

## Maintenance

### Cleanup

```bash
# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Full cleanup
docker system prune -a --volumes
```

### Backup

```bash
# Backup cache data
tar -czf backup-cache-$(date +%Y%m%d).tar.gz backend/cache/

# Backup volumes
docker run --rm -v scout_cache-data:/data -v $(pwd):/backup \
  alpine tar -czf /backup/cache-backup.tar.gz /data
```

### Updates

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose down
docker-compose up -d --build
```

## Support

For issues with Docker deployment:
1. Check logs: `docker-compose logs`
2. Verify health: `docker-compose ps`
3. Review configuration: `docker-compose config`
4. Check Docker resources: `docker stats`

For application-specific issues, see the main README.md
