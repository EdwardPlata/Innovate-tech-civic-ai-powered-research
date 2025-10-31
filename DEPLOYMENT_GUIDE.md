# Deployment Guide - Scout Data Discovery Platform

This guide provides comprehensive instructions for deploying Scout Data Discovery Platform using various methods.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Deployment Options](#deployment-options)
- [Option 1: Quick Start with run.sh](#option-1-quick-start-with-runsh)
- [Option 2: Standard Deployment](#option-2-standard-deployment)
- [Option 3: Docker Deployment](#option-3-docker-deployment)
- [Optimization Scripts](#optimization-scripts)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Prerequisites

### For Local Deployment (Options 1 & 2)
- **Python 3.8+** (Python 3.10+ recommended)
- **pip** package manager
- **curl** (for health checks)
- **4GB+ RAM** recommended
- **2GB+ free disk space**

### For Docker Deployment (Option 3)
- **Docker 20.10+**
- **Docker Compose 2.0+**
- **4GB+ RAM** recommended
- **4GB+ free disk space** (for images)

---

## Deployment Options

Scout Data Discovery Platform offers three flexible deployment methods:

| Method | Best For | Setup Time | Customization |
|--------|----------|------------|---------------|
| **run.sh** | Quick testing, development | 2-3 min | Medium |
| **Standard** | Production, manual control | 3-5 min | High |
| **Docker** | Production, containerized apps | 5-10 min | Medium |

---

## Option 1: Quick Start with run.sh

**Recommended for**: Quick testing and development

### Step 1: Clone and Navigate
```bash
git clone https://github.com/EdwardPlata/Innovate-tech-civic-ai-powered-research.git
cd Innovate-tech-civic-ai-powered-research
```

### Step 2: Run the Application
```bash
# Basic startup
./run.sh

# With optimization (recommended for first run)
./run.sh --optimize
```

### Step 3: Access the Application
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs

### What run.sh Does:
1. ✅ Checks Python installation
2. ✅ Verifies and installs dependencies
3. ✅ Optionally runs optimization scripts
4. ✅ Starts backend and frontend services
5. ✅ Monitors service health

### Stopping Services
```bash
./start_scout.sh --stop
```

---

## Option 2: Standard Deployment

**Recommended for**: Production environments with manual control

### Step 1: Install Dependencies

```bash
# Backend dependencies
cd backend
pip install -r requirements.txt

# Frontend dependencies
cd ../frontend
pip install -r requirements.txt

# Scout core dependencies (if needed)
cd ../scout_data_discovery
pip install -r requirements.txt
```

### Step 2: Start Services

```bash
# Return to project root
cd ..

# Start both services
./start_scout.sh
```

### Step 3: Service Management

```bash
# Check status
./start_scout.sh --status

# Stop services
./start_scout.sh --stop

# View help
./start_scout.sh --help
```

### Manual Service Start (Alternative)

**Backend:**
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8080
```

**Frontend:**
```bash
cd frontend
streamlit run app.py --server.port 8501
```

---

## Option 3: Docker Deployment

**Recommended for**: Production, containerized environments, easy scaling

### Step 1: Prerequisites Check
```bash
# Verify Docker installation
docker --version
docker compose version
```

### Step 2: Build and Start
```bash
# Build and start all services
docker compose up -d

# View logs
docker compose logs -f

# Check status
docker compose ps
```

### Step 3: Access the Application
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs

### Docker Management

**View Logs:**
```bash
# All services
docker compose logs

# Specific service
docker compose logs backend
docker compose logs frontend

# Follow logs in real-time
docker compose logs -f
```

**Restart Services:**
```bash
# Restart all
docker compose restart

# Restart specific service
docker compose restart backend
```

**Stop Services:**
```bash
# Stop (can be restarted)
docker compose stop

# Stop and remove containers
docker compose down

# Stop and remove containers + volumes
docker compose down -v
```

**Rebuild After Changes:**
```bash
docker compose up -d --build
```

### Docker Deployment Advantages:
- ✅ Isolated environment
- ✅ Easy to scale
- ✅ Consistent across platforms
- ✅ Simple updates and rollbacks
- ✅ Built-in health checks

See [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) for advanced Docker configuration.

---

## Optimization Scripts

Located in `scripts/` directory, these tools help maintain and optimize your deployment.

### Dependency Optimizer

```bash
# Check and install missing dependencies
./scripts/optimize-deps.sh

# With pip upgrade
./scripts/optimize-deps.sh --upgrade-pip

# Clean pip cache to save space
./scripts/optimize-deps.sh --clean-cache

# Check for outdated packages
./scripts/optimize-deps.sh --check-outdated
```

### Cache Optimizer

```bash
# View cache usage
./scripts/optimize-cache.sh

# Clean old files (>7 days)
./scripts/optimize-cache.sh --clean-old

# Clean all cache
./scripts/optimize-cache.sh --clean-all

# Clean Python cache
./scripts/optimize-cache.sh --clean-python

# Show statistics
./scripts/optimize-cache.sh --stats
```

### Health Check

```bash
# Basic health check
./scripts/health-check.sh

# Verbose output with errors
./scripts/health-check.sh --verbose

# Check dependencies too
./scripts/health-check.sh --check-deps
```

See [scripts/README.md](scripts/README.md) for detailed script documentation.

---

## Troubleshooting

### Common Issues

#### 1. Port Already in Use

**Problem:** Port 8080 or 8501 is already in use

**Solution:**
```bash
# Check what's using the ports
lsof -i :8080
lsof -i :8501

# Kill the process or change ports in configuration
```

#### 2. Dependencies Not Installing

**Problem:** pip install fails

**Solution:**
```bash
# Upgrade pip
pip install --upgrade pip

# Try with no cache
pip install --no-cache-dir -r requirements.txt

# Use optimization script
./scripts/optimize-deps.sh --upgrade-pip
```

#### 3. Services Won't Start

**Problem:** Backend or frontend fails to start

**Solution:**
```bash
# Check health
./scripts/health-check.sh --verbose

# View logs
tail -f backend.log
tail -f frontend.log

# Clean cache and restart
./scripts/optimize-cache.sh --clean-all
./run.sh
```

#### 4. Docker Build Fails

**Problem:** Docker build errors

**Solution:**
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker compose build --no-cache

# Check disk space
df -h
```

#### 5. Out of Memory

**Problem:** System runs out of memory

**Solution:**
```bash
# For Docker, limit resources in docker-compose.yml:
# mem_limit: 2g

# For local, monitor usage
./scripts/health-check.sh
```

### Getting Help

1. **Check logs**: Always start by checking log files
2. **Run health check**: Use `./scripts/health-check.sh --verbose`
3. **Verify dependencies**: Run `./scripts/optimize-deps.sh --check-outdated`
4. **Check documentation**: See README.md and script docs
5. **Clean and retry**: Clean cache and rebuild

---

## Best Practices

### Development

1. **Use run.sh for quick iterations**
   ```bash
   ./run.sh --optimize
   ```

2. **Monitor health regularly**
   ```bash
   ./scripts/health-check.sh
   ```

3. **Clean cache during development**
   ```bash
   ./scripts/optimize-cache.sh --clean-old
   ```

### Production

1. **Use Docker for consistency**
   ```bash
   docker compose up -d
   ```

2. **Set up monitoring**
   ```bash
   # Schedule health checks
   */5 * * * * /path/to/scripts/health-check.sh
   ```

3. **Regular maintenance**
   ```bash
   # Weekly cache cleanup
   ./scripts/optimize-cache.sh --clean-old
   
   # Monthly dependency updates
   ./scripts/optimize-deps.sh --check-outdated
   ```

4. **Backup important data**
   ```bash
   # Backup cache
   tar -czf cache-backup-$(date +%Y%m%d).tar.gz backend/cache/
   ```

5. **Use log rotation**
   ```bash
   # Rotate logs to prevent disk fill
   logrotate -f /etc/logrotate.conf
   ```

### Security

1. **Keep dependencies updated**
2. **Use environment variables for secrets**
3. **Enable HTTPS in production**
4. **Limit resource usage**
5. **Regular security audits**

---

## Deployment Checklist

### Before Deployment

- [ ] Python 3.8+ installed
- [ ] All dependencies available
- [ ] Ports 8080 and 8501 free
- [ ] Adequate disk space (2GB+)
- [ ] Adequate RAM (4GB+)

### Initial Setup

- [ ] Clone repository
- [ ] Choose deployment method
- [ ] Install dependencies
- [ ] Configure environment (if needed)
- [ ] Test services locally

### Post-Deployment

- [ ] Verify all services are running
- [ ] Run health check
- [ ] Test frontend access
- [ ] Test backend API
- [ ] Check logs for errors
- [ ] Set up monitoring
- [ ] Configure backups

### Ongoing Maintenance

- [ ] Monitor service health
- [ ] Clean old cache files
- [ ] Check for outdated packages
- [ ] Review error logs
- [ ] Update dependencies
- [ ] Test after updates

---

## Performance Optimization

### For Local Deployment

1. **Optimize dependencies**
   ```bash
   ./scripts/optimize-deps.sh --clean-cache
   ```

2. **Clean cache regularly**
   ```bash
   ./scripts/optimize-cache.sh --clean-old
   ```

3. **Monitor resource usage**
   ```bash
   ./scripts/health-check.sh
   ```

### For Docker Deployment

1. **Limit container resources**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 2G
   ```

2. **Use multi-stage builds** (already implemented)

3. **Optimize image size**
   - Use slim base images
   - Clean cache in Dockerfile
   - Use .dockerignore

---

## Next Steps

After successful deployment:

1. **Explore the platform**: Visit http://localhost:8501
2. **Read the documentation**: Check docs/ folder
3. **Try the API**: Visit http://localhost:8080/docs
4. **Set up monitoring**: Schedule health checks
5. **Customize configuration**: Adjust settings as needed

---

## Support

For deployment issues:
1. Check this guide first
2. Review [scripts/README.md](scripts/README.md)
3. See [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) for Docker
4. Check application logs
5. Run health checks with verbose output

For application-specific help, see the main [README.md](README.md)
