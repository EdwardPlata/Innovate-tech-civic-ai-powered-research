# Scout Data Discovery - Quick Reference

Quick command reference for Scout Data Discovery Platform deployment and management.

## 🚀 Getting Started

```bash
# First-time setup (interactive)
./setup.sh

# Quick start
./run.sh

# Start with optimization
./run.sh --optimize
```

## 📦 Deployment Commands

### Standard Deployment
```bash
# Start services
./start_scout.sh

# Check status
./start_scout.sh --status

# Stop services
./start_scout.sh --stop

# Show help
./start_scout.sh --help
```

### Docker Deployment
```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down

# Rebuild
docker compose up -d --build
```

## 🔧 Optimization Scripts

### Dependencies
```bash
# Check and install
./scripts/optimize-deps.sh

# Upgrade pip first
./scripts/optimize-deps.sh --upgrade-pip

# Clean cache
./scripts/optimize-deps.sh --clean-cache

# Check outdated
./scripts/optimize-deps.sh --check-outdated
```

### Cache Management
```bash
# View usage
./scripts/optimize-cache.sh

# Clean old (>7 days)
./scripts/optimize-cache.sh --clean-old

# Clean all
./scripts/optimize-cache.sh --clean-all

# Clean Python cache
./scripts/optimize-cache.sh --clean-python

# Show stats
./scripts/optimize-cache.sh --stats

# Show help
./scripts/optimize-cache.sh --help
```

### Health Checks
```bash
# Basic check
./scripts/health-check.sh

# Verbose output
./scripts/health-check.sh --verbose

# Check dependencies
./scripts/health-check.sh --check-deps
```

## 🌐 Access URLs

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:8501 | Main web interface |
| Backend | http://localhost:8080 | API server |
| API Docs | http://localhost:8080/docs | Interactive API documentation |

## 🐳 Docker Commands

```bash
# Service management
docker compose ps              # List containers
docker compose logs backend    # Backend logs
docker compose logs frontend   # Frontend logs
docker compose restart         # Restart all
docker compose restart backend # Restart backend only

# Cleanup
docker compose down -v         # Stop and remove volumes
docker system prune -a         # Clean Docker cache

# Build
docker compose build           # Build images
docker compose build --no-cache # Force rebuild
```

## 📊 Monitoring

```bash
# Check logs
tail -f backend.log
tail -f frontend.log

# Process status
ps aux | grep uvicorn
ps aux | grep streamlit

# Port usage
lsof -i :8080
lsof -i :8501
```

## 🔍 Troubleshooting

```bash
# Clean and restart
./scripts/optimize-cache.sh --clean-all
./run.sh

# Check health with details
./scripts/health-check.sh --verbose --check-deps

# View recent errors
tail -100 backend.log | grep -i error
tail -100 frontend.log | grep -i error

# Kill stuck processes
pkill -f uvicorn
pkill -f streamlit
```

## 📁 Important Files

```
run.sh                    # Main deployment script
setup.sh                  # First-time setup
start_scout.sh            # Service manager
docker-compose.yml        # Docker orchestration

scripts/
├── optimize-deps.sh      # Dependency management
├── optimize-cache.sh     # Cache optimization
└── health-check.sh       # Health monitoring

DEPLOYMENT_GUIDE.md       # Comprehensive guide
DOCKER_DEPLOYMENT.md      # Docker details
scripts/README.md         # Script documentation
```

## 🎯 Common Workflows

### Development
```bash
# 1. Setup
./setup.sh

# 2. Start
./run.sh

# 3. Check health
./scripts/health-check.sh
```

### Production
```bash
# 1. Optimize
./scripts/optimize-deps.sh --clean-cache
./scripts/optimize-cache.sh --clean-all

# 2. Deploy with Docker
docker compose up -d

# 3. Monitor
docker compose logs -f
./scripts/health-check.sh
```

### Maintenance
```bash
# Weekly
./scripts/optimize-cache.sh --clean-old

# Monthly
./scripts/optimize-deps.sh --check-outdated

# As needed
./scripts/health-check.sh --verbose
```

## 💡 Tips

- Use `./setup.sh` for first-time setup
- Run `./run.sh --optimize` before production
- Schedule health checks: `*/5 * * * * /path/to/scripts/health-check.sh`
- Clean cache weekly: `./scripts/optimize-cache.sh --clean-old`
- Check updates monthly: `./scripts/optimize-deps.sh --check-outdated`
- Use Docker for production deployments
- Always check logs after deployment

## 📚 Documentation

- [README.md](README.md) - Main documentation
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Deployment options
- [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) - Docker details
- [scripts/README.md](scripts/README.md) - Script documentation

## 🆘 Quick Help

```bash
./setup.sh                          # Interactive setup
./run.sh --help                     # Run script help
./start_scout.sh --help             # Startup script help
./scripts/optimize-cache.sh --help  # Cache script help
docker compose --help               # Docker help
```

---

**Need more help?** See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.
