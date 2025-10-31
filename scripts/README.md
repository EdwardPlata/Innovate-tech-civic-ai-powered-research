# Scout Data Discovery - Deployment Scripts

This directory contains optimization and utility scripts for deploying and managing the Scout Data Discovery Platform.

## Available Scripts

### 🚀 Main Deployment Script

#### `../run.sh`
The main entry point for running the Scout platform. This script sets up and starts both backend and frontend services.

**Usage:**
```bash
# Basic startup
./run.sh

# With optimization
./run.sh --optimize

# Pass through to start_scout.sh
./run.sh --status
./run.sh --stop
```

**Features:**
- Checks Python installation
- Verifies dependencies
- Optionally runs optimization scripts
- Delegates to `start_scout.sh` for service management

---

## Optimization Scripts

### 📦 Dependency Optimizer (`optimize-deps.sh`)

Manages and optimizes Python dependencies across the platform.

**Usage:**
```bash
# Check and install missing dependencies
./scripts/optimize-deps.sh

# Upgrade pip before installing
./scripts/optimize-deps.sh --upgrade-pip

# Clean pip cache after installing
./scripts/optimize-deps.sh --clean-cache

# Check for outdated packages
./scripts/optimize-deps.sh --check-outdated
```

**Features:**
- Checks Scout core, backend, and frontend dependencies
- Installs only missing packages
- Reports package counts and status
- Can clean pip cache to save space
- Shows outdated packages for maintenance

**Example Output:**
```
[DEP-OPT] === Dependency Optimizer ===
[DEP-OPT] Checking pip version...
[DEP-OPT] Current pip version: 23.2.1
[DEP-OPT] Checking Backend dependencies...
[DEP-OPT] Total packages in Backend: 15
[SUCCESS] All Backend dependencies are installed
```

---

### 🗑️ Cache Optimizer (`optimize-cache.sh`)

Manages and optimizes cache directories to improve performance and save disk space.

**Usage:**
```bash
# Analyze cache usage
./scripts/optimize-cache.sh

# Clean old cache files (>7 days)
./scripts/optimize-cache.sh --clean-old

# Clean all cache
./scripts/optimize-cache.sh --clean-all

# Clean Python __pycache__ directories
./scripts/optimize-cache.sh --clean-python

# Optimize cache structure
./scripts/optimize-cache.sh --optimize

# Show cache statistics
./scripts/optimize-cache.sh --stats
```

**Features:**
- Analyzes cache sizes and file counts
- Removes old cached data
- Cleans Python bytecode files
- Optimizes cache directory structure
- Shows detailed cache statistics

**Example Output:**
```
[CACHE-OPT] === Cache Optimizer ===
[CACHE-OPT] Analyzing cache directories...
[CACHE-OPT] Backend cache: 45M (234 files)
[CACHE-OPT] Root cache: 12M (89 files)
[CACHE-OPT] Python __pycache__ directories: 15
```

---

### ❤️ Health Check (`health-check.sh`)

Verifies that Scout services are running properly and provides diagnostics.

**Usage:**
```bash
# Basic health check
./scripts/health-check.sh

# Verbose mode (show errors)
./scripts/health-check.sh --verbose

# Check dependencies too
./scripts/health-check.sh --check-deps
```

**Features:**
- Checks backend and frontend endpoints
- Verifies process status
- Monitors disk space
- Analyzes log files for errors
- Reports overall system health
- Checks critical Python dependencies

**Example Output:**
```
[HEALTH] === Scout Data Discovery Health Check ===

[HEALTH] Checking Backend API...
[✓] Backend API is healthy (HTTP 200)
[HEALTH] Backend details: {"status":"healthy","version":"1.0"}

[HEALTH] Checking Frontend App...
[✓] Frontend App is healthy (HTTP 200)

[✓] Backend process running (PID: 12345)
[✓] Frontend process running (PID: 12346)

════════════════════════════════════════
[✓] Overall Status: HEALTHY
════════════════════════════════════════
```

---

## Recommended Workflows

### Initial Setup
```bash
# 1. Install dependencies
./scripts/optimize-deps.sh

# 2. Optimize cache structure
./scripts/optimize-cache.sh --optimize

# 3. Start services
./run.sh
```

### Daily Operations
```bash
# Quick start with optimization
./run.sh --optimize

# Check health
./scripts/health-check.sh
```

### Maintenance
```bash
# Check for outdated packages
./scripts/optimize-deps.sh --check-outdated

# Clean old cache files
./scripts/optimize-cache.sh --clean-old

# Health check with dependency verification
./scripts/health-check.sh --check-deps
```

### Troubleshooting
```bash
# Full cleanup
./scripts/optimize-cache.sh --clean-all
./scripts/optimize-cache.sh --clean-python

# Reinstall dependencies
./scripts/optimize-deps.sh --clean-cache --upgrade-pip

# Detailed health check
./scripts/health-check.sh --verbose --check-deps
```

### Production Deployment
```bash
# 1. Optimize dependencies
./scripts/optimize-deps.sh --clean-cache

# 2. Clean cache
./scripts/optimize-cache.sh --clean-all --optimize

# 3. Start services
./run.sh

# 4. Verify health
./scripts/health-check.sh
```

---

## Exit Codes

All scripts follow standard exit codes:
- `0`: Success
- `1`: Error or failure
- `2`: Invalid usage

This allows for scripting and automation:

```bash
# Example automation
if ./scripts/health-check.sh; then
    echo "System is healthy"
else
    echo "System needs attention"
    ./scripts/health-check.sh --verbose
fi
```

---

## Integration with CI/CD

These scripts can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
steps:
  - name: Install dependencies
    run: ./scripts/optimize-deps.sh
    
  - name: Health check
    run: ./scripts/health-check.sh --check-deps
```

---

## Environment Variables

Scripts respect the following environment variables:

- `SCRIPT_DIR`: Override script directory
- `PROJECT_ROOT`: Override project root
- `TIMEOUT`: HTTP timeout for health checks (default: 5s)

---

## Tips & Best Practices

1. **Regular Maintenance**: Run cache optimizer weekly
2. **Health Monitoring**: Schedule health checks hourly
3. **Dependency Updates**: Check outdated packages monthly
4. **Clean Before Deploy**: Always clean cache before production deployment
5. **Log Rotation**: Monitor log file sizes (shown in health check)
6. **Disk Space**: Keep an eye on disk usage alerts

---

## Troubleshooting

### Scripts Won't Execute
```bash
# Make scripts executable
chmod +x scripts/*.sh
```

### Permission Denied on Cache
```bash
# Fix permissions
chmod -R 755 backend/cache
```

### Health Check Fails
```bash
# Check if services are running
./start_scout.sh --status

# View logs
tail -f backend.log
tail -f frontend.log
```

---

## Contributing

When adding new scripts:
1. Follow the same color coding scheme
2. Include help text (`--help`)
3. Return appropriate exit codes
4. Add documentation to this README

---

## Support

For issues with deployment scripts:
1. Check script output messages
2. Run with verbose flags when available
3. Review log files mentioned in output
4. Verify all prerequisites are met
