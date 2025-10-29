# Subrepos - External Repository Integration

This folder is designed to contain external repositories that this project integrates with or depends on. It provides a clean way to organize third-party code and workflows.

## Purpose

The `subrepos/` directory serves as a centralized location for:
- External repository integrations
- Third-party workflows and automations
- Submodules or vendored dependencies
- Integration configurations

## Current Contents

### n8n-workflows/
This directory is reserved for n8n workflow configurations and automations that integrate with the Scout Data Discovery platform.

## Usage Guidelines

### Adding a New Subrepo

To add a new external repository or integration:

1. **As a Git Submodule** (recommended):
   ```bash
   cd subrepos/
   git submodule add <repository-url> <directory-name>
   ```

2. **As a Regular Directory** (for non-git resources):
   ```bash
   cd subrepos/
   mkdir <directory-name>
   # Add your files
   ```

3. **Document the Integration**:
   Create a README.md in the new directory explaining:
   - What the subrepo contains
   - How it integrates with the main project
   - Any setup or configuration needed
   - Version/compatibility information

### Best Practices

1. **Keep Dependencies Isolated**: Each subrepo should be self-contained
2. **Version Control**: Use git submodules for proper version tracking
3. **Documentation**: Always include a README in each subrepo directory
4. **Naming Convention**: Use descriptive, lowercase names with hyphens (e.g., `data-pipeline-tools`)

### Example Structure

```
subrepos/
├── README.md (this file)
├── n8n-workflows/
│   ├── README.md
│   └── workflows/
├── data-processing-scripts/
│   ├── README.md
│   └── scripts/
└── external-apis/
    ├── README.md
    └── integrations/
```

## Integration with Main Project

To use code from a subrepo in the main project:

```python
import sys
from pathlib import Path

# Add subrepo to Python path
subrepo_path = Path(__file__).parent.parent / "subrepos" / "my-subrepo"
sys.path.append(str(subrepo_path))

# Now you can import from the subrepo
from my_subrepo import some_module
```

## Maintenance

- Keep subrepos updated regularly
- Document any breaking changes in the subrepo's README
- Test integrations after updating subrepos
- Consider using requirements.txt or similar for dependencies

## Questions?

For questions about subrepo organization or integration, refer to the main project documentation in `/docs`.
