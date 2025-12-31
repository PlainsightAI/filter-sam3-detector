# Contributing to filter-sam3-detector

Thank you for your interest in contributing to the SAM3 Object Detection Filter!

## Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd filter-sam3-detector

# Install with dev dependencies
pip install -e .[dev] \
    --index-url https://python.openfilter.io/simple \
    --extra-index-url https://pypi.org/simple

# Or using uv (recommended)
uv pip install -e .[dev]
```

## Project Structure

```
filter-sam3-detector/
├── Makefile                    # Build and run commands
├── pyproject.toml              # Package configuration
├── VERSION                     # Version file (update for releases)
├── README.md                   # User documentation
├── CONTRIBUTING.md             # This file
├── Dockerfile                  # Production Docker image
├── Dockerfile.dev              # Development Docker image
├── docker-compose.local.yaml   # Local development pipeline
├── filter_sam3_detector/
│   ├── __init__.py
│   └── filter.py               # Main filter implementation
├── sam3/                       # SAM3 model submodule
└── tests/
    └── test_filter.py
```

## Running Locally

### With OpenFilter CLI

```bash
# Install dependencies
make install

# Run with a text prompt
TEXT_PROMPT="person" make run
```

### With Docker Compose

```bash
# Build the development image
make docker-build

# Run the pipeline
TEXT_PROMPT="car" make docker-run

# View results at http://localhost:8001
```

## Testing

```bash
# Run all tests
make test

# Run specific test
pytest -vv tests/test_filter.py::test_process_with_image
```

## Code Style

This filter follows OpenFilter idioms:

1. **FilterConfig**: Define all configuration in a typed `FilterConfig` subclass
2. **normalize_config**: Validate parameters here, not external resources
3. **setup**: Initialize models and external resources here
4. **process**: Use `frame.rw_bgr.image` for writable BGR images
5. **Frame creation**: Always specify `format=` when creating new Frames

### Do's and Don'ts

**DO:**
```python
@classmethod
def normalize_config(cls, config):
    config = super().normalize_config(config)
    # Validate parameters
    if not 0 <= config.get('threshold', 0.5) <= 1:
        raise ValueError("threshold must be 0-1")
    return MyFilterConfig(**config)
```

**DON'T:**
```python
# Don't manually parse environment variables
threshold = float(os.environ.get('FILTER_THRESHOLD', '0.5'))

# Don't check external resources in normalize_config
if not Path(config.model_path).exists():  # Move to setup()
    raise ValueError("Model not found")
```

## Contribution Workflow

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/my-feature`
3. **Make changes** following the code style above
4. **Run tests**: `make test`
5. **Commit with DCO sign-off**: `git commit -s -m "Add feature"`
6. **Push and open a PR**

### DCO Sign-off

All commits must include Developer Certificate of Origin sign-off:

```bash
git commit -s -m "Add new feature"
```

This adds `Signed-off-by: Your Name <email@example.com>` to commit messages.

## Publishing (Automated via GitHub Actions)

Releases are automated via GitHub Actions using the `Create Release` workflow.

### Release Process

1. **Update the VERSION file** to match your release version:
   ```bash
   echo "v0.2.0" > VERSION
   ```

2. **Update RELEASE.md** with changelog entry:
   ```markdown
   ## v0.2.0

   ### Added
   - New feature description

   ### Fixed
   - Bug fix description
   ```

3. **Commit and push**:
   ```bash
   git add VERSION RELEASE.md
   git commit -s -m "Prepare release v0.2.0"
   git push
   ```

4. **Trigger the release workflow**:
   - Go to Actions → Create Release → Run workflow
   - Or use: `gh workflow run create-release.yaml`

5. **GitHub Actions will automatically**:
   - Run tests across Python 3.10, 3.11, 3.12
   - Validate VERSION matches RELEASE.md
   - Create a GitHub Release with changelog
   - Build and publish Python package to PyPI
   - Build and push Docker image to Docker Hub

### Required Repository Secrets

The following secrets must be configured:

| Secret | Environment | Description |
|--------|-------------|-------------|
| `PLAINSIGHT_PYPI_TOKEN` | `pypi-release` | PyPI API token (used with `__token__` username) |
| `DOCKERHUB_ACCESS_TOKEN` | (repo secret) | Docker Hub access token for image publishing |

The workflow uses:
- **PyPI publishing**: Standard PyPI with API token authentication
- **Docker publishing**: `PlainsightAI/gh-actions/publish-dockerhub@main` composite action

### Manual Publishing (if needed)

If you have direct registry access:

```bash
# Build wheel
python -m build --wheel

# Upload to PyPI
twine upload dist/*.whl

# Docker image
docker build -t plainsightai/filter-sam3-detector:v0.2.0 .
docker push plainsightai/filter-sam3-detector:v0.2.0
```

## Getting Help

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- For OpenFilter platform questions, see [OpenFilter documentation](https://openfilter.io/docs)
