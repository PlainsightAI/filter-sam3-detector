# Documentation

Complete documentation for the SAM3 detector filter.

## Contents

### Core Documentation

- **[API Reference](API.md)** - Complete API documentation with all classes, methods, and parameters
- **[Configuration Guide](configuration.md)** - Detailed configuration options and examples
- **[Advanced Usage](advanced-usage.md)** - Advanced use cases, patterns, and integrations
- **[Performance Tuning](performance.md)** - Optimization tips and best practices

### Quick Links

- [Main README](../README.md) - Getting started and basic usage
- [Scripts README](../scripts/README.md) - Example scripts and use cases
- [Environment Configuration](../env.example) - Configuration template
- [CHANGELOG](../CHANGELOG.md) - Version history and release notes

## Getting Started

1. **New to the project?** Start with the [Main README](../README.md)
2. **Need configuration help?** See the [Configuration Guide](configuration.md)
3. **Looking for examples?** Check [Scripts README](../scripts/README.md)
4. **Want to optimize?** Read [Performance Tuning](performance.md)

## Documentation Structure

```
docs/
├── README.md              # This file - documentation index
├── API.md                 # Complete API reference
├── configuration.md       # Configuration guide
├── advanced-usage.md      # Advanced patterns and examples
└── performance.md         # Performance optimization guide
```

## API Overview

### Main Classes

- **`FilterSAM3Detector`** - Main filter class for object detection
- **`FilterSAM3DetectorConfig`** - Configuration class

### Key Methods

- `setup(config)` - Initialize the filter
- `process(frames)` - Process frames and detect objects
- `shutdown()` - Clean up resources
- `normalize_config(config)` - Normalize and validate configuration

See [API.md](API.md) for complete documentation.

## Configuration Overview

### Prompt Modes

1. **Text Prompts**: Natural language descriptions
   ```python
   {"text_prompt": "person"}
   ```

2. **Exemplar Images**: Few-shot learning with examples
   ```python
   {"exemplars_path": "./examples/"}
   ```

### Key Parameters

- `confidence_threshold` - Detection confidence (0.0-1.0)
- `max_detections` - Maximum detections per frame
- `output_masks` - Include segmentation masks
- `device` - Processing device (cuda/cpu/mps)

See [configuration.md](configuration.md) for details.

## Common Use Cases

### Basic Detection

```python
from filter_sam3_detector import FilterSAM3Detector

config = {
    "text_prompt": "person",
    "confidence_threshold": 0.5,
}

filter = FilterSAM3Detector()
filter.setup(filter.normalize_config(config))
```

### Pipeline Integration

```python
from openfilter.filter_runtime.filter import Filter

filters = [
    ("VideoIn", {"sources": "file://input.mp4"}),
    (FilterSAM3Detector, {"text_prompt": "person"}),
    ("Recorder", {"path": "output.jsonl"}),
]

Filter.Runner(filters).join()
```

See [advanced-usage.md](advanced-usage.md) for more examples.

## Performance Tips

1. **Use GPU** for 10-50x speedup
2. **Resize inputs** to appropriate resolution (480p recommended)
3. **Disable masks** if not needed (saves memory)
4. **Limit detections** to reasonable numbers
5. **Optimize confidence threshold** for your use case

See [performance.md](performance.md) for detailed optimization guide.

## Troubleshooting

Common issues and solutions:

- **Slow processing**: Use GPU, resize inputs, limit detections
- **Memory errors**: Disable masks, reduce max_detections, use CPU
- **No detections**: Lower confidence threshold, check prompts
- **Import errors**: Ensure package is installed correctly

See the [Main README](../README.md) troubleshooting section for more help.

## Contributing

Found an issue or want to improve the documentation? 

1. Check existing issues
2. Create a new issue or pull request
3. Follow the contribution guidelines

## License

Apache-2.0
