# Performance Tuning

Optimization tips and best practices for the SAM3 detector filter.

## Performance Factors

The filter's performance is affected by:

1. **Input resolution** - Higher resolution = slower processing
2. **Number of detections** - More detections = more processing time
3. **Output options** - Masks are memory-intensive
4. **Device** - GPU is much faster than CPU
5. **Model size** - Larger models are slower but more accurate

## Optimization Strategies

### 1. Resize Input Images

Reduce input resolution for faster processing:

```python
from openfilter.filter_runtime.filters.video_in import VideoIn
from openfilter.filter_runtime.filters.resize import Resize

# In pipeline, resize before detection
filters = [
    (VideoIn, {
        "sources": "file://input.mp4",
        "outputs": ["tcp://127.0.0.1:5555"],
    }),
    (Resize, {
        "sources": "tcp://127.0.0.1:5555",
        "outputs": ["tcp://127.0.0.1:5556"],
        "width": 640,
        "height": 480,
    }),
    (FilterSAM3Detector, {
        "sources": "tcp://127.0.0.1:5556",
        "text_prompt": "person",
    }),
]
```

**Recommendations:**
- 480p (640x480): Good balance of speed and accuracy
- 720p (1280x720): Better accuracy, slower
- 1080p+: Only for high-accuracy requirements

### 2. Limit Detections

Reduce `max_detections` for faster processing:

```python
config = {
    "text_prompt": "person",
    "max_detections": 20,  # Instead of default 100
}
```

**Guidelines:**
- Single object scenes: `10-20`
- Moderate scenes: `30-50`
- Crowded scenes: `50-100`

### 3. Disable Masks When Not Needed

Masks are memory-intensive. Disable if you only need bounding boxes:

```python
config = {
    "text_prompt": "person",
    "output_masks": False,  # Saves significant memory and time
    "output_boxes": True,
    "output_scores": True,
}
```

**Memory Savings:**
- With masks: ~100MB per frame (depends on resolution)
- Without masks: ~1MB per frame

### 4. Use GPU

GPU acceleration provides 10-50x speedup:

```python
config = {
    "device": "cuda",  # Much faster than CPU
}
```

**Requirements:**
- NVIDIA GPU with CUDA support
- CUDA toolkit installed
- PyTorch with CUDA support

**Fallback:**
- If GPU unavailable, automatically falls back to CPU
- CPU mode is slower but works everywhere

### 5. Optimize Confidence Threshold

Lower thresholds find more objects but process more:

```python
# High precision (fewer detections, faster)
config = {
    "confidence_threshold": 0.7,
}

# Balanced (default)
config = {
    "confidence_threshold": 0.5,
}

# High recall (more detections, slower)
config = {
    "confidence_threshold": 0.3,
}
```

## Benchmarking

### Performance Metrics

Measure your pipeline performance:

```python
import time
from openfilter.filter_runtime.filter import Filter
from filter_sam3_detector import FilterSAM3Detector

class PerformanceMonitor(Filter):
    def setup(self, config):
        self.frame_count = 0
        self.total_time = 0
        self.start_time = None
    
    def process(self, frames):
        if self.start_time is None:
            self.start_time = time.time()
        
        frame_start = time.time()
        # Process frames
        frame_end = time.time()
        
        self.frame_count += len(frames)
        self.total_time += (frame_end - frame_start)
        
        avg_time = self.total_time / self.frame_count
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        if self.frame_count % 100 == 0:
            print(f"Processed {self.frame_count} frames, "
                  f"Avg: {avg_time:.3f}s/frame, FPS: {fps:.2f}")
        
        return frames
```

### Typical Performance

**GPU (NVIDIA RTX 3090):**
- 1080p: ~2-5 FPS
- 720p: ~5-10 FPS
- 480p: ~10-20 FPS

**CPU (Intel i7-9700K):**
- 1080p: ~0.1-0.3 FPS
- 720p: ~0.3-0.5 FPS
- 480p: ~0.5-1 FPS

**Note:** Performance varies based on:
- Number of objects in scene
- Model size
- Detection threshold
- Output options

## Memory Optimization

### Reduce Memory Usage

```python
# Memory-efficient configuration
config = {
    "text_prompt": "person",
    "output_masks": False,      # Disable masks
    "max_detections": 20,       # Limit detections
    "confidence_threshold": 0.6, # Higher threshold = fewer detections
}
```

### Batch Processing

Process videos in smaller batches:

```python
def process_in_batches(video_path, batch_size=100):
    from openfilter.filter_runtime.filters.video_in import VideoIn

    filters = [
        (VideoIn, {
            "sources": f"file://{video_path}",
            "outputs": ["tcp://127.0.0.1:5555"],
            "max_frames": batch_size,  # Process in batches
        }),
        (FilterSAM3Detector, {
            "sources": "tcp://127.0.0.1:5555",
            "text_prompt": "person",
        }),
    ]

    Filter.run_multi(filters)
```

## Model Selection

### Available Models

- `facebook/sam2-hiera-large` (default): Best accuracy, slower
- `facebook/sam2-hiera-base`: Balanced
- `facebook/sam2-hiera-small`: Faster, lower accuracy

```python
config = {
    "model_id": "facebook/sam2-hiera-base",  # Faster alternative
    "text_prompt": "person",
}
```

## Parallel Processing

### Multi-GPU Processing

```python
# Process different videos on different GPUs
import os

def process_on_gpu(video_path, gpu_id):
    from openfilter.filter_runtime.filters.video_in import VideoIn

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    filters = [
        (VideoIn, {"sources": f"file://{video_path}"}),
        (FilterSAM3Detector, {
            "text_prompt": "person",
            "device": "cuda",
        }),
    ]

    Filter.run_multi(filters)

# Process multiple videos in parallel
from multiprocessing import Process

videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
processes = []

for i, video in enumerate(videos):
    p = Process(target=process_on_gpu, args=(video, i))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
```

## Profiling

### Identify Bottlenecks

```python
import cProfile
import pstats

def profile_pipeline():
    filters = [
        (FilterSAM3Detector, {
            "text_prompt": "person",
        }),
    ]

    profiler = cProfile.Profile()
    profiler.enable()

    Filter.run_multi(filters)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

## Best Practices

1. **Start with defaults** and optimize based on your needs
2. **Profile first** to identify actual bottlenecks
3. **Use GPU** whenever possible
4. **Resize inputs** to appropriate resolution
5. **Disable unused outputs** (masks if not needed)
6. **Limit detections** to reasonable numbers
7. **Batch process** large datasets
8. **Monitor memory** usage and adjust accordingly

## Troubleshooting Performance

### Slow Processing

1. Check if GPU is being used: `nvidia-smi`
2. Reduce input resolution
3. Lower `max_detections`
4. Disable masks if not needed
5. Use smaller model variant

### High Memory Usage

1. Disable `output_masks`
2. Reduce `max_detections`
3. Process in smaller batches
4. Use CPU mode (slower but less memory)

### GPU Out of Memory

1. Reduce input resolution
2. Disable masks
3. Lower max detections
4. Use CPU mode as fallback

