# Advanced Usage

Advanced use cases and patterns for the SAM3 detector filter.

## Pipeline Integration

### Multi-Filter Pipelines

Combine SAM3 detector with other OpenFilter filters:

```python
from openfilter.filter_runtime.filter import Filter
from filter_sam3_detector import FilterSAM3Detector

filters = [
    # Input: Stream video frames
    ("VideoIn", {
        "sources": "file://input.mp4",
        "outputs": ["tcp://127.0.0.1:5555"],
    }),
    
    # Pre-processing: Resize for performance
    ("Resize", {
        "sources": "tcp://127.0.0.1:5555",
        "outputs": ["tcp://127.0.0.1:5556"],
        "width": 640,
        "height": 480,
    }),
    
    # Detection: SAM3 detector
    (FilterSAM3Detector, {
        "sources": "tcp://127.0.0.1:5556",
        "outputs": ["tcp://127.0.0.1:5557"],
        "text_prompt": "person",
        "confidence_threshold": 0.5,
    }),
    
    # Post-processing: Filter by confidence
    ("FilterByScore", {
        "sources": "tcp://127.0.0.1:5557",
        "outputs": ["tcp://127.0.0.1:5558"],
        "min_score": 0.7,
    }),
    
    # Output: Save results
    ("Recorder", {
        "sources": "tcp://127.0.0.1:5558",
        "path": "detections.jsonl",
        "format": "jsonl",
    }),
]

runner = Filter.Runner(filters)
runner.join()
```

### Topic Forwarding

Preserve original frames while adding detections:

```python
filters = [
    ("VideoIn", {
        "sources": "file://input.mp4",
        "outputs": ["tcp://127.0.0.1:5555", "tcp://127.0.0.1:5559"],  # Split output
    }),
    
    # Detection branch
    (FilterSAM3Detector, {
        "sources": "tcp://127.0.0.1:5555",
        "outputs": ["tcp://127.0.0.1:5556"],
        "text_prompt": "person",
    }),
    
    ("Recorder", {
        "sources": "tcp://127.0.0.1:5556",
        "path": "detections.jsonl",
    }),
    
    # Original frames branch (preserved)
    ("Recorder", {
        "sources": "tcp://127.0.0.1:5559",
        "path": "original_frames.jsonl",
    }),
]
```

## Custom Processing

### Accessing Detection Results

```python
from openfilter.filter_runtime.filter import Filter, Frame
from filter_sam3_detector import FilterSAM3Detector

class CustomProcessor(Filter):
    def process(self, frames: dict[str, Frame]) -> dict[str, Frame]:
        output_frames = {}
        
        for topic, frame in frames.items():
            # Access detections
            detections = frame.data.get('meta', {}).get('sam3_detections', [])
            
            # Process detections
            for det in detections:
                box = det['box']
                score = det['score']
                
                # Custom logic here
                if score > 0.8:
                    # High confidence detection
                    pass
            
            output_frames[topic] = frame
        
        return output_frames

# Use in pipeline
filters = [
    (FilterSAM3Detector, {
        "sources": "tcp://127.0.0.1:5555",
        "outputs": ["tcp://127.0.0.1:5556"],
        "text_prompt": "person",
    }),
    (CustomProcessor, {
        "sources": "tcp://127.0.0.1:5556",
        "outputs": ["tcp://127.0.0.1:5557"],
    }),
]
```

### Filtering Detections

```python
def filter_high_confidence(frames):
    """Keep only high-confidence detections."""
    output_frames = {}
    
    for topic, frame in frames.items():
        detections = frame.data.get('meta', {}).get('sam3_detections', [])
        
        # Filter by confidence
        filtered = [d for d in detections if d['score'] > 0.7]
        
        # Update frame
        frame.data.setdefault('meta', {})['sam3_detections'] = filtered
        output_frames[topic] = frame
    
    return output_frames
```

## Batch Processing

### Processing Multiple Videos

```python
from pathlib import Path

video_files = [
    "video1.mp4",
    "video2.mp4",
    "video3.mp4",
]

for video in video_files:
    filters = [
        ("VideoIn", {
            "sources": f"file://{Path(video).absolute()}",
            "outputs": ["tcp://127.0.0.1:5555"],
        }),
        (FilterSAM3Detector, {
            "sources": "tcp://127.0.0.1:5555",
            "outputs": ["tcp://127.0.0.1:5556"],
            "text_prompt": "person",
        }),
        ("Recorder", {
            "sources": "tcp://127.0.0.1:5556",
            "path": f"output_{Path(video).stem}.jsonl",
        }),
    ]
    
    runner = Filter.Runner(filters)
    runner.join()
```

## Dynamic Configuration

### Runtime Configuration Changes

```python
class DynamicDetector(Filter):
    def setup(self, config):
        self.base_config = config
        self.current_prompt = config.get("text_prompt", "person")
    
    def process(self, frames):
        # Change prompt based on frame content
        if self.should_switch_prompt(frames):
            self.current_prompt = "car"
        
        # Create detector with current prompt
        detector = FilterSAM3Detector({
            **self.base_config,
            "text_prompt": self.current_prompt,
        })
        detector.setup(detector.normalize_config({
            **self.base_config,
            "text_prompt": self.current_prompt,
        }))
        
        return detector.process(frames)
```

## Performance Optimization

### Async Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def process_video_async(video_path, prompt):
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor()
    
    def run_pipeline():
        filters = [
            ("VideoIn", {"sources": f"file://{video_path}"}),
            (FilterSAM3Detector, {"text_prompt": prompt}),
            ("Recorder", {"path": f"output_{video_path}.jsonl"}),
        ]
        runner = Filter.Runner(filters)
        return runner.join()
    
    return await loop.run_in_executor(executor, run_pipeline)

# Process multiple videos concurrently
async def main():
    videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
    tasks = [process_video_async(v, "person") for v in videos]
    results = await asyncio.gather(*tasks)

asyncio.run(main())
```

### Memory Management

```python
class MemoryEfficientDetector(Filter):
    def setup(self, config):
        # Disable masks to save memory
        config["output_masks"] = False
        config["max_detections"] = 20
        
        self.detector = FilterSAM3Detector()
        self.detector.setup(self.detector.normalize_config(config))
    
    def process(self, frames):
        # Process in batches
        batch_size = 10
        results = {}
        
        frame_list = list(frames.items())
        for i in range(0, len(frame_list), batch_size):
            batch = dict(frame_list[i:i+batch_size])
            batch_results = self.detector.process(batch)
            results.update(batch_results)
        
        return results
```

## Error Handling

### Robust Pipeline

```python
class RobustDetector(Filter):
    def setup(self, config):
        self.config = config
        self.detector = None
        self.error_count = 0
        self.max_errors = 10
    
    def process(self, frames):
        if self.detector is None:
            try:
                self.detector = FilterSAM3Detector()
                self.detector.setup(
                    self.detector.normalize_config(self.config)
                )
            except Exception as e:
                logger.error(f"Failed to initialize detector: {e}")
                return frames  # Forward unchanged
        
        try:
            return self.detector.process(frames)
        except Exception as e:
            self.error_count += 1
            logger.error(f"Processing error ({self.error_count}/{self.max_errors}): {e}")
            
            if self.error_count >= self.max_errors:
                logger.error("Too many errors, reinitializing detector")
                self.detector = None
                self.error_count = 0
            
            return frames  # Forward unchanged on error
```

## Integration Examples

### With Database

```python
import sqlite3

class DatabaseRecorder(Filter):
    def setup(self, config):
        self.db_path = config.get("db_path", "detections.db")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                frame_id INTEGER,
                box_x1 INTEGER, box_y1 INTEGER, box_x2 INTEGER, box_y2 INTEGER,
                score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def process(self, frames):
        for topic, frame in frames.items():
            detections = frame.data.get('meta', {}).get('sam3_detections', [])
            
            for det in detections:
                box = det['box']
                self.conn.execute(
                    "INSERT INTO detections (frame_id, box_x1, box_y1, box_x2, box_y2, score) VALUES (?, ?, ?, ?, ?, ?)",
                    (frame.data.get('frame_id', 0), box[0], box[1], box[2], box[3], det['score'])
                )
            
            self.conn.commit()
        
        return frames
```

### With Web API

```python
import requests

class APISender(Filter):
    def setup(self, config):
        self.api_url = config.get("api_url")
        self.api_key = config.get("api_key")
    
    def process(self, frames):
        for topic, frame in frames.items():
            detections = frame.data.get('meta', {}).get('sam3_detections', [])
            
            if detections:
                response = requests.post(
                    self.api_url,
                    json={"detections": detections},
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                response.raise_for_status()
        
        return frames
```

