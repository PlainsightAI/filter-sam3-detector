def extract_items(data: dict) -> list:
    """Extract detections items from canonical or legacy frame data."""
    if not isinstance(data, dict):
        return []
        
    if "detections" in data:
        dets_payload = data["detections"]
        if isinstance(dets_payload, dict) and "items" in dets_payload:
            return dets_payload["items"]
        elif isinstance(dets_payload, list) and dets_payload:
            return dets_payload
            
    meta = data.get("meta", {})
    if not isinstance(meta, dict):
        return []
        
    if "sam3_detections" in meta and meta["sam3_detections"]:
        return meta["sam3_detections"]
    elif "detections" in meta and meta["detections"]:
        return meta["detections"]
        
    return []
