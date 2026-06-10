def to_xyxy(det: dict) -> list[float] | None:
    """
    Extract a bounding box from a detection dict and return as [x1, y1, x2, y2].
    
    Handles both canonical DetectionSet schema format (bbox dict with x1,y1,x2,y2) 
    and legacy protege format (box list or bbox dict with x,y,width,height).
    """
    # Prefer legacy 'box' key if present (often a list of [x1, y1, x2, y2])
    if 'box' in det and isinstance(det['box'], (list, tuple)) and len(det['box']) >= 4:
        # Some legacy formats provide coordinates as strings
        return [float(x) for x in det['box'][:4]]

    bbox = det.get('bbox')
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return [float(bbox[0]), float(bbox[1]), float(bbox[0]) + float(bbox[2]), float(bbox[1]) + float(bbox[3])]

    if isinstance(bbox, dict):
        # Canonical DetectionSet schema format
        if 'x1' in bbox and 'y1' in bbox and 'x2' in bbox and 'y2' in bbox:
            return [float(bbox['x1']), float(bbox['y1']), float(bbox['x2']), float(bbox['y2'])]

        # Legacy Protege format (x, y, width, height)
        if 'x' in bbox and 'y' in bbox and 'width' in bbox and 'height' in bbox:
            x = float(bbox['x'])
            y = float(bbox['y'])
            w = float(bbox['width'])
            h = float(bbox['height'])
            return [x, y, x + w, y + h]

    # Fallback to rois if bbox is missing or malformed
    rois = det.get("rois")
    if isinstance(rois, list) and rois:
        roi0 = rois[0]
        if isinstance(roi0, (list, tuple)) and len(roi0) == 4:
            return [float(x) for x in roi0]

    return None

def to_xywh(det: dict) -> list[float] | None:
    """
    Extract a bounding box from a detection dict and return as [x, y, w, h].
    """
    xyxy = to_xyxy(det)
    if xyxy is None:
        return None
    x1, y1, x2, y2 = xyxy
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
