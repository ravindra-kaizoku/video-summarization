# FASTAPI ENDPOINT REFERENCE

Base URL: `http://localhost:8000/api`

---

## Endpoints

### 1. Upload Video
- **URL**: `POST /api/upload`
- **Body**: `multipart/form-data` with key `file`
- **Response**:
```json
{
  "task_id": "8f2d...3a",
  "filename": "lecture_30min.mp4",
  "video_url": "/uploads/8f2d...3a.mp4",
  "message": "Upload complete"
}
```

### 2. Process Video
- **URL**: `POST /api/process/{task_id}`
- **Response**:
```json
{
  "task_id": "8f2d...3a",
  "status": "processing_started"
}
```

### 3. Check Task Status
- **URL**: `GET /api/status/{task_id}`
- **Response**:
```json
{
  "status": "processing",
  "progress": 65,
  "message": "Performing multimodal temporal alignment",
  "file_path": "uploads/8f2d...3a.mp4"
}
```

### 4. Fetch Summary Result
- **URL**: `GET /api/result/{task_id}`
- **Response**:
```json
{
  "video_duration": "30.0 minutes",
  "summary": "This 30-minute video presents...",
  "key_points": [
    "Detailed coverage across the 30.0-minute timeline..."
  ],
  "important_sections": [
    {
      "start": "00:00",
      "end": "05:00",
      "start_seconds": 0,
      "end_seconds": 300,
      "title": "Key Section 1: 00:00 - 05:00 (Score: 0.94)",
      "summary": "This segment covers key topic discussions..."
    }
  ]
}
```
