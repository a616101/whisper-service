# WhisperX Subtitle Service API 文件

## 基本資訊

- **Base URL**: `http://localhost:8000/api/v1`
- **Content-Type**: `multipart/form-data`（上傳檔案）
- **Response Format**: JSON

---

## 端點列表

| 端點 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康檢查 |
| `/transcribe` | POST | 同步轉寫 |
| `/transcribe/async` | POST | 異步轉寫 |
| `/subtitle` | POST | 生成字幕檔 |
| `/correct` | POST | 文字校正 |
| `/align` | POST | 重新對齊 |
| `/task/{task_id}` | GET | 查詢任務狀態 |

---

## 1. 健康檢查

### GET /health

檢查服務狀態和 GPU 可用性。

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "gpu_available": true,
  "model_loaded": true,
  "diarization_available": true,
  "details": {
    "gpu_count": 2,
    "model_size": "large-v3",
    "compute_type": "float16"
  }
}
```

---

## 2. 同步轉寫

### POST /transcribe

將音訊/影片轉寫為文字，返回 JSON 結果。

**適用場景**: < 15 分鐘的音檔

**Parameters:**
| 參數 | 類型 | 必填 | 預設 | 說明 |
|------|------|------|------|------|
| file | File | ✓ | - | 音訊或影片檔案 |
| language | string | | null | 語言代碼（zh/en/ja），null 為自動偵測 |
| diarize | bool | | true | 是否進行說話者分離 |
| min_speakers | int | | null | 最少說話者數 |
| max_speakers | int | | null | 最多說話者數 |

**範例:**
```bash
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@meeting.mp4" \
  -F "diarize=true" \
  -F "language=zh"
```

**Response:**
```json
{
  "language": "zh",
  "duration": 1234.5,
  "segments": [
    {
      "id": 1,
      "start": 0.0,
      "end": 3.5,
      "text": "大家好，歡迎來到今天的會議",
      "speaker": "SPEAKER_00",
      "words": [
        {"start": 0.0, "end": 0.3, "word": "大家", "confidence": 0.98, "speaker": "SPEAKER_00"},
        {"start": 0.3, "end": 0.5, "word": "好", "confidence": 0.99, "speaker": "SPEAKER_00"}
      ]
    }
  ],
  "speakers": ["SPEAKER_00", "SPEAKER_01"]
}
```

---

## 3. 字幕生成

### POST /subtitle

生成指定格式的字幕檔案。

**Parameters:**
| 參數 | 類型 | 必填 | 預設 | 說明 |
|------|------|------|------|------|
| file | File | ✓ | - | 音訊或影片檔案 |
| format | string | | "srt" | 輸出格式：srt / vtt / json |
| diarize | bool | | true | 是否進行說話者分離 |
| include_speaker | bool | | true | 是否在字幕中顯示說話者 |
| speaker_format | string | | "prefix" | 說話者格式：prefix / tag / none |
| max_chars_per_line | int | | 42 | 每行最大字數 |

### 範例：SRT 格式

```bash
curl -X POST http://localhost:8000/api/v1/subtitle \
  -F "file=@video.mp4" \
  -F "format=srt" \
  -F "diarize=true" \
  -F "include_speaker=true" \
  -o output.srt
```

**輸出 (SRT):**
```
1
00:00:00,000 --> 00:00:03,500
SPEAKER_00: 大家好，歡迎來到今天的會議

2
00:00:03,500 --> 00:00:07,200
SPEAKER_01: 謝謝主持人，我來分享今天的報告
```

### 範例：VTT 格式

```bash
curl -X POST http://localhost:8000/api/v1/subtitle \
  -F "file=@video.mp4" \
  -F "format=vtt" \
  -o output.vtt
```

**輸出 (VTT):**
```
WEBVTT

00:00:00.000 --> 00:00:03.500
SPEAKER_00: 大家好，歡迎來到今天的會議

00:00:03.500 --> 00:00:07.200
SPEAKER_01: 謝謝主持人，我來分享今天的報告
```

### 範例：JSON 格式（video.js 相容）

```bash
curl -X POST http://localhost:8000/api/v1/subtitle \
  -F "file=@video.mp4" \
  -F "format=json"
```

**輸出 (JSON):**
```json
{
  "version": "1.0",
  "language": "zh",
  "duration": 1234.5,
  "segments": [...],
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "metadata": {
    "generator": "WhisperX Subtitle Service",
    "model": "large-v3"
  }
}
```

---

## 4. 文字校正

### POST /correct

校正轉寫文字中的錯誤。

**Parameters:**
| 參數 | 類型 | 必填 | 預設 | 說明 |
|------|------|------|------|------|
| text | string | ✓ | - | 要校正的文字 |
| glossary_json | string | | null | JSON 格式的詞彙對照表 |
| use_llm | bool | | false | 是否使用 LLM 校正 |

**範例:**
```bash
curl -X POST http://localhost:8000/api/v1/correct \
  -F "text=我們的 A I 系統使用 Chat G P T 技術" \
  -F 'glossary_json={"Whishper":"Whisper","威斯博":"Whisper"}'
```

**Response:**
```json
{
  "original_text": "我們的 A I 系統使用 Chat G P T 技術",
  "corrected_text": "我們的 AI 系統使用 ChatGPT 技術",
  "changes": [
    {"type": "glossary", "from": "A I", "to": "AI"},
    {"type": "glossary", "from": "Chat G P T", "to": "ChatGPT"}
  ],
  "applied_rules": ["common_fixes"]
}
```

---

## 5. 重新對齊

### POST /align

校正文字後，重新生成準確的時間軸。

**Parameters:**
| 參數 | 類型 | 必填 | 預設 | 說明 |
|------|------|------|------|------|
| file | File | ✓ | - | 原始音訊檔案 |
| corrected_text | string | ✓ | - | 校正後的完整文字 |
| language | string | | null | 語言代碼 |

**範例:**
```bash
curl -X POST http://localhost:8000/api/v1/align \
  -F "file=@video.mp4" \
  -F "corrected_text=這是校正後的文字內容，會重新對齊時間軸"
```

**Response:**
```json
{
  "language": "zh",
  "duration": 1234.5,
  "segments": [
    {
      "id": 1,
      "start": 0.0,
      "end": 2.5,
      "text": "這是校正後的文字內容",
      "words": [...]
    }
  ]
}
```

---

## 6. 異步轉寫

### POST /transcribe/async

提交長音檔轉寫任務，返回 task_id。

**適用場景**: > 15 分鐘的長音檔

**Parameters:**
| 參數 | 類型 | 必填 | 預設 | 說明 |
|------|------|------|------|------|
| file | File | ✓ | - | 音訊或影片檔案 |
| language | string | | null | 語言代碼 |
| diarize | bool | | true | 是否進行說話者分離 |
| webhook_url | string | | null | 完成時回調的 URL |

**範例:**
```bash
curl -X POST http://localhost:8000/api/v1/transcribe/async \
  -F "file=@long_meeting.mp4" \
  -F "diarize=true" \
  -F "webhook_url=https://your-server.com/webhook"
```

**Response:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Task created. Use /task/{task_id} to check status."
}
```

---

## 7. 查詢任務狀態

### GET /task/{task_id}

查詢異步任務的處理狀態。

**範例:**
```bash
curl http://localhost:8000/api/v1/task/550e8400-e29b-41d4-a716-446655440000
```

**Response (處理中):**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 0.45,
  "result": null,
  "error": null,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:32:00Z"
}
```

**Response (完成):**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 1.0,
  "result": {
    "language": "zh",
    "duration": 3600.5,
    "segments": [...],
    "speakers": ["SPEAKER_00", "SPEAKER_01"]
  },
  "error": null,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T11:00:00Z"
}
```

---

## 錯誤處理

所有端點在發生錯誤時返回以下格式：

```json
{
  "error": "Error message",
  "detail": "Detailed error information",
  "code": "ERROR_CODE"
}
```

### 常見錯誤碼

| HTTP Status | 說明 |
|-------------|------|
| 400 | 請求參數錯誤 |
| 404 | 任務不存在 |
| 413 | 檔案太大 |
| 500 | 伺服器內部錯誤 |
| 503 | 服務暫時不可用（GPU 忙碌）|

---

## video.js 整合範例

```html
<video id="my-video" class="video-js">
  <source src="video.mp4" type="video/mp4">
</video>

<script>
const player = videojs('my-video');

// 載入字幕
fetch('/api/v1/subtitle', {
  method: 'POST',
  body: formData  // 包含影片檔案
})
.then(res => res.json())
.then(data => {
  // 動態添加字幕軌道
  player.addRemoteTextTrack({
    kind: 'captions',
    srclang: data.language,
    label: '字幕',
    src: URL.createObjectURL(new Blob([JSON.stringify(data)]))
  });
});
</script>
```
