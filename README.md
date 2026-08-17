# Elder Monitoring System (VLM + Fall + Human Detection)

A real-time elder monitoring system that combines pose-based human detection, rule-based fall detection, and AI-powered scene analysis using a webcam feed.

## What it does

1. **Human Detection** ([human_detection.py](human_detection.py)) — Uses MediaPipe pose estimation to detect whether a person is present in the frame.
2. **Fall Detection** ([fall_detection.py](fall_detection.py)) — When a human is detected, analyzes body pose (torso angle, head drop, velocity, bounding box ratio, etc.) to detect falls in real time.
3. **VLM Analysis** ([vlm_analysis.py](vlm_analysis.py)) — Once per minute (only while a human is present), sends a frame to OpenAI's vision model to describe the person's activity and flag potential hazards (e.g. spills, fire, distress).

All three modules are orchestrated by [main.py](main.py), which runs the webcam loop, combines results, and displays a live dashboard with pose landmarks, alerts, and a status sidebar.

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=your-new-openai-api-key
```

(If no API key is provided, the system still runs with human + fall detection only; VLM analysis is disabled.)

## Usage

```bash
python main.py
```

- Press `q` to quit
- Press `s` to print full system status
- Press `h` / `f` / `v` to print individual system status (human / fall / VLM)

Each module (`human_detection.py`, `fall_detection.py`, `vlm_analysis.py`) can also be run standalone for testing.

## Data

Detection events are logged to local SQLite databases (`human_detection_logs.db`, `fall_detection_logs.db`, `vlm_analysis_logs.db`). VLM-analyzed frames are saved to `vlm_analyzed_frames/`.
