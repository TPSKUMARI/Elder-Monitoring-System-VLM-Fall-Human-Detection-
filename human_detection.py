import cv2
import mediapipe as mp
import numpy as np
import time
import sqlite3
import pytz
from datetime import datetime

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class HumanDetectionSystem:
    """Human Detection System using MediaPipe pose estimation"""
    
    def __init__(self, window_name="Human Detection", show_gui=True):
        self.window_name = window_name
        self.show_gui = show_gui
        
        # Initialize MediaPipe pose
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # Initialize database
        self.init_database()
        
        # Status tracking
        self.current_status = {
            'human_detected': False,
            'detection_confidence': 0.0,
            'landmark_count': 0,
            'bbox': None,
            'frames_processed': 0
        }
        
        # Performance tracking
        self.frame_count = 0
        self.last_detection_time = 0
        
        print("Human Detection System initialized")
    
    def init_database(self):
        """Initialize database for human detection logs"""
        try:
            self.conn = sqlite3.connect('human_detection_logs.db', check_same_thread=False)
            self.cursor = self.conn.cursor()
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS human_detection_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    human_detected BOOLEAN NOT NULL,
                    confidence REAL NOT NULL,
                    landmark_count INTEGER,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.commit()
            print("Human detection database initialized")
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def detect_human(self, frame):
        """Main human detection function"""
        self.frame_count += 1
        current_time = time.time()
        
        # Process frame with MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        
        # Initialize detection results
        human_detected = False
        detection_confidence = 0.0
        landmark_count = 0
        bbox = None
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_height, frame_width = frame.shape[:2]
            
            # Count visible landmarks
            visible_landmarks = [lm for lm in landmarks if lm.visibility >= 0.5]
            landmark_count = len(visible_landmarks)
            
            # Calculate detection confidence
            detection_confidence = landmark_count / len(landmarks)
            
            # Determine if human is detected (at least 30% landmarks visible)
            human_detected = detection_confidence > 0.3
            
            # Calculate bounding box if human detected
            if human_detected and visible_landmarks:
                x_coords = [lm.x * frame_width for lm in visible_landmarks]
                y_coords = [lm.y * frame_height for lm in visible_landmarks]
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))
                bbox = (x1, y1, x2, y2)
        
        # Update current status
        self.current_status.update({
            'human_detected': human_detected,
            'detection_confidence': detection_confidence,
            'landmark_count': landmark_count,
            'bbox': bbox,
            'frames_processed': self.frame_count
        })
        
        # Log detection (rate limited - every 5 seconds)
        if current_time - self.last_detection_time > 5.0:
            self.log_detection(human_detected, detection_confidence, landmark_count, bbox)
            self.last_detection_time = current_time
            
            if human_detected:
                print(f"Human detected - Confidence: {detection_confidence:.2f}, Landmarks: {landmark_count}")
        
        # Return results for use by other systems
        result = {
            'human_detected': human_detected,
            'detection_confidence': detection_confidence,
            'landmark_count': landmark_count,
            'bbox': bbox,
            'pose_landmarks': results.pose_landmarks,
            'pose_results': results  # Full MediaPipe results for fall detection
        }
        
        # Display GUI if enabled
        if self.show_gui:
            self.render_gui(frame, results, human_detected, detection_confidence, bbox)
        
        return result
    
    def log_detection(self, human_detected, confidence, landmark_count, bbox):
        """Log human detection to database"""
        try:
            slst_tz = pytz.timezone('Asia/Colombo')
            current_time = datetime.now(slst_tz).strftime('%Y-%m-%d %H:%M:%S')
            
            bbox_coords = bbox if bbox else (None, None, None, None)
            
            self.cursor.execute('''
                INSERT INTO human_detection_logs 
                (human_detected, confidence, landmark_count, bbox_x1, bbox_y1, bbox_x2, bbox_y2, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                human_detected,
                confidence,
                landmark_count,
                bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3],
                current_time
            ))
            self.conn.commit()
        except Exception as e:
            print(f"Human detection logging error: {e}")
    
    def render_gui(self, frame, results, human_detected, confidence, bbox):
        """Render the GUI display"""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Draw pose landmarks if human detected
        if results.pose_landmarks and human_detected:
            mp_drawing.draw_landmarks(
                display_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
            )
        
        # Draw bounding box
        if bbox:
            color = (0, 255, 0) if human_detected else (0, 0, 255)
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Status text
        status_text = "HUMAN DETECTED" if human_detected else "NO HUMAN"
        status_color = (0, 255, 0) if human_detected else (0, 0, 255)
        
        # Display information
        cv2.putText(display_frame, "Human Detection System", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Status: {status_text}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(display_frame, f"Confidence: {confidence:.2f}", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Landmarks: {self.current_status['landmark_count']}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # System info
        cv2.putText(display_frame, f"Frame: {self.frame_count}", (10, h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, "Human Detection Module", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.imshow(self.window_name, display_frame)
    
    def get_status(self):
        """Get current system status"""
        return self.current_status.copy()
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
            if hasattr(self, 'pose'):
                self.pose.close()
            print("Human Detection System cleaned up")
        except Exception as e:
            print(f"Cleanup error: {e}")

# For standalone testing
if __name__ == "__main__":
    print("Testing Human Detection System standalone...")
    
    # Test with camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera available for testing")
        exit(1)
    
    system = HumanDetectionSystem(show_gui=True)
    
    try:
        print("Starting human detection test...")
        print("Controls: Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Mirror frame for better user experience
            frame = cv2.flip(frame, 1)
            
            # Run detection
            result = system.detect_human(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        system.cleanup()
        cv2.destroyAllWindows()
        print("Human Detection System test complete")