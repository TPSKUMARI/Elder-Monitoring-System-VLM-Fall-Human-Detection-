import cv2
import os
import time
import sys
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Import the separate detection modules
try:
    from human_detection import HumanDetectionSystem
    from fall_detection import FallDetectionSystem, filter_keypoints_by_confidence, SYSTEM_CONFIG
    from vlm_analysis import VLMAnalysisSystem
except ImportError as e:
    print(f"Error importing detection modules: {e}")
    print("Make sure human_detection.py, fall_detection.py, and vlm_analysis.py are in the same directory")
    sys.exit(1)

class CompleteElderMonitoringSystem:
    """Complete Elder Monitoring System: Human Detection + Fall Detection + VLM Analysis"""
    
    def __init__(self, camera_id=0, openai_api_key=None):
        self.camera_id = camera_id
        self.cap = None
        self.running = False
        
        # Validate API key
        if not openai_api_key:
            print("Warning: No OpenAI API key provided. VLM analysis will be disabled.")
            print("To enable VLM analysis, provide your OpenAI API key when creating the system.")
            self.vlm_enabled = False
        else:
            self.vlm_enabled = True
        
        # Initialize all detection systems
        print("Initializing Complete Elder Monitoring System...")
        self.human_system = HumanDetectionSystem(show_gui=False)
        self.fall_system = FallDetectionSystem(show_gui=False)
        
        if self.vlm_enabled:
            self.vlm_system = VLMAnalysisSystem(
                api_key=openai_api_key, 
                analysis_interval=60,  # Exactly 1 frame per minute
                show_gui=False,
                save_frames=True,  # Enable frame saving
                frames_folder="vlm_analyzed_frames"  # Folder name
            )
        else:
            self.vlm_system = None
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.frame_count = 0
        
        # Complete system status
        self.monitoring_status = {
            'timestamp': None,
            'human_detected': False,
            'fall_detected': False,
            'human_confidence': 0.0,
            'fall_confidence': 0.0,
            'human_activity': 'No analysis yet',
            'hazard_detected': False,
            'hazard_description': 'No hazards detected',
            'vlm_confidence': 0.0,
            'landmark_count': 0,
            'triggered_rules': [],
            'bbox': None,
            'frames_processed': 0,
            'fps': 0.0,
            'vlm_enabled': self.vlm_enabled,
            'last_vlm_analysis': None,
            'time_until_next_vlm': 60
        }
        
        # Window name
        self.window_name = "Complete Elder Monitoring System"
        
        print("Complete Elder Monitoring System initialized successfully!")
        if self.vlm_enabled:
            print("VLM Analysis: ENABLED (exactly 1 frame per minute)")
        else:
            print("VLM Analysis: DISABLED (no API key provided)")
    
    def initialize_camera(self):
        """Initialize camera"""
        print(f"Initializing camera {self.camera_id}...")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Camera {self.camera_id} not available")
            return False
        
        # Configure camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("Camera initialized successfully!")
        return True
    
    def process_frame(self, frame):
        """Process frame through all systems following the complete workflow"""
        current_time = time.time()
        self.frame_count += 1
        
        # Step 1: Run Human Detection
        human_result = self.human_system.detect_human(frame)
        
        # Initialize results
        fall_result = {
            'fall_detected': False,
            'confidence': 0.0,
            'triggered_rules': [],
            'velocity': 0,
            'bbox': None,
            'fall_info': None
        }
        
        vlm_result = {
            'human_activity': self.monitoring_status['human_activity'],
            'hazard_detected': self.monitoring_status['hazard_detected'],
            'hazard_description': self.monitoring_status['hazard_description'],
            'analysis_confidence': self.monitoring_status['vlm_confidence'],
            'last_analysis_time': self.monitoring_status['last_vlm_analysis'],
            'time_until_next_analysis': self.monitoring_status['time_until_next_vlm']
        }
        
        # Step 2: Run Fall Detection only if human is detected
        if human_result['human_detected'] and human_result['pose_landmarks']:
            
            # Prepare frame data for fall detection
            frame_data = {
                'frame': frame,
                'timestamp': current_time,
                'frame_count': self.frame_count
            }
            
            # Run fall detection
            try:
                fall_detection_result = self.fall_system.process_frame(frame_data)
                
                # Extract fall information safely
                if fall_detection_result and isinstance(fall_detection_result, dict):
                    fall_info = fall_detection_result.get('fall_info', {})
                    fall_result = {
                        'fall_detected': fall_detection_result.get('fall_detected', False),
                        'confidence': fall_detection_result.get('confidence', 0.0),
                        'triggered_rules': fall_info.get('triggered_rules', []) if fall_info else [],
                        'velocity': fall_info.get('velocity', 0) if fall_info else 0,
                        'bbox': fall_detection_result.get('bbox'),
                        'fall_info': fall_info
                    }
                else:
                    print(f"Fall detection returned invalid result: {fall_detection_result}")
            except Exception as e:
                print(f"Error in fall detection: {e}")
        
        # Step 3: Run VLM Analysis (exactly 1 frame per minute) ONLY if human detected
        if self.vlm_enabled and self.vlm_system:
            try:
                # Pass human detection status to VLM system
                vlm_result = self.vlm_system.process_frame(frame, human_detected=human_result['human_detected'])
            except Exception as e:
                print(f"Error in VLM analysis: {e}")
        
        # Update complete monitoring status
        self.monitoring_status.update({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'human_detected': human_result['human_detected'],
            'fall_detected': fall_result['fall_detected'],
            'human_confidence': human_result['detection_confidence'],
            'fall_confidence': fall_result['confidence'],
            'human_activity': vlm_result['human_activity'],
            'hazard_detected': vlm_result['hazard_detected'],
            'hazard_description': vlm_result['hazard_description'],
            'vlm_confidence': vlm_result['analysis_confidence'],
            'landmark_count': human_result['landmark_count'],
            'triggered_rules': fall_result['triggered_rules'],
            'bbox': human_result['bbox'] or fall_result['bbox'],
            'frames_processed': self.frame_count,
            'fps': self.current_fps,
            'last_vlm_analysis': vlm_result['last_analysis_time'],
            'time_until_next_vlm': vlm_result.get('time_until_next_analysis', 60)
        })
        
        # Print alerts
        if fall_result['fall_detected']:
            print(f"🚨 FALL ALERT! Frame {self.frame_count} - Confidence: {fall_result['confidence']:.3f} - Rules: {fall_result['triggered_rules']}")
        
        if vlm_result['hazard_detected'] and vlm_result['hazard_description'] != 'No hazards detected':
            print(f"⚠️ HAZARD ALERT! {vlm_result['hazard_description']}")
        
        return human_result, fall_result, vlm_result
    
    def render_complete_display(self, frame, human_result, fall_result, vlm_result):
        """Render clean display with sidebar for information"""
        h, w = frame.shape[:2]
        
        # Create extended frame with sidebar
        sidebar_width = 300
        extended_frame = np.zeros((h, w + sidebar_width, 3), dtype=np.uint8)
        
        # Place original frame on the left
        extended_frame[:h, :w] = frame
        
        # Draw pose landmarks if human detected (only on main frame, no text overlay)
        if human_result['pose_landmarks'] and human_result['human_detected']:
            import mediapipe as mp
            mp_drawing = mp.solutions.drawing_utils
            mp_pose = mp.solutions.pose
            
            # Change colors based on alerts
            if fall_result['fall_detected'] or vlm_result['hazard_detected']:
                landmark_color = (0, 0, 255)  # Red for danger
                connection_color = (255, 0, 0)
            else:
                landmark_color = (0, 255, 0)  # Green for normal
                connection_color = (0, 255, 255)
            
            mp_drawing.draw_landmarks(
                extended_frame[:h, :w], human_result['pose_landmarks'], mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=connection_color, thickness=2, circle_radius=2)
            )
        
        # Draw bounding box (only on main frame)
        bbox = self.monitoring_status['bbox']
        if bbox:
            if fall_result['fall_detected'] or vlm_result['hazard_detected']:
                color = (0, 0, 255)  # Red for any danger
            elif human_result['human_detected']:
                color = (0, 255, 0)  # Green for normal human
            else:
                color = (0, 0, 255)  # Red for no human
            cv2.rectangle(extended_frame[:h, :w], (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Create dark sidebar background
        sidebar_bg_color = (40, 40, 40)
        extended_frame[:, w:] = sidebar_bg_color
        
        # Sidebar content
        sidebar_x = w + 10
        y_pos = 30
        line_height = 25
        
        # Title
        cv2.putText(extended_frame, "Elder Monitor", (sidebar_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_height + 10
        
        # Current Date and Time
        current_datetime = datetime.now()
        date_str = current_datetime.strftime('%Y-%m-%d')
        time_str = current_datetime.strftime('%H:%M:%S')
        cv2.putText(extended_frame, f"Date: {date_str}", (sidebar_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_pos += line_height
        cv2.putText(extended_frame, f"Time: {time_str}", (sidebar_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_pos += line_height
        
        # System uptime
        uptime_seconds = int(time.time() - self.fps_start_time + self.frame_count/30)  # Approximate uptime
        uptime_mins = uptime_seconds // 60
        uptime_secs = uptime_seconds % 60
        cv2.putText(extended_frame, f"Uptime: {uptime_mins:02d}:{uptime_secs:02d}", (sidebar_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_pos += line_height + 10
        
        # Human Detection Status
        human_status = "DETECTED" if self.monitoring_status['human_detected'] else "NOT DETECTED"
        human_color = (0, 255, 0) if self.monitoring_status['human_detected'] else (0, 0, 255)
        cv2.putText(extended_frame, "HUMAN STATUS:", (sidebar_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_pos += 18
        cv2.putText(extended_frame, human_status, (sidebar_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, human_color, 2)
        y_pos += line_height
        
        cv2.putText(extended_frame, f"Confidence: {self.monitoring_status['human_confidence']:.2f}", (sidebar_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_pos += line_height + 10
        
        # Fall Detection Status
        cv2.putText(extended_frame, "FALL STATUS:", (sidebar_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_pos += 18
        
        if self.monitoring_status['human_detected']:
            fall_status = "FALL DETECTED!" if self.monitoring_status['fall_detected'] else "NORMAL"
            fall_color = (0, 0, 255) if self.monitoring_status['fall_detected'] else (0, 255, 0)
            cv2.putText(extended_frame, fall_status, (sidebar_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, fall_color, 2)
            y_pos += line_height
            cv2.putText(extended_frame, f"Confidence: {self.monitoring_status['fall_confidence']:.2f}", (sidebar_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        else:
            cv2.putText(extended_frame, "WAITING...", (sidebar_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        y_pos += line_height + 10
        
        # VLM Analysis Status
        cv2.putText(extended_frame, "VLM ANALYSIS:", (sidebar_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_pos += 18
        
        if self.vlm_enabled:
            # Show different status based on human detection
            if not self.monitoring_status['human_detected']:
                cv2.putText(extended_frame, "PAUSED", (sidebar_x, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)  # Orange color
                y_pos += line_height
                cv2.putText(extended_frame, "(No human detected)", (sidebar_x, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
                y_pos += 15
                
                # Show next analysis countdown
                next_vlm = int(self.monitoring_status['time_until_next_vlm'])
                cv2.putText(extended_frame, f"Next when human: {next_vlm}s", (sidebar_x, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            else:
                # Activity (complete text with multiple lines) - only when human detected
                activity = self.monitoring_status['human_activity']
                max_chars_per_line = 28
                
                # Split activity into multiple lines
                activity_lines = []
                for i in range(0, len(activity), max_chars_per_line):
                    line = activity[i:i+max_chars_per_line]
                    # Try to break at word boundaries
                    if i + max_chars_per_line < len(activity) and activity[i+max_chars_per_line] != ' ':
                        last_space = line.rfind(' ')
                        if last_space > max_chars_per_line * 0.7:  # If space is reasonably close to end
                            activity_lines.append(line[:last_space])
                            i = i + last_space + 1 - max_chars_per_line  # Adjust index for next iteration
                        else:
                            activity_lines.append(line)
                    else:
                        activity_lines.append(line)
                
                # Display activity lines (max 4 lines to fit in sidebar)
                for i, line in enumerate(activity_lines[:4]):
                    cv2.putText(extended_frame, line.strip(), (sidebar_x, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                    y_pos += 15
                
                # If more than 4 lines, show "..."
                if len(activity_lines) > 4:
                    cv2.putText(extended_frame, "...", (sidebar_x, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                    y_pos += 15
                
                y_pos += 5
                
                # Hazard status
                hazard_status = "HAZARD DETECTED!" if self.monitoring_status['hazard_detected'] else "SAFE"
                hazard_color = (0, 0, 255) if self.monitoring_status['hazard_detected'] else (0, 255, 0)
                cv2.putText(extended_frame, hazard_status, (sidebar_x, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, hazard_color, 1)
                y_pos += line_height
                
                # Hazard description if there's a hazard
                if self.monitoring_status['hazard_detected'] and self.monitoring_status['hazard_description'] != 'No hazards detected':
                    hazard_desc = self.monitoring_status['hazard_description']
                    # Split hazard description into lines
                    hazard_lines = []
                    for i in range(0, len(hazard_desc), max_chars_per_line):
                        hazard_lines.append(hazard_desc[i:i+max_chars_per_line])
                    
                    for line in hazard_lines[:2]:  # Max 2 lines for hazard description
                        cv2.putText(extended_frame, line.strip(), (sidebar_x, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
                        y_pos += 15
                
                y_pos += 5
                
                # VLM timing information
                next_vlm = int(self.monitoring_status['time_until_next_vlm'])
                cv2.putText(extended_frame, f"Next analysis: {next_vlm}s", (sidebar_x, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                y_pos += 15
                
                if self.monitoring_status['last_vlm_analysis']:
                    last_analysis = self.monitoring_status['last_vlm_analysis'][-8:]
                    cv2.putText(extended_frame, f"Last: {last_analysis}", (sidebar_x, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        else:
            cv2.putText(extended_frame, "DISABLED", (sidebar_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            y_pos += line_height
            cv2.putText(extended_frame, "(No API Key)", (sidebar_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
        
        # Alert flash effect for main frame only
        if self.monitoring_status['fall_detected'] or self.monitoring_status['hazard_detected']:
            overlay = extended_frame[:h, :w].copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            extended_frame[:h, :w] = cv2.addWeighted(extended_frame[:h, :w], 0.9, overlay, 0.1, 0)
        
        cv2.imshow(self.window_name, extended_frame)
    
    def calculate_fps(self):
        """Calculate FPS"""
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            time_diff = current_time - self.fps_start_time
            self.current_fps = self.fps_counter / time_diff
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def print_system_status(self):
        """Print detailed system status"""
        print("\n" + "="*70)
        print("COMPLETE ELDER MONITORING SYSTEM STATUS")
        print("="*70)
        print(f"Timestamp: {self.monitoring_status['timestamp']}")
        print(f"Human Detected: {self.monitoring_status['human_detected']}")
        print(f"Human Confidence: {self.monitoring_status['human_confidence']:.3f}")
        print(f"Landmark Count: {self.monitoring_status['landmark_count']}")
        print(f"Fall Detected: {self.monitoring_status['fall_detected']}")
        print(f"Fall Confidence: {self.monitoring_status['fall_confidence']:.3f}")
        print(f"Triggered Rules: {self.monitoring_status['triggered_rules']}")
        print(f"Human Activity: {self.monitoring_status['human_activity']}")
        print(f"Hazard Detected: {self.monitoring_status['hazard_detected']}")
        print(f"Hazard Description: {self.monitoring_status['hazard_description']}")
        print(f"VLM Confidence: {self.monitoring_status['vlm_confidence']:.3f}")
        print(f"Last VLM Analysis: {self.monitoring_status['last_vlm_analysis']}")
        print(f"Time Until Next VLM: {int(self.monitoring_status['time_until_next_vlm'])}s")
        print(f"Frames Processed: {self.monitoring_status['frames_processed']}")
        print(f"Current FPS: {self.monitoring_status['fps']:.1f}")
        print(f"VLM Enabled: {self.monitoring_status['vlm_enabled']}")
        print("="*70 + "\n")
    
    def get_workflow_output(self):
        """Get final output in the exact format specified in workflow"""
        return {
            "timestamp": self.monitoring_status['timestamp'],
            "human_detected": self.monitoring_status['human_detected'],
            "fall_alert": self.monitoring_status['fall_detected'],
            "human_activity": self.monitoring_status['human_activity'],
            "hazard_detected": self.monitoring_status['hazard_detected'],
            "hazard_description": self.monitoring_status['hazard_description'],
            "frames_processed": self.monitoring_status['frames_processed'],
            "fps": self.monitoring_status['fps']
        }
    
    def run(self):
        """Main execution loop"""
        if not self.initialize_camera():
            return False
        
        # Start VLM processing if enabled
        if self.vlm_enabled and self.vlm_system:
            self.vlm_system.start_processing()
        
        print("\nComplete Elder Monitoring System Running")
        print("Workflow: Human Detection → Fall Detection → VLM Analysis (1 frame/60s)")
        print("Press 'q' to quit, 's' for status, 'h'/'f'/'v' for individual system status")
        print("-" * 80)
        
        self.running = True
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Mirror frame for better user experience
                frame = cv2.flip(frame, 1)
                
                # Process frame through all systems
                human_result, fall_result, vlm_result = self.process_frame(frame)
                
                # Render complete display
                self.render_complete_display(frame, human_result, fall_result, vlm_result)
                
                # Update FPS
                self.fps_counter += 1
                self.calculate_fps()
                
                # Print live output every 30 frames (1 second at 30 FPS)
                if self.frame_count % 30 == 0:
                    output = self.get_workflow_output()
                    activity_short = output['human_activity'][:25] + "..." if len(output['human_activity']) > 25 else output['human_activity']
                    print(f"Live: Human={output['human_detected']}, "
                          f"Fall={output['fall_alert']}, "
                          f"Activity='{activity_short}', "
                          f"Hazard={output['hazard_detected']}, "
                          f"FPS={output['fps']:.1f}")
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord('s'):
                    self.print_system_status()
                elif key == ord('h'):
                    # Show human detection system status
                    human_status = self.human_system.get_status()
                    print(f"Human System: {human_status}")
                elif key == ord('f'):
                    # Show fall detection system status
                    fall_status = self.fall_system.get_status()
                    print(f"Fall System: {fall_status}")
                elif key == ord('v'):
                    # Show VLM system status
                    if self.vlm_enabled:
                        vlm_status = self.vlm_system.get_status()
                        print(f"VLM System: {vlm_status}")
                    else:
                        print("VLM System: DISABLED (no API key)")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup all resources"""
        print("Cleaning up complete system...")
        self.running = False
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        # Cleanup all detection systems
        if hasattr(self, 'human_system'):
            self.human_system.cleanup()
        if hasattr(self, 'fall_system'):
            self.fall_system.cleanup()
        if hasattr(self, 'vlm_system') and self.vlm_system:
            self.vlm_system.cleanup()
        
        # Close windows
        cv2.destroyAllWindows()
        print("Complete system cleanup complete")

def main():
    """Load configuration and start the complete monitoring system."""
    print("Starting Complete Elder Monitoring System")
    print("Modules: human_detection.py + fall_detection.py + vlm_analysis.py")
    print("Orchestrated by: main.py")
    
    # Load local development settings from .env. Existing system environment
    # variables take precedence, which is useful in CI and production.
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("\nWARNING: OPENAI_API_KEY is not set.")
        print("Copy .env.example to .env and add a newly generated OpenAI API key.")
        print("System will run without VLM analysis (Human Detection + Fall Detection only)...")
    
    try:
        # Create complete system (change camera_id if needed)
        system = CompleteElderMonitoringSystem(
            camera_id=0, 
            openai_api_key=openai_api_key
        )
        system.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("System shutdown complete")

if __name__ == "__main__":
    main()
