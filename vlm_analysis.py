import cv2
import time
import base64
import sqlite3
import pytz
import json
import os
from datetime import datetime
from typing import Dict, Optional
import threading
import queue
from dotenv import load_dotenv

# OpenAI imports
try:
    from openai import OpenAI
except ImportError:
    print("OpenAI library not installed. Install with: pip install openai")
    exit(1)

class VLMAnalysisSystem:
    """Vision Language Model Analysis System using OpenAI API"""
    
    def __init__(self, api_key: str, analysis_interval=60, show_gui=False, save_frames=True, frames_folder="vlm_frames"):
        self.api_key = api_key
        self.analysis_interval = analysis_interval  # seconds (60 = 1 minute)
        self.show_gui = show_gui
        self.save_frames = save_frames
        self.frames_folder = frames_folder
        
        # Create frames folder if saving is enabled
        if self.save_frames:
            self.setup_frames_folder()
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize database
        self.init_database()
        
        # Timing control
        self.last_analysis_time = 0
        self.frame_count = 0
        
        # Status tracking
        self.current_status = {
            'human_activity': 'No analysis yet',
            'hazard_detected': False,
            'hazard_description': 'No hazards detected',
            'analysis_confidence': 0.0,
            'last_analysis_time': None,
            'frames_processed': 0,
            'api_calls_made': 0,
            'frames_saved': 0
        }
        
        # Analysis queue for processing
        self.analysis_queue = queue.Queue()
        self.processing_thread = None
        self.running = False
        
        print("VLM Analysis System initialized")
        if self.save_frames:
            print(f"Frame saving enabled - folder: {self.frames_folder}")
    
    def setup_frames_folder(self):
        """Create folder structure for saving frames"""
        try:
            if not os.path.exists(self.frames_folder):
                os.makedirs(self.frames_folder)
                print(f"Created frames folder: {self.frames_folder}")
            
            # Create subfolders for organization
            date_folder = os.path.join(self.frames_folder, datetime.now().strftime('%Y-%m-%d'))
            if not os.path.exists(date_folder):
                os.makedirs(date_folder)
                print(f"Created date folder: {date_folder}")
                
        except Exception as e:
            print(f"Error creating frames folder: {e}")
            self.save_frames = False
    
    def save_frame_to_disk(self, frame, analysis_result=None):
        """Save frame to disk with timestamp and analysis info"""
        if not self.save_frames:
            return None
            
        try:
            # Create filename with timestamp
            current_time = datetime.now()
            date_str = current_time.strftime('%Y-%m-%d')
            time_str = current_time.strftime('%H-%M-%S')
            
            # Create date subfolder
            date_folder = os.path.join(self.frames_folder, date_str)
            if not os.path.exists(date_folder):
                os.makedirs(date_folder)
            
            # Filename format: YYYY-MM-DD_HH-MM-SS_frame_XXXX.jpg
            filename = f"{date_str}_{time_str}_frame_{self.current_status['api_calls_made']:04d}.jpg"
            filepath = os.path.join(date_folder, filename)
            
            # Save frame
            success = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            if success:
                self.current_status['frames_saved'] += 1
                print(f"Frame saved: {filename}")
                
                # Save analysis metadata if available
                if analysis_result:
                    self.save_analysis_metadata(filepath, analysis_result)
                
                return filepath
            else:
                print(f"Failed to save frame: {filename}")
                return None
                
        except Exception as e:
            print(f"Error saving frame: {e}")
            return None
    
    def save_analysis_metadata(self, image_filepath, analysis_result):
        """Save analysis metadata as JSON file alongside the image"""
        try:
            # Create metadata filename
            base_name = os.path.splitext(image_filepath)[0]
            metadata_filepath = f"{base_name}_metadata.json"
            
            # Prepare metadata
            metadata = {
                'image_file': os.path.basename(image_filepath),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'analysis': {
                    'human_activity': analysis_result.get('human_activity', ''),
                    'hazard_detected': analysis_result.get('hazard_detected', False),
                    'hazard_description': analysis_result.get('hazard_description', ''),
                    'emotional_state': analysis_result.get('emotional_state', ''),
                    'confidence': analysis_result.get('confidence', 0.0),
                    'additional_notes': analysis_result.get('additional_notes', ''),
                    'api_call_success': analysis_result.get('api_call_success', False),
                    'processing_time': analysis_result.get('processing_time', 0.0)
                }
            }
            
            # Save metadata as JSON
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"Metadata saved: {os.path.basename(metadata_filepath)}")
            
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def init_database(self):
        """Initialize database for VLM analysis logs with separate columns"""
        try:
            self.conn = sqlite3.connect('vlm_analysis_logs.db', check_same_thread=False)
            self.cursor = self.conn.cursor()
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS vlm_analysis_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    human_activity TEXT,
                    hazard_detected BOOLEAN,
                    hazard_description TEXT,
                    emotional_state TEXT,
                    confidence REAL,
                    additional_notes TEXT,
                    raw_response TEXT,
                    api_call_success BOOLEAN,
                    processing_time REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.commit()
            print("VLM analysis database initialized with column-wise storage")
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def encode_frame_to_base64(self, frame):
        """Convert frame to base64 for OpenAI API"""
        try:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            # Convert to base64
            base64_image = base64.b64encode(buffer).decode('utf-8')
            return base64_image
        except Exception as e:
            print(f"Error encoding frame: {e}")
            return None
    
    def analyze_frame_with_openai(self, base64_image):
        """Send frame to OpenAI API for analysis"""
        start_time = time.time()
        api_success = False
        
        try:
            # Create the prompt for elder monitoring
            prompt = """
            Analyze this image for elder monitoring purposes. Please provide:
            
            1. HUMAN ACTIVITY: Describe what the person is currently doing (e.g., sitting, standing, walking, lying down, cooking, reading, etc.)
            
            2. HAZARD ASSESSMENT: Look for any dangerous situations such as:
               - Person has fallen or is falling
               - Sharp objects or obstacles in path
               - Spilled liquids on floor
               - Fire or smoke
               - Person appears distressed or in pain
               - Unsafe positioning (climbing, reaching dangerously)
               - Medical emergency signs
               - Environmental hazards
            
            3. EMOTIONAL STATE: Does the person appear calm, distressed, confused, or in need of help?
            
            Please respond in this exact JSON format:
            {
                "human_activity": "brief description of current activity",
                "hazard_detected": true/false,
                "hazard_description": "description of hazard if detected, or 'No hazards detected'",
                "emotional_state": "description of person's apparent emotional state",
                "confidence": 0.8,
                "additional_notes": "any other relevant observations"
            }
            """
            
            # Make API call to OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use gpt-4o-mini for cost efficiency, or "gpt-4o" for better accuracy
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.1  # Low temperature for consistent analysis
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            api_success = True
            
            # Parse response
            raw_response = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                # Extract JSON from response (in case there's extra text)
                json_start = raw_response.find('{')
                json_end = raw_response.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = raw_response[json_start:json_end]
                    analysis_result = json.loads(json_str)
                else:
                    # Fallback parsing if JSON format is not found
                    analysis_result = {
                        "human_activity": "Unable to parse activity from response",
                        "hazard_detected": False,
                        "hazard_description": "Unable to assess hazards",
                        "emotional_state": "Unknown",
                        "confidence": 0.0,
                        "additional_notes": raw_response[:200] if raw_response else "No response"
                    }
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                # Handle non-JSON responses
                analysis_result = {
                    "human_activity": "Response parsing error",
                    "hazard_detected": False,
                    "hazard_description": "Could not parse hazard assessment",
                    "emotional_state": "Unknown",
                    "confidence": 0.0,
                    "additional_notes": raw_response[:200] if raw_response else "No response"
                }
            
            # Add metadata
            analysis_result['raw_response'] = raw_response
            analysis_result['api_call_success'] = api_success
            analysis_result['processing_time'] = processing_time
            
            return analysis_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            print(f"OpenAI API error: {error_msg}")
            
            return {
                "human_activity": f"API Error: {error_msg[:50]}",
                "hazard_detected": False,
                "hazard_description": "API call failed",
                "emotional_state": "Unknown",
                "confidence": 0.0,
                "additional_notes": f"Error: {error_msg}",
                "raw_response": f"API Error: {error_msg}",
                "api_call_success": False,
                "processing_time": processing_time
            }
    
    def process_analysis_queue(self):
        """Background thread to process analysis requests"""
        while self.running:
            try:
                # Get frame from queue (with timeout)
                frame_data = self.analysis_queue.get(timeout=1)
                if frame_data is None:
                    continue
                
                frame = frame_data['frame']
                timestamp = frame_data['timestamp']
                
                print(f"Processing VLM analysis for frame at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}")
                
                # Save frame to disk BEFORE analysis (in case API fails)
                saved_filepath = None
                if self.save_frames:
                    saved_filepath = self.save_frame_to_disk(frame)
                
                # Encode frame for OpenAI
                base64_image = self.encode_frame_to_base64(frame)
                if base64_image is None:
                    continue
                
                # Analyze with OpenAI
                analysis_result = self.analyze_frame_with_openai(base64_image)
                
                # Update status
                self.current_status.update({
                    'human_activity': analysis_result.get('human_activity', 'Unknown activity'),
                    'hazard_detected': analysis_result.get('hazard_detected', False),
                    'hazard_description': analysis_result.get('hazard_description', 'No assessment available'),
                    'analysis_confidence': analysis_result.get('confidence', 0.0),
                    'last_analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'api_calls_made': self.current_status['api_calls_made'] + 1
                })
                
                # Save analysis metadata if frame was saved
                if saved_filepath and analysis_result:
                    self.save_analysis_metadata(saved_filepath, analysis_result)
                
                # Log to database
                self.log_analysis(analysis_result)
                
                # Print analysis result
                print(f"\nVLM Analysis Result:")
                print(f"Activity: {analysis_result.get('human_activity', 'Unknown')}")
                print(f"Hazard: {analysis_result.get('hazard_detected', False)} - {analysis_result.get('hazard_description', 'N/A')}")
                print(f"Confidence: {analysis_result.get('confidence', 0.0):.2f}")
                print(f"Emotional State: {analysis_result.get('emotional_state', 'Unknown')}")
                print(f"Processing Time: {analysis_result.get('processing_time', 0.0):.2f}s")
                if saved_filepath:
                    print(f"Frame saved: {os.path.basename(saved_filepath)}")
                
                # Mark task as done
                self.analysis_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in analysis processing: {e}")
    
    def log_analysis(self, analysis_result):
        """Log analysis result to database with separate columns"""
        try:
            slst_tz = pytz.timezone('Asia/Colombo')
            current_time = datetime.now(slst_tz).strftime('%Y-%m-%d %H:%M:%S')
            
            self.cursor.execute('''
                INSERT INTO vlm_analysis_logs 
                (human_activity, hazard_detected, hazard_description, emotional_state, 
                 confidence, additional_notes, raw_response, api_call_success, 
                 processing_time, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_result.get('human_activity', ''),
                analysis_result.get('hazard_detected', False),
                analysis_result.get('hazard_description', ''),
                analysis_result.get('emotional_state', ''),
                analysis_result.get('confidence', 0.0),
                analysis_result.get('additional_notes', ''),
                analysis_result.get('raw_response', ''),
                analysis_result.get('api_call_success', False),
                analysis_result.get('processing_time', 0.0),
                current_time
            ))
            self.conn.commit()
            
            print(f"VLM analysis logged to database successfully")
            
        except Exception as e:
            print(f"VLM logging error: {e}")
    
    def process_frame(self, frame, human_detected=False):
        """Process frame - analyze exactly one frame per minute ONLY when human is detected"""
        current_time = time.time()
        self.frame_count += 1
        
        # Update frames processed
        self.current_status['frames_processed'] = self.frame_count
        
        # Only proceed with VLM analysis if human is detected
        if not human_detected:
            # Calculate time until next potential analysis
            time_until_next = max(0, self.analysis_interval - (current_time - self.last_analysis_time))
            
            return {
                'human_activity': 'No human detected - VLM analysis paused',
                'hazard_detected': False,
                'hazard_description': 'No analysis - waiting for human detection',
                'analysis_confidence': 0.0,
                'last_analysis_time': self.current_status['last_analysis_time'],
                'frames_since_analysis': self.frame_count - (self.current_status['api_calls_made'] * (self.analysis_interval * 30)),
                'time_until_next_analysis': time_until_next
            }
        
        # Human is detected - check if it's time for analysis (exactly every 60 seconds)
        if current_time - self.last_analysis_time >= self.analysis_interval:
            print(f"\nProcessing VLM analysis for Frame {self.frame_count} (Human detected - 1 frame/minute)...")
            
            # Clear any existing items in queue to ensure only one frame is processed
            while not self.analysis_queue.empty():
                try:
                    self.analysis_queue.get_nowait()
                    self.analysis_queue.task_done()
                except queue.Empty:
                    break
            
            # Add only this current frame to analysis queue
            frame_data = {
                'frame': frame.copy(),
                'timestamp': current_time,
                'human_detected': True
            }
            
            # Add to queue (non-blocking)
            try:
                self.analysis_queue.put_nowait(frame_data)
                self.last_analysis_time = current_time
                print(f"VLM analysis queued - Next analysis in {self.analysis_interval} seconds (when human present)")
            except queue.Full:
                print("Analysis queue is full, clearing and adding current frame")
                # Force clear and add current frame
                while not self.analysis_queue.empty():
                    try:
                        self.analysis_queue.get_nowait()
                        self.analysis_queue.task_done()
                    except queue.Empty:
                        break
                self.analysis_queue.put_nowait(frame_data)
                self.last_analysis_time = current_time
        
        # Calculate time until next analysis
        time_until_next = max(0, self.analysis_interval - (current_time - self.last_analysis_time))
        
        # Return current status
        return {
            'human_activity': self.current_status['human_activity'],
            'hazard_detected': self.current_status['hazard_detected'],
            'hazard_description': self.current_status['hazard_description'],
            'analysis_confidence': self.current_status['analysis_confidence'],
            'last_analysis_time': self.current_status['last_analysis_time'],
            'frames_since_analysis': self.frame_count - (self.current_status['api_calls_made'] * (self.analysis_interval * 30)),
            'time_until_next_analysis': time_until_next
        }
    
    def start_processing(self):
        """Start the background processing thread"""
        self.running = True
        self.processing_thread = threading.Thread(target=self.process_analysis_queue, daemon=True)
        self.processing_thread.start()
        print("VLM processing thread started")
    
    def stop_processing(self):
        """Stop the background processing thread"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        print("VLM processing thread stopped")
    
    def get_status(self):
        """Get current system status"""
        return self.current_status.copy()
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.stop_processing()
            if hasattr(self, 'conn'):
                self.conn.close()
            print("VLM Analysis System cleaned up")
        except Exception as e:
            print(f"VLM cleanup error: {e}")

# Run a camera-based smoke test when this module is executed directly.
if __name__ == "__main__":
    print("Testing VLM Analysis System standalone...")

    # Read the secret from the local .env file or the process environment.
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        exit(1)
    
    # Test with camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera available for testing")
        exit(1)
    
    # Create system with shorter interval for testing (10 seconds instead of 60)
    system = VLMAnalysisSystem(api_key=api_key, analysis_interval=10, show_gui=True)
    system.start_processing()
    
    try:
        print("Starting VLM analysis test...")
        print("Frame will be analyzed every 10 seconds")
        print("Controls: Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Mirror frame for better user experience
            frame = cv2.flip(frame, 1)
            
            # Process frame
            result = system.process_frame(frame)
            
            # Display frame with analysis info
            cv2.putText(frame, f"Activity: {result['human_activity'][:30]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Hazard: {result['hazard_detected']}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if not result['hazard_detected'] else (0, 0, 255), 2)
            cv2.putText(frame, f"Last Analysis: {result['last_analysis_time'] or 'None'}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow("VLM Analysis Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        system.cleanup()
        cv2.destroyAllWindows()
        print("VLM Analysis System test complete")
