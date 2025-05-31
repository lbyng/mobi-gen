import time
import sys
import numpy as np
import zmq
import json
import threading
import cv2
import os
import h5py
from datetime import datetime
from scipy import interpolate
from tqdm import tqdm

import utils.config as config


class D1DataRecorder:
    def __init__(self):
        self.recording = False
        self.start_time = None
        
        # HDF5 output directory
        self.output_dir = config.HDF5_DIR
        
        # Base velocity data from spacemouse - raw
        self.base_velocities_raw = []  # [x, y, yaw]
        self.base_velocities_timestamps_raw = []
        
        # D1 robotic arm data storage - raw
        self.d1_command_positions_raw = []    
        self.d1_command_timestamps_raw = []   
        self.d1_actual_positions_raw = []     
        self.d1_actual_timestamps_raw = []    
        
        # Camera frame storage - raw
        self.front_camera_frames_raw = []           
        self.front_camera_timestamps_raw = []
        self.wrist_camera_frames_raw = []           
        self.wrist_camera_timestamps_raw = []       
        
        # Synchronized data storage
        self.sync_timestamps = []
        self.base_velocities = []
        self.d1_command_positions = []
        self.d1_actual_positions = []
        self.front_camera_frames = []
        self.wrist_camera_frames = []
        
        # Recording counters and status
        self.base_velocity_count = 0
        self.d1_command_count = 0
        self.d1_actual_count = 0
        self.front_camera_frame_count = 0
        self.wrist_camera_frame_count = 0
        
        # ZMQ context and sockets
        self.zmq_context = zmq.Context()
        self.zmq_spacemouse_socket = None
        self.zmq_command_socket = None
        self.zmq_actual_socket = None
        self.zmq_front_camera_socket = None
        self.zmq_wrist_camera_socket = None
        
        # Threads
        self.zmq_spacemouse_thread = None
        self.zmq_command_thread = None
        self.zmq_actual_thread = None
        self.zmq_front_camera_thread = None
        self.zmq_wrist_camera_thread = None
        self.stop_event = threading.Event()
        
        # Synchronization
        self.sync_sample_rate = config.SYNC_RATE
        self.sync_interval = 1.0 / self.sync_sample_rate
        
        # Camera params
        self.save_camera_frames = True
        self.display_camera_frames = config.CAMERA_DISPLAY

    def Init(self):
        # Initialize ZMQ connection for spacemouse data (6000)
        self.zmq_spacemouse_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_spacemouse_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.zmq_spacemouse_socket.setsockopt(zmq.RCVHWM, 1)
        self.zmq_spacemouse_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_spacemouse_socket.connect(config.GO2_CMD_ADDRESS)
        
        # Initialize ZMQ connection for D1 arm commands (5555)
        self.zmq_command_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_command_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.zmq_command_socket.setsockopt(zmq.RCVHWM, 1)
        self.zmq_command_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_command_socket.connect(config.D1_CMD_ADDRESS)
        
        # Initialize ZMQ connection for D1 arm actual positions (5556)
        self.zmq_actual_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_actual_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.zmq_actual_socket.setsockopt(zmq.RCVHWM, 1) 
        self.zmq_actual_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_actual_socket.connect(config.D1_ACT_ADDRESS)
        
        # Initialize ZMQ connection for front camera (5557)
        self.zmq_front_camera_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_front_camera_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.zmq_front_camera_socket.setsockopt(zmq.RCVTIMEO, 1000)
        self.zmq_front_camera_socket.setsockopt(zmq.RCVHWM, 1)
        self.zmq_front_camera_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_front_camera_socket.connect(config.FRONT_CAMERA_ADDRESS)
        
        # Initialize ZMQ connection for wrist camera (5558)
        self.zmq_wrist_camera_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_wrist_camera_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.zmq_wrist_camera_socket.setsockopt(zmq.RCVTIMEO, 1000)
        self.zmq_wrist_camera_socket.setsockopt(zmq.RCVHWM, 1)
        self.zmq_wrist_camera_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_wrist_camera_socket.connect(config.WRIST_CAMERA_ADDRESS)
        
        print(f"Data will be synchronized to {self.sync_sample_rate}Hz and saved directly to HDF5")        
        
        # Test camera connections
        try:
            self.zmq_front_camera_socket.recv(flags=zmq.NOBLOCK)
            print("Front camera connected successfully!")
        except zmq.Again:
            print("Waiting for front camera data...")
            
        try:
            self.zmq_wrist_camera_socket.recv(flags=zmq.NOBLOCK)
            print("Wrist camera connected successfully!")
        except zmq.Again:
            print("Waiting for wrist camera data...")

    def SpacemouseThread(self):
        """Thread to receive base velocity from spacemouse."""
        while not self.stop_event.is_set() and self.recording:
            try:
                try:
                    data = self.zmq_spacemouse_socket.recv_json(flags=zmq.NOBLOCK)
                    
                    # Extract x, y, yaw velocities
                    velocities = [data["x"], data["y"], data["yaw"]]
                    
                    # Record velocities and timestamp
                    self.base_velocities_raw.append(velocities)
                    self.base_velocities_timestamps_raw.append(time.time() - self.start_time)
                    self.base_velocity_count += 1
                    
                except zmq.Again:
                    pass
                except Exception as e:
                    print(f"Spacemouse thread error: {e}")

                time.sleep(0.001)  # Small sleep
                
            except Exception as e:
                print(f"Spacemouse thread error: {str(e)}")
        
        print("Spacemouse thread ended")

    def D1CommandThread(self):
        """Thread to receive and record D1 command positions."""
        while not self.stop_event.is_set() and self.recording:
            try:
                try:
                    msg = self.zmq_command_socket.recv_string(flags=zmq.NOBLOCK)
                    data = json.loads(msg)
                    
                    if "positions" in data:
                        # Record joint positions and timestamp
                        self.d1_command_positions_raw.append(data["positions"])
                        self.d1_command_timestamps_raw.append(time.time() - self.start_time)
                        self.d1_command_count += 1
                except zmq.Again:
                    pass
                except json.JSONDecodeError:
                    pass

                time.sleep(0.001)  # Small sleep
                
            except Exception as e:
                print(f"D1 command thread error: {str(e)}")
        
        print("D1 command thread ended")

    def D1ActualThread(self):
        """Thread to receive and record D1 actual positions."""
        while not self.stop_event.is_set() and self.recording:
            try:
                try:
                    msg_raw = self.zmq_actual_socket.recv_string(flags=zmq.NOBLOCK)
                    if msg_raw.startswith("{\"joint_positions\":"):
                        try:
                            data = json.loads(msg_raw)
                            if "joint_positions" in data:
                                self.d1_actual_positions_raw.append(data["joint_positions"])
                                self.d1_actual_timestamps_raw.append(time.time() - self.start_time)
                                self.d1_actual_count += 1
                        except json.JSONDecodeError:
                            pass
                except zmq.Again:
                    pass
                
                time.sleep(0.001)  # Small sleep
                
            except Exception as e:
                print(f"D1 actual thread error: {str(e)}")
        
        print("D1 actual thread ended")

    def FrontCameraThread(self):
        """Thread to receive and record front camera frames."""
        # For FPS calculation
        fps_frame_count = 0
        fps_start_time = time.time()
        fps = 0
        display_window_created = False
        
        # Main thread loop
        while not self.stop_event.is_set() and self.recording:
            try:
                try:
                    jpg_buffer = self.zmq_front_camera_socket.recv(flags=zmq.NOBLOCK)
                    
                    img_array = np.frombuffer(jpg_buffer, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        current_time = time.time()
                        fps_frame_count += 1
                        fps_elapsed = current_time - fps_start_time
                        
                        if fps_elapsed >= 1.0:
                            fps = fps_frame_count / fps_elapsed
                            fps_frame_count = 0
                            fps_start_time = current_time
                        
                        # Save frame
                        if self.save_camera_frames:
                            self.front_camera_frames_raw.append(frame.copy())
                            self.front_camera_timestamps_raw.append(current_time - self.start_time)
                            self.front_camera_frame_count += 1
                        
                        # Display frame if enabled
                        if self.display_camera_frames:
                            display_frame = frame.copy()
                            cv2.putText(
                                display_frame, 
                                f"Front Camera - Frame: {self.front_camera_frame_count} FPS: {fps:.1f}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                            )
                            
                            if not display_window_created:
                                cv2.namedWindow('Front Camera', cv2.WINDOW_NORMAL)
                                display_window_created = True
                                
                            cv2.imshow('Front Camera', display_frame)
                            cv2.waitKey(1)
                    
                except zmq.Again:
                    pass
                
                time.sleep(0.001)  # Small sleep
                
            except Exception as e:
                print(f"Front camera thread error: {str(e)}")
        
        print("Front camera thread ended")

    def WristCameraThread(self):
        """Thread to receive and record wrist camera frames."""
        # For FPS calculation
        fps_frame_count = 0
        fps_start_time = time.time()
        fps = 0
        display_window_created = False
        
        # Main thread loop
        while not self.stop_event.is_set() and self.recording:
            try:
                try:
                    jpg_buffer = self.zmq_wrist_camera_socket.recv(flags=zmq.NOBLOCK)
                    
                    img_array = np.frombuffer(jpg_buffer, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        current_time = time.time()
                        fps_frame_count += 1
                        fps_elapsed = current_time - fps_start_time
                        
                        if fps_elapsed >= 1.0:
                            fps = fps_frame_count / fps_elapsed
                            fps_frame_count = 0
                            fps_start_time = current_time
                        
                        # Save frame
                        if self.save_camera_frames:
                            self.wrist_camera_frames_raw.append(frame.copy())
                            self.wrist_camera_timestamps_raw.append(current_time - self.start_time)
                            self.wrist_camera_frame_count += 1
                        
                        # Display frame if enabled
                        if self.display_camera_frames:
                            display_frame = frame.copy()
                            cv2.putText(
                                display_frame, 
                                f"Wrist Camera - Frame: {self.wrist_camera_frame_count} FPS: {fps:.1f}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                            )
                            
                            if not display_window_created:
                                cv2.namedWindow('Wrist Camera', cv2.WINDOW_NORMAL)
                                display_window_created = True
                                
                            cv2.imshow('Wrist Camera', display_frame)
                            cv2.waitKey(1)
                    
                except zmq.Again:
                    pass
                
                time.sleep(0.001)  # Small sleep
                
            except Exception as e:
                print(f"Wrist camera thread error: {str(e)}")
        
        print("Wrist camera thread ended")

    def UpdateStatus(self):
        """Update recording status"""
        if self.recording:
            elapsed = time.time() - self.start_time
            base_rate = self.base_velocity_count / elapsed if elapsed > 0 else 0
            d1_cmd_rate = self.d1_command_count / elapsed if elapsed > 0 else 0
            d1_act_rate = self.d1_actual_count / elapsed if elapsed > 0 else 0
            front_cam_rate = self.front_camera_frame_count / elapsed if elapsed > 0 else 0
            wrist_cam_rate = self.wrist_camera_frame_count / elapsed if elapsed > 0 else 0
            
            status = f"\rRec: {elapsed:.1f}s | Base vel: {self.base_velocity_count} ({base_rate:.1f}Hz) | "
            status += f"D1 cmd: {self.d1_command_count} ({d1_cmd_rate:.1f}Hz) | "
            status += f"D1 act: {self.d1_actual_count} ({d1_act_rate:.1f}Hz) | "
            status += f"Front: {self.front_camera_frame_count} ({front_cam_rate:.1f}fps) | "
            status += f"Wrist: {self.wrist_camera_frame_count} ({wrist_cam_rate:.1f}fps)"
            
            print(status, end='', flush=True)

    def StartRecording(self):
        if self.recording:
            print("Already recording...")
            return
        
        # Clear previous data
        self._clear_all_data()
        
        # Start recording
        self.start_time = time.time()
        self.recording = True
        
        # Reset stop event and start threads
        self.stop_event.clear()
        
        # Start thread for spacemouse velocities
        self.zmq_spacemouse_thread = threading.Thread(target=self.SpacemouseThread)
        self.zmq_spacemouse_thread.daemon = True
        self.zmq_spacemouse_thread.start()
        
        # Start thread for D1 command positions
        self.zmq_command_thread = threading.Thread(target=self.D1CommandThread)
        self.zmq_command_thread.daemon = True
        self.zmq_command_thread.start()
        
        # Start thread for D1 actual positions
        self.zmq_actual_thread = threading.Thread(target=self.D1ActualThread)
        self.zmq_actual_thread.daemon = True
        self.zmq_actual_thread.start()
        
        # Start thread for front camera
        self.zmq_front_camera_thread = threading.Thread(target=self.FrontCameraThread)
        self.zmq_front_camera_thread.daemon = True
        self.zmq_front_camera_thread.start()
        
        # Start thread for wrist camera
        self.zmq_wrist_camera_thread = threading.Thread(target=self.WristCameraThread)
        self.zmq_wrist_camera_thread.daemon = True
        self.zmq_wrist_camera_thread.start()
        
        print("Starting data recording (will save directly to HDF5)...")

    def _clear_all_data(self):
        """Clear all data arrays"""
        # Clear raw data
        self.base_velocities_raw = []
        self.base_velocities_timestamps_raw = []
        
        self.d1_command_positions_raw = []
        self.d1_command_timestamps_raw = []
        self.d1_actual_positions_raw = []
        self.d1_actual_timestamps_raw = []
        
        self.front_camera_frames_raw = []
        self.front_camera_timestamps_raw = []
        self.wrist_camera_frames_raw = []
        self.wrist_camera_timestamps_raw = []
        
        # Clear synchronized data
        self.sync_timestamps = []
        self.base_velocities = []
        self.d1_command_positions = []
        self.d1_actual_positions = []
        self.front_camera_frames = []
        self.wrist_camera_frames = []
        
        # Reset counters
        self.base_velocity_count = 0
        self.d1_command_count = 0
        self.d1_actual_count = 0
        self.front_camera_frame_count = 0
        self.wrist_camera_frame_count = 0

    def StopRecording(self, save_data=True):
        if not self.recording:
            print("No active recording...")
            return
        
        # Stop recording
        self.recording = False
        self.stop_event.set()
        
        # Wait for threads to finish
        threads = [
            (self.zmq_spacemouse_thread, "Spacemouse"),
            (self.zmq_command_thread, "D1 command"),
            (self.zmq_actual_thread, "D1 actual"),
            (self.zmq_front_camera_thread, "Front camera"),
            (self.zmq_wrist_camera_thread, "Wrist camera")
        ]
        
        for thread, name in threads:
            if thread and thread.is_alive():
                print(f"Waiting for {name} thread to finish...")
                thread.join(timeout=2.0)
        
        # Calculate total elapsed time from raw data
        total_elapsed = self._calculate_total_elapsed_time()
        
        # Print recording statistics
        self._print_recording_statistics(total_elapsed)
        
        if not save_data:
            print("Data discarded as requested.")
            return
        
        # Synchronize data to common timeline
        print(f"Synchronizing all data to {self.sync_sample_rate}Hz timeline...")
        self.SynchronizeData()
        
        # Save data if any was recorded
        has_data = (self.base_velocity_count > 0 or self.d1_command_count > 0 or 
                   self.d1_actual_count > 0 or self.front_camera_frame_count > 0 or 
                   self.wrist_camera_frame_count > 0)
        
        if has_data:
            hdf5_file = self.SaveToHDF5()
            if hdf5_file:
                print(f"Data successfully saved to: {hdf5_file}")
            else:
                print("Failed to save HDF5 file")
        else:
            print("No data recorded, not saving file.")

    def _calculate_total_elapsed_time(self):
        """Calculate total recording duration from all data sources"""
        total_elapsed = 0
        if len(self.base_velocities_timestamps_raw) > 0:
            total_elapsed = max(total_elapsed, self.base_velocities_timestamps_raw[-1])
        if len(self.d1_command_timestamps_raw) > 0:
            total_elapsed = max(total_elapsed, self.d1_command_timestamps_raw[-1])
        if len(self.d1_actual_timestamps_raw) > 0:
            total_elapsed = max(total_elapsed, self.d1_actual_timestamps_raw[-1])
        if len(self.front_camera_timestamps_raw) > 0:
            total_elapsed = max(total_elapsed, self.front_camera_timestamps_raw[-1])
        if len(self.wrist_camera_timestamps_raw) > 0:
            total_elapsed = max(total_elapsed, self.wrist_camera_timestamps_raw[-1])
        return total_elapsed

    def _print_recording_statistics(self, total_elapsed):
        """Print recording statistics"""
        # Calculate actual sampling rates from raw data
        base_sample_rate = self.base_velocity_count / total_elapsed if total_elapsed > 0 else 0
        d1_cmd_sample_rate = self.d1_command_count / total_elapsed if total_elapsed > 0 else 0
        d1_act_sample_rate = self.d1_actual_count / total_elapsed if total_elapsed > 0 else 0
        front_camera_rate = self.front_camera_frame_count / total_elapsed if total_elapsed > 0 else 0
        wrist_camera_rate = self.wrist_camera_frame_count / total_elapsed if total_elapsed > 0 else 0
        
        print("")
        print(f"Recording stopped. Total time: {total_elapsed:.2f} seconds")
        print(f"Raw data collection rates:")
        print(f"- Base velocity: {self.base_velocity_count} samples ({base_sample_rate:.1f}Hz)")
        print(f"- D1 cmd: {self.d1_command_count} samples ({d1_cmd_sample_rate:.1f}Hz)")
        print(f"- D1 act: {self.d1_actual_count} samples ({d1_act_sample_rate:.1f}Hz)")
        print(f"- Front camera: {self.front_camera_frame_count} frames ({front_camera_rate:.1f}fps)")
        print(f"- Wrist camera: {self.wrist_camera_frame_count} frames ({wrist_camera_rate:.1f}fps)")

    def SynchronizeData(self):
        """Synchronize all data to a common timeline using interpolation"""
        # Check if we have any data to synchronize
        if not (len(self.base_velocities_timestamps_raw) > 0 or len(self.d1_command_timestamps_raw) > 0 or 
                len(self.d1_actual_timestamps_raw) > 0 or len(self.front_camera_timestamps_raw) > 0 or 
                len(self.wrist_camera_timestamps_raw) > 0):
            print("No data to synchronize")
            return
            
        # Find the earliest and latest timestamps across all data sources
        start_times = []
        end_times = []
        
        if len(self.base_velocities_timestamps_raw) > 0:
            start_times.append(self.base_velocities_timestamps_raw[0])
            end_times.append(self.base_velocities_timestamps_raw[-1])
            
        if len(self.d1_command_timestamps_raw) > 0:
            start_times.append(self.d1_command_timestamps_raw[0])
            end_times.append(self.d1_command_timestamps_raw[-1])
            
        if len(self.d1_actual_timestamps_raw) > 0:
            start_times.append(self.d1_actual_timestamps_raw[0])
            end_times.append(self.d1_actual_timestamps_raw[-1])
            
        if len(self.front_camera_timestamps_raw) > 0:
            start_times.append(self.front_camera_timestamps_raw[0])
            end_times.append(self.front_camera_timestamps_raw[-1])
            
        if len(self.wrist_camera_timestamps_raw) > 0:
            start_times.append(self.wrist_camera_timestamps_raw[0])
            end_times.append(self.wrist_camera_timestamps_raw[-1])
        
        # Use the latest start time and earliest end time to ensure all data is available
        sync_start = max(start_times) if start_times else 0
        sync_end = min(end_times) if end_times else 0
        
        if sync_end <= sync_start:
            print("WARNING: Invalid time range for synchronization")
            return
        
        # Create common timeline at sync Hz
        self.sync_timestamps = np.arange(sync_start, sync_end, self.sync_interval)
        
        # Synchronize base velocity data if available
        if len(self.base_velocities_timestamps_raw) > 1:
            print("Synchronizing base velocity data...")
            base_timestamps = np.array(self.base_velocities_timestamps_raw)
            base_velocities_array = np.array(self.base_velocities_raw)
            self.base_velocities = self._interpolate_array(base_timestamps, base_velocities_array, self.sync_timestamps)
        
        # Synchronize D1 command data if available
        if len(self.d1_command_timestamps_raw) > 1 and len(self.d1_command_positions_raw) > 0:
            print("Synchronizing D1 command data...")
            d1_cmd_timestamps = np.array(self.d1_command_timestamps_raw)
            d1_cmd_positions = np.array(self.d1_command_positions_raw)
            self.d1_command_positions = self._interpolate_array(d1_cmd_timestamps, d1_cmd_positions, self.sync_timestamps)
        
        # Synchronize D1 actual data if available
        if len(self.d1_actual_timestamps_raw) > 1 and len(self.d1_actual_positions_raw) > 0:
            print("Synchronizing D1 actual data...")
            d1_act_timestamps = np.array(self.d1_actual_timestamps_raw)
            d1_act_positions = np.array(self.d1_actual_positions_raw)
            self.d1_actual_positions = self._interpolate_array(d1_act_timestamps, d1_act_positions, self.sync_timestamps)
        
        # Synchronize camera frames if available
        if len(self.front_camera_timestamps_raw) > 1 and len(self.front_camera_frames_raw) > 0:
            print("Synchronizing front camera frames...")
            self.front_camera_frames = self._nearest_frames(self.front_camera_timestamps_raw, self.front_camera_frames_raw, self.sync_timestamps)
            
        if len(self.wrist_camera_timestamps_raw) > 1 and len(self.wrist_camera_frames_raw) > 0:
            print("Synchronizing wrist camera frames...")
            self.wrist_camera_frames = self._nearest_frames(self.wrist_camera_timestamps_raw, self.wrist_camera_frames_raw, self.sync_timestamps)
        
        print(f"Synchronized {len(self.sync_timestamps)} samples at {self.sync_sample_rate}Hz")

    def _interpolate_array(self, src_timestamps, src_values, target_timestamps):
        """Interpolate array data to target timestamps"""
        if len(src_timestamps) != len(src_values):
            print(f"ERROR: Timestamp and value arrays have different lengths: {len(src_timestamps)} vs {len(src_values)}")
            return []
            
        result = []
        # If src_values is 1D, use simple interpolation
        if len(src_values.shape) == 1:
            interp_func = interpolate.interp1d(src_timestamps, src_values, axis=0, 
                                              bounds_error=False, fill_value="extrapolate")
            result = interp_func(target_timestamps)
        # If src_values is 2D, interpolate each component
        elif len(src_values.shape) == 2:
            try:
                interp_func = interpolate.interp1d(src_timestamps, src_values, axis=0, 
                                                bounds_error=False, fill_value="extrapolate")
                result = interp_func(target_timestamps)
            except Exception as e:
                print(f"Interpolation error: {e}")
                # Fallback method: use nearest neighbor for each point
                result = np.zeros((len(target_timestamps), src_values.shape[1]))
                for i, t in enumerate(target_timestamps):
                    idx = np.abs(src_timestamps - t).argmin()
                    result[i] = src_values[idx]
        else:
            print(f"ERROR: Unsupported array shape for interpolation: {src_values.shape}")
            
        return result

    def _nearest_frames(self, src_timestamps, src_frames, target_timestamps):
        """Select nearest camera frames for target timestamps"""
        result = []
        
        # Check if we have enough source frames
        if len(src_timestamps) < 1 or len(src_frames) < 1:
            return result
            
        # Convert source timestamps to numpy array if not already
        src_timestamps_array = np.array(src_timestamps)
        
        for target_time in target_timestamps:
            # Find index of nearest timestamp
            idx = np.abs(src_timestamps_array - target_time).argmin()
            # Add the corresponding frame to the result
            result.append(src_frames[idx].copy())
            
        return result

    def SaveToHDF5(self):
        """Save synchronized data directly to HDF5 format"""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate timestamp filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"{timestamp}.hdf5")
        
        # Check if we have synchronized data
        if len(self.sync_timestamps) == 0:
            print("No synchronized data to save")
            return None
        
        num_samples = len(self.sync_timestamps)
        print(f"Saving {num_samples} synchronized samples to HDF5: {output_file}")
        
        try:
            with h5py.File(output_file, 'w') as f:
                # Metadata
                f.attrs['sim'] = False
                f.attrs['compress'] = False
                f.attrs['robot_type'] = 'd1_with_base_velocity'
                f.attrs['num_episodes'] = 1
                
                # === OBSERVATIONS ===
                obs_group = f.create_group('observations')
                
                # 1. Arm state (D1)
                if len(self.d1_actual_positions) > 0:
                    arm_group = obs_group.create_group('arm')
                    arm_group.create_dataset('joint_positions', data=np.array(self.d1_actual_positions))
                    print(f"Added D1 actual positions to observations: {np.array(self.d1_actual_positions).shape}")
                elif len(self.d1_command_positions) > 0:
                    arm_group = obs_group.create_group('arm')
                    arm_group.create_dataset('joint_positions', data=np.array(self.d1_command_positions))
                    print(f"Added D1 command positions to observations (no actual positions available): {np.array(self.d1_command_positions).shape}")
                
                # 2. Camera observations
                if len(self.front_camera_frames) > 0 or len(self.wrist_camera_frames) > 0:
                    images_group = obs_group.create_group('images')
                    
                    # Save front camera images
                    if len(self.front_camera_frames) > 0:
                        print("Processing front camera frames...")
                        front_frames_rgb = []
                        for frame in tqdm(self.front_camera_frames, desc="Converting front camera frames"):
                            # Convert BGR to RGB
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            front_frames_rgb.append(frame_rgb)
                        
                        front_array = np.array(front_frames_rgb)
                        images_group.create_dataset('front_image', data=front_array)
                        print(f"Added front camera images: {front_array.shape}")
                    
                    # Save wrist camera images
                    if len(self.wrist_camera_frames) > 0:
                        print("Processing wrist camera frames...")
                        wrist_frames_rgb = []
                        for frame in tqdm(self.wrist_camera_frames, desc="Converting wrist camera frames"):
                            # Convert BGR to RGB
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            wrist_frames_rgb.append(frame_rgb)
                        
                        wrist_array = np.array(wrist_frames_rgb)
                        images_group.create_dataset('wrist_image', data=wrist_array)
                        print(f"Added wrist camera images: {wrist_array.shape}")
                
                # === ACTIONS ===
                action_components = []
                action_description = []
                
                # Add base velocities (3D: x, y, yaw)
                if len(self.base_velocities) > 0:
                    base_vel_array = np.array(self.base_velocities, dtype=np.float32)
                    action_components.append(base_vel_array)
                    action_description.append(f"BaseVel({base_vel_array.shape[1]}D)")
                
                # Add D1 actions
                if len(self.d1_command_positions) > 0:
                    arm_actions = np.array(self.d1_command_positions, dtype=np.float32)
                    action_components.append(arm_actions)
                    action_description.append(f"D1({arm_actions.shape[1]}D)")
                
                # Combine all actions
                if len(action_components) > 0:
                    # Combine Base velocities and D1
                    combined_actions = np.concatenate(action_components, axis=1)
                    f.create_dataset('action', data=combined_actions)
                    
                    # Print action info
                    action_desc = " + ".join(action_description)
                    print(f"Added combined action: {combined_actions.shape} [{action_desc}]")
                    
                    if len(self.base_velocities) > 0:
                        print(f"  Base velocity format: [x_vel(0), y_vel(1), yaw_vel(2)]")
                    if len(self.d1_command_positions) > 0:
                        print(f"  D1 action format: [joint_positions(3-9)]")
                
                # === EPISODE INFO ===
                f.attrs['episode_length'] = num_samples
                f.attrs['dt'] = float(self.sync_timestamps[1] - self.sync_timestamps[0]) if len(self.sync_timestamps) > 1 else 0.0
                f.attrs['total_time'] = float(self.sync_timestamps[-1] - self.sync_timestamps[0]) if len(self.sync_timestamps) > 1 else 0.0
                f.attrs['sync_sample_rate'] = self.sync_sample_rate
                f.attrs['recorded_at'] = datetime.now().isoformat()
                
                # Add action dimensions for easy access
                f.attrs['total_action_dim'] = combined_actions.shape[1] if 'combined_actions' in locals() else 0
                f.attrs['base_velocity_dim'] = 3 if len(self.base_velocities) > 0 else 0
                f.attrs['d1_action_dim'] = 7 if len(self.d1_command_positions) > 0 else 0
                
                # Add observation dimensions
                if len(self.d1_command_positions) > 0:
                    f.attrs['d1_obs_dim'] = 7
                
                print(f"Episode info: {num_samples} steps, {f.attrs['total_time']:.2f}s, dt={f.attrs['dt']:.4f}s")
                print(f"Action structure: BaseVel(3D) + D1(7D) = Total({f.attrs['total_action_dim']}D)")
            
            return output_file
            
        except Exception as e:
            print(f"Error saving HDF5 file: {str(e)}")
            return None
    
    def Cleanup(self):
        if self.display_camera_frames:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            time.sleep(0.1)
        
        # Close ZMQ sockets
        sockets = [
            (self.zmq_spacemouse_socket, "Spacemouse"),
            (self.zmq_command_socket, "D1 command"),
            (self.zmq_actual_socket, "D1 actual"),
            (self.zmq_front_camera_socket, "Front camera"),
            (self.zmq_wrist_camera_socket, "Wrist camera")
        ]
        
        for socket, name in sockets:
            if socket:
                socket.close()
                print(f"Closed {name} socket")
        
        # Terminate ZMQ context
        if self.zmq_context:
            self.zmq_context.term()
            print("ZMQ context terminated")


def main():
    print(f"Data will be synchronized to {config.SYNC_RATE}Hz and saved to HDF5")

    # Create and initialize recorder
    recorder = D1DataRecorder()
    recorder.Init()

    print("Ready, press Enter to start a new recording...")

    try:
        while True:
            # Wait for user input
            user_input = input().strip().lower()
            
            if not recorder.recording:
                # Start recording regardless of input when not recording
                recorder.StartRecording()
                print("Recording... Options:")
                print("  - Press Enter to SAVE")
                print("  - Type 'discard' or 'd' to DISCARD")
            else:
                # Stop recording with different options
                if user_input in ['discard', 'd']:
                    print("Discarding recorded data...")
                    recorder.StopRecording(save_data=False)
                    print("\nReady, press Enter to start a new recording...")
                else:
                    # Default behavior: save data (Enter or any other input)
                    print("Saving recorded data...")
                    recorder.StopRecording(save_data=True)
                    print("\nReady, press Enter to start a new recording...")
                
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, shutting down...")
        if recorder.recording:
            print("Discarding current recording due to forced exit...")
            recorder.StopRecording(save_data=False)
        recorder.Cleanup()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("Program terminated by user")
        
    sys.exit(0)


if __name__ == "__main__":
    main()