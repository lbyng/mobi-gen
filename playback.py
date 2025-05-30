import time
import sys
import numpy as np
import zmq
import threading
import h5py
import os
import argparse

import config

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
import unitree_legged_const as go2
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

class HDF5Replayer:
    def __init__(self):
        # Go2 low level control
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        
        # HDF5 data
        self.hdf5_data = None
        self.actions = None
        self.num_samples = 0
        self.current_index = 0
        
        # Action dimensions
        self.go2_action_dim = 36
        self.d1_action_dim = 7
        self.total_action_dim = 43
        
        # Replay control
        self.start_time = None
        self.is_playing = False
        self.stop_event = threading.Event()
        
        # Rate control
        self.data_dt = 1/50.0  # Will be read from file
        
        # Go2 thread
        self.go2_cmdThreadPtr = None
        self.crc = CRC()
        
        # D1 socket
        self.zmq_context = None
        self.zmq_socket = None
                
        # Control gains
        self.default_kp = 70.0
        self.default_kd = 5.0

    def init(self):
        print("Initializing HDF5 replayer...")
        
        # Initialize Go2
        self.init_low_cmd()
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        
        self.sc = SportClient()  
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        
        # Check and release current mode
        status, result = self.msc.CheckMode()
        while result['name']:
            print(f"Releasing current mode: {result['name']}")
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)
            
        # Initialize D1 mechanical arm socket
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.bind("tcp://*:5555")
        
        print("Initialization complete")
        print("ZMQ publisher bound to tcp://*:5555")

    def init_low_cmd(self):
        """Initialize low-level command structure"""
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q = go2.PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = go2.VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def load_hdf5(self, filename):
        """Load HDF5 file"""
        try:
            print(f"Loading HDF5 file: {filename}")
            self.hdf5_data = h5py.File(filename, 'r')
            
            if 'action' not in self.hdf5_data:
                print("Error: No action data in HDF5 file")
                return False
                
            self.actions = self.hdf5_data['action'][:]
            self.num_samples = self.actions.shape[0]
            
            print(f"Successfully loaded HDF5 data: {self.num_samples} samples")
            print(f"Action data dimensions: {self.actions.shape}")
            
            # Read dt from file
            if 'dt' in self.hdf5_data.attrs:
                self.data_dt = float(self.hdf5_data.attrs['dt'])
                print(f"Using dt from file: {self.data_dt:.4f}s ({1/self.data_dt:.1f}Hz)")
            
            # Validate expected format
            if self.actions.shape[1] != 43:
                print(f"Warning: Expected 43D actions (36D Go2 + 7D D1), got {self.actions.shape[1]}D")
            
            duration = self.num_samples * self.data_dt
            print(f"Data duration: {duration:.2f} seconds")
            
            self.current_index = 0
            return True
            
        except Exception as e:
            print(f"Error loading HDF5 data: {e}")
            return False

    def start_replay(self):
        """Start replaying data"""
        if self.hdf5_data is None:
            print("Error: Please load HDF5 data first")
            return False
            
        print("Preparing to start replay...")
        print("Warning: Make sure there are no obstacles around the robot and it's ready to move!")
        input("Press Enter to start replay...")
            
        self.is_playing = True
        self.stop_event.clear()
        self.start_time = time.time()
        self.current_index = 0
        
        # Start Go2 thread at file's original frequency
        go2_thread_rate = 1.0/self.data_dt
        self.go2_cmdThreadPtr = RecurrentThread(
            interval=1.0/go2_thread_rate,
            target=self.go2_command_thread,
            name="go2_replay_cmd_thread"
        )
        self.go2_cmdThreadPtr.Start()
        
        # Start D1 thread
        self.d1_thread = threading.Thread(target=self.d1_replay_thread, name="d1_replay_thread")
        self.d1_thread.daemon = True
        self.d1_thread.start()
        print("D1 replay thread started")
        
        print(f"Replay started at rate: {go2_thread_rate:.1f}Hz")
        return True
            
    def stop_replay(self):
        """Stop replay"""
        if not self.is_playing:
            return
        
        print("\nStopping replay...")
        
        self.is_playing = False
        self.stop_event.set()
        
        # Wait for D1 thread to complete
        if hasattr(self, 'd1_thread') and self.d1_thread.is_alive():
            self.d1_thread.join(timeout=2.0)
        
        # Switch Go2 to AI mode
        try:
            self.msc.SelectMode('ai')
            print("AI motion mode enabled")
            time.sleep(2)
        except Exception as e:
            print(f"Warning: Could not switch to AI mode: {e}")
        
        print("Replay stopped")

    def go2_command_thread(self):
        """Go2 command thread"""
        if not self.is_playing or self.current_index >= self.num_samples:
            return
               
        # Get action data for current index
        current_action = self.actions[self.current_index]
        
        # Extract Go2 data (first 36 dimensions)
        positions = current_action[:12]
        velocities = current_action[12:24]
        torques = current_action[24:36]
        
        kp_values = np.full(12, self.default_kp)
        kd_values = np.full(12, self.default_kd)
        
        if self.current_index == 0:
            print(f"kp={self.default_kp}, kd={self.default_kd}")
        
        # Update commands
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = float(positions[i])
            self.low_cmd.motor_cmd[i].dq = float(velocities[i])
            self.low_cmd.motor_cmd[i].kp = float(kp_values[i])
            self.low_cmd.motor_cmd[i].kd = float(kd_values[i])
            self.low_cmd.motor_cmd[i].tau = float(torques[i])
            
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)
        
        # Update index based on elapsed time
        elapsed_time = time.time() - self.start_time
        target_index = int(elapsed_time * (1.0 / self.data_dt))
        
        if target_index > self.current_index:
            self.current_index = min(target_index, self.num_samples - 1)
            
            # Progress display
            if self.current_index % 50 == 0:
                progress = (self.current_index / self.num_samples) * 100
                elapsed = elapsed_time
                remaining = (self.num_samples - self.current_index) * self.data_dt
                print(f"\rReplay: {progress:.1f}% | Index: {self.current_index}/{self.num_samples} | "
                      f"Time: {elapsed:.1f}s | Remaining: {remaining:.1f}s", end="", flush=True)
        
        # Check for completion
        if self.current_index >= self.num_samples - 1:
            print("\nReplay complete")
            self.stop_replay()

    def d1_replay_thread(self):
        """D1 mechanical arm replay thread """        
        print(f"D1 replay thread started (7DOF)")
        
        try:
            last_sent_index = -1
            
            while self.is_playing and self.current_index < self.num_samples:
                current_index = self.current_index
                
                if current_index > last_sent_index:
                    # Extract D1 data (dimensions 36-42)
                    d1_command = self.actions[current_index, 36:43].tolist()
                    
                    # Send D1 command
                    msg = {"positions": d1_command}
                    try:
                        self.zmq_socket.send_json(msg, zmq.NOBLOCK)
                        last_sent_index = current_index
                    except zmq.Again:
                        pass  # Skip if can't send immediately
                    except Exception as e:
                        print(f"Error sending D1 command: {e}")
                
                # Sleep at original data rate
                time.sleep(self.data_dt)
                
                if self.stop_event.is_set():
                    break
            
        except Exception as e:
            print(f"D1 replay thread error: {e}")
        finally:
            print("D1 replay thread ended")

    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        
        # Stop Go2 thread
        if self.go2_cmdThreadPtr:
            self.go2_cmdThreadPtr = None
        
        # Close HDF5 file
        if self.hdf5_data is not None:
            self.hdf5_data.close()
        
        # Close ZMQ socket
        if self.zmq_socket:
            self.zmq_socket.close()
        
        # Terminate ZMQ context
        if self.zmq_context:
            self.zmq_context.term()
        
        print("Cleanup complete")

def main():
    parser = argparse.ArgumentParser(description='HDF5 Replayer')
    parser.add_argument('hdf5_file', type=str, help='Path to HDF5 file to replay')    
    args = parser.parse_args()
    
    if not os.path.exists(args.hdf5_file):
        print(f"Error: HDF5 file not found {args.hdf5_file}")
        sys.exit(1)
    
    # Initialize Unitree SDK
    ChannelFactoryInitialize(0, config.NETWORK_INTERFACE)
    
    # Create replayer
    replayer = HDF5Replayer()
    replayer.init()
    
    # Load data
    if not replayer.load_hdf5(args.hdf5_file):
        print("Failed to load data, exiting")
        replayer.cleanup()
        sys.exit(1)
    
    try:
        # Start replay
        replayer.start_replay()
        
        # Wait for replay to complete
        while replayer.is_playing:
            time.sleep(0.1)
        
        print("Replay finished")
        print("Press Enter to exit")
        input()
        
    except KeyboardInterrupt:
        print("\nUser interrupt")
        if replayer.is_playing:
            replayer.stop_replay()
    finally:
        replayer.cleanup()
        print("Program exit")

if __name__ == '__main__':
    main()