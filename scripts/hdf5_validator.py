#!/usr/bin/env python3

import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse

def validate_hdf5_file(file_path):
    """Validate HDF5 file structure"""
    print(f"Validating: {os.path.basename(file_path)}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Check data lengths for consistency
            lengths = {}
            
            # Check action data
            if 'action' in f:
                action = f['action']
                lengths['action'] = len(action)
                total_dim = action.shape[1]
                
                # Validate expected 43D format
                if total_dim != 43:
                    print(f"⚠️  WARNING: Expected 43D actions (36D Go2 + 7D D1), got {total_dim}D")
                else:
                    print(f"✅ Action format: 43D (36D Go2 + 7D D1)")
            else:
                print("❌ FAIL: Missing 'action' dataset")
                return False
            
            # Check observations
            if 'observations' not in f:
                print("❌ FAIL: Missing 'observations' group")
                return False
            
            obs = f['observations']
            
            # Check Go2 base observations
            if 'base' in obs and 'joint_positions' in obs['base']:
                lengths['base_joints'] = len(obs['base']['joint_positions'])
                go2_joints = obs['base']['joint_positions'].shape[1]
                if go2_joints != 12:
                    print(f"⚠️  WARNING: Expected 12 Go2 joints, got {go2_joints}")
            
            # Check D1 arm observations  
            if 'arm' in obs and 'joint_positions' in obs['arm']:
                lengths['arm_joints'] = len(obs['arm']['joint_positions'])
                d1_joints = obs['arm']['joint_positions'].shape[1]
                if d1_joints != 7:
                    print(f"⚠️  WARNING: Expected 7 D1 joints, got {d1_joints}")
            
            # Check camera data
            if 'images' in obs:
                for cam_name in obs['images'].keys():
                    lengths[f'camera_{cam_name}'] = len(obs['images'][cam_name])
            
            # Check length consistency
            if len(set(lengths.values())) > 1:
                print("❌ FAIL: Data length inconsistency")
                for name, length in lengths.items():
                    print(f"     {name}: {length}")
                return False
            
            # Basic info
            data_length = list(lengths.values())[0]
            file_size_mb = os.path.getsize(file_path) / (1024*1024)
            
            # Get metadata
            dt = f.attrs.get('dt', 0.02)
            duration = data_length * dt
            
            print(f"✅ PASS: {data_length} timesteps, {duration:.1f}s, {file_size_mb:.1f}MB")
            
            # Show available data types
            data_types = []
            if 'base' in obs:
                data_types.append("Go2")
            if 'arm' in obs:
                data_types.append("D1")
            if 'images' in obs:
                cam_list = list(obs['images'].keys())
                data_types.append(f"Cameras({','.join(cam_list)})")
            
            print(f"     Data: {' + '.join(data_types)}")
            print(f"     Action: Go2(36D: pos+vel+torque) + D1(7D: joints)")
            
            return True
            
    except Exception as e:
        print(f"❌ FAIL: Error reading file - {str(e)}")
        return False

def visualize_robot_trajectories(file_path, save_plots=False):
    """Visualize robot trajectories"""
    print(f"Generating trajectory plots...")
    
    with h5py.File(file_path, 'r') as f:
        obs = f['observations']
        action = f['action'][:]
        
        # Create figure
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        time_steps = np.arange(len(action))
        
        # Go2 joint positions (observations)
        if 'base' in obs and 'joint_positions' in obs['base']:
            joint_pos = obs['base']['joint_positions'][:]
            
            ax1 = fig.add_subplot(gs[0, 0])
            for i in range(12):
                ax1.plot(time_steps, joint_pos[:, i], alpha=0.7, linewidth=0.8)
            ax1.set_title('Go2 Joint Positions (Obs)')
            ax1.set_xlabel('Time Steps')
            ax1.set_ylabel('Position (rad)')
            ax1.grid(True, alpha=0.3)
            
            # IMU orientation
            if 'orientation' in obs['base']:
                ax2 = fig.add_subplot(gs[0, 1])
                orientation = obs['base']['orientation'][:]
                labels = ['qw', 'qx', 'qy', 'qz']
                for i in range(4):
                    ax2.plot(time_steps, orientation[:, i], label=labels[i], alpha=0.8)
                ax2.set_title('Go2 IMU Orientation')
                ax2.set_xlabel('Time Steps')
                ax2.set_ylabel('Quaternion')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # D1 arm joint positions (observations)
        if 'arm' in obs and 'joint_positions' in obs['arm']:
            arm_pos = obs['arm']['joint_positions'][:]
            
            ax3 = fig.add_subplot(gs[0, 2])
            for i in range(7):
                ax3.plot(time_steps, arm_pos[:, i], label=f'J{i}', alpha=0.7)
            ax3.set_title('D1 Arm Joint Positions (Obs)')
            ax3.set_xlabel('Time Steps')
            ax3.set_ylabel('Position (rad)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Go2 position actions (0-11)
        ax4 = fig.add_subplot(gs[1, 0])
        go2_pos_actions = action[:, :12]
        for i in range(12):
            ax4.plot(time_steps, go2_pos_actions[:, i], alpha=0.6, linewidth=0.8)
        ax4.set_title('Go2 Position Actions')
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Target Position (rad)')
        ax4.grid(True, alpha=0.3)
        
        # Go2 velocity actions (12-23)
        ax5 = fig.add_subplot(gs[1, 1])
        go2_vel_actions = action[:, 12:24]
        for i in range(12):
            ax5.plot(time_steps, go2_vel_actions[:, i], alpha=0.6, linewidth=0.8)
        ax5.set_title('Go2 Velocity Actions')
        ax5.set_xlabel('Time Steps')
        ax5.set_ylabel('Target Velocity (rad/s)')
        ax5.grid(True, alpha=0.3)
        
        # Go2 torque actions (24-35)
        ax6 = fig.add_subplot(gs[1, 2])
        go2_torque_actions = action[:, 24:36]
        for i in range(12):
            ax6.plot(time_steps, go2_torque_actions[:, i], alpha=0.6, linewidth=0.8)
        ax6.set_title('Go2 Torque Actions')
        ax6.set_xlabel('Time Steps')
        ax6.set_ylabel('Target Torque (Nm)')
        ax6.grid(True, alpha=0.3)
        
        # D1 arm actions (36-42)
        ax7 = fig.add_subplot(gs[2, 0])
        d1_actions = action[:, 36:43]
        for i in range(7):
            ax7.plot(time_steps, d1_actions[:, i], label=f'J{i}', alpha=0.7)
        ax7.set_title('D1 Arm Actions (7DOF)')
        ax7.set_xlabel('Time Steps')
        ax7.set_ylabel('Target Position (rad)')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Action magnitude analysis
        ax8 = fig.add_subplot(gs[2, 1])
        go2_magnitude = np.linalg.norm(action[:, :36], axis=1)
        d1_magnitude = np.linalg.norm(action[:, 36:43], axis=1)
        ax8.plot(time_steps, go2_magnitude, label='Go2', alpha=0.8)
        ax8.plot(time_steps, d1_magnitude, label='D1', alpha=0.8)
        ax8.set_title('Action Magnitude')
        ax8.set_xlabel('Time Steps')
        ax8.set_ylabel('L2 Norm')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # Statistics
        ax9 = fig.add_subplot(gs[2, 2])
        dt = f.attrs.get('dt', 0.02)
        stats_text = f"""Episode Statistics:

Timesteps: {len(time_steps)}
Duration: {len(time_steps)*dt:.1f}s
Sample Rate: {1/dt:.1f}Hz
File size: {os.path.getsize(file_path)/(1024*1024):.1f}MB

Action Structure:
  Total: 43D
  Go2: 36D
    - Positions: 0-11
    - Velocities: 12-23  
    - Torques: 24-35
  D1: 7D
    - Positions: 36-42"""
        
        ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace')
        ax9.axis('off')
        
        plt.suptitle(f'Go2+D1 Trajectory Analysis - {os.path.basename(file_path)}', fontsize=14)
        
        if save_plots:
            plot_path = file_path.replace('.hdf5', '_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {plot_path}")
        
        plt.show()

def visualize_camera_data(file_path, save_images=False, max_frames=6):
    """Visualize camera data"""
    print(f"Generating camera visualization...")
    
    with h5py.File(file_path, 'r') as f:
        if 'observations' not in f or 'images' not in f['observations']:
            print("No camera data found")
            return
        
        images_group = f['observations']['images']
        camera_names = list(images_group.keys())
        
        if not camera_names:
            print("No camera data found")
            return
        
        # Create visualization
        num_cameras = len(camera_names)
        total_frames = len(images_group[camera_names[0]])
        num_frames_to_show = min(max_frames, total_frames)
        
        # Select frames evenly distributed
        frame_indices = np.linspace(0, total_frames-1, num_frames_to_show, dtype=int)
        
        fig, axes = plt.subplots(num_cameras, num_frames_to_show, 
                                figsize=(num_frames_to_show*2.5, num_cameras*2.5))
        
        if num_cameras == 1:
            axes = axes.reshape(1, -1)
        if num_frames_to_show == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'Camera Data - {os.path.basename(file_path)}', fontsize=14)
        
        for cam_idx, cam_name in enumerate(camera_names):
            cam_data = images_group[cam_name]
            
            for frame_idx, actual_frame in enumerate(frame_indices):
                ax = axes[cam_idx, frame_idx]
                
                # Read and display image
                img = cam_data[actual_frame]
                
                if img.dtype != np.uint8:
                    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                
                ax.imshow(img)
                ax.set_title(f'{cam_name}\n#{actual_frame}')
                ax.axis('off')
                
                # Save individual images if requested
                if save_images:
                    img_save_path = file_path.replace('.hdf5', f'_{cam_name}_{actual_frame:04d}.jpg')
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(img_save_path, img_bgr)
        
        plt.tight_layout()
        
        if save_images:
            plot_path = file_path.replace('.hdf5', '_cameras.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Camera plot saved: {plot_path}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='HDF5 Validator for Go2+D1 (43D format)')
    parser.add_argument('file_path', type=str, help='HDF5 file path')
    parser.add_argument('--save-plots', action='store_true', help='Save plots')
    parser.add_argument('--save-images', action='store_true', help='Save images')
    parser.add_argument('--max-frames', type=int, default=6, help='Max frames to show')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualizations')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"❌ File not found: {args.file_path}")
        return
    
    try:
        # Validate file
        is_valid = validate_hdf5_file(args.file_path)
        
        if not is_valid:
            print(f"\n❌ File failed validation")
            return
        
        # Show visualizations if requested
        if not args.no_viz:
            print(f"\nGenerating visualizations...")
            visualize_robot_trajectories(args.file_path, args.save_plots)
            visualize_camera_data(args.file_path, args.save_images, args.max_frames)
        
        print(f"\n✅ File ready for training")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == '__main__':
    main()