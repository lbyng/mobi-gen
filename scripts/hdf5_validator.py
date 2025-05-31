#!/usr/bin/env python3

import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse

def validate_hdf5_file(file_path):
    """Validate HDF5 file structure for D1 mobile manipulation"""
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
                
                # Validate expected 10D format
                if total_dim != 10:
                    print(f"⚠️  WARNING: Expected 10D actions (3D base velocity + 7D D1), got {total_dim}D")
                else:
                    print(f"✅ Action format: 10D (3D base velocity + 7D D1 positions)")
            else:
                print("❌ FAIL: Missing 'action' dataset")
                return False
            
            # Check observations
            if 'observations' not in f:
                print("❌ FAIL: Missing 'observations' group")
                return False
            
            obs = f['observations']
            
            # Check D1 arm observations  
            if 'arm' in obs and 'joint_positions' in obs['arm']:
                lengths['arm_joints'] = len(obs['arm']['joint_positions'])
                d1_joints = obs['arm']['joint_positions'].shape[1]
                if d1_joints != 7:
                    print(f"⚠️  WARNING: Expected 7 D1 joints, got {d1_joints}")
                else:
                    print(f"✅ Observation format: 7D (D1 joint positions)")
            else:
                print("❌ FAIL: Missing D1 arm observations")
                return False
            
            # Check camera data
            if 'images' in obs:
                for cam_name in obs['images'].keys():
                    lengths[f'camera_{cam_name}'] = len(obs['images'][cam_name])
                    print(f"✅ Found camera: {cam_name}")
            else:
                print("⚠️  WARNING: No camera data found")
            
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
            dt = f.attrs.get('dt', 0.033)  # 30Hz default
            duration = data_length * dt
            sync_rate = f.attrs.get('sync_sample_rate', 30)
            
            print(f"✅ PASS: {data_length} timesteps, {duration:.1f}s, {file_size_mb:.1f}MB")
            print(f"     Sync rate: {sync_rate}Hz")
            
            # Show data structure
            print(f"     Observations: D1 arm (7D)")
            print(f"     Actions: Base velocity (3D: x,y,yaw) + D1 positions (7D)")
            
            # Check robot type
            robot_type = f.attrs.get('robot_type', 'unknown')
            if robot_type == 'd1_with_base_velocity':
                print(f"✅ Robot type: {robot_type}")
            else:
                print(f"⚠️  WARNING: Unexpected robot type: {robot_type}")
            
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
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        time_steps = np.arange(len(action))
        dt = f.attrs.get('dt', 0.033)
        time_seconds = time_steps * dt
        
        # 在这里计算duration！
        duration = len(time_steps) * dt
        
        # D1 arm joint positions (observations)
        if 'arm' in obs and 'joint_positions' in obs['arm']:
            arm_pos = obs['arm']['joint_positions'][:]
            
            ax1 = fig.add_subplot(gs[0, :2])
            for i in range(7):
                ax1.plot(time_seconds, arm_pos[:, i], label=f'Joint {i+1}', alpha=0.8)
            ax1.set_title('D1 Arm Joint Positions (Observations)', fontsize=12)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Position (rad)')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
        
        # Base velocity actions (0-2)
        ax2 = fig.add_subplot(gs[0, 2])
        base_vel_actions = action[:, :3]
        labels = ['x velocity', 'y velocity', 'yaw velocity']
        colors = ['red', 'green', 'blue']
        for i in range(3):
            ax2.plot(time_seconds, base_vel_actions[:, i], label=labels[i], 
                    color=colors[i], alpha=0.8, linewidth=1.5)
        ax2.set_title('Base Velocity Commands', fontsize=12)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s, rad/s)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # D1 position actions (3-9)
        ax3 = fig.add_subplot(gs[1, :2])
        d1_actions = action[:, 3:10]
        for i in range(7):
            ax3.plot(time_seconds, d1_actions[:, i], label=f'Joint {i+1}', alpha=0.8)
        ax3.set_title('D1 Position Commands (Actions)', fontsize=12)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Target Position (rad)')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # Base velocity magnitude
        ax4 = fig.add_subplot(gs[1, 2])
        linear_vel_magnitude = np.linalg.norm(base_vel_actions[:, :2], axis=1)
        angular_vel = np.abs(base_vel_actions[:, 2])
        ax4.plot(time_seconds, linear_vel_magnitude, label='Linear (x,y)', 
                color='purple', alpha=0.8, linewidth=1.5)
        ax4.plot(time_seconds, angular_vel, label='Angular (yaw)', 
                color='orange', alpha=0.8, linewidth=1.5)
        ax4.set_title('Base Velocity Magnitude', fontsize=12)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Speed (m/s, rad/s)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # D1 tracking error (if actual matches command)
        if 'arm' in obs and 'joint_positions' in obs['arm']:
            ax5 = fig.add_subplot(gs[2, :2])
            tracking_error = d1_actions - arm_pos
            for i in range(7):
                ax5.plot(time_seconds, tracking_error[:, i], 
                        label=f'Joint {i+1}', alpha=0.7)
            ax5.set_title('D1 Tracking Error (Command - Actual)', fontsize=12)
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Error (rad)')
            ax5.legend(loc='upper right')
            ax5.grid(True, alpha=0.3)
            ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Statistics
        ax6 = fig.add_subplot(gs[2, 2])
        sync_rate = f.attrs.get('sync_sample_rate', 30)
        
        stats_text = f"""Episode Statistics:

Timesteps: {len(time_steps)}
Duration: {duration:.1f}s
Sync Rate: {sync_rate}Hz
dt: {dt:.3f}s
File size: {os.path.getsize(file_path)/(1024*1024):.1f}MB

Action Structure (10D):
  Base Velocity: 3D
  D1 Positions: 7D

Observation Structure (7D):
  D1 Joint Positions: 7D

Base Velocity Stats:
  Max linear: {linear_vel_magnitude.max():.3f} m/s
  Max angular: {angular_vel.max():.3f} rad/s"""
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        ax6.axis('off')
        
        plt.suptitle(f'D1 Mobile Manipulation Analysis - {os.path.basename(file_path)}', 
                    fontsize=14, fontweight='bold')
        
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
                                figsize=(num_frames_to_show*3, num_cameras*2.5))
        
        if num_cameras == 1:
            axes = axes.reshape(1, -1)
        if num_frames_to_show == 1:
            axes = axes.reshape(-1, 1)
        
        dt = f.attrs.get('dt', 0.033)
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
                time_sec = actual_frame * dt
                ax.set_title(f'{cam_name}\nt={time_sec:.1f}s (#{actual_frame})')
                ax.axis('off')
                
                # Add frame border based on camera type
                if 'front' in cam_name.lower():
                    for spine in ax.spines.values():
                        spine.set_edgecolor('blue')
                        spine.set_linewidth(2)
                elif 'wrist' in cam_name.lower():
                    for spine in ax.spines.values():
                        spine.set_edgecolor('green')
                        spine.set_linewidth(2)
                
                # Save individual images if requested
                if save_images:
                    img_save_path = file_path.replace('.hdf5', f'_{cam_name}_{actual_frame:04d}.jpg')
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(img_bgr, img_save_path)
        
        plt.tight_layout()
        
        if save_images:
            plot_path = file_path.replace('.hdf5', '_cameras.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Camera plot saved: {plot_path}")
        
        plt.show()

def print_detailed_structure(file_path):
    """Print detailed HDF5 structure"""
    print("\nDetailed HDF5 Structure:")
    print("="*50)
    
    def print_attrs(name, obj):
        if hasattr(obj, 'attrs'):
            attrs = dict(obj.attrs)
            if attrs:
                print(f"{name} attributes: {attrs}")
    
    def print_dataset_info(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
    
    with h5py.File(file_path, 'r') as f:
        print("File attributes:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        print()
        
        f.visititems(print_dataset_info)
        f.visititems(print_attrs)

def main():
    parser = argparse.ArgumentParser(description='HDF5 Validator for D1 Mobile Manipulation (10D format)')
    parser.add_argument('file_path', type=str, help='HDF5 file path')
    parser.add_argument('--save-plots', action='store_true', help='Save plots')
    parser.add_argument('--save-images', action='store_true', help='Save images')
    parser.add_argument('--max-frames', type=int, default=6, help='Max frames to show')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualizations')
    parser.add_argument('--structure', action='store_true', help='Print detailed structure')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"❌ File not found: {args.file_path}")
        return
    
    try:
        # Validate file
        is_valid = validate_hdf5_file(args.file_path)
        
        if args.structure:
            print_detailed_structure(args.file_path)
        
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
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()