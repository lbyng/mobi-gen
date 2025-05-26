# recorder_playback

## HDF5 File Format

```
YYYYMMDD_HHMMSS.hdf5
    Attributes
        sim: False
        compress: False  
        robot_type: 'go2_d1_system'
        num_episodes: 1
        episode_length: N (timesteps)
        dt: 0.02 (time interval, seconds)
        total_time: X.X (total duration, seconds)
        sync_sample_rate: XX (sync sampling rate, Hz)
        recorded_at: '2024-XX-XXTXX:XX:XX'
        total_action_dim: 43 (total action dimensions)
        go2_action_dim: 36 (Go2 action dimensions)
        d1_action_dim: 7 (D1 action dimensions)
        go2_obs_dim: 54 (Go2 observation dimensions)
        d1_obs_dim: 7 (D1 observation dimensions)

    action [N × 43]
        [0:12]      # Go2 joint target positions (rad)
        [12:24]     # Go2 joint target velocities (rad/s)  
        [24:36]     # Go2 joint target torques (Nm)
        [36:43]     # D1 arm joint positions (rad)

    observations/
        base/
            joint_positions [N × 12]     # joint positions (rad)
            joint_velocities [N × 12]    # joint velocities (rad/s)
            joint_torques [N × 12]       # joint torques (Nm)
            orientation [N × 4]          # IMU quaternion (qw,qx,qy,qz)
            angular_velocity [N × 3]     # IMU angular velocity (rad/s)
            linear_acceleration [N × 3]  # IMU linear acceleration (m/s²)

        arm/
            joint_positions [N × 7]      # arm joint positions (rad)

        images/
            front_image [N × H(480) × W(640) × 3]  # front camera (RGB)
            wrist_image [N × H(480) × W(640) × 3]  # wrist camera (RGB)
```

## HDF5 File Validator

![hdf5_validator](images/hdf5_validator.png)