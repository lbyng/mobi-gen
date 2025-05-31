import time
import pyspacemouse
import zmq
import json
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

class spacemouse_go2:
    def __init__(self):
        self.success = None
        self.state = None

        self.sport_client = None
        self.msc = None

        self.spacemouse_pos = [0.0, 0.0, 0.0]
        self.mapped_spacemouse_pos = [0.0, 0.0, 0.0]
        self.buttons = [0, 0]
        self.if_Damp = 0

        self.zmq_context = zmq.Context()
        self.zmq_socket = None
        self.zmq_port = 6000

        return None

    def connect_spacemouse(self):
        self.success = pyspacemouse.open()
        self.state = pyspacemouse.read()
        print("Spacemouse connected")
        return None
    
    def connect_robot(self):
        ChannelFactoryInitialize(0, "enp14s0")
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        print("Go2 Robot connected")

        self.msc = MotionSwitcherClient()
        self.msc.SelectMode("ai")
        print("ai motion mode enabled")
        return None
    
    def init_zmq_publisher(self):
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.bind(f"tcp://*:{self.zmq_port}")
        print(f"ZeroMQ publisher initialized on port {self.zmq_port}")
        time.sleep(0.5)
        return None
    
    def update_spacemouse(self):
        self.state = pyspacemouse.read()
        self.spacemouse_pos[0] = self.state.x
        self.spacemouse_pos[1] = self.state.y
        self.spacemouse_pos[2] = self.state.yaw
        self.mapped_spacemouse_pos = map_position(self.spacemouse_pos)

        self.buttons[0] = self.state.buttons[0]
        self.buttons[1] = self.state.buttons[1]

        self.if_damp = self.state.z
        return None

    def update_robot(self):
        self.sport_client.Move(self.mapped_spacemouse_pos[1], 
                               self.mapped_spacemouse_pos[0],
                               self.mapped_spacemouse_pos[2])
        if self.buttons[1]:
            self.sport_client.BalanceStand()
        
        if self.buttons[0]:
            self.sport_client.StandDown()

        if self.if_damp == -1.0:
            self.sport_client.Damp()
        return None 
    
    def send_position_via_zmq(self):
        if self.zmq_socket is not None:
            position_data = {
                "x": self.mapped_spacemouse_pos[0],
                "y": self.mapped_spacemouse_pos[1],
                "yaw": self.mapped_spacemouse_pos[2],
            }
            
            try:
                self.zmq_socket.send_json(position_data, zmq.NOBLOCK)
            except zmq.Again:
                pass
            except Exception as e:
                print(f"Error sending ZMQ message: {e}")
        return None

    def cleanup(self):
        if self.zmq_socket:
            self.zmq_socket.close()
        if self.zmq_context:
            self.zmq_context.term()
        print("ZeroMQ resources cleaned up")
        return None


def map_position_helper(x, ctrl_max, ctrl_min, mech_max, mech_min):
    return (x - ctrl_min) / (ctrl_max - ctrl_min) * (mech_max - mech_min) + mech_min

def map_position(spacemouse_pos):
    x, y, yaw = spacemouse_pos

    deadzone = 0.1
    if abs(x) < deadzone: x = 0
    if abs(y) < deadzone: y = 0
    if abs(yaw) < deadzone: yaw = 0

    x = map_position_helper(x, -0.6, 0.6, 0.6, -0.6)
    y = map_position_helper(y, -0.6, 0.6, -0.4, 0.4)
    yaw = map_position_helper(yaw, -1.0, 1.0, 0.8,  -0.8)

    return [x, y, yaw]

def main():
    robot = spacemouse_go2()
    
    try:
        robot.connect_spacemouse()
        robot.connect_robot()
        robot.init_zmq_publisher()

        while True:
            robot.update_spacemouse()
            robot.update_robot()
            robot.send_position_via_zmq()
            time.sleep(0.002)

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        robot.cleanup()

if __name__ == "__main__":
    main()