import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


OVR_ARKIT_BLENDSHAPES_MAP = {
    "19": "jawOpen",
    "50": "cheekPuff",  # Note: Cheek Suck L/R in old naming
    "88": "browInnerUp",  # Note: Used for both L/R in old naming
    "89": "browDown_L",
    "90": "browDown_R",
    "91": "browOuterUp_L",
    "92": "browOuterUp_R",
    "93": "eyeLookUp_L",
    "94": "eyeLookUp_R",
    "95": "eyeLookDown_L",
    "96": "eyeLookDown_R",
    "97": "eyeLookIn_L",
    "98": "eyeLookOut_L",
    "99": "eyeLookIn_R",
    "100": "eyeLookOut_R",
    "101": "eyeBlink_L",
    "102": "eyeBlink_R",
    "103": "eyeSquint_L",
    "104": "eyeSquint_R",
    "105": "eyeWide_L",
    "106": "eyeWide_R",
    "107": "cheekPuff",  # Note: Used for both L/R in old naming
    "108": "cheekSquint_L",
    "109": "cheekSquint_R",
    "110": "noseSneer_L",
    "111": "noseSneer_R",
    "113": "jawForward",
    "114": "jawLeft",
    "115": "jawRight",
    "116": "mouthFunnel",  # Note: Used for all funnel variants in old naming
    "117": "mouthPucker",  # Note: Used for both L/R in old naming
    "118": "mouthLeft",
    "119": "mouthRight",
    "120": "mouthRollUpper",  # Note: Lip Suck LT/RT in old naming
    "121": "mouthRollLower",  # Note: Lip Suck LB/RB in old naming
    "122": "mouthShrugUpper",
    "123": "mouthShrugLower",
    "124": "mouthClose",
    "125": "mouthSmile_L",
    "126": "mouthSmile_R",
    "127": "mouthFrown_L",
    "128": "mouthFrown_R",
    "129": "mouthDimple_L",
    "130": "mouthDimple_R",
    "131": "mouthUpperUp_L",
    "132": "mouthUpperUp_R",
    "133": "mouthLowerDown_L",
    "134": "mouthLowerDown_R",
    "135": "mouthPress_L",
    "136": "mouthPress_R",
    "137": "mouthStretch_L",
    "138": "mouthStretch_R"
}

class ROS2Subscriber(Node):
    def __init__(self, viewer):
        super().__init__('blendshape_subscriber')
        self.viewer = viewer
        
        # Initialize time tracking
        self.last_msg_time = self.get_clock().now()
        self.desired_period = 1.0/12.0  # 5Hz = 0.2 seconds
        
        self.subscription = self.create_subscription(
            JointState,
            '/operator/face/expressions',
            self.callback,
            1
        )

    def callback(self, msg):
        current_time = self.get_clock().now()
        if (current_time - self.last_msg_time).nanoseconds / 1e9 >= self.desired_period:
            self.get_logger().info('Received blendshape message:', throttle_duration_sec=1)
            blendshape_data = {OVR_ARKIT_BLENDSHAPES_MAP.get(index, index): value/100.0 for index, value in zip(msg.name, msg.position)}
            self.viewer.update_blendshapes_from_ros(blendshape_data)
            self.get_logger().info(f"{list(blendshape_data.keys())[0]}: {list(blendshape_data.values())[0]}", throttle_duration_sec=1)
            self.last_msg_time = current_time

def main(viewer):
    rclpy.init()
    subscriber = ROS2Subscriber(viewer)
    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()