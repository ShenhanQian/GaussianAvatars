#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import random
import time

# ARKit blendshape mapping from OVR indices
OVR_ARKIT_BLENDSHAPES_MAP = {
    "19": ("jawOpen", 1.5),
    "50": ("cheekPuff", 1.0),  # Note: Cheek Suck L/R in old naming
    "88": ("browInnerUp", 1.0),  # Note: Used for both L/R in old naming
    "89": ("browDown_L", 1.0),
    "90": ("browDown_R", 1.0),
    "91": ("browOuterUp_L", 1.0),
    "92": ("browOuterUp_R", 1.0),
    "93": ("eyeLookUp_L", 1.0),
    "94": ("eyeLookUp_R", 1.0),
    "95": ("eyeLookDown_L", 1.0),
    "96": ("eyeLookDown_R", 1.0),
    "97": ("eyeLookIn_L", 1.0),
    "98": ("eyeLookOut_L", 1.0),
    "99": ("eyeLookIn_R", 1.0),
    "100": ("eyeLookOut_R", 1.0),
    "101": ("eyeBlink_L", 1.0),
    "102": ("eyeBlink_R", 1.0),
    "103": ("eyeSquint_L", 1.0),
    "104": ("eyeSquint_R", 1.0),
    "105": ("eyeWide_L", 1.0),
    "106": ("eyeWide_R", 1.0),
    "107": ("cheekPuff", 1.0),  # Note: Used for both L/R in old naming
    "108": ("cheekSquint_L", 1.0),
    "109": ("cheekSquint_R", 1.0),
    "110": ("noseSneer_L", 1.0),
    "111": ("noseSneer_R", 1.0),
    "113": ("jawForward", 1.5),
    "114": ("jawLeft", 1.5),
    "115": ("jawRight", 1.5),
    "116": ("mouthFunnel", 1.5),  # Note: Used for all funnel variants in old naming
    "117": ("mouthPucker", 1.5),  # Note: Used for both L/R in old naming
    "118": ("mouthLeft", 1.5),
    "119": ("mouthRight", 1.5),
    "120": ("mouthRollUpper", 1.5),  # Note: Lip Suck LT/RT in old naming
    "121": ("mouthRollLower", 1.5),  # Note: Lip Suck LB/RB in old naming
    "122": ("mouthShrugUpper", 1.5),
    "123": ("mouthShrugLower", 1.5),
    "124": ("mouthClose", 1.5),
    "125": ("mouthSmile_L", 1.5),
    "126": ("mouthSmile_R", 1.5),
    "127": ("mouthFrown_L", 1.5),
    "128": ("mouthFrown_R", 1.5),
    "129": ("mouthDimple_L", 1.5),
    "130": ("mouthDimple_R", 1.5),
    "131": ("mouthUpperUp_L", 1.5),
    "132": ("mouthUpperUp_R", 1.5),
    "133": ("mouthLowerDown_L", 1.5),
    "134": ("mouthLowerDown_R", 1.5),
    "135": ("mouthPress_L", 1.5),
    "136": ("mouthPress_R", 1.5),
    "137": ("mouthStretch_L", 1.5),
    "138": ("mouthStretch_R", 1.5)
}

class BlendshapePublisher(Node):
    """
    ROS2 node that publishes random blendshape values for facial expressions.
    This simulates facial tracking data for avatar animation systems.
    """
    
    def __init__(self):
        super().__init__('blendshape_publisher')
        
        # Create publisher for facial expressions
        self.publisher = self.create_publisher(
            JointState,
            '/operator/face/expressions',
            10  # QoS profile depth
        )
        
        # Timer for periodic publishing
        self.timer_period = 0.033  # ~30Hz
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        # Initialize blend shape values
        self.blend_shapes = {key: 0.0 for key in OVR_ARKIT_BLENDSHAPES_MAP.keys()}
        
        # Track blink state for natural blinking
        self.blink_state = 0.0
        self.blink_direction = 0
        self.time_to_next_blink = random.uniform(1.0, 5.0)
        self.last_update_time = time.time()
        
        self.get_logger().info('Blendshape publisher initialized')

    def generate_random_blendshapes(self):
        """
        Generate random blendshape values with some constraints to make
        the expressions look more natural and less chaotic.
        """
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Handle natural blinking
        self.time_to_next_blink -= delta_time
        if self.time_to_next_blink <= 0 and self.blink_direction == 0:
            # Start a new blink
            self.blink_direction = 1
            self.time_to_next_blink = random.uniform(2.0, 5.0)
        
        if self.blink_direction == 1:
            # Closing eyes
            self.blink_state += delta_time * 10.0  # Speed of blink
            if self.blink_state >= 1.0:
                self.blink_state = 1.0
                self.blink_direction = -1
        elif self.blink_direction == -1:
            # Opening eyes
            self.blink_state -= delta_time * 5.0  # Slower opening
            if self.blink_state <= 0.0:
                self.blink_state = 0.0
                self.blink_direction = 0
        
        # Apply blink to both eyes
        self.blend_shapes["101"] = self.blink_state * 100.0  # Left eye
        self.blend_shapes["102"] = self.blink_state * 100.0  # Right eye
        
        # Randomly update other blendshapes occasionally
        for key in self.blend_shapes.keys():
            # Skip eyes during blink
            if key in ["101", "102"] and self.blink_direction != 0:
                continue
                
            # Only update a small subset of blendshapes each frame for more natural movement
            if random.random() < 0.05:  # 5% chance to change a value
                # Add some randomness but bias toward 0 for more neutral expressions
                target = random.uniform(0, 40.0) if random.random() < 0.7 else random.uniform(40.0, 100.0)
                
                # Smooth transition to new value
                current = self.blend_shapes[key]
                # Move 10% toward target value
                self.blend_shapes[key] = current + (target - current) * 0.1
        
        # Ensure some constraints for natural expressions
        # For example, don't smile and frown at the same time
        if self.blend_shapes["125"] > 20 and self.blend_shapes["127"] > 20:  # smile_L and frown_L
            if random.random() < 0.5:
                self.blend_shapes["125"] *= 0.8  # Reduce smile
            else:
                self.blend_shapes["127"] *= 0.8  # Reduce frown

    def timer_callback(self):
        """
        Create and publish a JointState message with random blendshape values.
        """
        # Update blendshape values
        self.generate_random_blendshapes()
        
        # Create message
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Add all blendshapes to the message
        for key, value in self.blend_shapes.items():
            msg.name.append(key)
            msg.position.append(value/50.0)
            
        # Publish message
        self.publisher.publish(msg)
        
        # Log occasionally (not every frame to avoid spamming)
        if random.random() < 0.01:  # ~1% of frames
            sample_key = random.choice(list(self.blend_shapes.keys()))
            blendshape_name = OVR_ARKIT_BLENDSHAPES_MAP[sample_key][0]
            self.get_logger().info(
                f'Publishing blendshapes. Example: {blendshape_name} = {self.blend_shapes[sample_key]:.2f}'
            )

def main(args=None):
    rclpy.init(args=args)
    
    # Create and run the publisher node
    publisher = BlendshapePublisher()
    
    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        publisher.get_logger().info('Keyboard interrupt, shutting down')
    finally:
        # Clean up
        publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()