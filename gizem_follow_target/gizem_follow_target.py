from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
from isaacsim.robot.manipulators.examples.franka.tasks import FollowTarget as FollowTargetTask
from isaacsim.core.utils.types import ArticulationAction

import rclpy # Isaac's bundled rclpy — no import tricks needed
from rclpy.node import Node
from sensor_msgs.msg import Joy
import numpy as np


GRIPPER_BTN = 30

class JoySubscriberNode(Node):
    """Minimal rclpy node that lives inside Isaac's Python process."""
    def __init__(self):
        super().__init__("isaac_joy_subscriber")
        self.axes = []
        self.buttons = []
        self.create_subscription(Joy, "/joy", self._cb, 10)

    def _cb(self, msg: Joy):
        self.axes   = list(msg.axes)
        self.buttons = list(msg.buttons)
        print("button-axis30: ", self.buttons[30])

class FollowTarget(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._controller = None
        self._articulation_controller = None
        self._joy_node = None
        print("Started Gizem")

    def setup_scene(self):
        world = self.get_world()
        world.add_task(FollowTargetTask())

    async def setup_pre_reset(self):
        world = self.get_world()
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")
        self._controller.reset()

    def world_cleanup(self): ## Maybe not needed?
        self._controller = None
        if self._joy_node is not None:
            self._joy_node.destroy_node()
            self._joy_node = None
        # Do NOT call rclpy.shutdown() here — Isaac owns the process

    async def setup_post_load(self):
        self._franka_task = list(self._world.get_current_tasks().values())[0]
        self._task_params = self._franka_task.get_params()
        my_franka = self._world.scene.get_object(self._task_params["robot_name"]["value"])
        self._controller = RMPFlowController(
            name="target_follower_controller", robot_articulation=my_franka
        )
        self._articulation_controller = my_franka.get_articulation_controller()

        # Joystick - gripper thingy
        self._gripper_open = False          # current gripper state
        self._prev_gripper_btn = False         # last seen state of button[32]
        self._gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]

        # Target cube
        target_name = self._task_params["target_name"]["value"]
        self._target_cube = self._world.scene.get_object(target_name)
        current_pos, _ = self._target_cube.get_world_pose()
        self._cube_pos = np.array(current_pos, dtype=np.float64)
        print(f"[FollowTarget] Cube init pos: {self._cube_pos}")

        # Get finger joint indices once at load time
        dof_names = my_franka.dof_names
        print(f"[DOF names] {dof_names}")  # keep this until you've confirmed indices
        self._finger_idx = [
            dof_names.index("panda_finger_joint1"),
            dof_names.index("panda_finger_joint2"),
        ]
        print(f"[Gripper] Finger joint indices: {self._finger_idx}")

        # Init rclpy once (guard against double-init if scene reloads)
        if not rclpy.ok():
            rclpy.init()
        self._joy_node = JoySubscriberNode()

        self._joy_speed = 0.05   # m per physics step — tune this

        

    async def _on_follow_target_event_async(self, val):
        world = self.get_world()
        if val:
            await world.play_async()
            world.add_physics_callback("sim_step", self._on_follow_target_simulation_step)
        else:
            world.remove_physics_callback("sim_step")

    def _on_follow_target_simulation_step(self, step_size):
        # 1. Pump ROS2 callbacks (non-blocking)
        rclpy.spin_once(self._joy_node, timeout_sec=0)

        # --- Gripper toggle (debounced) ---
        buttons = self._joy_node.buttons
        if len(buttons) > 32:
            gripper_btn = bool(buttons[GRIPPER_BTN])
            # Only act on the rising edge (press, not hold)
            if gripper_btn and not self._prev_gripper_btn:
                self._gripper_open = not self._gripper_open
                self._apply_gripper(self._gripper_open)
            self._prev_gripper_btn = gripper_btn  


        # 2. Map axes → delta XYZ
        #    Standard gamepad layout (verify with `ros2 topic echo /joy`):
        #      axes[0] = left stick horizontal  → Y world axis
        #      axes[1] = left stick vertical    → X world axis
        #      axes[2] = right stick vertical   → Z world axis  (up/down)
        axes = self._joy_node.axes
        if len(axes) >= 4:
            dx =  axes[1] * self._joy_speed
            dy =  axes[0] * self._joy_speed
            dz =  axes[2] * self._joy_speed
            self._cube_pos += np.array([dx, dy, dz])

        # 3. Move target cube
        self._target_cube.set_world_pose(position=self._cube_pos)

        # 4. RMPFlow: Franka follows cube
        observations = self._world.get_observations()
        actions = self._controller.forward(
            target_end_effector_position=observations[
                self._task_params["target_name"]["value"]
            ]["position"],
            target_end_effector_orientation=observations[
                self._task_params["target_name"]["value"]
            ]["orientation"],
        )
        self._articulation_controller.apply_action(actions)

    def _apply_gripper(self, open: bool):
        from isaacsim.core.utils.types import ArticulationAction
        import numpy as np

        target = 0.04 if open else 0.0

        # Build a full-length array of None, set only the finger joints
        num_dofs = len(self._articulation_controller._articulation_view.dof_names)
        joint_positions = [None] * num_dofs
        for idx in self._finger_idx:
            joint_positions[idx] = target

        action = ArticulationAction(joint_positions=np.array(
            [t if t is not None else 0.0 for t in joint_positions],   # can't pass None in np array
        ))
        # Only the finger indices will be written — but we need the joint_indices arg:
        action = ArticulationAction(
            joint_positions=np.array([target, target]),
            joint_indices=np.array(self._finger_idx),
        )
        self._articulation_controller.apply_action(action)
        print(f"[Gripper] {'OPEN' if open else 'CLOSED'} (joints {self._finger_idx} → {target})")

    def _on_add_obstacle_event(self): ## No need so far but keep for future
        world = self.get_world()
        current_task = list(world.get_current_tasks().values())[0]
        cube = current_task.add_obstacle()
        self._controller.add_obstacle(cube)
        return

    def _on_remove_obstacle_event(self): ## No need so far but keep for future
        world = self.get_world()
        current_task = list(world.get_current_tasks().values())[0]
        obstacle_to_delete = current_task.get_obstacle_to_delete()
        self._controller.remove_obstacle(obstacle_to_delete)
        current_task.remove_obstacle()
        return

    def _on_logging_event(self, val): ## No need so far but keep for future
        world = self.get_world()
        data_logger = world.get_data_logger()
        if not world.get_data_logger().is_started():
            robot_name = self._task_params["robot_name"]["value"]
            target_name = self._task_params["target_name"]["value"]

            def frame_logging_func(tasks, scene):
                return {
                    "joint_positions": scene.get_object(robot_name).get_joint_positions().tolist(),
                    "applied_joint_positions": scene.get_object(robot_name)
                    .get_applied_action()
                    .joint_positions.tolist(),
                    "target_position": scene.get_object(target_name).get_world_pose()[0].tolist(),
                }

            data_logger.add_data_frame_logging_func(frame_logging_func)
        if val:
            data_logger.start()
        else:
            data_logger.pause()
        return

    def _on_save_data_event(self, log_path): ## No need so far but keep for future
        world = self.get_world()
        data_logger = world.get_data_logger()
        data_logger.save(log_path=log_path)
        data_logger.reset()
        return

