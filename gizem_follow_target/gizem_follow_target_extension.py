import omni.ext
import omni.ui as ui
from .follow_target import GizemFollowTarget # We'll rename your class

class GizemFollowTargetExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        self._window = ui.Window("Gizem Robotics", width=300, height=300)
        with self._window.frame:
            with ui.VerticalStack():
                ui.Label("Follow Target with Joystick")
                ui.Button("Launch Scenario", clicked_fn=self._on_launch)

    def _on_launch(self):
        # This calls your logic
        self.scenario = GizemFollowTarget()
        self.scenario.load_scenario()

    def on_shutdown(self):
        self._window = None
