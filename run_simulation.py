import time
import mujoco
import mujoco.viewer

model_path = 'models/robot.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

with mujoco.viewer.launch(model, data) as viewer:

    start_time = time.time()
    while viewer.is_running():
        # Apply control inputs here
        # TODO

        mujoco.mj_step(model, data)

        viewer.sync()

        if time.time() - start_time < data.time:
            time.sleep(data.time - time.time() + start_time)
