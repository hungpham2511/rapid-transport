import pytest, time
import toppra_app
import matplotlib.pyplot as plt
import numpy as np

@pytest.fixture(scope="module")
def envcage(setup):
    env = setup
    env.Reset()
    db = toppra_app.database.Database()
    env.Load(toppra_app.utils.expand_and_join(
        db.get_model_dir(), "caged_denso_ft_sensor_suction.env.xml"))
    env.SetViewer("qtosg")
    robot = env.GetRobots()[0]
    robot.SetActiveManipulator("denso_suction_cup")
    robot.SetActiveDOFs(range(6))
    yield robot

@pytest.mark.parametrize("qinit", [
    [0, 1, 1, 0, 0, 0],
])
def test_basic(envcage, qinit):
    robot = envcage
    robot.SetActiveDOFValues(qinit)
    traj_array = toppra_app.generate_twist_at_active_conf(robot, max_angle=0.5)

    np.testing.assert_allclose(traj_array[0], qinit, atol=1e-8)
    np.testing.assert_allclose(traj_array[-1], qinit, atol=1e-8)
    # No duplicate
    for i in range(traj_array.shape[0] - 1):
        assert np.linalg.norm(traj_array[i + 1] - traj_array[i]) > 1e-5

    for q in traj_array:
        robot.SetActiveDOFValues(q)
        time.sleep(0.1)

    plt.plot(traj_array)
    fig = plt.gcf()
    timer = fig.canvas.new_timer( interval=3000)  # creating a timer object and setting an interval of 3000 milliseconds
    timer.start()
    timer.add_callback(lambda : plt.close())
    plt.show()

