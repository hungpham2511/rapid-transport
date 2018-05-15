import pytest, time, sys
import transport
import matplotlib.pyplot as plt
import numpy as np

@pytest.fixture(scope="module")
def envcage(setup):
    env = setup
    env.Reset()
    db = transport.database.Database()
    env.Load(transport.utils.expand_and_join(
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
    traj_array = transport.generate_twist_at_active_conf(robot, max_angle=0.5)

    np.testing.assert_allclose(traj_array[0], qinit, atol=1e-8)
    np.testing.assert_allclose(traj_array[-1], qinit, atol=1e-8)
    # No duplicate
    for i in range(traj_array.shape[0] - 1):
        assert np.linalg.norm(traj_array[i + 1] - traj_array[i]) > 1e-5

    # for q in traj_array:
    #     robot.SetActiveDOFValues(q)
    #     time.sleep(0.1)

    # plt.plot(traj_array)
    # fig = plt.gcf()
    # timer = fig.canvas.new_timer(interval=2000)  # creating a timer object and setting an interval of 3000 milliseconds
    # timer.start()
    # timer.add_callback(lambda : plt.close())
    # plt.title("Joint values")
    # plt.show()


@pytest.mark.console
def test_gen_twist_console(monkeypatch, setup):
    env = setup
    # path argv and raw_input
    monkeypatch.setattr(sys, "argv", ["", '-q', "[0, 1, 1, 0, 0, 0]"])
    monkeypatch.setattr("__builtin__.raw_input", lambda s: "abc")
    transport.console.gen_twist_main(env)



