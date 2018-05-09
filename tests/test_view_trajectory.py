import numpy as np
import pytest, toppra_app
import openravepy as orpy


@pytest.mark.parametrize("traj_id", ["suctioncup_traj4", "suctioncup_traj4_f1918e634c"])
@pytest.mark.parametrize("object_id", ["kindlebox", "small_alum_block", "heavy_alum_1"])
def test_view_basic(setup, traj_id, object_id):
    env = setup
    view_traj = toppra_app.ViewTrajectory(
        env,
        traj_id,
        "caged_denso_ft_sensor_suction.env.xml",
        "denso",
        object_id,
        "denso_suction_cup",
        np.eye(4)
    )

    view_traj.run_cmd("")
    view_traj.run_cmd("r")
    view_traj.run_cmd("1")

    with pytest.raises(SystemExit) as e:
        view_traj.run_cmd("q")
    assert e.type == SystemExit
    assert e.value.code == 42
    
