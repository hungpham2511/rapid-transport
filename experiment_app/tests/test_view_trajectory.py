from ..view_trajectory import ViewTrajectory
import numpy as np
import pytest


def test_view_basic():
    view_traj = ViewTrajectory(
        "suctioncup_traj1",
        "caged_denso_ft_sensor_suction.env.xml",
        "denso",
        "kindlebox",
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
    
