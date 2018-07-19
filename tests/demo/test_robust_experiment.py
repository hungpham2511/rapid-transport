import pytest, time
import transport.console.robust_experiment as robust_experiment
import transport
import numpy as np

transport.utils.setup_logging("DEBUG")


def test_run(setup, monkeypatch):
    monkeypatch.setattr("__builtin__.raw_input", lambda s: "")
    scene_path = "/home/hung/git/toppra-object-transport/models/robust-exp.env.xml"

    success = robust_experiment.main(setup, scene_path, "denso",
                                     "bigcup_bluenb_1608823082",
                                     "bluenb", "denso_suction_cup2",
                                     np.array([[0, -1, 0, 0e-3],
                                               [-1, 0, 0, 0],
                                               [0, 0, -1, 28.5e-3],
                                               [0, 0, 0, 1]]),
                                     "traj1", "kin_only", 1.0, 0, False, "hotqpoases")
    assert success


