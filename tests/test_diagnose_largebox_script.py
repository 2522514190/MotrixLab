# Copyright Motphys Technology Co., Ltd. 2026
# SPDX-License-Identifier: Apache-2.0
"""Force-balance invariants: unilateral forces, actuation limits, cone relaxation."""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "diagnose_largebox", Path(__file__).parents[1] / "scripts/motion/diagnose_largebox.py"
)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def test_physical_force_balance_fixtures():
    results = diagnostics.self_check()
    assert results["static_supported"]["rho"] < 1
    for case in ("unsupported_box", "actuator_over_limit", "friction_violation"):
        assert results[case]["rho"] > 1


def test_contact_force_is_unilateral_and_equal_opposite():
    contacts = [(np.array([[-1.0], [1.0]]), np.zeros(5))]
    actuators = np.empty((2, 0))
    bounds = np.empty((0, 2))
    tolerance = np.full(2, 0.01)
    pushed = diagnostics.solve_balance(np.array([-2.0, 2.0]), contacts, actuators, bounds, tolerance)
    pulled = diagnostics.solve_balance(np.array([2.0, -2.0]), contacts, actuators, bounds, tolerance)
    net_force = diagnostics.solve_balance(np.array([2.0, 2.0]), contacts, actuators, bounds, tolerance)
    assert pushed["rho"] < 1
    assert pulled["rho"] > 1
    assert net_force["rho"] > 1


def test_outer_cone_cannot_make_feasibility_worse():
    # The square outer cone admits diagonal tangential force that the model's
    # pyramidal inner cone rejects; this prevents reversed relaxation labels.
    args = (
        np.array([1.0, 0.75, 0.75]),
        [(np.eye(3), np.ones(5))],
        np.empty((3, 0)),
        np.empty((0, 2)),
        np.full(3, 0.01),
    )
    inner = diagnostics.solve_balance(*args)
    outer = diagnostics.solve_balance(*args, outer=True)
    assert inner["rho"] > 1
    assert outer["rho"] < 1


def test_support_polygon_and_degenerate_edge():
    assert diagnostics.support_distance(np.array([0.0, 0.0]), [[-1, -1], [1, -1], [1, 1], [-1, 1]]) < 0
    assert diagnostics.support_distance(np.array([2.0, 0.0]), [[-1, -1], [1, -1], [1, 1], [-1, 1]]) > 0
    assert np.isclose(diagnostics.support_distance(np.array([0.0, 1.0]), [[-1, 0], [1, 0]]), 1)
    assert diagnostics.support_distance(np.zeros(2), []) is None
