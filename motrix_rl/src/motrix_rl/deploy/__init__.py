# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Training-side policy deployment integration."""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from motrix_deploy.profile import DeploymentProfile
from motrix_rl.config import OnnxParityConfig
from motrix_rl.deploy.api import (
    DeploymentExportResult,
    OnnxExportReport,
    OnnxExportRequest,
    OnnxExportResult,
    OnnxModelArtifact,
    OnnxParityMetrics,
    OnnxPolicyExporter,
    PolicyTensorSpec,
    SourceRolloutResult,
)


def export_onnx_model(
    run_dir: str | Path,
    *,
    opset: int = 18,
    checkpoint: str | Path | None = None,
    validation_seed: int = 1,
    validation_samples: int = 32,
    atol: float = 1e-4,
    rtol: float = 1e-5,
) -> OnnxModelArtifact:
    """Return a validated in-memory ONNX policy without loading optional dependencies eagerly."""
    from motrix_rl.deploy.onnx import export_onnx_model as _export_onnx_model

    return _export_onnx_model(
        run_dir,
        opset=opset,
        checkpoint=checkpoint,
        validation_seed=validation_seed,
        validation_samples=validation_samples,
        atol=atol,
        rtol=rtol,
    )


def export_onnx(
    run_dir: str | Path,
    output: str | Path | None = None,
    *,
    opset: int = 18,
    checkpoint: str | Path | None = None,
    validation_seed: int = 1,
    validation_samples: int = 32,
    atol: float = 1e-4,
    rtol: float = 1e-5,
) -> OnnxExportResult:
    """Persist a validated ONNX policy without loading optional dependencies eagerly."""
    from motrix_rl.deploy.onnx import export_onnx as _export_onnx

    return _export_onnx(
        run_dir,
        output,
        opset=opset,
        checkpoint=checkpoint,
        validation_seed=validation_seed,
        validation_samples=validation_samples,
        atol=atol,
        rtol=rtol,
    )


def export_deploy_run(
    run_dir: str | Path,
    output: str | Path,
    *,
    profile_builder: Callable[[str], DeploymentProfile],
    checkpoint: str | Path | None = None,
    validation_seed: int = 1,
    validation_samples: int = 32,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> DeploymentExportResult:
    """Compose a metadata-backed run into a deployment artifact."""
    from motrix_rl.deploy.service import export_deploy_run as _export_deploy_run

    return _export_deploy_run(
        run_dir,
        output,
        profile_builder=profile_builder,
        checkpoint=checkpoint,
        validation_seed=validation_seed,
        validation_samples=validation_samples,
        atol=atol,
        rtol=rtol,
    )


def validate_motrixsim_source_rollout(
    artifact_path: str | Path,
    *,
    env_name: str,
    steps: int,
    seed: int,
    command: np.ndarray,
) -> SourceRolloutResult:
    """Run an optional bounded source-environment regression rollout."""
    from motrix_rl.deploy.source_rollout import validate_motrixsim_source_rollout as _validate

    return _validate(
        artifact_path,
        env_name=env_name,
        steps=steps,
        seed=seed,
        command=command,
    )


__all__ = [
    "DeploymentExportResult",
    "DeploymentProfile",
    "OnnxExportReport",
    "OnnxExportRequest",
    "OnnxExportResult",
    "OnnxModelArtifact",
    "OnnxParityConfig",
    "OnnxParityMetrics",
    "OnnxPolicyExporter",
    "PolicyTensorSpec",
    "SourceRolloutResult",
    "export_deploy_run",
    "export_onnx",
    "export_onnx_model",
    "validate_motrixsim_source_rollout",
]
