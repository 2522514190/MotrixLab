# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Metadata-driven public ONNX export service."""

import os
import tempfile
from pathlib import Path

from motrix_rl import checkpoints, frameworks
from motrix_rl.config import OnnxParityConfig
from motrix_rl.deploy.api import OnnxExportRequest, OnnxExportResult, OnnxModelArtifact
from motrix_rl.runs import open_run_context, read_task_config


def export_onnx_model(
    run_dir: str | Path,
    *,
    checkpoint: str | Path | None = None,
    opset: int = 18,
    validation_seed: int = 1,
    validation_samples: int = 32,
    atol: float = 1e-4,
    rtol: float = 1e-5,
) -> OnnxModelArtifact:
    """Return a validated in-memory ONNX policy for deployment composition."""
    _, exported = _export_onnx_model(
        run_dir,
        checkpoint=checkpoint,
        opset=opset,
        validation_seed=validation_seed,
        validation_samples=validation_samples,
        atol=atol,
        rtol=rtol,
    )
    return exported


def export_onnx(
    run_dir: str | Path,
    output: str | Path | None = None,
    *,
    checkpoint: str | Path | None = None,
    opset: int = 18,
    validation_seed: int = 1,
    validation_samples: int = 32,
    atol: float = 1e-4,
    rtol: float = 1e-5,
) -> OnnxExportResult:
    """Export the best policy in a training run to a self-contained ONNX model."""
    checkpoint, exported = _export_onnx_model(
        run_dir,
        checkpoint=checkpoint,
        opset=opset,
        validation_seed=validation_seed,
        validation_samples=validation_samples,
        atol=atol,
        rtol=rtol,
    )
    output_path = Path(output) if output is not None else checkpoint.with_name("policy.onnx")
    if output_path.suffix.lower() != ".onnx":
        raise ValueError(f"ONNX output path must end in .onnx: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_path.parent, prefix=f".{output_path.name}.", delete=False) as file:
            temporary_path = Path(file.name)
            file.write(exported.model_bytes)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return OnnxExportResult(
        path=output_path,
        report=exported.report,
    )


def _export_onnx_model(
    run_dir: str | Path,
    *,
    checkpoint: str | Path | None = None,
    opset: int,
    validation_seed: int,
    validation_samples: int,
    atol: float,
    rtol: float,
) -> tuple[Path, OnnxModelArtifact]:
    """Resolve one run and return its best checkpoint plus validated model."""
    run = open_run_context(run_dir)
    metadata = run.metadata
    framework = frameworks.get_framework(metadata.rllib)
    task_config = read_task_config(run.run_dir, framework.get_config_type(metadata.algo))
    if (
        task_config.task.env != metadata.env_name
        or task_config.task.rllib != metadata.rllib
        or task_config.task.algo != metadata.algo
        or task_config.task.train_backend not in (None, metadata.train_backend)
    ):
        task_identity = (
            task_config.task.env,
            task_config.task.rllib,
            task_config.task.train_backend,
            task_config.task.algo,
        )
        metadata_identity = (metadata.env_name, metadata.rllib, metadata.train_backend, metadata.algo)
        raise ValueError(f"task_config identity {task_identity} does not match run metadata {metadata_identity}")

    checkpoint = (
        checkpoints.resolve_checkpoint_path(checkpoint, metadata=metadata)
        if checkpoint is not None
        else checkpoints.best_policy(metadata, run.run_dir)
    )
    request = OnnxExportRequest(
        run=run,
        checkpoint=checkpoint,
        task_config=task_config,
        opset=opset,
        parity=OnnxParityConfig(
            seed=validation_seed,
            samples=validation_samples,
            atol=atol,
            rtol=rtol,
        ),
    )
    return checkpoint, framework.export_policy(request)
