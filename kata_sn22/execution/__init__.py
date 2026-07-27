"""SN22 execution-backend selection and the sealed-room client seam."""

from kata_sn22.execution.policy import (
    EXECUTION_BACKEND_ENV,
    resolve_execution_backend,
    tee_execution_enabled,
)

__all__ = ["EXECUTION_BACKEND_ENV", "resolve_execution_backend", "tee_execution_enabled"]
