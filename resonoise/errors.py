"""Exception hierarchy. Each error carries an HTTP status and a stable machine-readable code."""
from __future__ import annotations


class ResoNoiseError(Exception):
    status = 400
    code = "bad_request"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class UnsupportedFormatError(ResoNoiseError):
    status = 415
    code = "unsupported_format"


class InvalidAudioError(ResoNoiseError):
    status = 422
    code = "invalid_audio"


class ValidationError(ResoNoiseError):
    status = 400
    code = "invalid_parameters"


class NotFoundError(ResoNoiseError):
    status = 404
    code = "not_found"


class ProcessingError(ResoNoiseError):
    status = 500
    code = "processing_failed"
