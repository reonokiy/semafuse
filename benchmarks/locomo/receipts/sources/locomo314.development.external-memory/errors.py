"""Errors exposed by the external-memory protocol and driver."""


class ExternalMemoryError(RuntimeError):
    """Base class for adapter framework failures."""


class ProtocolError(ExternalMemoryError):
    """A request, response, or worker state violates the protocol."""


class ConfigurationError(ExternalMemoryError):
    """Adapter configuration is invalid or unsafe."""


class AdapterUnavailableError(ExternalMemoryError):
    """A pinned upstream implementation is not installed."""


class RequestTimeoutError(ExternalMemoryError):
    """The worker did not answer before the configured deadline."""


class ProcessExitedError(ExternalMemoryError):
    """The worker exited without a valid response."""


class RemoteAdapterError(ExternalMemoryError):
    """The worker returned a structured, fail-closed error."""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.retryable = retryable
