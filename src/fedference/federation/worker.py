"""Federation worker: serializes a local belief, sends it, receives consensus."""

from __future__ import annotations

import logging
import queue
from typing import Any

import numpy as np

from .server import DEFAULT_TIMEOUT

_log = logging.getLogger(__name__)


class FederationWorker:
    """Sends a belief to the server and receives the consensus."""

    def __init__(
        self,
        worker_id: int,
        request_queue: Any,
        response_queue: Any,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.timeout = timeout
        self.consensus: np.ndarray | None = None

    def send_belief(self, belief: np.ndarray) -> None:
        """Serialize ``belief`` and enqueue it tagged with this worker's id."""
        from fedference.federation.transport import serialize_belief

        self.request_queue.put((self.worker_id, serialize_belief(belief)))

    def receive_consensus(self, timeout: float | None = None) -> np.ndarray:
        """Block for the broadcast result, store and return consensus.

        Args:
            timeout: Maximum seconds to wait. Defaults to ``self.timeout``.

        Raises:
            queue.Empty: If the server does not respond within the timeout,
                propagated with a logged warning.
        """
        from fedference.federation.transport import deserialize_result

        t = self.timeout if timeout is None else timeout
        try:
            data = self.response_queue.get(timeout=t)
        except queue.Empty:
            _log.warning(
                "FederationWorker %d: timed out waiting for consensus after %.1fs.",
                self.worker_id, t,
            )
            raise
        self.consensus = deserialize_result(data)["consensus"]
        return self.consensus
