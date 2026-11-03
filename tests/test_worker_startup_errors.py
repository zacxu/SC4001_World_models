"""Backend EOF is an error; control-pipe EOF is normal worker shutdown."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from world_models.env.workers import RawParticipantState, WorkerConfig, participant_worker


class Connection:
    def __init__(self, requests, *, reply_failure=False):
        self.requests, self.reply_failure = list(requests), reply_failure
        self.sent, self.close_count, self.send_count = [], 0, 0

    def recv(self):
        if not self.requests:
            raise EOFError("control parent disconnected")
        return self.requests.pop(0)

    def send(self, reply):
        self.send_count += 1
        if self.reply_failure:
            raise BrokenPipeError("control parent no longer accepts replies")
        self.sent.append(reply)

    def close(self): self.close_count += 1


class Backend:
    def __init__(self, failing_operation=None, exception=EOFError):
        self.failing_operation, self.exception = failing_operation, exception
        self.calls, self.close_count = [], 0

    def _operation(self, name):
        self.calls.append(name)
        if name == self.failing_operation:
            raise self.exception(f"native backend {name} disconnected")
        return RawParticipantState(0, 1, None)

    def start(self): return self._operation("start")
    def reset(self, seed=None): return self._operation("reset")
    def step(self, action): return self._operation("step")
    def close(self): self.close_count += 1


def request(command, *, episode=0, decision=0, dispatch=1):
    return {"command": command, "episode_id": episode, "decision_index": decision,
            "dispatch_id": dispatch, "substep_index": 0, "substep_count": 1, "action": 0}


class WorkerStartupErrorTests(unittest.TestCase):
    def run_worker(self, connection, backend=None, *, factory=None):
        with tempfile.TemporaryDirectory() as directory:
            config = WorkerConfig(0, "unchanged-scenario.wad", 45000, 42, directory)
            original_cwd = Path.cwd()
            with patch("world_models.env.workers.os.chdir"), patch("world_models.env.workers.os.setsid", create=True):
                participant_worker(connection, config, factory or (lambda config: backend))
            self.assertEqual(Path.cwd(), original_cwd)
        self.assertEqual(connection.close_count, 1)
        if backend is not None:
            self.assertEqual(backend.close_count, 1)

    def test_startup_backend_eof_is_typed_error_with_traceback(self):
        backend = Backend("start")
        connection = Connection([request("start")])
        self.run_worker(connection, backend)
        self.assertEqual(len(connection.sent), 1)
        reply = connection.sent[0]
        self.assertEqual(reply["status"], "error")
        self.assertEqual(reply["error_type"], "EOFError")
        self.assertEqual(reply["operation"], "start")
        self.assertIn("native backend start disconnected", reply["error"])
        self.assertIn("state = backend.start()", reply["traceback"])
        self.assertEqual((reply["episode_id"], reply["decision_index"]), (-1, -1))

    def test_step_and_reset_backend_eof_are_not_silently_swallowed(self):
        for operation in ("step", "reset"):
            with self.subTest(operation=operation):
                backend = Backend(operation)
                second = request(operation, episode=1 if operation == "reset" else 0,
                                 decision=0 if operation == "reset" else 1, dispatch=2)
                connection = Connection([request("start"), second])
                self.run_worker(connection, backend)
                self.assertEqual([reply["status"] for reply in connection.sent], ["ok", "error"])
                error = connection.sent[-1]
                self.assertEqual(error["operation"], operation)
                self.assertEqual(error["error_type"], "EOFError")
                self.assertIn(f"native backend {operation} disconnected", error["traceback"])
                self.assertEqual((error["episode_id"], error["decision_index"]), (0, 0))

    def test_true_control_eof_and_explicit_close_are_clean_shutdown(self):
        for requests in ([], [request("close")], [request("start")]):
            with self.subTest(requests=requests):
                backend, connection = Backend(), Connection(requests)
                self.run_worker(connection, backend)
                self.assertFalse(any(reply["status"] == "error" for reply in connection.sent))

    def test_reply_pipe_closure_still_closes_backend_without_retrying_send(self):
        backend, connection = Backend(), Connection([request("start")], reply_failure=True)
        self.run_worker(connection, backend)
        self.assertEqual(connection.send_count, 1)
        self.assertEqual(backend.calls, ["start"])

    def test_backend_broken_pipe_is_reported_as_backend_failure(self):
        backend, connection = Backend("start", BrokenPipeError), Connection([request("start")])
        self.run_worker(connection, backend)
        self.assertEqual(connection.sent[0]["error_type"], "BrokenPipeError")
        self.assertEqual(connection.sent[0]["operation"], "start")

    def test_backend_construction_eof_is_reported_without_unconstructed_cleanup(self):
        def failed_factory(config): raise EOFError("backend constructor disconnected")
        connection = Connection([])
        self.run_worker(connection, factory=failed_factory)
        self.assertEqual(connection.sent[0]["status"], "error")
        self.assertEqual(connection.sent[0]["error_type"], "EOFError")
        self.assertEqual(connection.sent[0]["operation"], "construct_backend")


if __name__ == "__main__":
    unittest.main()
