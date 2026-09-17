"""Exercise the prepared upstream arrival dispatcher without loading Torch."""
import ast
import heapq
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(os.environ.get("SGLANG_ROOT",
    Path(__file__).resolve().parents[2] / "HBFSim/tmp/sglang-native"))
DISPATCHER = SOURCE / "tools/sglang-simulator/src/sglang_simulator/simulation/sglang/scheduler.py"


def dispatch_order(source, arrivals):
    tree = ast.parse(source)
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ReqDispatcher")
    modes = SimpleNamespace(OFFLINE="offline", BLOCKING="blocking")
    namespace = {"heapq": heapq, "SimulationMode": modes,
        "logger": logging.getLogger(__name__),
        "time": SimpleNamespace(time_ns=lambda: 1, sleep=lambda _: None)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "upstream-dispatcher", "exec"), namespace)
    dispatcher = namespace["ReqDispatcher"](modes.OFFLINE)
    request_type = type("TokenizedGenerateReqInput", (), {})
    requests = []
    for index, arrival in enumerate(arrivals):
        request = request_type()
        request.source_index = index
        request.sampling_params = SimpleNamespace(custom_params={"simulation": {
            "created_time": arrival, "total_request": len(arrivals)}})
        requests.append(request)
    for start in range(0, len(requests), 127):
        dispatcher.add(requests[start:start+127])
    result = [heapq.heappop(dispatcher.future_queue)[2].source_index for _ in requests]
    return result


@unittest.skipUnless(DISPATCHER.exists(), "set SGLANG_ROOT to the prepared pinned checkout")
class ArrivalDispatcherTest(unittest.TestCase):
    def test_large_tied_trace_uses_stable_input_order_with_colliding_host_ticks(self):
        arrivals = [index // 300 for index in range(12031)]
        self.assertEqual(dispatch_order(DISPATCHER.read_text(), arrivals), list(range(len(arrivals))))

    def test_arrival_timestamp_precedes_input_tiebreaker(self):
        self.assertEqual(dispatch_order(DISPATCHER.read_text(), [2, 1, 2, 0, 1]), [3, 1, 4, 0, 2])


if __name__ == "__main__":
    unittest.main()
