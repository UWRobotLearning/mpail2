"""
Minimal communication test — no real robot needed.

Run in THREE separate terminals:

  Terminal 1 (mock external planner, HTTP):
      python -m lerobot.async_inference.test_communication planner

  Terminal 2 (bridge server):
      python -m lerobot.async_inference.test_communication bridge

  Terminal 3 (fake robot that sends one observation and prints the action it gets back):
      python -m lerobot.async_inference.test_communication robot
"""

import json
import pickle
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import grpc
import numpy as np

from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from .helpers import TimedObservation


# ---------------------------------------------------------------------------
# Terminal 1: mock external planner (HTTP server)
# ---------------------------------------------------------------------------

class MockPlannerHandler(BaseHTTPRequestHandler):
    """Receives an observation JSON, prints it, returns dummy actions."""

    def do_POST(self):  # noqa: N802
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        obs = json.loads(body)

        print("\n[PLANNER] Got observation:")
        print(f"  timestep : {obs['timestep']}")
        print(f"  must_go  : {obs['must_go']}")
        state = obs["observation"].get("observation.state", [])
        print(f"  state    : {state}")
        task = obs["observation"].get("task", "")
        print(f"  task     : {task}")

        # Return 3 dummy actions (same dimensionality as the fake state)
        action_dim = len(state) if state else 6
        actions = [[float(i) * 0.01] * action_dim for i in range(3)]
        response = json.dumps({"actions": actions}).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)
        print("[PLANNER] Sent back 3 dummy actions.\n")

    def log_message(self, *args):  # suppress default access log
        pass


def run_planner(host="0.0.0.0", port=9090):
    print(f"[PLANNER] Mock external planner listening on {host}:{port}")
    HTTPServer((host, port), MockPlannerHandler).serve_forever()


# ---------------------------------------------------------------------------
# Terminal 2: bridge server (ExternalPlannerServer)
# ---------------------------------------------------------------------------

def run_bridge(host="0.0.0.0", port=8080, planner_url="http://127.0.0.1:9090/plan"):
    from lerobot.async_inference.external_planner_server import serve
    print(f"[BRIDGE] Starting ExternalPlannerServer on {host}:{port}")
    print(f"[BRIDGE] Will forward observations to {planner_url}")
    serve(host=host, port=port, planner_url=planner_url)


# ---------------------------------------------------------------------------
# Terminal 3: fake robot client (sends one observation, prints action)
# ---------------------------------------------------------------------------

def run_robot(server_address="127.0.0.1:8080"):
    print(f"[ROBOT] Connecting to bridge server at {server_address}")
    channel = grpc.insecure_channel(server_address, grpc_channel_options())
    stub = services_pb2_grpc.AsyncInferenceStub(channel)

    # 1. Handshake
    stub.Ready(services_pb2.Empty())
    print("[ROBOT] Handshake OK")

    # 2. Send fake policy instructions (bridge logs them, doesn't use them)
    from .helpers import RemotePolicyConfig
    fake_config = RemotePolicyConfig(
        policy_type="act",
        pretrained_name_or_path="test/model",
        lerobot_features={},
        actions_per_chunk=3,
        device="cpu",
    )
    stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(fake_config)))
    print("[ROBOT] Policy instructions sent")

    # 3. Build a fake observation (6-DOF joint state)
    fake_raw_obs = {
        "observation.state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
        "task": "test communication",
    }
    timed_obs = TimedObservation(
        timestamp=time.time(),
        timestep=0,
        observation=fake_raw_obs,
        must_go=True,
    )

    obs_bytes = pickle.dumps(timed_obs)
    obs_iterator = send_bytes_in_chunks(obs_bytes, services_pb2.Observation, silent=True)
    stub.SendObservations(obs_iterator)
    print("[ROBOT] Observation sent")

    # 4. Request actions back
    print("[ROBOT] Waiting for actions...")
    actions_msg = stub.GetActions(services_pb2.Empty())

    if not actions_msg.data:
        print("[ROBOT] Received empty response (timeout or error on bridge/planner side)")
    else:
        timed_actions = pickle.loads(actions_msg.data)  # nosec
        print(f"[ROBOT] Received {len(timed_actions)} actions:")
        for a in timed_actions:
            print(f"  timestep={a.get_timestep()}  action={a.get_action().tolist()}")

    channel.close()
    print("[ROBOT] Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

USAGE = """
Usage:
  python -m lerobot.async_inference.test_communication planner   # Terminal 1
  python -m lerobot.async_inference.test_communication bridge    # Terminal 2
  python -m lerobot.async_inference.test_communication robot     # Terminal 3
"""

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    role = sys.argv[1]
    if role == "planner":
        run_planner()
    elif role == "bridge":
        run_bridge()
    elif role == "robot":
        run_robot()
    else:
        print(f"Unknown role '{role}'.{USAGE}")
        sys.exit(1)
