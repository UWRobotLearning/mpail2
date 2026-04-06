"""Adapted from `frankz/network/server.py` in the frankz repository.

Source: https://github.com/memmelma/frankz/tree/mpail2
Author: memmelma

Modifications have been made to fit this project.
"""

import argparse
import pickle
import traceback

import zmq

from ..hardware import Franka


class FrankaServer:
    def __init__(self, robot_ip="172.16.0.2", bind_address="tcp://*:5555"):
        """
        Initialize the ZMQ server that wraps FrankaServer.

        Args:
            robot_ip: IP address of the Franka robot
            bind_address: ZMQ bind address (e.g., "tcp://*:5555")
        """
        self.robot_ip = robot_ip
        self.bind_address = bind_address

        # Initialize ZMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(bind_address)

        # FrankaServer will be initialized when client connects with init command
        self.franka_server = None

        print(f"ZMQ server listening on {bind_address}")
        print("Waiting for client connection...")

    def _handle_command(self, message):
        """
        Handle a command from the client.

        Args:
            message: Dictionary with "command" and "data" keys

        Returns:
            Response dictionary with "result" or "error"
        """
        try:
            command = message["command"]
            data = message.get("data")

            if command == "init":
                if self.franka_server is not None:
                    return {"result": {"status": "initialized"}}

                # Initialize the FrankaServer
                control_mode = data.get("control_mode", "cartesian_delta")
                dynamics_factor = data.get("dynamics_factor", 0.1)
                reset_qpos = data.get("reset_qpos", None)

                print(f"Initializing FrankaServer with control_mode={control_mode}, dynamics_factor={dynamics_factor}")
                try:
                    self.franka_server = Franka(
                        robot_ip=self.robot_ip,
                        control_mode=control_mode,
                        dynamics_factor=dynamics_factor,
                        reset_qpos=reset_qpos,
                    )
                    return {"result": {"status": "initialized"}}
                except Exception as e:
                    error_msg = f"Failed to initialize FrankaServer: {type(e).__name__}: {str(e)}"
                    print(f"Error: {error_msg}")
                    print(traceback.format_exc())
                    self.franka_server = None
                    return {"error": error_msg}

            # Check if server is initialized
            if self.franka_server is None:
                return {"error": "Server not initialized. Send 'init' command first."}

            if command == "step":
                action = data["action"]
                blocking = data.get("blocking", False)
                for _ in range(3):
                    try:
                        self.franka_server.step(action, blocking=blocking)
                        return {"result": None}
                    except RuntimeError as e:
                        error_msg = e
                self.franka_server.step(action, blocking=blocking)
                return {"result": None}

            elif command == "reset":
                self.franka_server.reset()
                return {"result": None}

            elif command == "get_obs":
                result = self.franka_server.get_obs()
                return {"result": result}

            else:
                return {"error": f"Unknown command: {command}"}

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            print(f"Error handling command: {error_msg}")
            return {"error": error_msg}

    def run(self):
        """
        Run the server loop.
        """
        print("Server ready to receive commands")

        while True:
            # Wait for request
            message = pickle.loads(self.socket.recv())
            print(f"Received command: {message['command']}")

            # Process request
            response = self._handle_command(message)

            # Send response
            self.socket.send(pickle.dumps(response))

    def close(self):
        """Close the ZMQ connection."""
        if self.franka_server is not None:
            try:
                print("Cleaning up FrankaServer...")
                self.franka_server.close()
            except Exception as e:
                print(f"Warning: Error cleaning up FrankaServer: {e}")
            finally:
                self.franka_server = None

        self.socket.close()
        self.context.term()
        print("Server closed")


def main():
    parser = argparse.ArgumentParser(description="Franka ZMQ Server")
    parser.add_argument("--robot_ip", type=str, default="172.16.0.2", help="Robot IP address")
    parser.add_argument("--bind_address", type=str, default="tcp://*:5555", help="ZMQ bind address")

    args = parser.parse_args()

    server = FrankaServer(
        robot_ip=args.robot_ip,
        bind_address=args.bind_address,
    )
    server.run()


if __name__ == "__main__":
    main()
