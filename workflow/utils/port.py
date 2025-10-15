import socket


def find_available_port(start_port: int = 5000, end_port: int = 5500) -> int:
    """Find an available port in the specified range"""
    for port in range(start_port, end_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("localhost", port))
                return port
        except OSError:
            # Port is already in use, try next one
            continue

    # If no port is available in the range, raise an exception
    raise RuntimeError(f"No available port found in range {start_port}-{end_port}")
