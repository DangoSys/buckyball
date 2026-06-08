import socket

HOST = "127.0.0.1"


def find_available_port(start_port: int = 5000, end_port: int = 5500) -> int:
    """Find an available port in the specified range"""
    for port in range(start_port, end_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((HOST, port))
                return port
        except OSError:
            continue

    raise RuntimeError(f"No available port found in range {start_port}-{end_port}")


def _port_order(start_port: int, end_port: int, preferred_port: int | None = None):
    if preferred_port is None:
        yield from range(start_port, end_port + 1)
        return

    if preferred_port < start_port or preferred_port > end_port:
        raise ValueError("preferred_port must be within the requested range")

    yield from range(preferred_port, end_port + 1)
    yield from range(start_port, preferred_port)


def reserve_port(
    start_port: int = 5000,
    end_port: int = 5500,
    preferred_port: int | None = None,
) -> tuple[int, socket.socket]:
    """Reserve a port by binding and holding the socket open.

    The caller must close the returned socket once the engine has taken over the port.
    This prevents TOCTOU races when multiple bbdev instances start concurrently.
    """
    for port in _port_order(start_port, end_port, preferred_port):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((HOST, port))
            return port, sock
        except OSError:
            sock.close()
            continue

    raise RuntimeError(f"No available port found in range {start_port}-{end_port}")
