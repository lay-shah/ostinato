"""The operator's remote: `python -m ostinato.remote --robot 192.168.1.20 --key key.txt`.

The rover moves only while this window is open and sending, and the operator has
pressed G. Space stops it and holds it stopped. If the window closes, the laptop
sleeps, or Wi-Fi drops, the heartbeats stop and the rover stops within 0.5 s. The
hardware emergency stop is still the one to trust.

Make a shared key once with `--make-key key.txt`, and copy the file to the rover.
Messages are signed with it. Each heartbeat carries a number the rover gave out in
the last 2 s, so recorded heartbeats cannot be played back later.
"""

import argparse
import hashlib
import hmac
import os
import secrets
import socket
import time
from pathlib import Path

PORT = 47800
EVERY = 0.1  # seconds between heartbeats
NONCE_LIFE = 1.0  # seconds; the current and previous numbers are accepted
MAX = 512  # bytes in a message


def load_key(path):
    text = Path(path).read_text().strip()
    try:
        key = bytes.fromhex(text)
    except ValueError:
        raise ValueError("the key file must hold 64 hex digits") from None
    if len(key) != 32:
        raise ValueError("the key file must hold 64 hex digits")
    return key


def make_key(path):
    path = Path(path)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as out:
        out.write(secrets.token_hex(32) + "\n")
    return path


def sign(key, body):
    tag = hmac.new(key, body.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{body}|{tag}".encode()


def opened(key, data):
    """The body of a correctly signed message, or None."""
    if len(data) > MAX:
        return None
    try:
        body, tag = data.decode("ascii").rsplit("|", 1)
    except (UnicodeDecodeError, ValueError):
        return None
    good = hmac.new(key, body.encode(), hashlib.sha256).hexdigest()[:32]
    return body if hmac.compare_digest(tag, good) else None


class Listener:
    """The rover's end. Call `poll` every control step."""

    def __init__(self, key, port=PORT, host="0.0.0.0", sock=None):
        self.key = key
        self.sock = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if sock is None:
            self.sock.bind((host, port))
        self.sock.setblocking(False)
        self.heard = None  # robot time of the last good heartbeat
        self.go = False
        self._nonces, self._since = [secrets.token_hex(8)], None
        self._count, self._last_nonce = -1, None

    @property
    def port(self):
        return self.sock.getsockname()[1]

    def poll(self, t, guard, status=""):
        """Read heartbeats, tell the guard, and answer with `status`."""
        if self._since is None or t - self._since >= NONCE_LIFE:
            self._nonces, self._since = [secrets.token_hex(8), self._nonces[0]], t
        while True:
            try:
                data, sender = self.sock.recvfrom(MAX + 1)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                break
            body = opened(self.key, data)
            if body is None:
                continue  # unsigned or damaged: no reply, nothing learned
            self._reply(sender, status)
            parts = body.split(",")
            if (
                len(parts) != 4
                or parts[0] != "H"
                or parts[1] not in self._nonces
                or parts[3] not in ("0", "1")
            ):
                continue
            try:
                count = int(parts[2])
            except ValueError:
                continue
            # Only ever forward: a later count under the same number, or the newest number
            # once. Whoever sends it, an older heartbeat played back is ignored.
            newer = parts[1] == self._last_nonce and count > self._count
            fresh = parts[1] == self._nonces[0] and self._last_nonce != self._nonces[0]
            if not (newer or fresh):
                continue
            self._last_nonce, self._count = parts[1], count
            go = parts[3] == "1"
            if go and not self.go:
                guard.clear()  # the operator pressed G
            if not go:
                guard.emergency_stop()
            self.go, self.heard = go, t
        if self.heard is not None and self.go and t - self.heard <= EVERY * 3:
            guard.link(t)
        return self.heard is not None and t - self.heard <= EVERY * 3

    def _reply(self, sender, status):
        body = f"S,{self._nonces[0]},{status[:200].replace('|', '/')}"
        try:
            self.sock.sendto(sign(self.key, body), sender)
        except OSError:
            pass

    def close(self):
        self.sock.close()


class Remote:
    """The operator's end. Call `step` about every EVERY seconds."""

    def __init__(self, key, robot, port=PORT, sock=None):
        self.key, self.robot = key, (robot, port)
        self.sock = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.go, self.count, self.nonce = False, 0, "-"
        self.status, self.heard = "no answer yet", None

    def step(self, now):
        while True:
            try:
                data, sender = self.sock.recvfrom(MAX + 1)
            except (BlockingIOError, InterruptedError, OSError):
                break
            body = opened(self.key, data)
            if body is None or sender[1] != self.robot[1]:
                continue
            parts = body.split(",", 2)
            if len(parts) == 3 and parts[0] == "S":
                self.nonce, self.status, self.heard = parts[1], parts[2], now
        self.count += 1
        try:
            self.sock.sendto(sign(self.key, f"H,{self.nonce},{self.count},{int(self.go)}"), self.robot)
        except OSError:
            pass

    def connected(self, now):
        return self.heard is not None and now - self.heard < 1.0

    def close(self):
        self.sock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--robot", help="the rover's address on the network")
    parser.add_argument("--key", type=Path, help="the shared key file")
    parser.add_argument("--make-key", type=Path, metavar="FILE", help="make a new key file and stop")
    args = parser.parse_args(argv)
    if args.make_key:
        try:
            print(f"Saved {make_key(args.make_key)}. Copy it to the rover; keep it private.")
        except OSError as error:
            parser.exit(1, f"{error}\n")
        return
    if not (args.robot and args.key):
        parser.error("give --robot and --key, or --make-key")
    try:
        remote = Remote(load_key(args.key), args.robot)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    import cv2
    import numpy as np

    title = "Ostinato remote"
    try:
        while True:
            now = time.monotonic()
            remote.step(now)
            screen = np.full((220, 760, 3), 30, np.uint8)
            link = "connected" if remote.connected(now) else "NOT CONNECTED: the rover is stopping"
            state = "GO: moving allowed" if remote.go else "STOPPED"
            for i, (text, color) in enumerate(
                [
                    (state, (80, 200, 80) if remote.go else (60, 60, 230)),
                    (link, (220, 220, 220)),
                    (remote.status[:70], (200, 200, 200)),
                    ("G go   Space stop   Q quit (stops it)", (150, 150, 150)),
                ]
            ):
                cv2.putText(
                    screen, text, (20, 45 + 48 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8 if i else 1.1, color, 2
                )
            cv2.imshow(title, screen)
            key = cv2.waitKey(max(1, int(EVERY * 1000))) & 0xFF
            if key == ord("g"):
                remote.go = True
            elif key == ord(" "):
                remote.go = False
            elif key == ord("q") or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        remote.go = False
        for _ in range(3):
            remote.step(time.monotonic())
        remote.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
