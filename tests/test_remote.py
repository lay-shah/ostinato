import socket
import time

import pytest

from ostinato import remote
from ostinato.remote import Listener, Remote, load_key, make_key, opened, sign
from ostinato.rover import STOP, Drive, Guard

KEY = bytes(range(32))


@pytest.fixture
def pair():
    listener = Listener(KEY, port=0, host="127.0.0.1")
    r = Remote(KEY, "127.0.0.1", port=listener.port)
    yield listener, r
    listener.close()
    r.close()


def settle(listener, r, guard, t, go, steps=3):
    """Exchange a few messages, as about 0.3 s would."""
    r.go = go
    for _ in range(steps):
        r.step(t)
        t += 0.1
        time.sleep(0.01)  # let the message arrive
        listener.poll(t, guard, "status here")
        time.sleep(0.01)
    r.step(t)  # read the last answer
    return t


def allowed(guard, t):
    guard.sensors(t, 5, 5)
    guard.person(t, 3, False)
    guard.allow(t, Drive(0.1, 0))
    return guard.allow(t + 0.05, Drive(0.1, 0)) != STOP


def test_go_and_stop(pair):
    listener, r = pair
    guard = Guard()
    guard.emergency_stop()
    t = settle(listener, r, guard, 0.0, go=False)
    assert r.connected(t) and r.status == "status here" and not allowed(guard, t)
    t = settle(listener, r, guard, t, go=True)
    assert listener.go and allowed(guard, t)
    t = settle(listener, r, guard, t, go=False)
    assert guard.stopped and not allowed(guard, t)


def test_silence_stops_it(pair):
    listener, r = pair
    guard = Guard()
    t = settle(listener, r, guard, 0.0, go=True)
    assert allowed(guard, t)
    time.sleep(0.01)
    listener.poll(t, guard)  # the last heartbeat sent
    t += 0.6
    listener.poll(t, guard)
    assert not allowed(guard, t) and guard.reason == "no news from link"


def test_ignores_unsigned_wrong_key_and_replayed(pair):
    listener, r = pair
    guard = Guard()
    t = settle(listener, r, guard, 0.0, go=True)
    nonce = r.nonce
    t = settle(listener, r, guard, t, go=False)
    assert guard.stopped
    sneaky = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = ("127.0.0.1", listener.port)
    for data in (
        f"H,{nonce},999,1".encode(),
        sign(bytes(32), f"H,{nonce},999,1"),
        sign(KEY, f"H,{nonce},1,1"),  # an old count
        sign(KEY, "H,0000000000000000,999,1"),  # a number it never gave out
        sign(KEY, f"H,{nonce},x,1"),
        b"\xff" * 600,
    ):
        sneaky.sendto(data, target)
        time.sleep(0.01)
        t += 0.05
        listener.poll(t, guard)
    sneaky.close()
    assert guard.stopped and not listener.go


def test_a_restarted_remote_is_heard_again(pair):
    listener, first = pair
    guard = Guard()
    t = settle(listener, first, guard, 0.0, go=True)
    second = Remote(KEY, "127.0.0.1", port=listener.port)
    try:
        for _ in range(30):
            t = settle(listener, second, guard, t, go=False, steps=1)
        assert guard.stopped
    finally:
        second.close()


def test_signing():
    body = "H,ab,1,1"
    assert opened(KEY, sign(KEY, body)) == body
    assert opened(KEY, sign(KEY, body).replace(b"1|", b"0|")) is None
    assert opened(bytes(32), sign(KEY, body)) is None
    assert opened(KEY, b"no bar") is None


def test_key_files(tmp_path):
    path = make_key(tmp_path / "key.txt")
    assert len(load_key(path)) == 32 and (path.stat().st_mode & 0o777) == 0o600
    with pytest.raises(FileExistsError):
        make_key(path)
    (tmp_path / "bad.txt").write_text("1234")
    with pytest.raises(ValueError):
        load_key(tmp_path / "bad.txt")
    remote.main(["--make-key", str(tmp_path / "other.txt")])
    with pytest.raises(SystemExit):
        remote.main(["--robot", "127.0.0.1"])
