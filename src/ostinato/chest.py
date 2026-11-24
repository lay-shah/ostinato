"""Track vertical chest motion in video, relative to the background."""

import cv2
import numpy as np

from .response import body_from_landmarks

MIN_VISIBILITY = 0.7
MIN_POINTS = 12
MIN_CHEST = 0.4  # least chest height in view, in shoulder widths
MAX_STEP = 0.08  # chest motion per frame, in shoulder widths
MAX_CAMERA_STEP = 0.03


class PoseFinder:
    """Locate one person's shoulders and hips. Runs on the device."""

    def __init__(self):
        import mediapipe as mp

        self.body = None  # arm positions from the latest frame

        self._pose = mp.solutions.pose.Pose(model_complexity=1)

    def __call__(self, bgr):
        """Return pixel array [l_shoulder, r_shoulder, l_hip, r_hip] or None."""
        self.body = None
        result = self._pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not result.pose_landmarks:
            return None
        marks = result.pose_landmarks.landmark
        h, w = bgr.shape[:2]
        self.body = body_from_landmarks([(m.x * w, m.y * h) for m in marks], [m.visibility for m in marks], (w, h))
        if min(marks[11].visibility, marks[12].visibility) < MIN_VISIBILITY:
            return None
        return np.array([[marks[i].x * w, marks[i].y * h] for i in (11, 12, 23, 24)])

    def close(self):
        self._pose.close()


def chest_box(points, shape):
    """Return (x0, y0, x1, y1, shoulder_width), or None if not usable."""
    h, w = shape[:2]
    left, right = points[0], points[1]
    width = float(np.linalg.norm(left - right))
    if width < 60:  # too distant or side-on to measure
        return None
    cx, cy = (left + right) / 2
    x0, x1 = cx - 0.4 * width, cx + 0.4 * width
    y0, y1 = cy + 0.05 * width, cy + 0.7 * width
    y1 = min(y1, h)  # a close webcam often cuts off the lower chest
    if x0 < 0 or y0 < 0 or x1 > w or y1 - y0 < MIN_CHEST * width:
        return None
    return int(x0), int(y0), int(x1), int(y1), width


class ChestTracker:
    """Turn frames into cumulative displacement in shoulder widths."""

    def __init__(self, finder):
        self.finder = finder
        self.reset()

    def reset(self):
        self._gray = None
        self._box = None
        self.position = 0.0
        self.background = 0.0  # vertical motion subtracted so far
        self.box = None

    def _median_flow(self, gray, mask):
        pts = cv2.goodFeaturesToTrack(self._gray, 120, 0.01, 5, mask=mask)
        if pts is None or len(pts) < MIN_POINTS:
            return None
        nxt, ok, _ = cv2.calcOpticalFlowPyrLK(self._gray, gray, pts, None)
        if nxt is None:
            return None
        back, ok2, _ = cv2.calcOpticalFlowPyrLK(gray, self._gray, nxt, None)
        error = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
        keep = (ok.ravel() == 1) & (ok2.ravel() == 1) & (error < 0.5)
        if keep.sum() < MIN_POINTS:
            return None
        return np.median((nxt - pts).reshape(-1, 2)[keep], axis=0)

    def update(self, bgr):
        """Return (displacement or NaN, reason)."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        points = self.finder(bgr)
        box = chest_box(points, bgr.shape) if points is not None else None
        prev_box = self._box
        value, reason = np.nan, "chest not visible"
        if box is not None and prev_box is not None:
            value, reason = self._measure(gray, prev_box)
        elif box is not None:
            reason = "starting"
        self._gray, self._box, self.box = gray, box, box
        return value, reason

    def _measure(self, gray, box):
        x0, y0, x1, y1, width = box
        chest = np.zeros_like(gray)
        chest[y0:y1, x0:x1] = 255
        flow = self._median_flow(gray, chest)
        if flow is None:
            return np.nan, "not enough chest detail"

        # Exclude a generous column around the person from the background.
        background = np.full_like(gray, 255)
        bx0, bx1 = max(0, int(x0 - 0.6 * width)), int(x1 + 0.6 * width)
        background[:, bx0:bx1] = 0
        camera = self._median_flow(gray, background)
        if camera is None:
            return np.nan, "cannot check camera motion"
        if np.linalg.norm(camera) / width > MAX_CAMERA_STEP:
            return np.nan, "camera moving"

        step = (flow - camera) / width
        if np.linalg.norm(step) > MAX_STEP:
            return np.nan, "too much movement"
        self.position += float(step[1])
        self.background += float(camera[1]) / width
        return self.position, "ok"
