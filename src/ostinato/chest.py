"""Track vertical chest motion in video, relative to the background."""

import cv2
import numpy as np

from .response import body_from_landmarks

MIN_VISIBILITY = 0.7
MIN_POINTS = 12
MIN_CHEST = 0.4  # least chest height in view, in shoulder widths
MAX_STEP = 0.08  # chest motion per frame, in shoulder widths
MAX_CAMERA_STEP = 0.03
LYING_TILT = 40.0  # degrees between the torso and image-down
MIN_TORSO = 0.6  # torso length needed to trust its direction, in shoulder widths
TORSO_RATIO = 1.5  # typical torso length, in shoulder widths
MIN_IN_FRAME = 0.6  # share of a lying person's chest that must be in view


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
    if not np.isfinite(points[:2]).all():
        return None
    h, w = shape[:2]
    if lying_axes(points) is not None:
        return lying_box(points, shape)
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


def lying_axes(points):
    """Return (shoulder center, unit vector toward the hips, scale), or None if upright.

    Seen from the side, a lying person's shoulders overlap, so the torso's length
    sets the scale when it implies wider shoulders than can be seen.
    """
    if not (np.abs(points) < 1e5).all():  # also rejects NaN
        return None
    across = points[0] - points[1]
    width = float(np.linalg.norm(across))
    center = (points[0] + points[1]) / 2
    torso = (points[2] + points[3]) / 2 - center
    length = float(np.linalg.norm(torso))
    if length < MIN_TORSO * width or length < 1:
        return None
    down = torso / length
    if np.degrees(np.arccos(np.clip(down[1], -1, 1))) < LYING_TILT:
        return None
    return center, down, max(width, length / TORSO_RATIO)


def lying_regions(points, shape):
    """Return (chest, head, body strip) polygons and the scale, or None if not usable."""
    center, down, scale = lying_axes(points)
    h, w = shape[:2]
    across = points[0] - points[1]
    if scale < 60 or np.linalg.norm(across) < 0.3 * scale:  # too distant or side-on
        return None
    side = np.array([-down[1], down[0]])

    def polygon(half, start, end):
        a, b = center + start * scale * down, center + end * scale * down
        return np.array([a - half, a + half, b + half, b - half])

    chest = polygon(0.4 * across, 0.05, 0.7)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, np.int32(np.round(chest)), 1)
    area = cv2.contourArea(np.float32(chest))
    if area < 1 or mask.sum() < MIN_IN_FRAME * area:
        return None
    reach = 0.4 * abs(float(across @ side)) + 0.6 * scale
    # Far past both ends, so that the head, legs, and feet are excluded too.
    strip = polygon(reach * side, -20, 20)
    return chest, polygon(0.2 * scale * side, -0.75, -0.3), strip, scale


def lying_box(points, shape):
    """Return the chest's bounding box and scale, or None if not usable."""
    regions = lying_regions(points, shape)
    if regions is None:
        return None
    chest, _, _, scale = regions
    h, w = shape[:2]
    x0, y0 = np.maximum(chest.min(0), 0)
    x1, y1 = np.minimum(chest.max(0), [w, h])
    return int(x0), int(y0), int(x1), int(y1), scale


class ChestTracker:
    """Turn frames into cumulative displacement in shoulder widths."""

    def __init__(self, finder):
        self.finder = finder
        self.reset()

    def reset(self):
        self._gray = None
        self._box = None
        self._points = None
        self.position = 0.0
        self.head = 0.0  # head motion on the same scale, while it can be seen
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
        if self._gray is not None and self._gray.shape != gray.shape:
            self._gray = self._box = None  # the camera changed resolution
        prev_box = self._box
        value, reason = np.nan, "chest not visible"
        if box is not None and prev_box is not None:
            value, reason = self._measure(gray, prev_box, self._points)
        elif box is not None:
            reason = "starting"
        self._gray, self._box, self.box = gray, box, box
        self._points = points
        return value, reason

    def _masks(self, gray, box, points):
        """Return masks for the chest, the background, and the head."""
        x0, y0, x1, y1, width = box
        chest, head = np.zeros_like(gray), np.zeros_like(gray)
        background = np.full_like(gray, 255)
        if lying_axes(points) is not None:
            chest_area, head_area, strip, _ = lying_regions(points, gray.shape)
            cv2.fillConvexPoly(chest, np.int32(np.round(chest_area)), 255)
            cv2.fillConvexPoly(head, np.int32(np.round(head_area)), 255)
            cv2.fillConvexPoly(background, np.int32(np.round(strip)), 0)
            return chest, background, head

        chest[y0:y1, x0:x1] = 255
        # Exclude a generous column around the person from the background.
        bx0, bx1 = max(0, int(x0 - 0.6 * width)), int(x1 + 0.6 * width)
        background[:, bx0:bx1] = 0
        # Just above the shoulders, which is the head when the person is upright.
        cx, top = (x0 + x1) // 2, int(y0 - 0.05 * width)
        hy0, hy1 = max(0, int(top - 0.75 * width)), max(0, int(top - 0.3 * width))
        hx0, hx1 = max(0, int(cx - 0.2 * width)), int(cx + 0.2 * width)
        head[hy0:hy1, hx0:hx1] = 255
        return chest, background, head

    def _measure(self, gray, box, points):
        width = box[4]
        chest, background, head_mask = self._masks(gray, box, points)
        flow = self._median_flow(gray, chest)
        if flow is None:
            return np.nan, "not enough chest detail"
        camera = self._median_flow(gray, background)
        if camera is None:
            return np.nan, "cannot check camera motion"
        if np.linalg.norm(camera) / width > MAX_CAMERA_STEP:
            return np.nan, "camera moving"

        # A chest rises away from the ground, which is image-up for an upright person
        # and for a lying person seen from the side.
        step = (flow - camera) / width
        if np.linalg.norm(step) > MAX_STEP:
            return np.nan, "too much movement"
        self.position += float(step[1])
        self.background += float(camera[1]) / width

        head = self._median_flow(gray, head_mask)
        if head is not None:
            self.head += float(head[1] - camera[1]) / width
        return self.position, "ok"
