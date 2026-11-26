"""Track vertical chest motion in video, relative to the background."""

import cv2
import numpy as np

from .response import body_from_landmarks, lying_axes

MIN_VISIBILITY = 0.7
MIN_POINTS = 12
MIN_CHEST = 0.4  # least chest height in view, in shoulder widths
MAX_STEP = 0.08  # chest motion per frame, in shoulder widths
MAX_CAMERA_STEP = 0.03
MAX_JUMP = 0.6  # shift of the shoulders between frames, in shoulder widths: maybe someone else
MAX_RESIZE = 1.5
SWITCH_FRAMES = 4  # frames a new position must hold before it is followed instead
MIN_SIDE_POINTS = 5
MAX_SCATTER = 1.0  # pixels from the median, for points that should move together
FEATURE_MARGIN = 8  # pixels around a region, so that its corners are judged the same
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
        self.body = body_from_landmarks(
            [(m.x * w, m.y * h) for m in marks], [m.visibility for m in marks], (w, h)
        )
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
        self._person = None  # shoulder center, width, and scale last followed
        self._away = 0
        self.switched = False  # now following what may be someone else
        self.position = 0.0
        self.head = 0.0  # head motion on the same scale, while it can be seen
        self.sides = [0.0] * 4  # halves of the chest: right, left, and its two ends
        self.sideways = 0.0  # chest motion across the image, on the same scale
        self.background = 0.0  # vertical motion subtracted so far
        self.box = None

    def _features(self, mask):
        """Corners to follow in the previous frame. Only the mask's surroundings are searched."""
        x, y, w, h = cv2.boundingRect(mask)
        x0, y0 = max(0, x - FEATURE_MARGIN), max(0, y - FEATURE_MARGIN)
        area = np.s_[y0 : y + h + FEATURE_MARGIN, x0 : x + w + FEATURE_MARGIN]
        pts = cv2.goodFeaturesToTrack(self._gray[area], 120, 0.01, 5, mask=mask[area]) if w else None
        if pts is None or len(pts) < MIN_POINTS:
            return np.empty((0, 1, 2), np.float32)
        return pts + np.float32([x0, y0])

    def _median_flows(self, gray, masks):
        """Return the median motion in each mask, or None. All are followed at once.

        Also keeps the first mask's points and motion, to compare parts of the chest.
        """
        self._followed = None
        groups = [self._features(mask) for mask in masks]
        pts = np.concatenate(groups)
        if not len(pts):
            return [None] * len(masks)
        nxt, ok, _ = cv2.calcOpticalFlowPyrLK(self._gray, gray, pts, None)
        back, ok2, _ = cv2.calcOpticalFlowPyrLK(gray, self._gray, nxt, None)
        error = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
        keep = (ok.ravel() == 1) & (ok2.ravel() == 1) & (error < 0.5)
        moved = (nxt - pts).reshape(-1, 2)
        flows, start = [], 0
        for group in groups:
            part = slice(start, start + len(group))
            start += len(group)
            good = moved[part][keep[part]]
            flow = np.median(good, axis=0) if len(good) >= MIN_POINTS else None
            # On a plain surface the "corners" are camera noise, and scatter.
            if flow is not None and np.median(np.linalg.norm(good - flow, axis=1)) > MAX_SCATTER:
                flow = None
            if flow is not None and not flows:
                self._followed = pts[part][keep[part]].reshape(-1, 2), good
            flows.append(flow)
        return flows

    def update(self, bgr):
        """Return (displacement or NaN, reason)."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        points = self.finder(bgr)
        self.switched = False
        if self._gray is not None and self._gray.shape != gray.shape:
            self._person = None
        box = chest_box(points, bgr.shape) if points is not None else None
        if self._gray is not None and self._gray.shape != gray.shape:
            self._gray = self._box = None  # the camera changed resolution
        if box is not None and self._elsewhere(points, box[4]):
            box = None
        prev_box = self._box
        value, reason = np.nan, "chest not visible"
        if box is not None and prev_box is not None:
            value, reason = self._measure(gray, prev_box, self._points)
        elif box is not None:
            reason = "starting"
        self._gray, self._box, self.box = gray, box, box
        self._points = points
        return value, reason

    def _elsewhere(self, points, scale):
        """True while the pose has jumped, so it may be someone else or a glitch."""
        center = (points[0] + points[1]) / 2
        # Not `scale`, which changes when a tilted person counts as lying down.
        width = float(np.linalg.norm(points[0] - points[1]))
        if self._person is not None:
            old, old_width, old_scale = self._person
            near = np.linalg.norm(center - old) <= MAX_JUMP * max(old_scale, scale)
            if not (near and 1 / MAX_RESIZE <= width / old_width <= MAX_RESIZE):
                self._away += 1
                if self._away < SWITCH_FRAMES:
                    return True
                self.switched = True
        self._person, self._away = (center, width, scale), 0
        return False

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
        flow, camera, head = self._median_flows(gray, self._masks(gray, box, points))
        if flow is None:
            return np.nan, "not enough chest detail"
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
        where, moved = self._followed
        across = points[0] - points[1]
        along = np.array([-across[1], across[0]])
        # Parts: right, left, then the two ends of the chest, split at its middle point.
        for k, axis in enumerate((across, along)):
            place = where @ axis
            for side in (0, 1):
                part = moved[(place > np.median(place)) == bool(side)]
                # With too few points in one part, it is taken to move with the rest.
                own = np.median(part[:, 1]) if len(part) >= MIN_SIDE_POINTS else flow[1]
                self.sides[2 * k + side] += float(own - camera[1]) / width
        self.sideways += float(step[0])
        self.background += float(camera[1]) / width

        if head is not None:
            self.head += float(head[1] - camera[1]) / width
        return self.position, "ok"
