# Breathing estimator: how it decides, and what it has been tested on

Everything here is synthetic. It shows how the estimator behaves on motion we made up. Accuracy on people needs [counted-breath trials](VIEWPOINT_TRIALS.md).

## How a rate is accepted

1. At least 25 s of chest motion, tracked in at least 80% of frames, with no gap over 1 s.
2. Enough motion, with no lurch that dwarfs it.
3. The signal must not mirror the background. A curtain or fan is otherwise subtracted as camera motion and looks like breathing.
4. The head must not move with the chest by about as much. That is rocking. This check needs a visible head.
5. Displacement and velocity must agree on a rate between 6 and 42 breaths/min, and the signal must repeat itself one breath later.
6. Motion must not fade for about a breath anywhere in the window, so an old rate does not linger.
7. Either the rhythm is very clear and steady, or at least four breaths can be counted from both peaks and troughs, with similar length and depth, and agree with the spectrum.

Otherwise it reports "unable to assess", which says nothing about whether the person is breathing.

## Benchmark

`python -m ostinato.benchmark --trials 500 --seed 777` gives:

| Synthetic breathing, 8-30/min | Windows given a rate | Mean error | Largest error |
|---|---:|---:|---:|
| Regular | 97% | 0.2 | 0.9 |
| Typical unevenness | 75% | 0.4 | 1.5 |
| Long pauses | 40% | 0.4 | 1.6 |
| Irregular | 4% | 0.4 | 1.0 |

Errors are in breaths/min and cover only windows given a rate. Each is a single 30 s window; the live app keeps trying.

| Other motion | Given a false rate |
|---|---:|
| Tracking drift, white noise, fidgeting | 0% |
| Posture shifts | 0.4% |
| Random sway across 6-36/min | 0% |
| Random sway within 12-24/min | 4.2% |
| Random sway within 6-18/min | 3.8% |

| Seconds after motion stops | 8/min | 12/min | 16/min | 24-32/min |
|---|---:|---:|---:|---:|
| 4 s | 14% | 16% | 1% | 0% |
| 6 s | 7% | 1% | 0.2% | 0% |
| 8 s | 1% | 0.4% | 0% | 0% |

These are the shares of windows that still showed a rate.

## Lying down

When the torso is tilted 40 degrees or more in the image, the chest, head, and background regions turn with it. Only motion up and down the image is measured. That suits a level camera looking from the side, because the chest rises away from the ground. Straight overhead, the chest mostly moves toward the camera and is likely to be declined.

On 24 synthetic clips per row, with the torso at 50-180 degrees:

| Clip | Given a rate before | Now |
|---|---:|---:|
| Breathing, lying | 7 | 24 |
| Breathing, upright | 24 | 24 |
| Rocking (false rate) | 7 | 0 |
| Still, moving background, or moving arm (false rate) | 0 | 0 |

These are drawn textures with given landmarks. Nobody lying down has been measured.

## Known weaknesses

- Lying down, a hand raised toward the ceiling counts when seen from the side, up to about 40 degrees above level. From higher, or with the head or feet toward the camera, it may be missed.

- Motion that rises and falls regularly within the breathing band can pass, such as steady sway or wind on clothing.
- Irregular breathing is almost always declined.
- A slow breather can show an old rate for several seconds after motion stops.
- One person only. The app cannot tell whose chest or hand it is seeing.
