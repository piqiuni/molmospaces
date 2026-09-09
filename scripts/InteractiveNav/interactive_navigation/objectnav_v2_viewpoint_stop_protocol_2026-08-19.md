# ObjectNav-v2 VIEW_POINTS and STOP protocol (2026-08-19)

## Official runtime contract

- The Habitat Challenge 2023 ObjectNav task describes success as calling STOP at a pose that is within 1.0 m of a target-category instance and from which an oracle can view that object by rotating/looking without translating. Source: [Habitat Challenge 2023](https://github.com/facebookresearch/habitat-challenge#evaluation).
- The exact `challenge-2023` Habitat-Lab configuration used locally sets `distance_to: VIEW_POINTS` and `success_distance: 0.1`: `habitat/config/habitat/task/objectnav_v2.yaml`.
- `DistanceToGoal` collects every stored `goal.view_points[*].agent_state.position` and computes the geodesic distance from the current pose to the nearest one: `habitat/tasks/nav/nav.py:962-989`.
- `Success` is 1 only when STOP has been called and that distance is strictly smaller than 0.1 m: `habitat/tasks/nav/nav.py:531-545`.

Therefore 0.1 m is a tolerance around a precomputed legal standing/viewing pose. It is not a requirement to put the robot 0.1 m from the object surface.

## Distance from stored viewpoints to the object reference position

The public v2 val archive stores an object `position` and many viewpoint positions, but it does not store the distance from each viewpoint to the nearest point on the object's mesh/surface. The object reference position is especially misleading for long sofas and beds. The following numbers are horizontal XZ Euclidean distances from each stored viewpoint to that stored object reference position, not the official success metric and not surface clearance.

Computed over 290,219 public HM3D-v2 val viewpoints:

| statistic | XZ distance to stored object position |
|---|---:|
| minimum | 0.061 m |
| median | 1.219 m |
| p95 | 2.200 m |
| p99 | 2.634 m |
| maximum | 3.451 m |

Focused scenes/categories:

| scene / category | median | p95 | maximum |
|---|---:|---:|---:|
| 00803 / sofa | 1.458 m | 1.939 m | 2.184 m |
| 00808 / chair | 1.004 m | 1.334 m | 1.475 m |
| 00813 / tv_monitor | 0.988 m | 1.356 m | 1.425 m |

These values do not contradict the challenge's 1.0 m object-proximity prose: a large object's stored reference position is not its nearest surface. The executable benchmark never compares the agent to that reference position for v2 success; it compares against the precomputed legal viewpoints.

## Implication for a public-observation STOP policy

A public RGB-D policy cannot read `DistanceToGoal` or the stored viewpoints. It must estimate whether it occupies a suitable viewing/interaction stance. Official distance, goal centers, viewpoints and top-down navmesh may only be used by evaluator-side posthoc diagnostics.

