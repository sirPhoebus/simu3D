simu3d

A Rust voxel engine with one gravitational acceleration, structural collapse, and a bazooka that breaks hit blocks into cubic-grid fragments. Up to 350 fragments eject from each block; the rest remain in one rigid chunk. Ejected fragments despawn after 5 seconds.

Run

  cargo run --release
  cargo run --release -- --self-test

ZQSD moves. Those are the AZERTY letters. W and A do not move. Mouse movement looks while the window is focused. The pointer is captured, so look does not stop when it would have left the window. Esc releases it; click the window to capture it again. Player height is adjustable with F1 or `player_height_m` in `simu3d.toml`. F1 opens settings. MAP, with the left and right arrows, loads the villa or the kilometre biome. The biome ground does not break. Trees, bushes, and rocks do. Mouse sensitivity in that panel runs from 0.0001 to 0.05 radians per count. Left mouse fires the bazooka. R reloads. Esc releases the mouse or closes settings. F3 prints counts.

A metal fence runs around the plot. It does not break, and it is taller than a jump, so walking off the edge is blocked. The grass, soil, terrace, and house floors do break. There is no hidden floor under the plot: a hole is a fall.

The demo is a two-storey villa. The south face is a glass pavilion. The east bedroom wing is a slab carried by walls, not by an anchor flag. Sever that path and the wing falls under the same gravity as a rocket. The west carport roof does the same. You start on the south terrace, facing the glass.

Gravity

The acceleration is constant, default 9.80665 m/s², pointed down. It is the same value for the player, rockets, fragments, and any structure that has lost a face-connected path to the anchored ground course. Altitude variation of g across a house is about 0.0003 percent and is ignored.

Rapier 0.36 integrates free flight with substepped symplectic Euler, which does not land on the closed form. After each step, bodies with no contact are overwritten with

  y = y0 + v0 t - ½ g t²
  v = v0 - g t

Bodies in contact keep Rapier's impulse solver. That is the part that can stack. There is no air drag. The blast is not a shockwave solver: it is a radial impulse that falls off as (1 - d/R)², clamped so a fragment does not leave at an absurd speed.

Measured by `cargo run --no-default-features -- --self-test` on this machine:

  free fall, g = 9.80665, t = 1 s: position error 3.3e-6 m, velocity error 1.1e-5 m/s
  free fall, g = 2.5, t = 1 s: position error 2.4e-6 m
  unsupported column, t = 0.5 s: position error 5.4e-7 m
  one concrete block in a 0.12 m blast: ejected fragments plus retained cubes equal 1000
  ejected fragments: present at 4.9 s, gone at 5.05 s
  1000-way dissection of a 0.215 m cube: volume error 4.3e-8 m³, overlap 3.5e-12 m³

Fragments use an equal cubic grid, with resolution rounded to the nearest cube root of `fragment_count` and capped at 10 per axis. The default 1000 fragments tile a cubic block exactly. Mass is density times volume, and each piece receives its share of the impulse.

Configuration

simu3d.toml, or flags that override it:

  --fragment-lifetime 5
  --fragment-count 10
  --gravity 9.80665
  --explosion-radius 2.8

`fragment_count` selects the nearest cubic grid from 1³ to 10³. Splitting stops when a child cube would be smaller than `min_fragment_extent` (default 0.005 m). A 0.215 m block produces 1000 pieces with the default settings.

What this is not

It is not a per-atom simulation, and it is not a port of Three.js. The camera, mesh, and material split are the ideas taken from that kind of engine. Drawing is wgpu. Contacts are Rapier. The standing house is a greedy mesh plus merged static colliders. Only broken and unsupported material becomes a rigid body, because a villa of separate dynamic blocks does not run.
