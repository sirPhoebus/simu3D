# Review and optimization plan

- Preserve current gravity, blast impulses, support flood-fill, and collider accuracy.
- Remove unreachable legacy physics code and fields identified by compiler warnings.
- Reduce redundant render work for fused shell clusters without changing their mass or collision shapes.
- Verify the physics self-tests and inspect lint output; keep unrelated style changes out of scope.

Results: dead physics paths removed; fused-cluster render instances reduced from about 351k to 149k in the villa blast fixture; fragment count/minimum size now govern splitting; fragment impulses use actual mass. Debug and release suites pass, and the release self-test reports `RESULT=PASS`.

Known limits: the existing irregular collapsed-body fallback still uses a bounding cuboid when greedy decomposition yields more than four boxes. Debug-profile physics timing is not representative; the 8 ms budget is enforced only in release tests.
