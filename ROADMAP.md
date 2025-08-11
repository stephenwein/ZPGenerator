# ZPGenerator Core Rewrite (Stable API)

## Goals
- Keep public API (`Pulse`, `Source`, `Circuit`, `Detector`, `Processor`) unchanged.
- Preserve tutorial outputs exactly (rtol=1e-9, atol=1e-12).
- Rebuild internals around:
  1. Resolve → Plan (deferred-by-default parameters + runtime defaults)
  2. Virtual batching for sweeps / quasi-static noise
  3. PropagatorStore for interval composition (global timeline; TD/TI unified)

## Milestones
1. **PR1:** Golden-output tests from tutorials
2. **PR2:** Config/Resolve/Plan scaffolding (use old numerics)
3. **PR3:** Global scheduler + virtual batch axis
4. **PR4:** PropagatorStore (prefix strategy) + NumPy backend
5. **PR5:** New evolve + reconstruction with parity
6. **PR6:** Optimisation + perf gate

## Invariants
- Units: ? (validate at resolve)
- RNG: per-batch stream from (global_seed, batch_index)
- API behaviour must match goldens
