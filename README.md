# TinyAerialVLN Final V8 / NavLab Studio

**Start with [START_HERE.md](START_HERE.md).**

Research + engineering handoff for: instruction-conditioned monocular RGB aerial VLN, compact C++/CUDA student, real physics simulation, and a later SG2002/ESP32 deployment. The primary UI is NavLab Studio v3, replacing the V7.1 dashboard. This package is a functional management scaffold, not a trained navigation system.

**Verified locally:** 36 stdlib tests; C++17 gradient check; deterministic static semantic SDF generation; same-origin UI/API and preflight blocking. **Not verified:** actual Gazebo flight, UE4 AirSim on RTX 5090, CUDA trainer, SG2002 NPU, firmware integration, real UAV autonomy.

Commands: `./run.sh doctor`, `./run.sh test`, `./run.sh cpp-test`, `./run.sh scene --difficulty S1 --seed 42`, `./run.sh ui`. The only HTTP bind is `127.0.0.1:8765` by default. Do not use port-forwarding without trusted SSH authentication or expose ports publicly.

## License

Proprietary and confidential; internal research use only. No external license or redistribution rights are granted. See [LICENSE](LICENSE). Third-party components under `references/` and `legacy/` retain their own licenses and restrictions.
