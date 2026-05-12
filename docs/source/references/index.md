# Reference Documentation

These pages contain the original standalone documentation that previously lived as individual `README_*.md` files. They are preserved here as detailed references.

- [Training Code Structure](training_code.md) — training codebase architecture, pipeline flow, configuration system, and key classes
- [Deployment Code & Program Flow](deployment_code.md) — CLI arguments, logging, observation config, and visualization tools
- [Observation Configuration](observation_config.md) — YAML config format, all observation types with dimensions, and how to create custom observations
- [Motion Reference Data](motion_reference.md) — reference motion file format, conversion, verification, and usage
- [Kinematic Planner ONNX Model](planner_onnx.md) — detailed input/output specification for the ONNX-exported kinematic planner
- [X2 Heuristic Locomotion Planner](x2_heuristic_planner.md) — Python heuristic planner that fills the role of the (unreleased) X2 kinematic planner: motion-primitive curator, 50 Hz state-machine daemon, scripted/keyboard/ZMQ command surface, and the layered validation pyramid
- [JetPack 6 Flashing Guide](jetpack6.md) — flash the Orin NX on the Unitree G1
- [Decoupled WBC (N1.5 / N1.6)](decoupled_wbc.md) — the earlier Decoupled WBC stack used in Gr00t N1.5 and N1.6
- [X2 ↔ Isaac-GR00T Data Contract](x2_isaac_groot_data_contract.md) — the LeRobot v2.1 schema, ModalityConfig surface, and CLI knobs the X2 VLA pipeline must match (M0 reference)
- [X2 Deploy ZMQ Wire Protocol](x2_zmq_protocol.md) — the binding ZMQ message envelope (1280 B JSON header + binary payload) plus the `pose` / `command` / `planner` / `x2_debug` / `robot_config` schemas shared by C++ deploy, Python VLA, and mock helpers (M2 reference)
- [X2 Deploy: C++ ZMQ Port Plan](x2_zmq_cpp_port_plan.md) — concrete handoff plan for porting the G1 deploy's ZMQ input/output stack to the X2 deploy (drop-in `ZmqPoseInputSource`, `ZmqDebugPublisher`, CLI/CMake wiring, sim smoke gate; M2 closeout work)
