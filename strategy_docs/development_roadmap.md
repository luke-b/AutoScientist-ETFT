# AutoScientist-ETFT: Development Roadmap
## Pathway to SOTA TinyML for Global Health

This roadmap outlines the practical, incremental steps to transition the AutoScientist-ETFT repository from its current state into a fully operational engine capable of generating high-impact TinyML models on standard developer hardware (VMs/PCs).

---

### Phase 1: Foundation & Sandbox Finalization (Months 1-2)
**Goal:** Stabilize the core ETFT framework and ensure the Agentic Research Loop (ARL) can execute complex workflows reliably on local, non-GPU hardware.

*   **Milestone 1.1: Robust Local Sandboxing**
    *   Solidify the `DockerSandbox` (`etft/sandbox.py`) to fully support CPU-only execution with strict memory constraints, mimicking edge devices.
    *   Integrate open-source edge ML compilation tools (e.g., TFLite Micro, microTVM) into the sandbox base image.
*   **Milestone 1.2: Tooling & Observability Enhancements**
    *   Enhance `run_logger.py` to specifically track FLOPs, model parameter count, and binary size (`.tflite` or `.h` file size) alongside standard accuracy metrics.
    *   Refine the CLI (`etft-report`) to display these hardware-centric metrics.

### Phase 2: The Healthcare Edge Corpus (Months 3-4)
**Goal:** Build the foundational datasets (`D_Gen`, `D_Rationale`) necessary to teach the LLM how algorithms evolve under extreme memory constraints.

*   **Milestone 2.1: Data Acquisition & Preprocessing**
    *   Select a robust, open-source biomedical time-series dataset (e.g., MIT-BIH Arrhythmia Database for ECG, or open respiration datasets).
    *   Create a standardized data loader accessible to the empirical agents.
*   **Milestone 2.2: Reverse-Engineering the TinyML Trajectory**
    *   Utilize the `corpus/regression_pipeline/` to de-optimize modern edge models back to baseline implementations.
    *   **Trajectory Steps:**
        1. Base: Simple statistical thresholding (Moving Average/Variance).
        2. Evolution 1: Lightweight Decision Trees/Random Forests.
        3. Evolution 2: Minimal 1D-CNNs.
        4. Evolution 3: SOTA TinyML (e.g., depthwise separable convolutions or micro-attention blocks).
    *   Generate the `D_Rationale` focusing heavily on *why* specific ops (like depthwise convolutions) save memory without destroying accuracy.

### Phase 3: "GPU-Poor" ETFT Deployment (Months 5-6)
**Goal:** Execute the full ETFT pipeline locally to generate a novel SOTA+1 architecture for the chosen healthcare task.

*   **Milestone 3.1: Pareto Optimization Calibration**
    *   Tune the Retrospective 80/20 Δ Analysis (`analysis/pareto_delta/`) to prioritize bottlenecks related to activation memory footprints, not just loss/accuracy.
    *   Train the Probabilistic Heuristic Filter (`D_Perf`) to aggressively reject candidates that exceed a 256KB memory threshold.
*   **Milestone 3.2: Launch the Agentic Research Loop**
    *   Deploy literature agents (`agents/literature/`) to query ArXiv for the latest techniques in "ultra-low-power time series classification" and "hardware-aware neural architecture search."
    *   Run empirical agents to perform micro-experiments, validating small architectural perturbations on the CPU sandbox.
*   **Milestone 3.3: SOTA+1 Synthesis & Calibration**
    *   Pass the Evolutionary Replay stage-gate.
    *   Synthesize the SOTA+1 architecture, verifying that it improves on the baseline efficiency/accuracy Pareto frontier.

### Phase 4: Hardware Validation & Open Release (Months 7-8)
**Goal:** Prove the physical viability of the synthesized models and distribute the results for global impact.

*   **Milestone 4.1: Physical Device Deployment**
    *   Flash the ETFT-generated SOTA+1 model onto physical commodity hardware (e.g., ESP32, Arduino Nano 33 BLE).
    *   Measure real-world power draw, inference latency, and memory utilization to validate the "in-silico" sandbox metrics.
*   **Milestone 4.2: Open Source "Health-Edge" Release**
    *   Publish the specific optimized models (C++ headers/TFLite payloads) to an open model zoo.
    *   Release a comprehensive case study documenting the ETFT framework's ability to autonomously engineer a life-saving AI application using only consumer-grade compute.
*   **Milestone 4.3: Community Enablement**
    *   Provide easy-to-use Docker compose setups allowing any developer to replicate the entire TinyML ETFT workflow on their personal laptop.

---

By the end of this roadmap, AutoScientist-ETFT will have transformed from a theoretical framework into a proven, deployed system that demonstrably accelerates innovation in a domain that directly benefits human health, all while operating efficiently within the constraints of a standard development VM.