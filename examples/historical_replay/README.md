# Historical Replay example

This directory is reserved for the smallest executable ETFT falsification test.

Planned shape:

```
examples/historical_replay/
├── README.md
├── fixtures/
│   └── tiny_trajectory.jsonl
├── run_baseline.py
├── run_etft.py
└── evaluate.py
```

The first implementation should keep the model, candidate budget, evaluator, and benchmark identical across conditions.

Conditions:

1. `B0`: unordered/final-state evidence;
2. `B1`: ordered trajectory without causal metadata;
3. `ETFT`: ordered trajectory + causal transition metadata.

The command should eventually produce a machine-readable `results.json` containing at least:

- condition;
- transition_id;
- candidate_evaluations;
- tokens_used;
- threshold_reached;
- best_score;
- valid_candidate_count;
- discovery_yield.

See [../../EXPERIMENT.md](../../EXPERIMENT.md) for the frozen research protocol.
