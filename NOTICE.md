# Origin notice

The initial HBServe codebase was extracted from the MIT-licensed HBFSim working
tree based on commit `b44a7960ea44e20252aa6c8e758e8f9ed2d7a4b5`. It was renamed,
made independently installable, and stripped of production evidence and local
experiment history for this repository.

HBServe retains the original HBFSim copyright notice in `LICENSE`.

The unified fixed-window frontend and shared client incorporate HBFSim commit
`70c3fd5` (serving/window frontend unification). This standalone port preserves
HBServe's package name, public model schema, and external-simulator interface.
The 70B descriptor preserves the original per-output-channel W8 scale and
embedding storage assumptions through the public model ledger. Paper-specific
analysis runners, production evidence, and local experiment history are not
included.

The simulation-session client incorporates protocol and controller-HBM contract
updates from HBFSim `2a59b7f13461356d33c00ad1f25fdee4e0bf7fb1`, including
logical page invalidation and wear-output arguments required by native SGLang. Digest performance changes,
transaction storage optimizations and optional zone commands are not part
of this compatibility update. See docs/configuration.md.

The client additionally incorporates Host DRAM transaction/accounting and wear
snapshot v2 validation from HBFSim
`60e3f6669c49a7e8c0a0bd299955de2527ec6a70`. The original OCP profile provenance
remains pinned to `2a59b7f`; those profile values are unchanged at `60e3f66`.
