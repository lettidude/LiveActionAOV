"""Execution backends.

`LocalExecutor` is the v1 implementation (single-GPU, in-process).
`DeadlineExecutorStub` is declared now so the architecture flows through a
distributed backend from day one — v2 swaps the stub for a real adapter
(design §14, decision 1).
"""
