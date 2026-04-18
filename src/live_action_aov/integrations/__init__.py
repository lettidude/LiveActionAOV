"""Pipeline integrations.

v1 ships `StandaloneAdapter` (the default no-op) and declared-but-stubbed
Prism / ShotGrid / OpenPype adapters. Integrations are discovered via the
`live_action_aov.integrations` entry-point group (design §14, decision 4).
"""
