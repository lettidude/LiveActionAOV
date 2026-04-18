# Developing Plugins

*This guide is a stub for Phase 0 and will be filled in in Phase 6.*

## The pass contract

Every pass subclasses `live_action_aov.core.pass_base.UtilityPass` and is
registered via the `live_action_aov.passes` entry point group.

```python
# my_plugin/pass.py
from live_action_aov.core.pass_base import (
    UtilityPass, License, PassType, TemporalMode, ChannelSpec,
)

class MyDepthPass(UtilityPass):
    name = "my_depth"
    version = "0.1.0"
    license = License(spdx="MIT", commercial_use=True)
    pass_type = PassType.GEOMETRIC
    temporal_mode = TemporalMode.PER_FRAME
    produces_channels = [ChannelSpec(name="Z")]

    def preprocess(self, frames): ...
    def infer(self, tensor): ...
    def postprocess(self, tensor): ...
```

```toml
# pyproject.toml
[project.entry-points."live_action_aov.passes"]
my_depth = "my_plugin.pass:MyDepthPass"
```

A complete cookiecutter template ships in `plugin-template/` at v1.0.
