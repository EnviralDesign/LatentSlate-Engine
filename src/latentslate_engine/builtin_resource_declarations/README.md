# Built-in resource declarations

Engine-owned resource declaration TOML files live here. This directory is part of
the installed package and is loaded before user-owned declarations in the Engine
data directory. Built-in declarations describe known artifacts but never make an
absent artifact available; availability is derived from the artifact on disk.

The catalog is intentionally empty until a recipe and its runtime contract have
been independently verified. Declarations should use exact, credential-free source
identities; credentials remain external to the catalog.
