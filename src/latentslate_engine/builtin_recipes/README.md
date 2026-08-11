# Built-in recipes

Engine-owned, Comfy-referenced recipe TOML files live here. This directory is part of
the installed package and is read before local/private catalogs. Built-ins must use
exact resource identities and fail closed when their transitive resource closure is
not installed or provisionable.

The catalog is intentionally empty until each operation's artifact contract and
runtime adapter are independently verified. See `docs/RECIPES.md` for the pinned
reference source and migration policy.
