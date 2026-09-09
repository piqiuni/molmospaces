"""External, navigation-only Habitat ObjectNav-v2 adapter.

Nothing in this package modifies or imports private Habitat task state into the
policy.  It only consumes the public RGB-D, GPS, compass and object-goal
observations exposed by the official Challenge 2023 configuration.
"""

from .policy import HabitatInteractiveNavM2Policy, PolicyConfig

__all__ = ["HabitatInteractiveNavM2Policy", "PolicyConfig"]
