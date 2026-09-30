"""Isaac Lab / FORGE server-side code.

Everything in this subpackage imports isaaclab/isaacsim and must only ever
be run from .venv-isaaclab, never imported from the main .venv (which
doesn't have Isaac Sim installed). The client side (expo_ft/env/isaaclab_env.py)
lives outside this subpackage precisely to keep that boundary explicit.
"""
