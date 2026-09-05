"""External Repository Integrations & Bridges."""

__all__ = ["IchigoAncBridge"]

def __getattr__(name: str):
    if name == "IchigoAncBridge":
        from src.integrations.ichigo_bridge import IchigoAncBridge
        return IchigoAncBridge
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
