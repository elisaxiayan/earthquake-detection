from typing import Any

from mcp.server.fastmcp import FastMCP

from disaster_core import create_detector_from_env


mcp = FastMCP("disaster-verifier", log_level="ERROR")
detector = create_detector_from_env()


@mcp.tool()
def verify_disaster_text(text: str) -> dict[str, Any]:
    """Verify one disaster statement for factual conflicts."""
    return detector.run(text)


@mcp.tool()
def verify_disaster_batch(texts: list[str]) -> dict[str, Any]:
    """Verify multiple disaster statements in batch."""
    return detector.run_batch(texts)


if __name__ == "__main__":
    mcp.run(transport="stdio")
