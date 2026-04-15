import logging
import warnings


def configure_logging(level: str) -> None:
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    warnings.simplefilter("default")
    logging.captureWarnings(True)
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging.getLogger(logger_name).setLevel(resolved_level)
