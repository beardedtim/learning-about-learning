import logging
import sys
import structlog

# 1. Configure standard logging to use two handlers: one for the file, one for the console
log_file = "simulation_history.log"

logging.basicConfig(
    level=logging.NOTSET,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_file),         # Routes output to the file
        logging.StreamHandler(sys.stdout)      # Routes output to the console
    ]
)

# 2. Configure structlog to act as the formatter and pass data to standard logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        
        # We turn off colors here because ANSI color codes (e.g., ^[[32mINFO^[[0m) 
        # look like garbage when written into a plain text file.
        structlog.dev.ConsoleRenderer(colors=False)
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True
)

Log = structlog.get_logger()