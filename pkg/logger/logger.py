import logging
import logging.handlers
import os
from config.config import Config

def setup_logger(name):
    log_config = Config.get("log", {})
    log_file = f"./logs/{name}.log"
    log_level = log_config.get("level", "info").upper()

    max_bytes = int(log_config.get("max_size", 100)) * 1024 * 1024
    backup_count = int(log_config.get("max_backups", 5))
    compress = str(log_config.get("compress", "false")).lower() == "true"

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Ensure logs directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if compress:
        try:
            import shutil
            import glob

            class CompressingHandler(logging.Handler):
                def emit(self, record):
                    log_dir = os.path.dirname(log_file)
                    for file in glob.glob(f"{log_file}.*"):
                        if not file.endswith(".gz"):
                            shutil.make_archive(file, 'gztar', root_dir=log_dir, base_dir=os.path.basename(file))
                            os.remove(file)

            logger.addHandler(CompressingHandler())
        except Exception as e:
            logger.warning(f"Compression not enabled: {e}")
    
    return logger