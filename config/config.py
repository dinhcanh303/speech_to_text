import os
import re
import yaml
from dotenv import load_dotenv

class Config:
    _config_data = None

    @classmethod
    def _env_constructor(cls, loader, node):
        value = loader.construct_scalar(node)

        def replacer(match):
            key_default = match.group(1)
            if '=' in key_default:
                key, default = key_default.split('=', 1)
            else:
                key, default = key_default, ''
            return os.environ.get(key, default)

        return re.sub(r'\$\{([^}^{]+)\}', replacer, value)

    @classmethod
    def _load_yaml(cls, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        yaml.add_constructor('!ENV', cls._env_constructor, Loader=yaml.SafeLoader)
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @classmethod
    def load(cls, config_path="config/config.yml"):
        cls._config_data = cls._load_yaml(config_path)

    @classmethod
    def get(cls, key, default=None):
        if cls._config_data is None:
            cls.load()
        keys = key.split(".")
        value = cls._config_data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        if value is None:
            return default
        # Auto convert to boolean if applicable
        if isinstance(value, str):
            if value.lower() in ("true", "1"):
                return True
            elif value.lower() in ("false", "0"):
                return False
        return value
