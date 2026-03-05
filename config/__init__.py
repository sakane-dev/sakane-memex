import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).parent.parent
_DEFAULT_CONFIG = _ROOT / 'config' / 'settings.yaml'


def load_config(path=None):
    config_path = Path(path or os.getenv('SAKANE_CONFIG', str(_DEFAULT_CONFIG)))
    if not config_path.exists():
        raise FileNotFoundError(f'Config not found: {config_path}')
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    base = config_path.parent.parent
    for key, val in cfg.get('paths', {}).items():
        cfg['paths'][key] = str(base / val)
    return cfg


def setup_logging(cfg):
    log_cfg = cfg.get('logging', {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get('level', 'INFO')),
        format=log_cfg.get('format', '%(asctime)s | %(levelname)s | %(name)s | %(message)s'),
    )
