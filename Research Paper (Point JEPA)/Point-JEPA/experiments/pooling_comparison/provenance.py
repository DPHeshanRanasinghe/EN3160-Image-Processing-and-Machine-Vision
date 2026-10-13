"""Explicit, hash-bound relocation of retained historical runs."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def binding_for_checkpoint(path):
    """Return a binding only for an exact recorded checkpoint location and hash."""
    manifest = json.loads((EXPERIMENT / 'relocation_manifest.json').read_text())
    if manifest['version'] != 1:
        raise ValueError('Unsupported relocation manifest')
    for binding in manifest['bindings']:
        for checkpoint in binding['checkpoints'].values():
            if path.resolve() == (ROOT / checkpoint['path']).resolve():
                for prefix in ('summary', 'evaluation'):
                    artifact = ROOT / binding[prefix + '_path']
                    if sha256(artifact) != binding[prefix + '_sha256']:
                        raise ValueError('Historical provenance record hash mismatch')
                if sha256(path) != checkpoint['sha256']:
                    raise ValueError('Historical checkpoint hash mismatch')
                return binding
    return None


def validate_cache_binding(configuration, cache_path, binding):
    """Relocation accepts only the named cache with its retained evaluation hash."""
    recorded = configuration['cache_path']
    if binding is None:
        expected = Path(recorded)
        if not expected.is_absolute():
            expected = ROOT / expected
        if expected.resolve() != cache_path.resolve():
            raise ValueError('Evaluation cache path differs from training cache path')
        return
    if recorded != binding['historical_cache_path']:
        raise ValueError('Historical training cache path is not explicitly mapped')
    if cache_path.resolve() != (ROOT / binding['cache_path']).resolve():
        raise ValueError('Unregistered cache relocation or substitution')
    if sha256(cache_path) != binding['cache_sha256']:
        raise ValueError('Relocated cache hash differs from retained evaluation provenance')
