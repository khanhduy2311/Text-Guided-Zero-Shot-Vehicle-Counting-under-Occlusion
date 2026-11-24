"""Load official CountGD++ safetensors without silently missing active weights."""
import torch
from safetensors import safe_open
from safetensors.torch import load_file


def restore_shared_aliases(model, state, metadata):
    state = dict(state)
    tensors = dict(model.named_parameters(remove_duplicate=False))
    tensors.update(model.named_buffers(remove_duplicate=False))
    restored = {}
    for alias, canonical in (metadata or {}).items():
        if alias not in tensors:
            continue  # Non-tensor safetensors metadata, e.g. "format".
        if canonical not in tensors or tensors[alias] is not tensors[canonical]:
            raise ValueError(f'Safetensors alias is not shared by this model: {alias} -> {canonical}')
        if canonical not in state:
            raise ValueError(f'Missing canonical shared weight: {canonical}')
        if alias in state and not torch.equal(state[alias], state[canonical]):
            raise ValueError(f'Conflicting shared weights: {alias}')
        state[alias] = state[canonical]
        restored[alias] = canonical
    return state, restored


def load_official_countgdpp(model, path):
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        metadata = handle.metadata()
    state, aliases = restore_shared_aliases(model, load_file(str(path), device='cpu'), metadata)
    expected = model.state_dict()
    missing = sorted(set(expected)-set(state))
    unexpected = sorted(set(state)-set(expected))
    # The GitHub batch-forward class still declares this unused legacy module.
    # combine_features() never calls it. The official HF app class omits it.
    allowed_missing = [name for name in missing if name.startswith('feature_map_encoder.')]
    if missing != allowed_missing or unexpected:
        raise ValueError(f'CountGD++ active weight mismatch: missing={missing}, unexpected={unexpected}')
    model.load_state_dict(state, strict=not missing)
    if missing:
        model.feature_map_encoder.requires_grad_(False)
    return {'checkpoint_format': 'official safetensors', 'shared_aliases_restored': aliases,
            'unused_legacy_missing_keys': missing, 'missing_active_keys': [], 'unexpected_keys': [],
            'loaded_state_keys_after_alias_expansion': len(state),
            'note': 'All active weights loaded. Only the uncalled legacy feature_map_encoder is absent/frozen.'}
