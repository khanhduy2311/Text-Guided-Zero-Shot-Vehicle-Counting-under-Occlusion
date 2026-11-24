"""Canonical annotations use continuous pixel coordinates, never normalized storage."""
import math
from pathlib import PurePosixPath

SCHEMA_VERSION = 1
SOURCES = {"FSC147", "VisDrone", "UA_DETRAC"}
TARGETS = {"CARPK", "TRANCOS"}
VEHICLES = {"car", "van", "truck", "bus", "motorcycle", "tricycle", "awning_tricycle", "other_vehicle"}


def selected_instances(record, query=None):
    query = record['prompt'] if query is None else query
    if query == record['prompt']:
        classes = set(record['query_classes'])
    elif record['dataset'] in {'VisDrone', 'UA_DETRAC'} and query == 'vehicle':
        classes = VEHICLES
    elif query in record['annotated_classes']:
        classes = {query}
    else:
        raise ValueError(f"Query {query!r} has no exhaustive annotation in {record['id']}")
    return [i for i in record['instances'] if i['class'] in classes]


def validate_record(r):
    if r['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Unsupported schema')
    if r['dataset'] in TARGETS and r['role'].startswith('source_'):
        raise ValueError('Protected target marked as source')
    path = PurePosixPath(r['image'])
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Image paths must be relative to workspace root')
    w, h = r['width'], r['height']
    if w <= 0 or h <= 0:
        raise ValueError('Invalid image size')
    ids = set()
    for instance in r['instances']:
        if instance['id'] in ids:
            raise ValueError('Duplicate instance ID')
        ids.add(instance['id'])
        x, y = instance['point']
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= w and 0 <= y <= h):
            raise ValueError(f"Invalid point in {r['id']}")
        b = instance['box']
        if b is not None:
            check_box(b, w, h)
        if r['supervision'] == 'point' and b is not None:
            raise ValueError('Point-only annotations cannot contain invented boxes')
        if r['supervision'] == 'box' and b is None:
            raise ValueError('Missing instance box')
    for region in r['ignore_regions']:
        check_box(region['box'], w, h)
    for b in r['exemplar_boxes']:
        check_box(b, w, h)
    if r['count'] != len(selected_instances(r)):
        raise ValueError('Count does not match the prompt-selected instances')


def check_box(b, w, h):
    if len(b) != 4 or not all(math.isfinite(v) for v in b):
        raise ValueError('Invalid box values')
    if not (0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h):
        raise ValueError(f'Invalid xyxy box {b} for {w}x{h}')
