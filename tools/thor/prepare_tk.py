"""Prepare a local include overlay for pinned TK ARM64/compiler-order fixes.

The submodule stays at its recorded commit. Two pack helpers use explicit
signed-char conversion; CUDA 13's char2 has signed-char components while plain
char on ARM64 is unsigned. Semaphore wait asm also receives memory clobbers
so the compiler cannot move protected accesses across waits.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def prepare():
    source = ROOT / 'ThunderKittens/include'
    target = ROOT / 'local/thunderkittens/include'
    relative = Path('common/base_types.cuh')
    text = (source / relative).read_text()
    old = 'return char2{i, i};'
    if text.count(old) != 2:
        raise RuntimeError('pinned TK pack helpers changed; compatibility fix needs review')
    modified = text.replace(old, 'return char2{static_cast<signed char>(i), static_cast<signed char>(i)};')
    patches = {relative: (text, modified)}
    for file, expected_count in [('ops/thread/memory/util/util.cuh', 3),
                                 ('ops/thread/memory/util/tma.cuh', 1)]:
        path = Path(file)
        original = (source / path).read_text()
        before = '"r"(kPhaseBit)\n    );'
        after = '"r"(kPhaseBit) : "memory"\n    );'
        if original.count(before) != expected_count:
            raise RuntimeError(f'pinned TK wait asm changed: {file}')
        patches[path] = (original, original.replace(before, after))
    for path in source.rglob('*'):
        out = target / path.relative_to(source)
        if path.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        elif path.relative_to(source) not in patches and not out.exists():
            out.symlink_to(path)
    records = []
    for path, (original, patched) in patches.items():
        out = target / path
        if out.is_symlink():
            out.unlink()
        out.write_text(patched)
        records.append({'file': str(path),
                        'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
                        'patched_sha256': hashlib.sha256(patched.encode()).hexdigest()})
    manifest = {'source': str(source), 'patches': records,
                'reason': 'ARM64 signed-char2 conversion and compiler memory barriers on semaphore waits'}
    (target.parent / 'overlay.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(target.parent)


if __name__ == '__main__':
    prepare()
