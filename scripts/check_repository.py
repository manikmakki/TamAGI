"""Check repository contents and maintained Markdown links before publishing."""
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import unicodedata
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIRECTORIES = {
    '.platform', 'data', 'workspace', '.venv', 'venv', '.agents', '.claude',
    '.codex', '.aws', '.ruff_cache', '.pytest_cache', 'build', 'dist', '__pycache__',
}
GENERATED_SUFFIXES = {'.log', '.jsonl', '.pyc', '.pyo', '.sqlite', '.sqlite3', '.gpg', '.gguf', '.safetensors'}
PERSONAL_NAME = re.compile(r'\b(?:Connor|Ember|Echo)\b')
INSTALLATION_ID = re.compile(r'\b(?:resident|installation|household)-[0-9a-f]{32}\b')
ARTWORK_CATALOG = Path('docs/sprite-customization.md')
LOCAL_FILENAMES = {'config.yaml', 'config.platform.yaml', '.session_secret', 'secrets.enc', '.secrets.key'}


def headings(text):
    anchors = set()
    duplicates = {}
    fenced = False
    for line in text.splitlines():
        if line.startswith('```'):
            fenced = not fenced
        if fenced or not re.match(r'^#{1,6} ', line):
            continue
        title = re.sub(r'<[^>]+>', '', line.split(' ', 1)[1]).strip().lower()
        slug = ''.join(c for c in title if c in '-_ ' or unicodedata.category(c)[0] in 'LN').replace(' ', '-')
        count = duplicates.get(slug, 0)
        duplicates[slug] = count + 1
        anchors.add(slug + ('-' + str(count) if count else ''))
    return anchors


def check():
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT,
    ).decode().split('\0')
    # Deleted working-tree files are not part of the proposed branch contents.
    paths = sorted({Path(name) for name in names if name and (ROOT / name).is_file()})
    errors = []
    documents = {}
    links = 0
    for relative in paths:
        if (set(relative.parts) & LOCAL_DIRECTORIES
                or relative.parts[:2] == ('docs', 'qualification')
                or relative.suffix in GENERATED_SUFFIXES
                or relative.name in LOCAL_FILENAMES
                or relative.name == '.env' or relative.name.startswith('.env.')):
            errors.append(f'{relative}: local state or generated output belongs outside tracked source')
        if relative.suffix == '.md':
            documents[relative] = (ROOT / relative).read_text()
    for relative, content in documents.items():
        fenced = False
        for number, line in enumerate(content.splitlines(), 1):
            if relative != ARTWORK_CATALOG and PERSONAL_NAME.search(line):
                errors.append(f'{relative}:{number}: use generic resident/operator identities; artwork labels belong in the appearance catalog')
            if INSTALLATION_ID.search(line):
                errors.append(f'{relative}:{number}: use an identity placeholder instead of a live installation ID')
            if line.startswith('```'):
                fenced = not fenced
                continue
            if fenced:
                continue
            for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', line):
                link = link.split(' "', 1)[0].strip('<>')
                if urlsplit(link).scheme or link.startswith('//'):
                    continue
                links += 1
                filename, _, anchor = link.partition('#')
                target = (ROOT / relative.parent / unquote(filename)).resolve() if filename else ROOT / relative
                if not target.exists():
                    errors.append(f'{relative}:{number}: missing link target {link}')
                elif anchor and target.suffix == '.md' and unquote(anchor) not in headings(target.read_text()):
                    errors.append(f'{relative}:{number}: missing heading {link}')
        if fenced:
            errors.append(f'{relative}: unclosed Markdown code fence')
    for error in errors:
        print(error)
    print(f'Checked {len(paths)} source files, {len(documents)} Markdown documents and {links} relative links.')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(check())
