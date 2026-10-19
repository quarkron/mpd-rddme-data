"""Exit 0 if a figure folder's combined PDF was saved by Adobe Illustrator (hand edits to protect), else 1.

    python tools/illustrator_edited.py main/fig5      # render_all.sh calls this before each folder

Looks at the combined figure PDFs (plots/*.pdf and plots/pdf/*.pdf, not panels/ or published/) and their
/Creator; a hand-edited figure keeps Illustrator's name there. render_all.sh then skips the folder unless FORCE=1.
"""
import glob
import os
import sys

import fitz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def edited(folder):
    hits = []
    for f in glob.glob(os.path.join(ROOT, folder, 'plots', '*.pdf')) + glob.glob(os.path.join(ROOT, folder, 'plots', 'pdf', '*.pdf')):
        try:
            if 'illustrator' in (fitz.open(f).metadata.get('creator') or '').lower():
                hits.append(os.path.relpath(f, ROOT))
        except Exception:
            pass
    return hits


if __name__ == '__main__':
    h = edited(sys.argv[1])
    if h:
        print('  ' + ', '.join(h))
    sys.exit(0 if h else 1)
