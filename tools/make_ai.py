"""Write an Illustrator file beside every figure and panel PDF (2026-10-19).

    python tools/make_ai.py            # from the repository root; render_all.sh calls it last

A modern .ai file is a PDF with Illustrator's private data added; Illustrator opens a PDF-content .ai like any other
.ai, with the vector paths and the live Arial text editable. This copies each plots/**/*.pdf to the same name with
.ai, so a figure and each of its panels can be opened from Finder or Explorer straight into Illustrator; File > Save
there turns it into a native .ai. Copies are refreshed only when the PDF is newer.
"""
import glob
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    n = 0
    for pdf in glob.glob(os.path.join(ROOT, '*', '*', 'plots', '**', '*.pdf'), recursive=True):
        if os.sep + 'published' + os.sep in pdf:
            continue
        ai = pdf[:-4] + '.ai'
        if not os.path.exists(ai) or os.path.getmtime(ai) < os.path.getmtime(pdf):
            shutil.copyfile(pdf, ai); n += 1
    print('  make_ai: %d .ai files written or refreshed' % n)


if __name__ == '__main__':
    main()
