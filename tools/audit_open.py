"""Run a figure script and record every file it opens from OUTSIDE this repository (reproducibility check).

    python tools/audit_open.py <script.py> [args...]      # from the script's own folder, as render_all.sh does

Python's audit hook sees every open(); paths under the repo, the Python installation, the system and temp folders
are ignored, and anything else is appended to $AUDIT_LOG (default: tools/external_opens.log). An empty log after
render_all.sh AUDIT=1 means the figures are built from the repository alone.
"""
import os
import runpy
import sys
import sysconfig
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.environ.get('AUDIT_LOG', os.path.join(ROOT, 'tools', 'external_opens.log'))
OK = [ROOT, sys.prefix, sys.base_prefix, sysconfig.get_paths()['stdlib'], tempfile.gettempdir(), '/usr', '/etc',
      '/proc', '/sys', '/dev', '/tmp', os.path.expanduser('~/.cache'), os.path.expanduser('~/.config'),
      os.path.expanduser('~/.matplotlib'), os.path.expanduser('~/.local/share/fonts'), os.path.expanduser('~/.fonts'),
      os.environ.get('TMPDIR', '/tmp')]                               # installed fonts (Arial) are system files
TEX = [p for p in os.environ.get('PATH', '').split(':') if 'texlive' in p]
OK += [os.path.dirname(os.path.dirname(os.path.dirname(p))) for p in TEX]      # .../texlive/2026
seen = set()


def hook(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = os.path.abspath(os.fsdecode(args[0]))
    if p in seen or any(p.startswith(o + os.sep) or p == o for o in OK if o):
        return
    seen.add(p)
    with open(LOG, 'a') as f:
        f.write('%s\t%s\n' % (sys.argv[0], p))


script = os.path.abspath(sys.argv[1])
sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(script))
sys.addaudithook(hook)
runpy.run_path(script, run_name='__main__')
