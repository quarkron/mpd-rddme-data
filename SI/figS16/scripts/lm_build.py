"""Resolve which `lm` extension build a script imports -- in order, once, and out loud.

WHY THIS EXISTS

The scripts in studies/lmrddme_validation/mpd_crddme/ used to choose a build like this:

    for _p in [os.environ.get('LM_BUILD',''), '/lm/build_container', '/lm26',
               '/lm/build_v1', '/lmsrc/build_container']:
        if _p and os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)

`sys.path.insert(0, ...)` prepends, so calling it once per candidate builds sys.path in
REVERSE list order: the last existing candidate wins. The list reads as a priority order and
behaves as its opposite. Because it prepends, it also outranks PYTHONPATH -- so the
`PYTHONPATH=/lm/build_container` that every runbook prescribes had no effect against it.

On a machine where both /lm/build_container (Aug 31) and /lm/build_v1 (Aug 5) existed, that
silently ran an eight-week-old binary for an entire benchmark campaign. It produced a result
that is impossible for a correct solver -- shifting a periodic field by one lattice site
changed a fitted log-slope by 0.16 -- and two debugging sessions blamed the CUDA kernels
before anyone checked which .so was loaded.

The defect was not the wrong default. It was that the choice was never observable. Hence: the
first existing candidate wins and we stop, and the resolved build is ALWAYS reported with
enough identity (mtime, size) to tell two builds apart in a log read months later.
"""
import os
import sys

# Preserved verbatim from the original preamble, including order. Correcting the mechanism
# already moves these scripts from build_v1 to build_container; re-ordering as well would make
# it impossible to attribute a changed number to one cause or the other.
DEFAULT_CANDIDATES = (
    "LM_BUILD",                  # environment variable, honoured first if set
    "/lm/build_container",
    "/lm26",
    "/lm/build_v1",
    "/lmsrc/build_container",
)


class BuildResolutionError(RuntimeError):
    """An explicitly named build could not be used."""


def resolve_lm_build(candidates=DEFAULT_CANDIDATES, verbose=True, env=None, stream=None):
    """Put the first existing candidate on sys.path and return it.

    `candidates` entries are directory paths, except the literal string "LM_BUILD", which is
    replaced by that environment variable's value. If LM_BUILD is set but does not name an
    existing directory this raises: naming a build explicitly and silently getting a different
    one is the failure mode this module exists to prevent. LM_BUILD unset is a normal case.

    Returns the resolved directory, or None if no candidate exists (the caller may be running
    against an already-importable `lm`, e.g. an installed package or a pytest conftest).
    """
    env = os.environ if env is None else env
    stream = sys.stderr if stream is None else stream

    resolved = None
    for cand in candidates:
        if cand == "LM_BUILD":
            named = (env.get("LM_BUILD") or "").strip()
            if not named:
                continue                       # unset is fine; fall through to the defaults
            if not os.path.isdir(named):
                raise BuildResolutionError(
                    "LM_BUILD=%r does not exist. Refusing to fall through to another build: "
                    "an explicitly named build is honoured or the run stops." % named)
            resolved = named
            break
        if cand and os.path.isdir(cand):
            resolved = cand
            break

    if resolved is not None and resolved not in sys.path:
        sys.path.insert(0, resolved)           # exactly one entry, at the front

    if verbose:
        _report(resolved, stream)
    return resolved


def _describe_module(mod):
    """Identity of a loaded extension: path, mtime, size. A path alone is not identity --
    the same path names a different binary before and after a rebuild, which is exactly the
    ambiguity that hid this bug."""
    import datetime
    path = getattr(mod, "__file__", None)
    if not path or not os.path.exists(path):
        return "path unknown"
    st = os.stat(path)
    when = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
    return "%s (%s, %d bytes)" % (path, when, st.st_size)


def _report(resolved, stream):
    if resolved is None:
        stream.write("[lm_build] no candidate build directory found; "
                     "relying on whatever `lm` is already importable\n")
    else:
        stream.write("[lm_build] resolved build: %s\n" % resolved)
    already = sys.modules.get("lm")
    if already is not None:
        stream.write("[lm_build] lm already imported: %s\n" % _describe_module(already))
    stream.flush()


def report_imported_lm(stream=None):
    """Call AFTER `import lm` to record the binary that actually loaded.

    resolve_lm_build() runs before the import, so it can only report its intent. This reports
    the outcome -- which is what matters if something else is shadowing the path.
    """
    stream = sys.stderr if stream is None else stream
    mod = sys.modules.get("lm")
    if mod is None:
        stream.write("[lm_build] lm is not imported\n")
    else:
        stream.write("[lm_build] imported lm: %s\n" % _describe_module(mod))
    stream.flush()
