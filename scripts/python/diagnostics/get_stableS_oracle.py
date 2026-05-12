"""ctypes oracle around mgcv's ``C_get_stableS`` for byte-faithful validation
of a Rust port.

What this provides
------------------
``gam_reparam_oracle`` mirrors the R wrapper at
``mgcv/R/gam.fit3.r:9-63`` (``mgcv:::gam.reparam``). It loads the installed
mgcv shared object (found at runtime via R itself, since user/system R
libpaths vary), looks up the exported ``get_stableS`` symbol (the
``C_<name>`` form in ``init.c`` is just R's registration tag; the linker
symbol is plain ``get_stableS``), and invokes it via ``ctypes``.

Calling convention notes
------------------------
mgcv's ``get_stableS`` is reached through R's ``.C()`` interface, which
passes *every* argument by pointer regardless of underlying type. That
means every ``int`` is a ``ctypes.POINTER(ctypes.c_int)`` and every
``double`` (scalar or array) is a ``ctypes.POINTER(ctypes.c_double)``. The
matrix arguments (``S``, ``Qf``, the flattened ``sqrtS``, ``det2``) are
column-major like Fortran/R, so we use ``order="F"`` for the buffers and
transpose on the way in/out for ``rS`` (which is a list of column-major
matrices laid out contiguously).

``sqrtS`` is modified in place by mgcv; we hand it a freshly-allocated
copy and then unpack the transformed roots from that same buffer.

``fixed_penalty`` semantics
---------------------------
When ``fixed_penalty=True``:
  * ``rS`` has ``M+1`` entries; the last one is the (square root of the)
    fixed component and its implicit ``sp`` is hard-coded to 1.0 inside
    the C routine.
  * ``log_sp`` has length ``M`` (one per *non-fixed* penalty); ``rSncol``
    has length ``M+1``.
  * ``det1`` / ``det2`` are length ``M`` / shape ``(M, M)`` -- only over
    the smoothing parameters, never the fixed term (see gdi.c:766-777).

So ``fixed_penalty`` consumes one extra ``rS`` slot but *no* extra
``log_sp`` slot.

Validation block (``__main__``)
-------------------------------
Builds a synthetic 2-penalty problem (q=10, M=2, each rS is 10x9, rank 9),
calls the oracle at ``log_sp = [log(1.5), log(0.7)]`` with ``deriv=2``,
then shells out to ``Rscript`` to run ``mgcv:::gam.reparam`` on the same
inputs and asserts agreement to 1e-12 on every output.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Locate and load mgcv.so
# ---------------------------------------------------------------------------
def _find_mgcv_so() -> str:
    """Ask R itself where its installed mgcv shared object lives.

    User and system R libpaths can vary; the install location of mgcv may
    or may not be on the user libpath. Using R as an oracle avoids hard
    coding paths.
    """
    out = subprocess.check_output(
        ["Rscript", "-e",
         'cat(list.files(system.file("libs", package="mgcv"), '
         'pattern="mgcv\\\\.(so|dylib|dll)$", full.names=TRUE))'],
        text=True,
    ).strip()
    if not out or not os.path.exists(out):
        raise RuntimeError(f"could not locate mgcv shared object (got: {out!r})")
    return out


_MGCV_SO_PATH = _find_mgcv_so()
_MGCV_LIB = ctypes.CDLL(_MGCV_SO_PATH, mode=ctypes.RTLD_GLOBAL)

# The symbol exported in the .so is the bare C name; the `C_get_stableS`
# token in mgcv's R sources is the registration tag from src/init.c.
if not hasattr(_MGCV_LIB, "get_stableS"):
    raise RuntimeError(
        f"loaded {_MGCV_SO_PATH} but no symbol 'get_stableS'; "
        "available registered names use the R-side 'C_' prefix only."
    )

_get_stableS = _MGCV_LIB.get_stableS
_get_stableS.restype = None
_get_stableS.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # S       (q*q, col-major)
    ctypes.POINTER(ctypes.c_double),  # Qf      (q*q, col-major)
    ctypes.POINTER(ctypes.c_double),  # sp      (M doubles)
    ctypes.POINTER(ctypes.c_double),  # sqrtS   (q * sum(rSncol), col-major, modified)
    ctypes.POINTER(ctypes.c_int),     # rSncol  (Mf ints)
    ctypes.POINTER(ctypes.c_int),     # q       (int*)
    ctypes.POINTER(ctypes.c_int),     # M       (int*)
    ctypes.POINTER(ctypes.c_int),     # deriv   (int*)
    ctypes.POINTER(ctypes.c_double),  # det     (double*)
    ctypes.POINTER(ctypes.c_double),  # det1    (M doubles)
    ctypes.POINTER(ctypes.c_double),  # det2    (M*M, col-major)
    ctypes.POINTER(ctypes.c_double),  # d_tol   (double*)
    ctypes.POINTER(ctypes.c_double),  # r_tol   (double*)
    ctypes.POINTER(ctypes.c_int),     # fixed_penalty (int*)
]


# ---------------------------------------------------------------------------
# Oracle entry point
# ---------------------------------------------------------------------------
def gam_reparam_oracle(
    rS: list[np.ndarray],
    log_sp: np.ndarray,
    deriv: int,
    fixed_penalty: bool = False,
) -> dict:
    """Call mgcv's ``C_get_stableS`` via ctypes.

    Parameters
    ----------
    rS
        List of square-root penalty matrices. Each entry is shape
        ``(q, rSncol[i])``. If ``fixed_penalty=True`` the last entry is the
        fixed-penalty root and is *not* paired with a smoothing parameter.
    log_sp
        Length-M log smoothing parameters. ``M == len(rS)`` when
        ``fixed_penalty=False`` and ``M == len(rS) - 1`` when
        ``fixed_penalty=True``.
    deriv
        0, 1, or 2 -- order of derivatives required.
    fixed_penalty
        See module docstring.

    Returns
    -------
    dict with keys ``S``, ``Qs``, ``rS``, ``det``, ``det1``, ``det2``.
    """
    if deriv not in (0, 1, 2):
        raise ValueError("deriv must be 0, 1, or 2")
    if not rS:
        raise ValueError("rS must be non-empty")

    log_sp = np.asarray(log_sp, dtype=np.float64).reshape(-1)
    M = log_sp.size
    Mf = len(rS)
    if fixed_penalty:
        if Mf != M + 1:
            raise ValueError(
                f"fixed_penalty=True requires len(rS) == len(log_sp)+1, "
                f"got len(rS)={Mf}, len(log_sp)={M}"
            )
    else:
        if Mf != M:
            raise ValueError(
                f"fixed_penalty=False requires len(rS) == len(log_sp), "
                f"got len(rS)={Mf}, len(log_sp)={M}"
            )

    q = rS[0].shape[0]
    rSncol = np.empty(Mf, dtype=np.int32)
    for i, R in enumerate(rS):
        if R.shape[0] != q:
            raise ValueError(
                f"rS[{i}] has {R.shape[0]} rows; expected q={q}")
        rSncol[i] = R.shape[1]

    # Flatten rS list into a single (q, sum(rSncol)) col-major buffer.
    sqrtS_flat = np.empty(q * int(rSncol.sum()), dtype=np.float64)
    off = 0
    for i, R in enumerate(rS):
        block = q * int(rSncol[i])
        # column-major layout: R[:, j] occupies positions [j*q .. j*q+q-1]
        sqrtS_flat[off:off + block] = np.asarray(R, dtype=np.float64, order="F").reshape(-1, order="F")
        off += block
    assert off == sqrtS_flat.size

    # Output buffers
    S_buf = np.zeros(q * q, dtype=np.float64)
    Qf_buf = np.zeros(q * q, dtype=np.float64)
    sp = np.exp(log_sp).astype(np.float64, copy=True)
    det = np.zeros(1, dtype=np.float64)
    det1 = np.zeros(max(M, 1), dtype=np.float64)
    det2 = np.zeros(max(M * M, 1), dtype=np.float64)

    # Tolerances chosen to match gam.reparam exactly:
    #   d.tol = .Machine$double.eps^.3
    #   r.tol = .Machine$double.eps^.75
    eps = np.finfo(np.float64).eps
    d_tol = np.array([eps ** 0.3], dtype=np.float64)
    r_tol = np.array([eps ** 0.75], dtype=np.float64)

    q_arr = np.array([q], dtype=np.int32)
    M_arr = np.array([M], dtype=np.int32)
    deriv_arr = np.array([deriv], dtype=np.int32)
    fp_arr = np.array([1 if fixed_penalty else 0], dtype=np.int32)

    _ctype_dbl = ctypes.POINTER(ctypes.c_double)
    _ctype_int = ctypes.POINTER(ctypes.c_int)

    def _dp(a):
        return a.ctypes.data_as(_ctype_dbl)

    def _ip(a):
        return a.ctypes.data_as(_ctype_int)

    _get_stableS(
        _dp(S_buf),
        _dp(Qf_buf),
        _dp(sp),
        _dp(sqrtS_flat),
        _ip(rSncol),
        _ip(q_arr),
        _ip(M_arr),
        _ip(deriv_arr),
        _dp(det),
        _dp(det1),
        _dp(det2),
        _dp(d_tol),
        _dp(r_tol),
        _ip(fp_arr),
    )

    # Reassemble outputs in column-major order.
    S = S_buf.reshape((q, q), order="F").copy()
    # The R wrapper symmetrises; replicate exactly.
    S = 0.5 * (S + S.T)
    Qs = Qf_buf.reshape((q, q), order="F").copy()

    new_rS: list[np.ndarray] = []
    off = 0
    for i in range(Mf):
        cols = int(rSncol[i])
        block = q * cols
        mat = sqrtS_flat[off:off + block].reshape((q, cols), order="F").copy()
        new_rS.append(mat)
        off += block

    out = {
        "S": S,
        "Qs": Qs,
        "rS": new_rS,
        "det": float(det[0]),
        "det1": det1.reshape(M).copy() if deriv >= 1 else None,
        "det2": det2.reshape((M, M), order="F").copy() if deriv >= 2 else None,
    }
    return out


# ---------------------------------------------------------------------------
# Validation harness
# ---------------------------------------------------------------------------
def _make_validation_problem(seed: int = 0) -> tuple[list[np.ndarray], np.ndarray]:
    """Synthetic 2-penalty problem: q=10, M=2, each rS is 10x9 rank 9.

    Constructed so the total penalty is full rank (rS[0] hits cols 1..9
    and rS[1] hits cols 0..8, giving overlap on cols 1..8 and full
    coverage of all 10 coefficient slots when summed).
    """
    rng = np.random.default_rng(seed)
    q = 10

    # rS[0]: rank-9 rooted in coefficient slots 1..9.
    rS0 = np.zeros((q, 9), dtype=np.float64)
    A0 = rng.standard_normal((9, 9))
    rS0[1:10, :9] = A0  # full rank 9 block

    # rS[1]: rank-9 rooted in coefficient slots 0..8.
    rS1 = np.zeros((q, 9), dtype=np.float64)
    A1 = rng.standard_normal((9, 9))
    rS1[0:9, :9] = A1

    rS = [rS0, rS1]
    log_sp = np.array([np.log(1.5), np.log(0.7)], dtype=np.float64)
    return rS, log_sp


def _run_R_reparam(rS: list[np.ndarray], log_sp: np.ndarray, deriv: int) -> dict:
    """Invoke mgcv:::gam.reparam via Rscript on the same inputs."""
    tmpdir = Path(tempfile.mkdtemp(prefix="stableS_oracle_"))

    # Save inputs as plain text so R can read them deterministically.
    q = rS[0].shape[0]
    Mf = len(rS)
    rSncol = [int(r.shape[1]) for r in rS]
    np.savetxt(tmpdir / "log_sp.txt", log_sp)
    np.savetxt(tmpdir / "rSncol.txt", np.asarray(rSncol, dtype=np.int32), fmt="%d")
    for i, R in enumerate(rS):
        np.savetxt(tmpdir / f"rS_{i}.txt", R)
    meta = dict(q=int(q), Mf=int(Mf), deriv=int(deriv))
    (tmpdir / "meta.json").write_text(json.dumps(meta))

    r_script = f"""
suppressMessages(library(mgcv))
meta <- jsonlite::fromJSON('{tmpdir}/meta.json')
q <- as.integer(meta$q); Mf <- as.integer(meta$Mf); deriv <- as.integer(meta$deriv)
log_sp <- scan('{tmpdir}/log_sp.txt', quiet = TRUE)
rSncol <- scan('{tmpdir}/rSncol.txt', quiet = TRUE, what = integer())
rS <- vector('list', Mf)
for (i in seq_len(Mf)) {{
  m <- as.matrix(read.table(sprintf('{tmpdir}/rS_%d.txt', i-1)))
  dimnames(m) <- NULL
  storage.mode(m) <- 'double'
  rS[[i]] <- m
}}
res <- mgcv:::gam.reparam(rS, log_sp, deriv)
# Save outputs
write.table(res$S,  '{tmpdir}/out_S.txt',  row.names=FALSE, col.names=FALSE)
write.table(res$Qs, '{tmpdir}/out_Qs.txt', row.names=FALSE, col.names=FALSE)
for (i in seq_along(res$rS)) {{
  write.table(res$rS[[i]], sprintf('{tmpdir}/out_rS_%d.txt', i-1),
              row.names=FALSE, col.names=FALSE)
}}
cat(sprintf('%.17g\\n', res$det), file='{tmpdir}/out_det.txt')
if (!is.null(res$det1)) write.table(matrix(res$det1, nrow=1),
   '{tmpdir}/out_det1.txt', row.names=FALSE, col.names=FALSE)
if (!is.null(res$det2)) write.table(res$det2,
   '{tmpdir}/out_det2.txt', row.names=FALSE, col.names=FALSE)
cat('R OK\\n')
"""
    r_file = tmpdir / "run.R"
    r_file.write_text(r_script)

    res = subprocess.run(
        ["Rscript", "--vanilla", str(r_file)],
        check=True, capture_output=True, text=True,
    )
    if "R OK" not in res.stdout:
        raise RuntimeError(f"R run did not signal OK\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}")

    S = np.loadtxt(tmpdir / "out_S.txt")
    Qs = np.loadtxt(tmpdir / "out_Qs.txt")
    new_rS = []
    for i in range(Mf):
        new_rS.append(np.loadtxt(tmpdir / f"out_rS_{i}.txt"))
    det = float((tmpdir / "out_det.txt").read_text().strip())
    det1 = None
    det2 = None
    if deriv >= 1:
        det1 = np.loadtxt(tmpdir / "out_det1.txt").reshape(-1)
    if deriv >= 2:
        det2 = np.loadtxt(tmpdir / "out_det2.txt")

    return {"S": S, "Qs": Qs, "rS": new_rS, "det": det, "det1": det1, "det2": det2}


def _max_abs_diff(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.max(np.abs(a - b)))


def _validate(tol: float = 1e-12) -> int:
    rS, log_sp = _make_validation_problem()
    deriv = 2

    py = gam_reparam_oracle(rS, log_sp, deriv=deriv, fixed_penalty=False)
    r = _run_R_reparam(rS, log_sp, deriv=deriv)

    diffs = {}
    diffs["S"] = _max_abs_diff(py["S"], r["S"])
    diffs["Qs"] = _max_abs_diff(py["Qs"], r["Qs"])
    for i, (pa, ra) in enumerate(zip(py["rS"], r["rS"])):
        diffs[f"rS[{i}]"] = _max_abs_diff(pa, ra)
    diffs["det"] = abs(py["det"] - r["det"])
    diffs["det1"] = _max_abs_diff(py["det1"], r["det1"])
    diffs["det2"] = _max_abs_diff(py["det2"], r["det2"])

    print("max |Py - R| diffs:")
    for k, v in diffs.items():
        print(f"  {k:10s} = {v:.3e}")

    worst = max(diffs.values())
    if worst <= tol:
        print(f"PASS (worst = {worst:.3e} <= tol {tol:.0e})")
        return 0
    print(f"FAIL (worst = {worst:.3e} > tol {tol:.0e})")
    return 1


if __name__ == "__main__":
    sys.exit(_validate())
