===========================
Performance: numba vs Rust
===========================

This page records why ``pythermalcomfort`` uses numba rather than a compiled
extension written in Rust (or C, C++), and where the real speed problems are.
The measurements come from one machine on one date, which are listed with each
table.

Summary
=======

We compared numba and Rust on the same model. Both compile code to machine code
with the same compiler toolkit (LLVM), so the same algorithm runs at almost the
same speed in both. In our test Rust was 6% faster on one core and 3% faster on
14 cores. A Rust core would also need a second build system, compiled wheels for
every platform, and contributors who read Rust, so the package stays with numba.

Most of the time spent in the slower models comes from three sources, and none
of them depends on the language:

1. ``np.vectorize`` wrappers, which call the compiled kernel from Python once
   per point.
2. Kernels that run on a single CPU core. Numba can spread the same kernel over
   all cores with ``@vectorize(target="parallel")`` or ``prange``.
3. Slow algorithms, such as nested root finders with very tight tolerances.

The follow-up work is tracked in issues #496, #497, #498 and #357.

Test 1: the same PMV kernel in numba and Rust
=============================================

We ported ``_pmv_ppd_optimized`` line for line to Rust (release build, LTO,
``codegen-units = 1``), called it from Python with ``ctypes``, and ran both
versions on 5 million random inputs. The outputs were identical to the last
digit.

Measured on 2026-10-26, Apple M3 Max (14 cores), numba 0.68, numpy 2.5,
rustc 1.99, pythermalcomfort 4.6.2.

==========================================  ============
Version                                     ns per point
==========================================  ============
``pmv_ppd_iso`` (public function)           426
numba kernel, 1 core (current code)         411
Rust kernel, 1 core                         386
numba kernel, 14 cores                      37
Rust kernel, 14 threads                     36
==========================================  ============

The numba parallel version needs no new code. It compiles the existing Python
kernel a second time with a parallel target:

.. code-block:: python

    from numba import float64, vectorize

    from pythermalcomfort.models._pmv_ppd_optimized import _pmv_ppd_optimized

    py_func = _pmv_ppd_optimized._dispatcher.py_func
    pmv_parallel = vectorize([float64(*[float64] * 7)], target="parallel")(py_func)

The first Rust version used 58.2 for ``met_to_w_m2`` instead of the package's
58.15, and its PMV values differed by about 0.003 until the constant was
corrected. A port into another language duplicates every constant and formula,
so it has to be checked against the validation data like the original.

Test 2: cost of each model at global-grid scale
===============================================

A 0.25° global grid with hourly data for one year has about 1 million cells ×
8,760 hours, so roughly 9 billion points. The table shows the time for that
workload on one core (same machine as above).

.. list-table::
   :header-rows: 1

   * - Model
     - ns per point
     - One global year
     - Cause
   * - ``utci``
     - 187
     - ~0.5 h
     - fine
   * - ``pmv_ppd_iso``
     - 426
     - ~1 h
     - fine
   * - ``set_tmp`` / ``two_nodes_gagge``
     - 6,200
     - ~15 h
     - ``np.vectorize`` around the kernel (#496)
   * - ``heat_index_lu``
     - 53,000
     - ~5 days
     - nested bisection, tolerance 1e-8 (#497)
   * - ``pet_steady``
     - 630,000
     - ~2 months
     - plain-Python solver via ``np.vectorize`` (#357)

For ``two_nodes_gagge`` we also measured the effect of changing only the
wrapper while keeping the existing kernel. The cost dropped from 6,233 ns per
point (``np.vectorize``) to 3,404 ns (numba loop, 1 core) and 362 ns (numba
``prange`` loop, 14 cores), which is 17× faster with no change to the physics.

Gridded data in practice
========================

One year of one variable on the grid above is about 70 GB of float64 values, so
it does not fit in memory and is usually processed in chunks with xarray and
dask. When the numba kernels also run in parallel, dask workers and numba
threads compete for the same cores. The number of numba threads can be limited
with ``numba.set_num_threads`` or the ``NUMBA_NUM_THREADS`` environment
variable. Issue #498 tracks how the package exposes this.

When Rust could become worthwhile
=================================

The comparison above would change if one of these became true:

- numba stopped supporting a Python or numpy version the package needs;
- the models had to be shipped to another language (for example a web app in
  WebAssembly or an R package), where one shared Rust core would replace
  several copies of the same code;
- a profile showed a bottleneck that numba cannot compile, rather than a slow
  algorithm or a Python wrapper.
