"""
Data Acquisition (FA-0) for the category classification feature pipeline.

Extracted from category_classification_f.ipynb for brevity.
"""
from __future__ import annotations

# Standard library imports for datetime handling, IO, math, concurrency, and path manipulation.
import datetime
import io
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Third-party imports for interactive widgets, numerical computing, dataframes, and Arrow/Parquet.
import ipywidgets as widgets
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import remotezip
from IPython.display import display
from tqdm.auto import tqdm as _tqdm


def setup_ui(cfg: dict) -> dict:
    """Create and display the FA-0 interactive widget UI.

    This function builds a two-mode interface inside a Jupyter notebook:
    1) Download from CloudFront — specify a date range and target row count.
    2) Load from folder — browse local storage, refresh the file list,
       and select one or more CSV / Parquet files.

    Parameters
    ----------
    cfg : dict
        Loaded configuration dict containing the ``fa0`` key with defaults.

    Returns
    -------
    dict
        A dict of widget references needed by :func:`run_acquisition`.
    """
    # Pull out the fa0 sub-configuration and CloudFront routing strings.
    _fa0_cfg = cfg["fa0"]
    _FA0_BASE    = _fa0_cfg["cloudfront_base"]
    _FA0_SUBPATH = _fa0_cfg["cloudfront_subpath"]

    # ------------------------------------------------------------------
    # Mode selector (download vs local load)
    # ------------------------------------------------------------------
    _fa0_mode = widgets.RadioButtons(
        options=["⬇  Download from CloudFront", "📂  Load from folder"],
        value="⬇  Download from CloudFront",
        layout=widgets.Layout(width="auto"),
    )

    # ------------------------------------------------------------------
    # Download-mode widgets: date range + target rows
    # ------------------------------------------------------------------
    _acq_start = widgets.Text(
        description="Start date", value=_fa0_cfg.get("download_start_date", "2026-04-01"),
        placeholder="YYYY-MM-DD", layout=widgets.Layout(width="230px"),
    )
    _acq_end = widgets.Text(
        description="End date", value=_fa0_cfg.get("download_end_date", "2026-04-07"),
        placeholder="YYYY-MM-DD", layout=widgets.Layout(width="230px"),
    )
    _acq_n_rows = widgets.BoundedIntText(
        value=_fa0_cfg.get("default_target_rows", 240_000), min=1, max=10_000_000, step=10_000,
        description="Target rows:", layout=widgets.Layout(width="210px"),
    )
    # Assemble the download controls into a visible VBox.
    _download_box = widgets.VBox(
        [
            widgets.HBox([_acq_start, _acq_end, _acq_n_rows]),
            widgets.Label("↑ Set date range and target row count, then run the cell below."),
        ],
        layout=widgets.Layout(display=""),
    )

    # ------------------------------------------------------------------
    # Load-mode widgets: folder text, refresh button, target rows, file selector
    # ------------------------------------------------------------------
    _fa0_folder = widgets.Text(
        description="Folder:",
        value=_fa0_cfg.get("data_folder", "/home/adrian/Documents/data/full"),
        placeholder="Relative or absolute path",
        layout=widgets.Layout(width="380px"),
    )
    _fa0_refresh_btn = widgets.Button(
        description="↺ Refresh",
        layout=widgets.Layout(width="90px"),
    )
    _fa0_n_rows = widgets.BoundedIntText(
        value=_fa0_cfg.get("default_target_rows", 240_000), min=1, max=10_000_000, step=10_000,
        description="Target rows:",
        layout=widgets.Layout(width="210px"),
    )
    _fa0_file_select = widgets.SelectMultiple(
        options=[],
        description="Files:",
        layout=widgets.Layout(width="660px", height="180px"),
    )
    _fa0_file_info = widgets.Label("—")
    # Assemble the local-load controls into a hidden VBox (shown only when mode switches).
    _load_box = widgets.VBox(
        [
            widgets.HBox([_fa0_folder, _fa0_refresh_btn]),
            widgets.HBox([
                _fa0_n_rows,
                widgets.Label("↑ Rows to sample; fraction derived from total source rows."),
            ]),
            _fa0_file_select,
            _fa0_file_info,
        ],
        layout=widgets.Layout(display="none"),
    )

    # ------------------------------------------------------------------
    # Folder scanner — populates the SelectMultiple with CSV / Parquet files
    # ------------------------------------------------------------------
    def _fa0_scan(_=None):
        """Scan the folder in ``_fa0_folder`` and update the file list."""
        # Resolve the folder path and guard against missing directories.
        folder = Path((_fa0_folder.value or ".").strip())
        if not folder.exists():
            _fa0_file_select.options = []
            _fa0_file_info.value = f"Folder '{folder}' does not exist."
            return
        # Supported extensions: parquet, csv, and gzipped csv.
        _EXTS = {".parquet", ".csv"}
        files = sorted(
            p for p in folder.rglob("*")
            if p.is_file()
            and (p.suffix.lower() in _EXTS or str(p).lower().endswith(".csv.gz"))
            and not p.name.startswith(".")          # skip hidden files
        )
        if not files:
            _fa0_file_select.options = []
            _fa0_file_info.value = f"No CSV / parquet files found under '{folder}'."
            return
        # Build (label, absolute_path) tuples for the selector.
        options = []
        for p in files:
            try:
                label = str(p.relative_to(folder))
            except ValueError:
                label = str(p)
            options.append((label, str(p.resolve())))
        _fa0_file_select.options = options
        # Default to selecting every file.
        _fa0_file_select.value = [v for _, v in options]
        _fa0_file_info.value = (
            f"{len(files)} file(s) found — Ctrl / Cmd + click to select multiple."
        )

    # Wire the refresh button to the scanner.
    _fa0_refresh_btn.on_click(_fa0_scan)

    # ------------------------------------------------------------------
    # Mode toggle — swap visibility of download vs load boxes
    # ------------------------------------------------------------------
    def _fa0_on_mode(change):
        """Callback fired when the user switches acquisition mode."""
        if change["new"].startswith("⬇"):
            _download_box.layout.display = ""
            _load_box.layout.display     = "none"
        else:
            _download_box.layout.display = "none"
            _load_box.layout.display     = ""
            _fa0_scan()   # auto-populate file list when switching to load mode

    # Attach observer and run an initial scan (so load mode is ready immediately).
    _fa0_mode.observe(_fa0_on_mode, names="value")
    _fa0_scan()

    # Render the complete UI in the notebook output cell.
    display(widgets.VBox([
        _fa0_mode,
        _download_box,
        _load_box,
        widgets.Label("↑ Configure source above, then run the cell below."),
    ]))

    # Return every widget and constant that ``run_acquisition`` will need.
    return dict(
        _fa0_mode=_fa0_mode,
        _download_box=_download_box,
        _load_box=_load_box,
        _fa0_folder=_fa0_folder,
        _fa0_refresh_btn=_fa0_refresh_btn,
        _fa0_n_rows=_fa0_n_rows,
        _fa0_file_select=_fa0_file_select,
        _fa0_file_info=_fa0_file_info,
        _acq_start=_acq_start,
        _acq_end=_acq_end,
        _acq_n_rows=_acq_n_rows,
        _FA0_BASE=_FA0_BASE,
        _FA0_SUBPATH=_FA0_SUBPATH,
    )


def run_acquisition(
    cfg: dict,
    merge_arrow_schema,
    read_data_file,
    w: dict,
) -> str | None:
    """Execute FA-0 acquisition (download or load from folder).

    This is the back-end counterpart to :func:`setup_ui`.  It reads the
    current widget values, decides which branch to take, and produces a
    single merged Parquet file on disk.

    Parameters
    ----------
    cfg : dict
        Loaded TOML configuration.
    merge_arrow_schema : callable
        Schema-merging function from ``category_classification``.
        Used in the local-load branch when schemas diverge across files.
    read_data_file : callable
        File-read function from ``category_classification``.
        Expected signature ``(idx, path, do_sample, stride) -> (idx, pa.Table)``.
    w : dict
        Widget dict returned by :func:`setup_ui`.

    Returns
    -------
    str or None
        Absolute path to the output parquet file, or ``None`` if nothing was collected.
    """
    # ------------------------------------------------------------------
    # Unpack configuration and widget handles
    # ------------------------------------------------------------------
    _fa0_cfg = cfg["fa0"]
    _FA0_BASE    = w["_FA0_BASE"]
    _FA0_SUBPATH = w["_FA0_SUBPATH"]
    _fa0_mode    = w["_fa0_mode"]
    _acq_start   = w["_acq_start"]
    _acq_end     = w["_acq_end"]
    _acq_n_rows  = w["_acq_n_rows"]
    _fa0_folder  = w["_fa0_folder"]
    _fa0_n_rows  = w["_fa0_n_rows"]
    _fa0_file_select = w["_fa0_file_select"]

    # ==================================================================
    # BRANCH 1 — Download from CloudFront
    # ==================================================================
    if _fa0_mode.value.startswith("⬇"):
        # Worker counts: one pool for days, another (nested) pool for chunks.
        _FA0_DL_WORKERS    = _fa0_cfg.get("dl_workers", 8)
        _FA0_CHUNK_WORKERS = _fa0_cfg.get("chunk_workers", 4)

        # ------------------------------------------------------------------
        # Per-day sampler: downloads a zipped daily archive, samples chunks,
        # and returns a single Arrow Table for that date.
        # ------------------------------------------------------------------
        def _fa0_sample_day_arrow(date_str: str, fraction: float, base_url: str, subpath: str, chunk_workers: int = 4):
            """Sample a single day's data from a remote zip of Parquet chunks.

            Parameters
            ----------
            date_str : str
                ISO date, e.g. ``"2026-04-01"``.
            fraction : float
                Target fraction of rows to retain (0 < fraction ≤ 1).
            base_url : str
                CloudFront origin base URL.
            subpath : str
                Path suffix after the base URL.
            chunk_workers : int
                Number of parallel chunk downloads *inside* this day.

            Returns
            -------
            tuple
                ``(date_str, pa.Table | None, dict)`` — metadata or error info.
            """
            # Build the remote zip URL and the internal directory prefix for chunks.
            url    = f"{base_url}{subpath}/sor-global-{date_str}-full.parquet.zip"
            prefix = f"global___full/daily_dumps_chunked/sor-global-{date_str}-full/"
            try:
                # Open a remote zip handle and list only the Parquet entries under ``prefix``.
                with remotezip.RemoteZip(url) as rz:
                    chunk_files = sorted(
                        n for n in rz.namelist()
                        if n.startswith(prefix) and n.endswith(".parquet")
                    )
                # If the archive exists but contains no matching chunks, skip the day.
                if not chunk_files:
                    return date_str, None, {}

                # Decide row-sampling strategy: random mask for large fractions,
                # fixed stride for small fractions (more efficient).
                step = 1 if fraction > 0.5 else math.ceil(1.0 / fraction)

                def _sample(tbl: pa.Table) -> pa.Table:
                    """Apply the target fraction to an in-memory Arrow Table."""
                    n = len(tbl)
                    if fraction > 0.5:
                        # Deterministic randomness seeded by the date string.
                        np.random.seed(hash(date_str) % 2**32)
                        mask = np.random.random(n) < fraction
                        idx = np.nonzero(mask)[0]
                        if len(idx):
                            return tbl.take(pa.array(idx))
                        return tbl.slice(0, 0)
                    step = math.ceil(1.0 / fraction)
                    if step > 1:
                        return tbl.take(pa.array(range(0, n, step)))
                    return tbl

                def _fetch(name: str):
                    """Download one named chunk, parse it, and down-sample."""
                    with remotezip.RemoteZip(url) as rz:
                        _t0 = time.perf_counter()
                        _raw = rz.read(name)
                        _t1 = time.perf_counter()
                        tbl = pq.read_table(io.BytesIO(_raw))
                    tbl = _sample(tbl)
                    return tbl, len(_raw), (_t1 - _t0)

                # ------------------------------------------------------------------
                # Parallel chunk fetch with a ThreadPoolExecutor
                # ------------------------------------------------------------------
                _total_bytes = 0
                _total_dl_time = 0.0
                with ThreadPoolExecutor(max_workers=chunk_workers) as _cpool:
                    # Map each Future back to its chunk index so we can insert in order.
                    _chunk_futs = {_cpool.submit(_fetch, n): i
                                   for i, n in enumerate(chunk_files)}
                    tables = [None] * len(chunk_files)
                    with _tqdm(total=len(chunk_files), desc=f"  {date_str}",
                               unit="chunk", leave=False) as _cpbar:
                        for _cf in as_completed(_chunk_futs):
                            _tbl, _b, _dt = _cf.result()
                            tables[_chunk_futs[_cf]] = _tbl
                            _total_bytes += _b
                            _total_dl_time += _dt
                            _cpbar.update(1)

                # Concatenate all chunk tables (schema promotion handles minor mismatches).
                day_table = pa.concat_tables(tables, promote_options="default")
                del tables
                # Compute average download throughput for diagnostics.
                _speed = (_total_bytes / _total_dl_time / 1024 / 1024) if _total_dl_time > 0 else 0.0
                return date_str, day_table, {
                    "date": date_str, "total_chunks": len(chunk_files),
                    "step": step, "sample_rows": len(day_table),
                    "total_bytes": _total_bytes, "dl_time": _total_dl_time,
                    "speed_mbps": _speed,
                }
            except Exception as exc:
                # Capture any network or parsing error and return it gracefully.
                return date_str, None, {"date": date_str, "error": str(exc)}

        # ------------------------------------------------------------------
        # Helper to coerce user-provided date strings into ``datetime.date``
        # ------------------------------------------------------------------
        def _fa0_parse_date(s):
            try:
                return datetime.date.fromisoformat(s.strip())
            except ValueError:
                raise ValueError(f"Invalid date '{s}' — use YYYY-MM-DD.")

        # Read the date range and target row count from the download widgets.
        _fa0_s    = _fa0_parse_date(_acq_start.value)
        _fa0_e    = _fa0_parse_date(_acq_end.value)
        _target_n = _acq_n_rows.value
        if _fa0_s > _fa0_e:
            raise ValueError("Start date must be ≤ end date.")

        # Ensure the save directory exists (default "sample" if blank).
        _fa0_save_dir = Path(_fa0_folder.value.strip() or "sample")
        _fa0_save_dir.mkdir(parents=True, exist_ok=True)

        # Build output filename from dates and target row count.
        OUT_PATH = str(
            _fa0_save_dir / (
                f"dsa_sample_{_fa0_s.strftime('%Y-%m-%d')}"
                f"_{_fa0_e.strftime('%Y-%m-%d')}_{_target_n}rows.parquet"
            )
        )
        # Generate an inclusive list of ISO date strings for the range.
        dates_str = [dt.strftime("%Y-%m-%d") for dt in pd.date_range(_fa0_s, _fa0_e, freq="D")]

        # ------------------------------------------------------------------
        # PROBE PHASE: estimate total rows before downloading
        # ------------------------------------------------------------------
        print(f"Probing {len(dates_str)} day(s) for row estimate...")
        _probe_total_chunks = 0
        _probe_url_first = _probe_chunk_first = None
        for _pd in dates_str:
            _pu = f"{_FA0_BASE}{_FA0_SUBPATH}/sor-global-{_pd}-full.parquet.zip"
            _pp = f"global___full/daily_dumps_chunked/sor-global-{_pd}-full/"
            try:
                with remotezip.RemoteZip(_pu) as _rz:
                    _dc = sorted(n for n in _rz.namelist()
                                 if n.startswith(_pp) and n.endswith(".parquet"))
                    _probe_total_chunks += len(_dc)
                    # Remember the first valid archive + chunk for a live row-count sample.
                    if _probe_url_first is None and _dc:
                        _probe_url_first, _probe_chunk_first = _pu, _dc[0]
            except Exception:
                pass   # missing archives are ignored; they’ll be reported later
        # Start with a conservative default, then refine with a real chunk.
        _rows_per_chunk = _fa0_cfg.get("rows_per_chunk_estimate", 50_000)
        if _probe_url_first:
            try:
                with remotezip.RemoteZip(_probe_url_first) as _rz:
                    _ptbl = pq.read_table(io.BytesIO(_rz.read(_probe_chunk_first)))
                    _rows_per_chunk = len(_ptbl)
                    del _ptbl
            except Exception:
                pass
        # Derive the per-day sampling fraction from the estimated total.
        _estimated_total = _rows_per_chunk * _probe_total_chunks
        _fa0_frac = min(1.0, _target_n / _estimated_total) if _estimated_total > 0 else 1.0
        print(f"  {_probe_total_chunks} chunks × ~{_rows_per_chunk:,} rows/chunk"
              f" → ~{_estimated_total:,} total  |  target {_target_n:,} → fraction {_fa0_frac:.4f}")
        print(f"Downloading: {_fa0_frac:.2%}/day · {len(dates_str)} day(s) → {OUT_PATH}"
              f"  [{_FA0_DL_WORKERS} day-workers × {_FA0_CHUNK_WORKERS} chunk-workers]")

        # ------------------------------------------------------------------
        # DOWNLOAD & MERGE PHASE: one Future per day, streaming to Parquet
        # ------------------------------------------------------------------
        _fa0_stats  = []
        _fa0_writer = None
        _fa0_schema = None
        try:
            with ThreadPoolExecutor(max_workers=_FA0_DL_WORKERS) as _dpool:
                # Submit all day-samplers in parallel.
                _futs = {_dpool.submit(_fa0_sample_day_arrow, ds, _fa0_frac, _FA0_BASE, _FA0_SUBPATH, _FA0_CHUNK_WORKERS): ds
                         for ds in dates_str}
                with _tqdm(total=len(dates_str), unit="day") as _pbar:
                    for _fut in as_completed(_futs):
                        _ds, _day_tbl, _stats = _fut.result()
                        if "error" in _stats:
                            _pbar.write(f"  {_ds}: ERROR — {_stats['error']}")
                        elif _day_tbl is not None:
                            # Lazily create the ParquetWriter once the first successful table arrives.
                            if _fa0_writer is None:
                                _fa0_schema = _day_tbl.schema
                                _fa0_writer = pq.ParquetWriter(
                                    OUT_PATH, _fa0_schema,
                                    write_batch_size=65536,
                                    data_page_size=1024 * 1024,
                                )
                            else:
                                # Cast to the initial schema if later days have evolved types.
                                if _day_tbl.schema != _fa0_schema:
                                    _day_tbl = _day_tbl.cast(_fa0_schema)
                            _fa0_writer.write_table(_day_tbl)
                            del _day_tbl
                            _fa0_stats.append(_stats)
                            _speed_str = f"{_stats['speed_mbps']:.2f} MB/s" if 'speed_mbps' in _stats else "N/A"
                            _pbar.write(
                                f"  {_ds}: {_stats['sample_rows']:,} rows "
                                f"({_stats['total_chunks']} chunks, step={_stats['step']}) — {_speed_str}"
                            )
                        _pbar.update(1)
        finally:
            # Always close the writer so the file footer is flushed.
            if _fa0_writer:
                _fa0_writer.close()

        # Summarise the run and decide whether to return a path or None.
        if _fa0_stats:
            _fa0_total = sum(s["sample_rows"] for s in _fa0_stats)
            print(f"\nTotal sample rows : {_fa0_total:,}  |  "
                  f"Days with data: {len(_fa0_stats)}/{len(dates_str)}")
            print(f"Saved → {OUT_PATH}")
        else:
            print("No data collected — check dates and network connectivity.")
            OUT_PATH = None

    # ==================================================================
    # BRANCH 2 — Load from local folder
    # ==================================================================
    else:
        import pyarrow.csv as _pacsv   # imported locally because it may not be needed in download branch

        # Number of parallel file readers for the local-load path.
        _FA0_N_WORKERS = _fa0_cfg.get("file_read_workers", 16)

        # Retrieve the user-selected file paths from the widget.
        _selected = list(_fa0_file_select.value)
        if not _selected:
            raise ValueError(
                "No files selected. Click '↺ Refresh' in the widget above "
                "and select one or more files."
            )

        # ------------------------------------------------------------------
        # Estimate total source rows to decide whether sampling is necessary
        # ------------------------------------------------------------------
        _target_n  = _fa0_n_rows.value
        _total_src = 0
        for _f in _selected:
            try:
                if _f.endswith(".parquet"):
                    # Fast path: read Parquet metadata without loading data.
                    _total_src += pq.read_metadata(_f).num_rows
                else:
                    # CSV path: count lines (handling .csv.gz transparently).
                    if _f.lower().endswith(".csv.gz"):
                        import gzip as _gz
                        with _gz.open(_f, "rt") as _gzf:
                            _total_src += sum(1 for _ in _gzf)
                    else:
                        # Plain CSV: use ``wc -l`` then subtract the header line.
                        import subprocess as _sp
                        _total_src += max(0, int(_sp.run(
                            ["wc", "-l", _f], capture_output=True, text=True
                        ).stdout.split()[0]) - 1)
            except Exception:
                pass   # ignore unreadable files; they’ll fail later in the pool
        _do_sample = _total_src > 0 and _target_n < _total_src
        _stride    = max(1, round(_total_src / _target_n)) if _do_sample else 1
        _pct       = min(100.0, _target_n / _total_src * 100.0) if _total_src > 0 else 100.0
        print(f"Source: {_total_src:,} rows  |  target {_target_n:,}" f"  →  stride {_stride} (~{_pct:.1f}%)")

        # ------------------------------------------------------------------
        # Fast path: single Parquet file, no sampling → pass through unchanged
        # ------------------------------------------------------------------
        if len(_selected) == 1 and _selected[0].lower().endswith(".parquet") and not _do_sample:
            OUT_PATH = _selected[0]
            _n_rows  = pq.read_metadata(OUT_PATH).num_rows
            print(f"\nReady → {OUT_PATH}  ({_n_rows:,} rows)")
        else:
            # Determine output directory (sibling "output" folder next to the input folder).
            _fa0_input_dir = Path(_fa0_folder.value.strip() or ".")
            _fa0_out_dir   = _fa0_input_dir.parent / "output"
            _fa0_out_dir.mkdir(parents=True, exist_ok=True)
            # Build a descriptive filename from the first selected file stem.
            _stem    = Path(_selected[0]).stem[:24]
            _suffix  = f"_and_{len(_selected)-1}_more" if len(_selected) > 1 else ""
            OUT_PATH = str(_fa0_out_dir / f"dsa_loaded_{_stem}{_suffix}.parquet")

            print(f"Loading {len(_selected)} file(s)"
                  f"{f' @ 1-in-{_stride} rows (~{_pct:.1f}%)' if _do_sample else ''}"
                  f"  [{_FA0_N_WORKERS} workers] → {OUT_PATH}")

            # ------------------------------------------------------------------
            # Parallel read → sequential write with schema evolution handling
            # ------------------------------------------------------------------
            _fa0_writer = None
            _fa0_schema = None
            _total_rows = 0
            try:
                with _tqdm(total=len(_selected), unit="file") as _pbar:
                    with ThreadPoolExecutor(max_workers=_FA0_N_WORKERS) as _pool:
                        # Submit all file reads; key the futures by the file path.
                        _futs = {_pool.submit(read_data_file, args, _do_sample, _stride): args[1]
                                 for args in enumerate(_selected)}
                        for _fut in as_completed(_futs):
                            _fpath = _futs[_fut]
                            _idx, _tbl = _fut.result()
                            # Update progress bar with the current filename and running row count.
                            _pbar.set_description(Path(_fpath).name[-30:])
                            _pbar.set_postfix(rows=f"{_total_rows:,}", refresh=True)
                            if _fa0_writer is None:
                                # First table seeds the output schema and writer.
                                _fa0_schema = _tbl.schema
                                _fa0_writer = pq.ParquetWriter(
                                    OUT_PATH, _fa0_schema,
                                    write_batch_size=131072,
                                    data_page_size=1024 * 1024,
                                )
                            else:
                                # Detect schema drift and, if necessary, rewrite everything
                                # with the merged (evolved) schema before continuing.
                                _evolved = merge_arrow_schema(_fa0_schema, _tbl.schema)
                                if _evolved != _fa0_schema:
                                    _fa0_writer.close()
                                    _prior = pq.read_table(OUT_PATH).cast(_evolved)
                                    _fa0_schema = _evolved
                                    _fa0_writer = pq.ParquetWriter(OUT_PATH, _fa0_schema)
                                    _fa0_writer.write_table(_prior)
                                    del _prior
                                    _pbar.write(
                                        f"  schema evolved: rewrote {_total_rows:,} prior rows"
                                    )
                                _tbl = _tbl.cast(_fa0_schema)
                            _fa0_writer.write_table(_tbl)
                            _total_rows += len(_tbl)
                            del _tbl
                            _pbar.update(1)
            finally:
                if _fa0_writer:
                    _fa0_writer.close()

            print(f"\nReady → {OUT_PATH}  ({_total_rows:,} rows)")

    return OUT_PATH
