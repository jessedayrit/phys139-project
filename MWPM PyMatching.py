"""
Analyze surf17_mwpm.db:
  - rebuild the same Stim circuit & detector model,
  - read dets/obs/pred from the DB,
  - unpack bits,
  - compute MWPM logical error rate.
"""

import sqlite3
from pathlib import Path

import numpy as np
import stim
import pymatching as pm


# =========================
# 1. Parameters (must match generator)
# =========================

db_path = Path("db")
mwpm_fname = "surf17_mwpm.db"
distance = 17
rounds = 500
p_phys = 0.01  # must match what you used when generating


# =========================
# 2. Rebuild circuit & DEM to know num_det / num_obs
# =========================

def build_stim_surface_code_circuit(
    d: int,
    t_rounds: int,
    p_phys_cycle: float,
) -> stim.Circuit:
    p_cycle = p_phys_cycle
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=t_rounds,
        before_round_data_depolarization=p_cycle / 3,
        after_clifford_depolarization=p_cycle / 3,
        after_reset_flip_probability=p_cycle / 6,
        before_measure_flip_probability=p_cycle / 6,
    )


circuit = build_stim_surface_code_circuit(distance, rounds, p_phys)
dem = circuit.detector_error_model(decompose_errors=True)
num_det = dem.num_detectors
num_obs = dem.num_observables

print(f"[INFO] num_det = {num_det}, num_obs = {num_obs}")


# =========================
# 3. Helpers to unpack bits from blobs
# =========================

def unpack_bits(blob: bytes, n_bits: int) -> np.ndarray:
    """Unpack a packed-bit blob into a 1D uint8 array of length n_bits."""
    if blob is None:
        return np.zeros(n_bits, dtype=np.uint8)
    b = np.frombuffer(blob, dtype=np.uint8)
    bits = np.unpackbits(b, bitorder="little")
    if bits.size < n_bits:
        raise ValueError(f"Not enough bits in blob: {bits.size} < {n_bits}")
    return bits[:n_bits].astype(np.uint8)


# =========================
# 4. Load data from DB and compute error rate
# =========================

def analyze_db(db_file: Path):
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()

    # Peek at schema (optional)
    cur.execute("PRAGMA table_info(info);")
    print("[INFO] info columns:", [r[1] for r in cur.fetchall()])

    cur.execute("SELECT * FROM info;")
    info_rows = cur.fetchall()
    print("[INFO] info row(s):", info_rows)

    cur.execute("PRAGMA table_info(data);")
    data_cols = [r[1] for r in cur.fetchall()]
    print("[INFO] data columns:", data_cols)

    cur.execute("SELECT shot_id, seed, dets, obs, pred FROM data;")
    rows = cur.fetchall()
    conn.close()

    n_shots = len(rows)
    print(f"[INFO] Loaded {n_shots} shots from {db_file}")

    dets_mat = np.zeros((n_shots, num_det), dtype=np.uint8)
    obs_mat  = np.zeros((n_shots, num_obs), dtype=np.uint8)
    pred_mat = np.zeros((n_shots, num_obs), dtype=np.uint8)

    for i, (shot_id, seed, dets_blob, obs_blob, pred_blob) in enumerate(rows):
        dets_mat[i] = unpack_bits(dets_blob, num_det)
        obs_mat[i]  = unpack_bits(obs_blob, num_obs)
        pred_mat[i] = unpack_bits(pred_blob, num_obs)

    # Logical error = decoded logical flips != actual logical flips
    mismatch = np.any(obs_mat != pred_mat, axis=1)
    logical_error_rate = mismatch.mean()

    print(f"[RESULT] MWPM logical error rate = {logical_error_rate/100}%")
    return logical_error_rate, dets_mat, obs_mat, pred_mat


if __name__ == "__main__":
    db_file = db_path / mwpm_fname
    analyze_db(db_file)
