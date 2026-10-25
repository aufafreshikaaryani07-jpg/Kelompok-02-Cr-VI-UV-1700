"""
SPECTRA LAB 2.0 — UV-Vis Analytical Workbench (Python / Streamlit, SATU BERKAS)
Politeknik AKA Bogor · Analisis Spektrofotometri · instrumen acuan Shimadzu UV-1700 PharmaSpec

Cara menjalankan:
    pip install -r requirements.txt
    streamlit run app.py

Seluruh pembacaan "instrumen" berasal dari MODEL SIMULASI; aplikasi tidak terhubung ke alat.

Struktur berkas
  BAGIAN 1  Inti perhitungan & model simulasi
            (statistik, regresi, uji Mandel, LOD/LOQ, Grubbs, adisi standar,
             multikomponen Vierordt/CLS, kinetika, kualifikasi kinerja instrumen)
  BAGIAN 2  Materi pembelajaran & kuis
  BAGIAN 3  Interpretasi otomatis, grafik laporan, PDF
  BAGIAN 4  Model 3D instrumen (HTML/JavaScript, disesuaikan dengan unit UV-1700 di laboratorium)
  BAGIAN 4B Praktikum virtual 3D: meja lab, kabel daya/data, UPS, CPU, monitor, tisu, kuvet, layar PC
  BAGIAN 5  Aplikasi Streamlit (navigasi berkelompok, 16 halaman)
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# BAGIAN 1 — INTI PERHITUNGAN & MODEL SIMULASI
# =============================================================================

# Nilai kritis Grubbs dua sisi, alfa = 0,05 (ISO 5725-2), n = 3..12
GRUBBS_CRIT = {3: 1.155, 4: 1.481, 5: 1.715, 6: 1.887, 7: 2.020, 8: 2.126,
               9: 2.215, 10: 2.290, 11: 2.355, 12: 2.412}


def t95(df: float) -> float:
    return float(stats.t.ppf(0.975, df)) if df >= 1 else float("nan")


def describe(values) -> dict:
    a = np.asarray(values, dtype=float)
    n = a.size
    mean = float(a.mean()) if n else float("nan")
    sd = float(a.std(ddof=1)) if n > 1 else float("nan")
    rsd = abs(sd / mean) * 100 if n > 1 and mean != 0 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": mean, "sd": sd, "rsd": rsd, "se": se,
            "ci95": t95(n - 1) * se if n > 1 else float("nan"),
            "min": float(a.min()) if n else float("nan"), "max": float(a.max()) if n else float("nan")}


def grubbs(values) -> dict:
    a = np.asarray(values, dtype=float)
    n = a.size
    if n < 3 or n > 12:
        return {"applicable": False, "n": n}
    m, s = a.mean(), a.std(ddof=1)
    if not s > 0:
        return {"applicable": False, "n": n}
    gv = np.abs(a - m) / s
    i = int(np.argmax(gv))
    return {"applicable": True, "n": n, "G": float(gv[i]), "Gcrit": GRUBBS_CRIT[n],
            "index": i, "value": float(a[i]), "outlier": bool(gv[i] > GRUBBS_CRIT[n])}


def horwitz(c_fraction: float) -> float:
    """PRSD_R (%) = 2^(1 − 0,5·log10 C), C sebagai fraksi massa."""
    return 2 ** (1 - 0.5 * math.log10(c_fraction)) if c_fraction > 0 else float("nan")


@dataclass
class Regression:
    ok: bool
    reason: str = ""
    n: int = 0
    df: int = 0
    slope: float = float("nan")
    intercept: float = float("nan")
    r: float = float("nan")
    r2: float = float("nan")
    sxy: float = float("nan")
    sa: float = float("nan")
    sb: float = float("nan")
    xbar: float = float("nan")
    ybar: float = float("nan")
    Sxx: float = float("nan")
    sumx2: float = float("nan")
    tcrit: float = float("nan")
    through_zero: bool = False
    xmin: float = float("nan")
    xmax: float = float("nan")
    yhat: np.ndarray = field(default_factory=lambda: np.array([]))
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    std_residuals: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def ci_slope(self):
        return self.tcrit * self.sa

    @property
    def ci_intercept(self):
        return 0.0 if self.through_zero else self.tcrit * self.sb

    @property
    def intercept_significant(self):
        return (not self.through_zero) and abs(self.intercept) > self.tcrit * self.sb

    @property
    def lod(self):
        return 3.3 * self.sxy / self.slope

    @property
    def loq(self):
        return 10 * self.sxy / self.slope

    def band(self, xs):
        """Setengah lebar pita kepercayaan 95 % garis regresi pada xs."""
        xs = np.asarray(xs, dtype=float)
        if self.through_zero:
            return self.tcrit * self.sxy * np.abs(xs) / math.sqrt(self.sumx2)
        return self.tcrit * self.sxy * np.sqrt(1 / self.n + (xs - self.xbar) ** 2 / self.Sxx)

    def equation(self, digits=5) -> str:
        if self.through_zero:
            return f"y = {self.slope:.{digits}g}x"
        sign = "−" if self.intercept < 0 else "+"
        return f"y = {self.slope:.{digits}g}x {sign} {abs(self.intercept):.{digits - 1}g}"


def linreg(x, y, through_zero: bool = False) -> Regression:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n != y.size:
        raise ValueError("Jumlah data x dan y tidak sama")
    min_n = 2 if through_zero else 3
    if n < min_n:
        return Regression(ok=False, reason=f"Minimal {min_n} titik data diperlukan")
    xbar, ybar = x.mean(), y.mean()
    Sxx = float(((x - xbar) ** 2).sum())
    Syy = float(((y - ybar) ** 2).sum())
    Sxy = float(((x - xbar) * (y - ybar)).sum())
    sumx2 = float((x * x).sum())
    if not Sxx > 0:
        return Regression(ok=False, reason="Semua konsentrasi bernilai sama")
    if through_zero:
        a = float((x * y).sum() / sumx2); b = 0.0; df = n - 1
    else:
        a = Sxy / Sxx; b = float(ybar - a * xbar); df = n - 2
    yhat = a * x + b
    res = y - yhat
    ss_res = float((res ** 2).sum())
    sxy = math.sqrt(ss_res / df) if df > 0 else float("nan")
    r = Sxy / math.sqrt(Sxx * Syy) if Syy > 0 else float("nan")
    r2 = 1 - ss_res / Syy if Syy > 0 else float("nan")
    if through_zero:
        sa = sxy / math.sqrt(sumx2); sb = 0.0
    else:
        sa = sxy / math.sqrt(Sxx)
        sb = sxy * math.sqrt(sumx2 / (n * Sxx))
    return Regression(ok=True, n=n, df=df, slope=a, intercept=b, r=r, r2=r2, sxy=sxy, sa=sa, sb=sb,
                      xbar=float(xbar), ybar=float(ybar), Sxx=Sxx, sumx2=sumx2, tcrit=t95(df), through_zero=through_zero,
                      xmin=float(x.min()), xmax=float(x.max()), yhat=yhat, residuals=res,
                      std_residuals=res / sxy if sxy > 0 else res * np.nan)


def inverse_predict(reg: Regression, y0: float, m: int = 1) -> dict:
    """x0 dari rata-rata m pembacaan y0, s(x0) dan SK 95 % (Miller & Miller)."""
    x0 = (y0 - reg.intercept) / reg.slope
    if reg.through_zero:
        sx0 = (reg.sxy / abs(reg.slope)) * math.sqrt(1 / m + y0 ** 2 / (reg.slope ** 2 * reg.sumx2))
    else:
        sx0 = (reg.sxy / abs(reg.slope)) * math.sqrt(1 / m + 1 / reg.n + (y0 - reg.ybar) ** 2 / (reg.slope ** 2 * reg.Sxx))
    return {"x0": x0, "sx0": sx0, "ci95": reg.tcrit * sx0, "extrapolated": x0 < reg.xmin or x0 > reg.xmax}


def mandel(x, y) -> dict | None:
    """Uji kecocokan Mandel (DIN 38402-51): model linear vs kuadratik, F(1, n−3) pada α = 0,01."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n = x.size
    if n < 5:
        return None
    lin = linreg(x, y)
    if not lin.ok:
        return None
    c = np.polyfit(x, y, 2)
    s2q = float(((y - np.polyval(c, x)) ** 2).sum()) / (n - 3)
    ds2 = (n - 2) * lin.sxy ** 2 - (n - 3) * s2q
    Fv = ds2 / s2q if s2q > 0 else float("inf")
    Fc = float(stats.f.ppf(0.99, 1, n - 3))
    return {"F": Fv, "Fcrit": Fc, "linear_ok": Fv <= Fc, "quad": c, "s_lin": lin.sxy, "s_quad": math.sqrt(s2q), "n": n}


def std_addition(x_add, y) -> dict | None:
    """Adisi standar: C_x = b/a; s(C_x) = (s_y/x/a)·√(1/n + ȳ²/(a²·Sxx))."""
    r = linreg(x_add, y)
    if not r.ok or not r.slope > 0:
        return None
    cx = r.intercept / r.slope
    s = (r.sxy / abs(r.slope)) * math.sqrt(1 / r.n + r.ybar ** 2 / (r.slope ** 2 * r.Sxx))
    return {"reg": r, "cx": cx, "s": s, "ci": r.tcrit * s}


def vierordt(aX1, aY1, aX2, aY2, A1, A2) -> dict | None:
    """Persamaan simultan dua komponen pada λ1, λ2."""
    M = np.array([[aX1, aY1], [aX2, aY2]], dtype=float)
    det = float(np.linalg.det(M))
    if abs(det) < 1e-14:
        return None
    cx, cy = np.linalg.solve(M, np.array([A1, A2], dtype=float))
    return {"cx": float(cx), "cy": float(cy), "det": det, "cond": float(np.linalg.cond(M))}


def cls_fit(lam, A_mix, pure: list[np.ndarray]) -> dict:
    """Kuadrat terkecil klasik (CLS) spektrum penuh: A_mix ≈ Σ c_i·k_i(λ)."""
    K = np.column_stack(pure)
    c, *_ = np.linalg.lstsq(K, A_mix, rcond=None)
    fit = K @ c
    rmse = float(np.sqrt(np.mean((A_mix - fit) ** 2)))
    return {"c": c, "fit": fit, "rmse": rmse}


def abs_to_t(A):
    return 100 * np.power(10.0, -np.asarray(A, dtype=float))


def t_to_abs(T):
    return -np.log10(np.asarray(T, dtype=float) / 100)


_SG = {5: np.array([-3, 12, 17, 12, -3]) / 35,
       7: np.array([-2, 3, 6, 7, 6, 3, -2]) / 21,
       9: np.array([-21, 14, 39, 54, 59, 54, 39, 14, -21]) / 231}


def smooth(y, w: int = 7) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    c = _SG.get(w)
    if c is None or y.size < w:
        return y.copy()
    h = (w - 1) // 2
    out = y.copy()
    out[h:-h] = np.convolve(y, c[::-1], mode="valid")
    return out


def _interp_x(x1, y1, x2, y2, yt):
    return x1 if y2 == y1 else x1 + (yt - y1) * (x2 - x1) / (y2 - y1)


def find_peaks(x, y, prominence: float = 0.01, min_height: float = 0.02, smooth_w: int = 7) -> list[dict]:
    x = np.asarray(x, dtype=float); y0 = np.asarray(y, dtype=float)
    if x.size < 5:
        return []
    order = np.argsort(x)
    xs, ys0 = x[order], y0[order]
    ys = smooth(ys0, smooth_w)
    n = xs.size
    peaks = []
    for i in range(1, n - 1):
        if not (ys[i] > ys[i - 1] and ys[i] >= ys[i + 1]) or ys[i] < min_height:
            continue
        lmin, j = ys[i], i - 1
        while j >= 0 and ys[j] <= ys[i]:
            lmin = min(lmin, ys[j]); j -= 1
        rmin, j = ys[i], i + 1
        while j < n and ys[j] <= ys[i]:
            rmin = min(rmin, ys[j]); j += 1
        prom = ys[i] - max(lmin, rmin)
        if prom < prominence:
            continue
        y1, y2, y3 = ys[i - 1], ys[i], ys[i + 1]
        den = y1 - 2 * y2 + y3
        off = 0.5 * (y1 - y3) / den if den != 0 else 0.0
        lam = xs[i] + off * (xs[i + 1] - xs[i - 1]) / 2
        half = ys[i] - prom / 2
        L = i
        while L > 0 and ys[L] > half:
            L -= 1
        R = i
        while R < n - 1 and ys[R] > half:
            R += 1
        fwhm = _interp_x(xs[R - 1], ys[R - 1], xs[R], ys[R], half) - _interp_x(xs[L], ys[L], xs[L + 1], ys[L + 1], half)
        peaks.append({"lambda": float(lam), "abs": float(ys0[i]), "prominence": float(prom), "fwhm": float(fwhm)})
    return sorted(peaks, key=lambda p: -p["abs"])


def interp_y(x, y, xt):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    if xt < x.min() or xt > x.max():
        return float("nan")
    return float(np.interp(xt, x[order], y[order]))


# --- kinetika ---------------------------------------------------------------
def kinetic_model(A0, Ainf, k, t):
    return Ainf + (A0 - Ainf) * np.exp(-k * np.asarray(t, dtype=float))


def initial_rate(t, A, n_pts: int = 5):
    k = min(n_pts, len(t))
    if k < 3:
        return None
    r = linreg(t[:k], A[:k])
    return {"rate": r.slope, "r2": r.r2, "n": k, "intercept": r.intercept} if r.ok else None


def first_order_fit(t, A, Ainf):
    t = np.asarray(t, dtype=float); A = np.asarray(A, dtype=float)
    d = np.abs(A - Ainf)
    mask = d > 1e-4
    if mask.sum() < 3:
        return None
    r = linreg(t[mask], np.log(d[mask]))
    if not r.ok:
        return None
    k = -r.slope
    return {"k": k, "half_life": math.log(2) / k if k > 0 else float("nan"), "r2": r.r2,
            "n": int(mask.sum()), "lnA0": r.intercept, "t": t[mask], "lny": np.log(d[mask])}


# --- parser angka & CSV -----------------------------------------------------
def parse_number(s) -> float:
    if isinstance(s, (int, float, np.floating)):
        return float(s)
    if s is None:
        return float("nan")
    s = re.sub(r"\s", "", str(s))
    if not s:
        return float("nan")
    if re.fullmatch(r"[-+]?\d{1,3}(\.\d{3})+(,\d+)?", s):
        s = s.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"[-+]?\d*,\d+(e[-+]?\d+)?", s, flags=re.I):
        s = s.replace(",", ".")
    if not re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?", s, flags=re.I):
        return float("nan")
    return float(s)


def parse_list(text: str) -> list[float]:
    parts = [p for p in re.split(r"[;\s|]+", text or "") if p]
    vals = [parse_number(p) for p in parts]
    if any(math.isnan(v) for v in vals):
        raise ValueError("Pisahkan nilai dengan titik koma, mis. 0,412; 0,415")
    return vals


def parse_csv(text: str) -> dict:
    lines = [ln for ln in text.replace("\r", "").split("\n") if ln.strip()]
    if not lines:
        return {"ok": False, "reason": "Berkas kosong"}
    sample = "\n".join(lines[:20])
    cnt = {";": sample.count(";"), "\t": sample.count("\t"), ",": sample.count(",")}
    delim = "\t" if cnt["\t"] >= cnt[";"] and cnt["\t"] > 0 else (";" if cnt[";"] > 0 else ",")
    header, data, skipped = None, [], 0
    for line in lines:
        cells = [c.strip().strip('"') for c in line.split(delim)]
        nums = [parse_number(c) for c in cells]
        if sum(not math.isnan(v) for v in nums) >= 2 and not math.isnan(nums[0]):
            data.append(nums)
        else:
            if not data:
                header = cells
            skipped += 1
    if not data:
        return {"ok": False, "reason": "Tidak ditemukan baris numerik (minimal 2 kolom angka)"}
    ncol = max(len(r) for r in data)
    cols = [np.array([r[c] if c < len(r) else np.nan for r in data]) for c in range(ncol)]
    return {"ok": True, "delimiter": delim, "header": header, "cols": cols, "nrows": len(data), "skipped": skipped}


# --- model simulasi instrumen ----------------------------------------------
ANALYTES = {
    "kmno4": dict(nama="Kalium permanganat", rumus="KMnO₄", pelarut="Akuades", lam_ref=525, abs_ref=0.0155,
                  bands=[(489, .22, 10), (507, .62, 8), (526, 1.0, 8.5), (546, .97, 8), (566, .46, 8.5), (311, .30, 20), (355, .12, 22)],
                  std=[10, 20, 30, 40, 50], reagent=None, warna="#8e2c8f",
                  catatan="Pita multipel 500–570 nm khas transfer muatan ligan→logam MnO₄⁻ dengan struktur vibronik."),
    "fephen": dict(nama="Fe(II)–1,10-fenantrolin", rumus="[Fe(phen)₃]²⁺", pelarut="Bufer asetat pH ≈ 4,5", lam_ref=510, abs_ref=0.199,
                   bands=[(514, 1.0, 22), (478, .30, 15), (400, .08, 30), (265, 1.6, 18)],
                   std=[0.5, 1, 2, 3, 4, 5], reagent=dict(nama="hidroksilamin + 1,10-fenantrolin", bands=[(265, .55, 18), (228, .9, 14)]),
                   warna="#c0392b", catatan="Kompleks oranye-merah (MLCT); Fe(III) direduksi dengan hidroksilamin."),
    "crdpc": dict(nama="Cr(VI)–difenilkarbazida", rumus="Cr(VI)–DPC", pelarut="H₂SO₄ (pH ≈ 1–2)", lam_ref=540, abs_ref=0.77,
                  bands=[(542, 1.0, 40), (360, .08, 30), (290, .35, 22)], std=[0.1, 0.2, 0.3, 0.4, 0.5],
                  reagent=dict(nama="difenilkarbazida", bands=[(285, .50, 18), (540, .006, 60), (235, .9, 15)]),
                  warna="#9b2d6f", catatan="Kompleks Cr(III)–difenilkarbazon merah-ungu; blanko reagen wajib."),
    "mb": dict(nama="Metilen biru", rumus="C₁₆H₁₈ClN₃S", pelarut="Akuades", lam_ref=664, abs_ref=0.23,
               bands=[(664, 1.0, 19), (612, .42, 24), (292, .62, 14), (246, .32, 14)], std=[0.5, 1, 2, 3, 4],
               reagent=None, warna="#1f5fbf", catatan="Bahu 612 nm berasal dari dimer; rasio A612/A664 naik dengan konsentrasi."),
    "k2cr2o7": dict(nama="Kalium dikromat (HClO₄ 0,001 M)", rumus="K₂Cr₂O₇", pelarut="HClO₄ 0,001 M", lam_ref=350, abs_ref=0.0107,
                    bands=[(257, 1.36, 17), (350, 1.0, 24), (440, .055, 28), (205, .9, 10)], std=[20, 40, 60, 80, 100],
                    reagent=None, warna="#d98a00", catatan="Lazim untuk verifikasi akurasi fotometrik."),
    "pct": dict(nama="Parasetamol (metanol)", rumus="C₈H₉NO₂", pelarut="Metanol", lam_ref=249, abs_ref=0.0668,
                bands=[(249, 1.0, 15), (205, .85, 9), (290, .06, 15)], std=[2, 4, 6, 8, 10, 12],
                reagent=None, warna="#5f6b7a", catatan="Daerah UV wajib kuvet kuarsa."),
}


def _band_sum(bands, lam):
    lam = np.asarray(lam, dtype=float)
    return sum(h * np.exp(-((lam - l0) ** 2) / (2 * w * w)) for l0, h, w in bands)


for _a in ANALYTES.values():
    _a["k"] = _a["abs_ref"] / float(_band_sum(_a["bands"], _a["lam_ref"]))


def absorptivity(analyte_id: str, lam):
    a = ANALYTES[analyte_id]
    return a["k"] * _band_sum(a["bands"], lam)


CUVETTES = {
    "kuarsa": dict(nama="Kuarsa", cutoff=190, A=lambda l: 0.02 + 2.5 / (1 + np.exp((l - 175) / 4))),
    "kaca": dict(nama="Kaca optik", cutoff=320, A=lambda l: 0.04 + 4 / (1 + np.exp((l - 318) / 7))),
    "plastik": dict(nama="Polistirena (PS)", cutoff=340, A=lambda l: 0.05 + 4 / (1 + np.exp((l - 330) / 6))),
}
SOLVENT_CUTOFF = {"Akuades": 190, "Metanol": 205, "HClO₄ 0,001 M": 195, "Bufer asetat pH ≈ 4,5": 230, "H₂SO₄ (pH ≈ 1–2)": 200}


def _solvent_A(name, lam):
    return 3 / (1 + np.exp((lam - SOLVENT_CUTOFF.get(name, 190)) / 5))


def lamp_energy(lam, switch_at=340.8):
    lam = np.asarray(lam, dtype=float)
    e = np.where(lam < switch_at,
                 0.15 + 0.85 * np.exp(-((lam - 240) ** 2) / (2 * 70 ** 2)),
                 0.08 + 0.92 * np.exp(-((lam - 850) ** 2) / (2 * 300 ** 2)))
    e = np.where(lam < 200, e * np.maximum(0.08, (lam - 185) / 15), e)
    return np.maximum(e, 0.02)


def instrument_baseline(lam, switch_at=340.8):
    lam = np.asarray(lam, dtype=float)
    return (0.014 + 1.6e-5 * (lam - 500) + 0.006 * np.sin(lam / 37) + np.where(lam < switch_at, 0.0035, 0)
            + 0.01 * np.exp(-((lam - 195) ** 2) / 60))


def _stray(lam):
    return np.where(np.asarray(lam) < 230, 0.0004, 0.0001)


def _photometric_sigma(lam, Ts, speed="sedang", lamp_switch=340.8):
    sig_t = {"cepat": 6e-4, "sedang": 3e-4, "lambat": 1.5e-4}[speed]
    E = lamp_energy(lam, lamp_switch) * Ts
    return np.minimum(0.4343 * sig_t / np.maximum(E, 1e-6), 0.5), E


def measure(lam, analyte_id="kmno4", conc=0.0, path=1.0, cuvette="kuarsa", blank="pelarut",
            baseline_done=True, speed="sedang", lamp_switch=340.8, matrix=0.0, rng=None):
    """Absorbansi terukur (simulasi): serapan kuvet/pelarut/reagen, blanko di sel referensi,
    sinar sesat, offset baseline bila koreksi belum dilakukan, derau fotometrik σA = 0,434·σT/T."""
    rng = rng or np.random.default_rng()
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    an = ANALYTES[analyte_id]
    cuv = CUVETTES[cuvette]
    sol = an["pelarut"]
    reagent = 0.25 * path * _band_sum(an["reagent"]["bands"], lam) if an["reagent"] else 0.0
    mat = matrix * np.exp(-((lam - 300) ** 2) / (2 * 120 ** 2)) * path
    sample_cell = cuv["A"](lam) + _solvent_A(sol, lam) + reagent + conc * absorptivity(analyte_id, lam) * path + mat
    if blank == "udara":
        ref_cell = np.zeros_like(lam)
    elif blank == "reagen":
        ref_cell = cuv["A"](lam) + _solvent_A(sol, lam) + reagent
    else:
        ref_cell = cuv["A"](lam) + _solvent_A(sol, lam)
    s = _stray(lam)
    Ts = (10 ** (-sample_cell) + s) / (1 + s)
    Tr = (10 ** (-ref_cell) + s) / (1 + s)
    A = -np.log10(Ts / Tr)
    if not baseline_done:
        A = A + instrument_baseline(lam, lamp_switch)
    sig_a, E = _photometric_sigma(lam, Ts, speed, lamp_switch)
    A = A + rng.normal(0, 1, lam.size) * sig_a
    return {"A": np.clip(A, -0.5, 4.0), "over": (A > 3.0) | (A < -0.5), "low_energy": E < 2e-3}


def mix_spectrum(lam, comps: dict, path=1.0, rng=None, speed="sedang"):
    """Spektrum campuran aditif Σ c_i·a_i(λ)·b dengan derau fotometrik (kuvet kuarsa, blanko pelarut)."""
    rng = rng or np.random.default_rng()
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    A = sum(c * absorptivity(k, lam) * path for k, c in comps.items())
    s = _stray(lam)
    Ts = (10 ** (-A) + s) / (1 + s)
    A = -np.log10(Ts)
    sig_a, _ = _photometric_sigma(lam, Ts, speed)
    return A + rng.normal(0, 1, lam.size) * sig_a


def simulate_scan(start, end, interval, **kw) -> dict:
    lo, hi = max(190, min(start, end)), min(1100, max(start, end))
    n = int(round((hi - lo) / interval)) + 1
    lam = np.round(hi - np.arange(n) * interval, 3)
    m = measure(lam, **kw)
    return {"lam": lam, "A": m["A"], "T": abs_to_t(m["A"]), "over": bool(m["over"].any()),
            "low_energy": bool(m["low_energy"].any()),
            "uv_cutoff": lo < CUVETTES[kw.get("cuvette", "kuarsa")]["cutoff"]}


def prep_conc(c: float, rng=None, rsd: float = 0.004) -> float:
    rng = rng or np.random.default_rng()
    return c * (1 + rsd * rng.normal())


def hidden_conc(analyte_id: str, seed: int) -> float:
    an = ANALYTES[analyte_id]
    r = np.random.default_rng((seed * 31 + sum(map(ord, analyte_id))) % (2 ** 32)).random()
    lo, hi = an["std"][1], an["std"][-2]
    return float(f"{lo + (hi - lo) * (0.15 + 0.7 * r):.3g}")


# --- analisis sampel --------------------------------------------------------
def dilution_factor(steps=None, manual=None) -> float:
    if manual is not None:
        return float(manual) if manual > 0 else float("nan")
    fct = 1.0
    for vp, vl in steps or []:
        if not (vp > 0 and vl > 0):
            return float("nan")
        fct *= vl / vp
    return fct


def sample_result(reg: Regression, reps, fp=1.0, mass_g=None, vol_ml=None, unit="mg/L",
                  a_min=0.1, a_max=1.0, rsd_max=2.0) -> dict | None:
    reps = [r for r in reps if not math.isnan(r)]
    if not reg.ok or not reps:
        return None
    d = describe(reps)
    ip = inverse_predict(reg, d["mean"], len(reps))
    c_final, ci_final = ip["x0"] * fp, ip["ci95"] * fp
    mass_ok = bool(mass_g and vol_ml and mass_g > 0 and vol_ml > 0 and unit in ("mg/L", "µg/mL", "ppm"))
    kadar = c_final * vol_ml / mass_g if mass_ok else float("nan")
    flags = []
    if ip["extrapolated"]:
        flags.append(f"C terukur ({ip['x0']:.3g}) di luar rentang standar {reg.xmin:.3g}–{reg.xmax:.3g} {unit}; hasil ekstrapolasi.")
    if d["mean"] < a_min or d["mean"] > a_max:
        flags.append(f"Ā = {d['mean']:.3f} di luar daerah kerja absorbansi ({a_min}–{a_max}).")
    if len(reps) > 1 and d["rsd"] > rsd_max:
        flags.append(f"RSD ulangan {d['rsd']:.2f} % melebihi batas {rsd_max} %.")
    if len(reps) < 2:
        flags.append("Hanya satu pembacaan; presisi tidak dapat dievaluasi.")
    if ip["x0"] < 0:
        flags.append("Konsentrasi negatif: absorbansi sampel di bawah intersep.")
    elif ip["x0"] < reg.loq:
        flags.append(f"C terukur di bawah LOQ ({reg.loq:.3g} {unit}); nilai indikatif.")
    return {"n": len(reps), "A_mean": d["mean"], "sd": d["sd"], "rsd": d["rsd"], "c_meas": ip["x0"],
            "sx0": ip["sx0"], "ci": ip["ci95"], "fp": fp, "c_final": c_final, "ci_final": ci_final,
            "kadar": kadar, "mass_ok": mass_ok, "flags": flags, "extrapolated": ip["extrapolated"]}


def recovery_spike(c_sample, c_spiked, c_added) -> float:
    return (c_spiked - c_sample) / c_added * 100


def recovery_reference(c_measured, c_reference) -> dict:
    return {"R": c_measured / c_reference * 100, "bias": c_measured - c_reference,
            "rel_bias": (c_measured - c_reference) / c_reference * 100}


def residual_runs(residuals) -> int:
    s = [np.sign(v) for v in residuals if v != 0]
    return (1 + sum(1 for a, b in zip(s, s[1:]) if a != b)) if s else 0


# --- kualifikasi kinerja instrumen (model) ----------------------------------
WL_LINES = [("Holmium perklorat", 241.15), ("Holmium perklorat", 287.15), ("Holmium perklorat", 361.50),
            ("Garis emisi D₂", 486.00), ("Holmium perklorat", 536.30), ("Garis emisi D₂", 656.10)]
# Ph. Eur. 2.2.25: absorbansi spesifik A(1 %, 1 cm) K2Cr2O7 dalam H2SO4 0,005 M (≈ 60 mg/L)
K2CR2O7_PE = {235: (124.5, 122.9, 126.2), 257: (144.5, 142.8, 146.2), 313: (48.6, 47.0, 50.3), 350: (107.3, 105.6, 109.0)}
TOLUENE_BANDS = [(248.3, .20, 1.4), (254.2, .46, 1.3), (260.6, .78, 1.2), (264.5, .38, 1.0), (268.7, .92, 1.2)]


def pq_wavelength(offset_nm: float, rng) -> list[dict]:
    out = []
    for src, ref in WL_LINES:
        meas = ref + offset_nm + rng.normal(0, 0.04)
        out.append({"sumber": src, "ref": ref, "meas": meas, "delta": meas - ref, "lim_pe": 1.0 if ref < 400 else 3.0})
    return out


def pq_photometric(bias_pct: float, rng, c_mg_l: float = 60.0) -> list[dict]:
    cf = c_mg_l / 1e4        # g per 100 mL
    out = []
    for lam, (nom, lo, hi) in K2CR2O7_PE.items():
        A = nom * cf * (1 + bias_pct / 100) + rng.normal(0, 0.0007)
        a1 = A / cf
        out.append({"lam": lam, "A": A, "a1": a1, "nom": nom, "lo": lo, "hi": hi, "pass": lo <= a1 <= hi})
    return out


def pq_stray(stray_pct: float, rng) -> float:
    """KCl 12 g/L pada 198 nm (A sejati ≫ 3): A_obs = −log(10^−A + s)."""
    s = stray_pct / 100
    return float(-np.log10(10 ** -5.5 + s) + rng.normal(0, 0.01))


def toluene_spectrum(sbw: float, lam=None):
    """Toluena 0,02 % v/v dalam heksana, dikonvolusikan dengan fungsi celah segitiga (FWHM = SBW) pada domain T."""
    fine = np.arange(240.0, 280.0001, 0.02)
    A0 = 0.35 * np.exp(-((fine - 261) / 9) ** 2) + _band_sum(TOLUENE_BANDS, fine)
    T = 10 ** -A0
    h = max(sbw, 0.02)
    xk = np.arange(-h, h + 1e-9, 0.02)
    ker = np.clip(1 - np.abs(xk) / h, 0, None); ker /= ker.sum()
    Tc = np.convolve(T, ker, mode="same")
    Ac = -np.log10(Tc)
    lam = fine if lam is None else lam
    return lam, np.interp(lam, fine, Ac)


def toluene_ratio(sbw: float) -> dict:
    lam, A = toluene_spectrum(sbw)
    w1 = (lam >= 268) & (lam <= 270); w2 = (lam >= 265) & (lam <= 267)
    amax = float(A[w1].max()); amin = float(A[w2].min())
    return {"Amax": amax, "Amin": amin, "ratio": amax / amin, "lam_max": float(lam[w1][np.argmax(A[w1])]),
            "lam_min": float(lam[w2][np.argmin(A[w2])])}


def lam_color(l: float) -> str:
    if l < 380:
        return "#a99cf0"
    if l > 780:
        return "#b0302a"
    r = g = b = 0.0
    if l < 440: r, b = -(l - 440) / 60, 1
    elif l < 490: g, b = (l - 440) / 50, 1
    elif l < 510: g, b = 1, -(l - 510) / 20
    elif l < 580: r, g = (l - 510) / 70, 1
    elif l < 645: r, g = 1, -(l - 645) / 65
    else: r = 1
    fct = 0.45 + 0.55 * (l - 380) / 40 if l < 420 else (0.45 + 0.55 * (780 - l) / 80 if l > 700 else 1)
    return "#" + "".join(f"{round(255 * (v * fct) ** 0.8):02x}" for v in (r, g, b))


def region(l: float) -> str:
    return "UV vakum (tepi)" if l < 200 else "UV" if l < 400 else "Tampak" if l <= 780 else "NIR"


# =============================================================================
# BAGIAN 2 — MATERI PEMBELAJARAN & KUIS
# =============================================================================
COMPONENTS = {
    "Lampu deuterium (D₂)": "Sumber kontinu UV ±190–350 nm. Energinya turun tajam di daerah tampak, sehingga alat berganti ke lampu WI di atas λ pergantian (bawaan 340,8 nm pada UV-1700).",
    "Lampu wolfram-iodin (WI)": "Sumber kontinu tampak–NIR ±340–1100 nm. Pendarannya terlihat sebagai garis oranye pada kisi ventilasi sisi kanan unit.",
    "Monokromator (celah + kisi difraksi)": "Celah masuk, kisi holografik, dan celah keluar memilih pita spektral ±1 nm. Kualitasnya menentukan sinar sesat.",
    "Pembagi berkas (double-beam)": "Membagi cahaya monokromatis menjadi berkas referensi dan sampel sehingga fluktuasi lampu terkompensasi.",
    "Kompartemen sampel": "Dua dudukan kuvet sejajar: referensi (belakang) dan sampel (depan). Tutup harus rapat agar cahaya ruangan tidak terbaca.",
    "Kuvet": "Kuarsa (≥190 nm), kaca optik (≥320 nm), plastik PS (≥340 nm). Pegang sisi buram, bilas dengan larutan uji, hindari gelembung.",
    "Detektor fotodioda Si": "Mengubah intensitas cahaya menjadi arus. UV-1700 memakai fotodioda silikon pada kedua jalur.",
    "Panel kendali & LCD": "Menampilkan λ, data (ABS/%T/konsentrasi), dan lampu aktif (D2/WI). Slot IC card dan tombol CONTRAST berada di sisi kanan.",
}

LESSONS = {
    "Prinsip kerja": """
Cahaya polikromatis diuraikan oleh **monokromator** menjadi pita sempit. Berkas dibagi menjadi berkas
**referensi** (melewati blanko) dan berkas **sampel**. Molekul dengan transisi elektronik (π→π*, n→π*,
transfer muatan) pada energi yang sesuai menyerap sebagian foton. Detektor membandingkan I dengan I₀:

**T = I / I₀** dan **A = −log T = log (I₀ / I)**.
""",
    "Fungsi blanko": """
Blanko memuat semua komponen larutan **kecuali analit**; I₀ diukur melaluinya sehingga serapan pelarut,
reagen, dan dinding kuvet terkoreksi.

- **Blanko pelarut**: analit yang menyerap sendiri (KMnO₄, dikromat).
- **Blanko reagen**: metode kolorimetri (Fe–fenantrolin, Cr(VI)–DPC).
- **Blanko sampel**: sampel tanpa reagen, mengoreksi warna/kekeruhan matriks.
- **Udara**: hanya untuk pemeriksaan kinerja alat.
""",
    "Faktor yang memengaruhi": """
- **pH dan pelarut** mengubah spesiasi sehingga λmaks dan ε bergeser.
- **Waktu dan kestabilan warna**: ukur dalam jendela waktu metode.
- **Suhu** memengaruhi kesetimbangan dan laju pembentukan kompleks.
- **Konsentrasi tinggi**: interaksi antarmolekul dan dimerisasi → penyimpangan linearitas.
- **Sinar sesat dan lebar pita** → kurva melengkung pada absorbansi tinggi.
- **Gangguan matriks**: kekeruhan, kromofor lain, ion pengganggu; efek multiplikatif dikoreksi dengan **adisi standar**.
- **Kuvet**: bahan tidak sesuai, goresan, sidik jari, orientasi berubah.
""",
    "Kesalahan umum": """
| Kesalahan | Akibat |
|---|---|
| Tidak melakukan baseline/auto zero | Offset instrumen terbaca; intersep besar |
| Kuvet kaca di daerah UV | Energi hampir nol < 320 nm; data hanya derau |
| Blanko tidak sesuai | Serapan reagen terbaca sebagai analit |
| Sidik jari atau gelembung | Hamburan menaikkan absorbansi |
| A di luar daerah kerja | Galat relatif meningkat |
| Mengekstrapolasi kurva | Hasil tidak dijamin linear |
| Lupa faktor pengenceran | Hasil akhir meleset sebesar FP |
| Mengabaikan efek matriks | Bias proporsional; gunakan adisi standar |
""",
    "Kualifikasi kinerja": """
Kualifikasi kinerja (PQ) memastikan instrumen layak dipakai sebelum data kuantitatif diterima.
Parameter minimum menurut Ph. Eur. 2.2.25 / USP ⟨857⟩:

| Parameter | Bahan acuan | Kriteria |
|---|---|---|
| Akurasi panjang gelombang | Holmium perklorat, garis D₂ 486,0/656,1 nm | ±1 nm (UV), ±3 nm (Vis) |
| Akurasi fotometrik | K₂Cr₂O₇ ≈ 60 mg/L dalam H₂SO₄ 0,005 M | A(1 %, 1 cm) dalam rentang tabel |
| Sinar sesat | KCl 12 g/L, 198 nm | A ≥ 2,0 |
| Resolusi | Toluena 0,02 % v/v dalam heksana | A₂₆₉/A₂₆₆ ≥ batas monografi (mis. 1,5) |
""",
}

QUIZ = [
    ("Absorbansi suatu larutan 0,500. Berapa persen transmitansinya?", ["50,0 %", "31,6 %", "5,0 %", "68,4 %"], 1,
     "%T = 100 × 10⁻ᴬ = 100 × 10⁻⁰·⁵ = 31,6 %. Hubungannya logaritmik."),
    ("Mengapa pengukuran kuantitatif dilakukan pada λmaks?", ["Derau lampu paling besar", "Kepekaan tertinggi dan dA/dλ ≈ 0", "Lambert–Beer hanya berlaku pada λmaks", "Agar A selalu < 0,1"], 1,
     "Di puncak, absorptivitas terbesar dan kemiringan spektrum mendekati nol sehingga galat akibat ketidaktepatan λ minimal."),
    ("Fungsi utama larutan blanko adalah…", ["Menaikkan absorbansi sampel", "Mengoreksi serapan pelarut, reagen, dan kuvet", "Mengkalibrasi λ", "Menghilangkan sinar sesat"], 1,
     "Blanko memuat semua komponen kecuali analit."),
    ("Kuvet kaca dipakai untuk parasetamol pada 249 nm. Akibatnya…", ["Tidak ada masalah", "Energi ke detektor sangat kecil; pembacaan tidak bermakna", "A tepat dua kali lipat", "Lampu WI rusak"], 1,
     "Kaca menyerap kuat di bawah ±320 nm; daerah UV memerlukan kuarsa."),
    ("Kurva y = 0,198x + 0,003 (mg/L). Sampel A = 0,450 setelah pengenceran 5×. C sampel asal?", ["2,26 mg/L", "11,29 mg/L", "0,45 mg/L", "2,29 mg/L"], 1,
     "x = (0,450 − 0,003)/0,198 = 2,258 mg/L; × 5 = 11,29 mg/L."),
    ("R² = 0,9990. Pernyataan paling tepat…", ["Linearitas pasti terbukti", "Plot residual dan uji kecocokan (mis. Mandel) tetap diperlukan", "Intersep pasti nol", "Tidak perlu ulangan"], 1,
     "R² tinggi tidak menjamin linearitas; kelengkungan sistematis dapat muncul pada R² > 0,999."),
    ("Penyebab penyimpangan negatif Lambert–Beer, kecuali…", ["Sinar sesat", "Dimerisasi", "Radiasi polikromatis", "Kuvet bersih dan blanko sesuai"], 3,
     "Kuvet bersih dan blanko sesuai adalah praktik yang benar."),
    ("Keuntungan utama sistem double-beam…", ["Tidak perlu blanko", "Fluktuasi lampu dan drift terkompensasi", "Rentang λ lebih lebar", "Tidak perlu monokromator"], 1,
     "Rasio I/I₀ diukur hampir bersamaan."),
    ("Galat relatif konsentrasi minimum (galat %T konstan) pada A sekitar…", ["0,043", "0,434", "1,5", "2,5"], 1,
     "Analisis Twyman–Lothian: minimum pada A = 0,434 (T = 36,8 %)."),
    ("Metode adisi standar terutama dipilih untuk…", ["Mempercepat analisis", "Mengoreksi efek matriks yang mengubah kepekaan (kemiringan)", "Menghilangkan kebutuhan blanko", "Menurunkan LOD instrumen"], 1,
     "Standar ditambahkan ke matriks sampel sendiri sehingga kemiringan kurva mencerminkan kepekaan dalam matriks tersebut."),
    ("Penentuan dua komponen dengan persamaan Vierordt paling andal bila…", ["Kedua spektrum identik", "Rasio absorptivitas kedua komponen pada λ1 dan λ2 sangat berbeda (bilangan kondisi kecil)", "Diukur pada satu λ", "Absorbansi > 2"], 1,
     "Determinan matriks absorptivitas yang jauh dari nol membuat sistem persamaan tidak sensitif terhadap derau."),
    ("Sinar sesat 0,1 % membatasi absorbansi terukur maksimum sekitar…", ["1,0", "2,0", "3,0", "4,0"], 2,
     "A_maks ≈ −log(0,001) = 3; di atasnya kurva mendatar."),
]


# =============================================================================
# BAGIAN 3 — INTERPRETASI, GRAFIK LAPORAN, PDF
# =============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

INK, IND, VIO, TEAL, MUTED = "#14123a", "#4f46e5", "#7c3aed", "#0d9488", "#5b6386"


def fnum(x, d=4, dec="koma"):
    if x is None or not isinstance(x, (int, float, np.floating, np.integer)) or not np.isfinite(x):
        return "—"
    s = f"{x:.{d}f}"
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    return s.replace(".", ",") if dec == "koma" else s


def fsig(x, sig_=4, dec="koma"):
    if x is None or not isinstance(x, (int, float, np.floating, np.integer)) or not np.isfinite(x):
        return "—"
    if x == 0:
        return fnum(0, sig_ - 1, dec)
    d = max(0, min(8, sig_ - 1 - int(np.floor(np.log10(abs(x))))))
    return fnum(x, d, dec)


def calib_interpretation(reg: Regression, unit: str, r2_min: float, std_res_max: float, dec="koma", md=None) -> list[str]:
    if not reg.ok:
        return []
    f_ = lambda x, d=4: fnum(x, d, dec)
    g_ = lambda x, s=4: fsig(x, s, dec)
    out = [f"Linearitas. Persamaan regresi {reg.equation()} dengan r = {f_(reg.r)} dan R² = {f_(reg.r2)}. "
           f"R² {'memenuhi' if reg.r2 >= r2_min else 'tidak memenuhi'} kriteria (≥ {f_(r2_min, 3)}); "
           f"{f_(reg.r2 * 100, 2)} % variasi absorbansi dijelaskan oleh model linear.",
           f"Kepekaan. Kemiringan {g_(reg.slope)} A per {unit} (± {g_(reg.ci_slope, 2)}, SK 95 %). Bila b = 1 cm dan satuan mg/L, "
           f"kemiringan setara absorptivitas a (L·mg⁻¹·cm⁻¹); ε = a × M (mg/mol)."]
    if not reg.through_zero:
        out.append(f"Intersep. b = {g_(reg.intercept, 3)} ± {g_(reg.ci_intercept, 2)} (SK 95 %). " +
                   ("Intersep berbeda nyata dari nol; telusuri serapan latar (blanko, reagen, offset baseline)."
                    if reg.intercept_significant else
                    "Intersep tidak berbeda nyata dari nol pada taraf 5 %, konsisten dengan hukum Lambert–Beer."))
    out.append(f"Batas deteksi. Dari s_y/x = {g_(reg.sxy, 3)}: LOD = 3,3·s_y/x/a = {g_(reg.lod, 3)} {unit} dan "
               f"LOQ = 10·s_y/x/a = {g_(reg.loq, 3)} {unit} (estimasi dari kurva; LOD formal memakai ulangan blanko).")
    if md:
        out.append(f"Uji Mandel. F = {f_(md['F'], 2)} vs F kritis(1; {md['n'] - 3}; 0,99) = {f_(md['Fcrit'], 2)}: " +
                   ("model kuadratik tidak memberi perbaikan nyata; model linear diterima." if md["linear_ok"] else
                    "model kuadratik memberi perbaikan nyata; persempit rentang kerja atau gunakan model nonlinear."))
    big = int((np.abs(reg.std_residuals) > std_res_max).sum())
    if big:
        out.append(f"Residual. {big} titik memiliki residual terstandar melebihi ±{f_(std_res_max, 1)}; periksa preparasi atau pembacaannya sebelum dieliminasi.")
    runs = residual_runs(reg.residuals)
    if reg.n >= 5 and runs <= 2:
        out.append(f"Pola residual. Tanda residual hanya berganti {runs - 1} kali (pola melengkung), indikasi ketidaklinearan.")
    if reg.n < 5:
        out.append(f"Jumlah titik. Kurva hanya memiliki {reg.n} titik; pedoman validasi menganjurkan minimal 5 tingkat konsentrasi.")
    return out


def _style(ax, xl, yl):
    ax.set_xlabel(xl, color=MUTED); ax.set_ylabel(yl, color=MUTED)
    ax.grid(True, color="#e7e9f5", lw=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#c4c9e2")
    ax.tick_params(colors=MUTED, labelsize=8)


def _png(fig) -> bytes:
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=180, bbox_inches="tight"); plt.close(fig)
    return buf.getvalue()


def spectrum_png(lam, A, name, peak=None) -> bytes:
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    o = np.argsort(lam)
    x, y = np.asarray(lam)[o], np.asarray(A)[o]
    ax.plot(x, y, color=IND, lw=1.6, label=name)
    ax.fill_between(x, y, color=IND, alpha=0.08)
    if peak:
        ax.axvline(peak, color=TEAL, ls="--", lw=1)
        ax.text(peak, ax.get_ylim()[1] * 0.95, " λmaks " + fnum(peak, 1) + " nm", color=TEAL, fontsize=8, va="top")
    _style(ax, "Panjang gelombang (nm)", "Absorbansi"); ax.legend(frameon=False, fontsize=8)
    return _png(fig)


def calib_png(x, y, reg: Regression, samples=(), unit="mg/L") -> bytes:
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    xs = np.linspace(min(0, reg.xmin), reg.xmax * 1.05, 60)
    yh = reg.slope * xs + reg.intercept; bd = reg.band(xs)
    ax.fill_between(xs, yh - bd, yh + bd, color=VIO, alpha=0.12, label="SK 95 % garis")
    ax.scatter(x, y, color=IND, zorder=3, label="Standar")
    ax.plot(xs, yh, color=VIO, lw=1.6, label=f"{reg.equation(4)}   R² = {reg.r2:.4f}")
    for name, cx, ay in samples:
        ax.scatter([cx], [ay], marker="D", color="#e11d48", zorder=4)
        ax.annotate(name, (cx, ay), textcoords="offset points", xytext=(6, 6), fontsize=8)
    _style(ax, f"Konsentrasi ({unit})", "Absorbansi"); ax.set_ylim(bottom=0); ax.legend(frameon=False, fontsize=8)
    return _png(fig)


def stdadd_png(x, y, res, unit="mg/L") -> bytes:
    r = res["reg"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    xs = np.array([-res["cx"] * 1.1, max(x) * 1.05])
    ax.plot(xs, r.slope * xs + r.intercept, color=VIO, lw=1.6)
    ax.scatter(x, y, color=IND, zorder=3)
    ax.scatter([-res["cx"]], [0], marker="X", color="#e11d48", zorder=4, s=60)
    ax.axhline(0, color="#9aa1c2", lw=0.8); ax.axvline(0, color="#9aa1c2", lw=0.8)
    _style(ax, f"C ditambahkan ({unit})", "Absorbansi")
    return _png(fig)


def _font():
    try:
        base = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(base, "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(base, "DejaVuSans-Bold.ttf")))
        return "DejaVu", "DejaVu-Bold"
    except Exception:
        return "Helvetica", "Helvetica-Bold"


def build_pdf(model: dict) -> bytes:
    F, FB = _font()
    st_ = {
        "h": ParagraphStyle("h", fontName=FB, fontSize=12, leading=15, textColor=colors.HexColor(INK), spaceBefore=10, spaceAfter=4),
        "p": ParagraphStyle("p", fontName=F, fontSize=9.5, leading=13.5, textColor=colors.HexColor("#1c1f3d"), alignment=TA_LEFT, spaceAfter=5),
        "note": ParagraphStyle("n", fontName=F, fontSize=8.5, leading=11.5, textColor=colors.HexColor(MUTED)),
        "cell": ParagraphStyle("c", fontName=F, fontSize=8.5, leading=11),
        "cellb": ParagraphStyle("cb", fontName=FB, fontSize=8, leading=10.5, textColor=colors.HexColor("#3b3f6b")),
    }
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=38 * mm, bottomMargin=18 * mm,
                            title="Laporan Analisis Spektrofotometri UV-Vis", author=model.get("praktikan") or "SPECTRA LAB")
    W = A4[0] - 36 * mm
    story = []
    if model.get("simulated"):
        t = Table([[Paragraph("<b>Mengandung data simulasi/contoh:</b> hasil latihan, bukan hasil pengujian laboratorium.", st_["cell"])]], colWidths=[W])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff4d6")), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        story += [t, Spacer(1, 6)]
    for b in model["blocks"]:
        kind = b["t"]
        if kind == "h":
            line = Table([[""]], colWidths=[W], rowHeights=[1.2]); line.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(IND))]))
            kt = KeepTogether([Paragraph(b["text"], st_["h"]), line, Spacer(1, 5)]); kt.keepWithNext = 1
            story.append(kt)
        elif kind == "p":
            story.append(Paragraph(b["text"], st_["p"]))
        elif kind == "note":
            story.append(Paragraph(b["text"], st_["note"]))
        elif kind == "kv":
            rows = [[Paragraph(k, st_["note"]), Paragraph(str(v), st_["cell"])] for k, v in b["rows"]]
            t = Table(rows, colWidths=[52 * mm, W - 52 * mm]); t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, -1), 2)]))
            story += [t, Spacer(1, 4)]
        elif kind == "table":
            data = [[Paragraph(str(h), st_["cellb"]) for h in b["head"]]] + [[Paragraph(str(c), st_["cell"]) for c in r] for r in b["rows"]]
            t = Table(data, repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef0fb")), ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#e1e4f2")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
            story += [t, Spacer(1, 6)]
        elif kind == "img":
            img = Image(io.BytesIO(b["png"])); ratio = img.imageHeight / img.imageWidth
            img.drawWidth = W; img.drawHeight = W * ratio
            story += [img, Paragraph(b["cap"], st_["note"]), Spacer(1, 6)]
    stamp = datetime.now().strftime("%d/%m/%Y %H:%M")
    rainbow = ["#7b3fe4", "#2f5bff", "#00b3d6", "#22c55e", "#facc15", "#f97316", "#ef4444"]

    def deco(canvas, doc_):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor(INK)); canvas.rect(0, A4[1] - 32 * mm, A4[0], 32 * mm, stroke=0, fill=1)
        seg = A4[0] / len(rainbow)
        for i, c in enumerate(rainbow):
            canvas.setFillColor(colors.HexColor(c)); canvas.rect(i * seg, A4[1] - 33.2 * mm, seg, 1.2 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.HexColor("#a5b4fc")); canvas.setFont(F, 8.5); canvas.drawString(18 * mm, A4[1] - 11 * mm, "SPECTRA LAB 2.0  ·  UV-Vis Analytical Workbench")
        canvas.setFillColor(colors.white); canvas.setFont(FB, 15); canvas.drawString(18 * mm, A4[1] - 19 * mm, "LAPORAN ANALISIS SPEKTROFOTOMETRI UV-VIS")
        canvas.setFillColor(colors.HexColor("#c7c9ef")); canvas.setFont(F, 9); canvas.drawString(18 * mm, A4[1] - 26 * mm, (model.get("judul") or "") + ("  ·  " + model["tanggal"] if model.get("tanggal") else ""))
        canvas.setStrokeColor(colors.HexColor("#e1e4f2")); canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
        canvas.setFillColor(colors.HexColor(MUTED)); canvas.setFont(F, 7.5)
        canvas.drawString(18 * mm, 9 * mm, f"SPECTRA LAB 2.0 · dibuat {stamp}"); canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Halaman {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=deco, onLaterPages=deco)
    return buf.getvalue()


# =============================================================================
# BAGIAN 4 — MODEL 3D INSTRUMEN (kanvas 2D, tanpa pustaka; disematkan lewat components.html)
# =============================================================================
MODEL3D_HTML = r'''<!doctype html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--fg:#14123a;--muted:#5b6386;--line:#e1e4f2;--bg:#ffffff;--soft:#eef0ff;--acc:#4f46e5}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:transparent;color:var(--fg);font:13px/1.45 Inter,system-ui,sans-serif}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 250px;gap:10px;height:100%;padding:2px}
.wrap.compact{grid-template-columns:1fr}.wrap.compact .side{display:none}
.stage{position:relative;border-radius:16px;overflow:hidden;min-height:260px;border:1px solid var(--line)}
canvas{width:100%;height:100%;display:block;cursor:grab;touch-action:none}
.tools{position:absolute;top:8px;right:8px;display:flex;gap:6px}
.views{position:absolute;left:8px;bottom:8px;display:flex;flex-wrap:wrap;gap:3px;background:rgba(255,255,255,.86);backdrop-filter:blur(6px);padding:4px;border-radius:10px;max-width:calc(100% - 16px)}
.badge{position:absolute;left:10px;top:10px;font:600 10px 'JetBrains Mono',monospace;letter-spacing:.1em;color:#4f46e5;background:rgba(255,255,255,.86);padding:4px 8px;border-radius:99px}
button{font:inherit;border:1px solid var(--line);background:#fff;color:var(--fg);border-radius:9px;padding:5px 9px;cursor:pointer}
button:hover{border-color:var(--acc)} .views button{border:0;background:none;color:var(--muted);font-size:12px} .views button.on{background:#4f46e5;color:#fff;font-weight:600}
.side{display:flex;flex-direction:column;gap:8px;overflow:auto}
.tg{display:grid;grid-template-columns:1fr 1fr;gap:6px}.tg button{text-align:left;font-size:12px;color:var(--muted)}
.tg button[aria-pressed=true]{border-color:var(--acc);background:var(--soft);color:var(--fg);font-weight:600}
.card{border:1px solid var(--line);border-radius:14px;padding:10px;background:#fff}
h4{margin:0 0 6px;font-size:13px}.hint{color:var(--muted);font-size:12px}
input[type=range]{width:100%;accent-color:var(--acc)}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-1px;margin-right:4px;border:1px solid rgba(0,0,0,.15)}
@media (max-width:640px){.wrap{grid-template-columns:1fr}.stage{height:360px}}
</style></head><body>
<div class="wrap" id="wrap"><div class="stage"><canvas id="c" aria-label="Model 3D spektrofotometer UV-1700"></canvas>
<div class="badge">UV-1700 · MODEL 3D</div>
<div class="tools"><button data-z="0.85" title="Perbesar">+</button><button data-z="1.18" title="Perkecil">−</button><button id="rs" title="Tampilan awal">⟲</button></div>
<div class="views"><button data-v="iso" class="on">Isometrik</button><button data-v="depan">Depan</button><button data-v="samping">Kanan</button><button data-v="belakang">Belakang</button><button data-v="atas">Atas</button><button data-v="kompartemen">Kompartemen</button></div></div>
<div class="side"><div class="card"><h4>Tampilan</h4><div class="tg">
<button data-t="lid" aria-pressed="false">Buka tutup</button><button data-t="xray" aria-pressed="false">Tembus pandang</button><button data-t="expl" aria-pressed="false">Terurai</button><button data-t="beam" aria-pressed="false">Jalur cahaya</button><button data-t="labels" aria-pressed="false">Label</button><button data-t="auto" aria-pressed="false">Putar otomatis</button></div></div>
<div class="card"><h4><span class="sw" id="sw"></span>λ = <span id="lv"></span> nm</h4><input type="range" id="lam" min="190" max="1100" step="1"><div class="hint" id="lamp"></div></div>
<div class="card"><h4 id="ti">Komponen</h4><div class="hint" id="info">Seret untuk memutar, roda/cubit untuk zoom, klik komponen untuk penjelasan.</div></div>
<div class="hint">Model skematis disederhanakan; proporsi mengikuti unit UV-1700 PharmaSpec di laboratorium, bukan gambar teknik pabrikan.</div></div></div>
<script>
(function (root) {
  'use strict';
  const SL = root.SL || (root.SL = {});
  const D2R = Math.PI / 180;
  function lamColor(l) {
    if (l < 380) return '#a99cf0'; if (l > 780) return '#b0302a';
    let r = 0, g = 0, b = 0;
    if (l < 440) { r = -(l - 440) / 60; b = 1; } else if (l < 490) { g = (l - 440) / 50; b = 1; }
    else if (l < 510) { g = 1; b = -(l - 510) / 20; } else if (l < 580) { r = (l - 510) / 70; g = 1; }
    else if (l < 645) { r = 1; g = -(l - 645) / 65; } else r = 1;
    const f = l < 420 ? 0.45 + 0.55 * (l - 380) / 40 : l > 700 ? 0.45 + 0.55 * (780 - l) / 80 : 1;
    const c = v => Math.round(255 * Math.pow(v * f, 0.8)).toString(16).padStart(2, '0');
    return '#' + c(r) + c(g) + c(b);
  }
  function hex(c) { c = c.replace('#', ''); if (c.length === 3) c = c.split('').map(x => x + x).join(''); const n = parseInt(c, 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
  function shade(col, k, emi) { const [r, g, b] = hex(col); const f = v => Math.max(0, Math.min(255, Math.round(v * k + (emi || 0) * 255))); return 'rgb(' + f(r) + ',' + f(g) + ',' + f(b) + ')'; }
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const norm = a => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
  function rotX(p, piv, ang) { const s = Math.sin(ang), c = Math.cos(ang); const y = p[1] - piv[1], z = p[2] - piv[2]; return [p[0], piv[1] + y * c - z * s, piv[2] + y * s + z * c]; }
  function rotY(p, piv, ang) { const s = Math.sin(ang), c = Math.cos(ang); const x = p[0] - piv[0], z = p[2] - piv[2]; return [piv[0] + x * c + z * s, p[1], piv[2] - x * s + z * c]; }

  const INFO = {
    base: ['Rangka dasar', 'Alas abu-abu gelap berkaki karet peredam getaran; panel belakang memuat port RS-232C, I/O-2, dan Centronics.'],
    body: ['Rumah instrumen', 'Melindungi optik dari cahaya luar dan debu. Kompartemen harus tertutup saat pengukuran.'],
    cover: ['Penutup atas', 'Panel atas yang menutupi bangku optik; pada tampilan terurai panel ini diangkat.'],
    lid: ['Tutup kompartemen sampel', 'Berengsel di sisi belakang, bagian dalam berwarna hitam untuk menekan pantulan. Tutup kembali sebelum membaca.'],
    compartment: ['Kompartemen sampel', 'Ruang hitam berisi dua dudukan kuvet: sel referensi (belakang) dan sel sampel (depan).'],
    holderR: ['Dudukan sel referensi', 'Tempat kuvet blanko; intensitas yang melewatinya menjadi I₀.'],
    holderS: ['Dudukan sel sampel', 'Tempat kuvet sampel; sisi bening kuvet harus menghadap berkas.'],
    cuvR: ['Kuvet blanko', 'Berisi pelarut atau blanko reagen.'],
    cuvS: ['Kuvet sampel', 'Berisi larutan uji; warna mengikuti larutan aktif.'],
    d2: ['Lampu deuterium (D₂)', 'Sumber kontinu UV ±190–350 nm; menyala pada model bila λ di bawah titik pergantian lampu.'],
    wi: ['Lampu wolfram-iodin (WI)', 'Sumber tampak–NIR ±340–1100 nm; pendarannya tampak oranye dari kisi ventilasi kanan.'],
    lamphouse: ['Rumah lampu', 'Ruang lampu berventilasi; label HOT SURFACE mengingatkan agar lampu didinginkan sebelum diganti.'],
    mirror: ['Cermin pemilih sumber', 'Berputar memilih berkas D₂ atau WI sesuai panjang gelombang.'],
    slit1: ['Celah masuk', 'Membatasi berkas masuk monokromator; menentukan lebar pita spektral (±1 nm).'],
    mono: ['Monokromator', 'Ruang tertutup berisi kisi difraksi; dinding gelap menekan sinar sesat.'],
    grating: ['Kisi difraksi holografik', 'Mendispersikan cahaya; sudut kisi menentukan λ keluaran.'],
    slit2: ['Celah keluar', 'Meneruskan pita sempit di sekitar λ terpilih.'],
    split: ['Pembagi berkas', 'Membagi cahaya monokromatis menjadi berkas referensi dan sampel.'],
    mirror2: ['Cermin pengarah', 'Membelokkan berkas sampel agar sejajar melewati sel sampel.'],
    detR: ['Detektor referensi', 'Fotodioda Si yang mengukur I₀.'],
    detS: ['Detektor sampel', 'Fotodioda Si yang mengukur I; A = log(I₀/I).'],
    pcb: ['Papan elektronik', 'Penguat, ADC, dan prosesor yang menghitung %T dan absorbansi.'],
    lcd: ['Layar LCD', 'Layar ungu-lavender menampilkan λ, data ABS/%T, dan lampu aktif (D2/WI).'],
    keypad: ['Papan tombol', 'MODE, GOTO WL, AUTO ZERO, START/STOP, dan tombol angka.'],
    vent: ['Kisi ventilasi', 'Membuang panas lampu dan catu daya. Jangan ditutup.'],
    card: ['Slot IC card · CONTRAST', 'Kartu memori metode/data dengan tombol EJECT; pengatur kontras LCD di sebelahnya.'],
    fan: ['Kipas & catu daya', 'Pendingin dan catu daya 230 V AC (dianjurkan melalui UPS/stabilizer).'],
    plate: ['Pelat identitas', 'UV-1700 PharmaSpec · UV-VIS SPECTROPHOTOMETER.'],
    warn: ['Label peringatan', 'WARNING — HOT SURFACE: dinginkan lampu hingga suhu ruang sebelum penggantian.']
  };
  const COMPART = new Set(['compartment', 'holderR', 'holderS', 'cuvR', 'cuvS']);

  class Model3D {
    constructor(canvas, opt) {
      this.cv = canvas; this.ctx = canvas.getContext('2d'); this.opt = opt || {};
      this.yaw = -32; this.pitch = 24; this.dist = 84; this.target = [0, 9, 0];
      this.lid = 0; this.lidT = 0; this.expl = 0; this.explT = 0; this.xray = 0; this.xrayT = 0;
      this.beam = true; this.labels = false; this.auto = false; this.sel = null;
      this.lam = 550; this.lampSwitch = 340.8; this.abs = 0; this.status = 'READY'; this.solColor = '#8e2c8f';
      this._build(); this._bind(); this._loop = this._loop.bind(this);
      this._ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(() => this.draw()) : null;
      if (this._ro) this._ro.observe(canvas.parentElement || canvas);
      this.draw();
    }
    _build() {
      const P = this.parts = [];
      const box = (id, grp, x0, x1, y0, y1, z0, z1, col, o) => {
        o = o || {};
        const faces = [
          ['f', [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]],
          ['b', [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]]],
          ['r', [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]]],
          ['l', [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]]],
          ['t', [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]]],
          ['u', [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]]
        ];
        const cell = o.cell || 4.5;
        for (const [k, q] of faces) {
          if (o.skip && o.skip.includes(k)) continue;
          const fc = (o.faceCol && o.faceCol[k]) || col;
          const lu = Math.hypot(...sub(q[1], q[0])), lv = Math.hypot(...sub(q[3], q[0]));
          const isText = o.text && k === o.text.face;
          const nu = isText ? 1 : Math.max(1, Math.ceil(lu / cell)), nv = isText ? 1 : Math.max(1, Math.ceil(lv / cell));
          for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
            const L = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
            const bl = (u, v) => L(L(q[0], q[1], u), L(q[3], q[2], u), v);
            P.push({ id, grp, col: fc, alpha: o.alpha == null ? 1 : o.alpha, emi: o.emi || 0, bias: o.bias || 0, tf: o.tf || null, key: k,
              v: [bl(i / nu, j / nv), bl((i + 1) / nu, j / nv), bl((i + 1) / nu, (j + 1) / nv), bl(i / nu, (j + 1) / nv)],
              text: (o.text && k === o.text.face && nu === 1 && nv === 1) ? o.text : null, dyn: o.dyn || null });
          }
        }
      };
      // palet mengikuti foto unit: krem-mustard, alas abu-abu arang, kompartemen hitam
      const BEIGE = '#d8bf7f', BEIGE2 = '#e2cb8e', GREY = '#4a5058', DARK = '#1e2126', CREAM = '#efe9d6';
      box('base', 'body', -23, 23, 0, 3, -24.5, 24.5, GREY, { cell: 6 });
      [[-19, 20], [19, 20], [-19, -20], [19, -20]].forEach(([x, z]) => box('base', 'body', x - 1.6, x + 1.6, -1.4, 0, z - 1.6, z + 1.6, '#363b42', { skip: ['u'] }));
      // port belakang (RS-232C merah, Centronics)
      box('base', 'body', 4, 7, 0.6, 2.2, -24.8, -24.5, '#c73a3a', { bias: -4, cell: 99 });
      box('base', 'body', 12, 18, 0.7, 2.0, -24.8, -24.5, '#2b2f35', { bias: -4, cell: 99 });
      const wallO = { cell: 3 };
      box('body', 'walls', -23, 23, 3, 13.5, 23.9, 24.5, BEIGE, wallO);
      box('body', 'walls', -23, 23, 3, 13.5, -24.5, -23.9, BEIGE, wallO);
      box('body', 'walls', -23, -22.4, 3, 13.5, -23.9, 23.9, BEIGE, wallO);
      box('body', 'walls', 22.4, 23, 3, 13.5, -23.9, 23.9, BEIGE, wallO);
      // panel depan kompartemen (pintu krem lebih terang) dan pelat identitas
      box('compartment', 'walls', -16, -1, 4, 12.5, 24.5, 24.8, '#cfb575', { bias: -4, skip: ['b'], cell: 99 });
      box('plate', 'walls', 9, 20, 6.5, 10.5, 24.5, 24.7, '#f7f6f0', { bias: -4.5, skip: ['b'], cell: 99,
        text: { face: 'f', w: 110, h: 40, lines: [['UV-1700', 8, 18, 'left'], ['PharmaSpec', 70, 18, 'left'], ['UV-VIS SPECTROPHOTOMETER', 8, 30, 'left']], color: '#22263a', font: '600 12px Arial' } });
      // sisi kanan: kisi ventilasi (satu kisi berpendar oranye saat WI menyala), slot IC card, EJECT, CONTRAST, label peringatan
      for (let i = 0; i < 8; i++) box('vent', 'walls', 23, 23.25, 5.6 + i * 0.95, 6.1 + i * 0.95, -20, -11, '#bfa868', { bias: -4, cell: 99, dyn: i === 4 ? 'glow' : null });
      box('card', 'walls', 23, 23.2, 11, 11.8, 1, 10, '#2b2f36', { bias: -4, cell: 99 });
      box('card', 'walls', 23, 23.35, 11.05, 11.75, -0.6, 0.8, '#f2efe4', { bias: -4.2, cell: 99 });
      box('card', 'walls', 23, 23.45, 10.9, 11.9, 11.2, 13.4, '#ebe6d6', { bias: -4.2, cell: 99 });
      box('warn', 'walls', 23, 23.15, 4.6, 8.4, 4, 13, '#fafafa', { bias: -4.3, cell: 99 });
      box('warn', 'walls', 23, 23.2, 7, 8.2, 7.5, 12.6, '#f6d21f', { bias: -4.5, cell: 99 });
      // bangku optik
      box('pcb', 'inner', -21.8, -12, 3, 3.6, -22, -6, '#1d6b45');
      [[-20, -19], [-17, -15], [-14, -10]].forEach(([a, b], k) => box('pcb', 'inner', a, b, 3.6, 4.4, -20 + k * 4, -18 + k * 4, '#15181d'));
      box('fan', 'inner', -21.6, -15, 3, 9, -23.6, -21.8, '#3a4048');
      box('lamphouse', 'inner', 14, 22.2, 3, 4, -23.4, -9, '#3a4048');
      box('lamphouse', 'inner', 21.6, 22.2, 4, 12, -23.4, -9, '#4a515b', { alpha: 0.9 });
      box('d2', 'inner', 18, 20.2, 4, 11.5, -21, -18.6, '#d9d4ff', { dyn: 'd2' });
      box('wi', 'inner', 18, 20.2, 4, 10.5, -14.2, -11.8, '#fff1b8', { dyn: 'wi' });
      box('mirror', 'inner', 15, 15.6, 7.5, 11.5, -18, -14, '#cfd8e6', { dyn: 'mirror' });
      box('slit1', 'inner', 11, 11.4, 6, 13, -18.5, -16.4, DARK); box('slit1', 'inner', 11, 11.4, 6, 13, -15.6, -13.5, DARK);
      box('mono', 'inner', -2, 10.5, 3, 4, -23.2, -9, '#2d323a');
      box('mono', 'inner', -2, 10.5, 4, 12.5, -23.2, -22.7, '#3a4048', { alpha: 0.55 });
      box('mono', 'inner', -2, -1.5, 4, 12.5, -22.7, -9, '#3a4048', { alpha: 0.55 });
      box('grating', 'inner', 1.5, 5.5, 5.5, 12.5, -21.8, -20.6, '#9fb3cf', { faceCol: { f: '#c9d6ea' } });
      box('slit2', 'inner', 1.5, 2.6, 6, 13, -9.4, -9, DARK); box('slit2', 'inner', 3.4, 4.5, 6, 13, -9.4, -9, DARK);
      box('split', 'inner', 2.4, 3.6, 6.5, 12, 1.4, 4.6, '#7fe6f7', { alpha: 0.7, tf: [['y', [3, 0, 3], 45]] });
      box('mirror2', 'inner', 2.4, 3.6, 6.5, 12, 11.4, 14.6, '#cfd8e6', { tf: [['y', [3, 0, 13], -45]] });
      box('compartment', 'inner', -15, -2, 3, 6, -2, 20, '#16181c', { cell: 4 });
      box('compartment', 'inner', -15, -14.5, 6, 13.5, -2, 20, '#1d2025', { alpha: 0.88 });
      box('compartment', 'inner', -2.5, -2, 6, 13.5, -2, 20, '#1d2025', { alpha: 0.88 });
      box('compartment', 'inner', -14.5, -2.5, 6, 13.5, -2, -1.5, '#1d2025');
      box('compartment', 'inner', -14.5, -2.5, 6, 13.5, 19.5, 20, '#1d2025');
      // dua sumbat karet bulat di lantai kompartemen (terlihat di foto)
      [[-12.5, 1], [-12.5, 16.5]].forEach(([x, z]) => box('compartment', 'inner', x - 1, x + 1, 6, 6.3, z - 1, z + 1, '#0f1013'));
      const holder = (id, z) => {
        box(id, 'inner', -10.6, -6.4, 6, 12.5, z - 2.1, z - 1.6, DARK); box(id, 'inner', -10.6, -6.4, 6, 12.5, z + 1.6, z + 2.1, DARK);
        box(id, 'inner', -10.6, -10.1, 6, 12.5, z - 1.6, z + 1.6, DARK, { alpha: 0.9 });
        box(id, 'inner', -11.5, -5.5, 6, 6.6, z - 2.6, z + 2.6, '#101215');
        box(id, 'inner', -8.9, -8.1, 12.5, 13.2, z - 0.4, z + 0.4, '#8b929c');
      };
      holder('holderR', 3); holder('holderS', 13);
      box('cuvR', 'inner', -9.8, -7.2, 6.6, 14.2, 1.7, 4.3, '#d9f3f7', { alpha: 0.35, bias: -0.4 });
      box('cuvS', 'inner', -9.8, -7.2, 6.6, 14.2, 11.7, 14.3, '#8e2c8f', { alpha: 0.55, bias: -0.4, dyn: 'cuvS' });
      box('detR', 'inner', -21.6, -19.6, 8, 11.5, 1.6, 4.4, '#2a3346'); box('detS', 'inner', -21.6, -19.6, 8, 11.5, 11.6, 14.4, '#2a3346');
      const covO = { cell: 4.5 };
      box('cover', 'cover', -23, -15, 13.5, 15, -24.5, 24.5, BEIGE2, covO);
      box('cover', 'cover', -2, 23, 13.5, 15, -24.5, 24.5, BEIGE2, covO);
      box('cover', 'cover', -15, -2, 13.5, 15, -24.5, -2, BEIGE2, covO);
      box('cover', 'cover', -15, -2, 13.5, 15, 20, 24.5, BEIGE2, covO);
      for (let i = 0; i < 14; i++) box('vent', 'cover', 6 + i * 1.15, 6.55 + i * 1.15, 15, 15.25, -23.5, -15.5, '#c6ae6c');
      box('lid', 'lid', -15.2, -1.8, 15, 16.6, -2.2, 20.2, '#dcc589', { cell: 4, faceCol: { u: '#111215' } });
      box('keypad', 'cover', 2, 21, 15, 16.2, 13, 23.5, CREAM, { bias: -0.6 });
      const keys = [];
      for (let r = 0; r < 4; r++) for (let c = 0; c < 6; c++) keys.push([2.8 + c * 3.0, 14.2 + r * 2.2]);
      keys.forEach(([x, z], k) => box('keypad', 'cover', x, x + 1.9, 16.2, 16.55, z, z + 1.3, k % 6 === 5 ? '#7c93c9' : '#cfd4dc', { bias: -0.8 }));
      box('lcd', 'cover', 9, 14, 15, 17.5, 7.5, 10, '#ddd6bd', { bias: -0.6 });
      const tilt = [['x', [11.5, 16.5, 9], -18]];
      box('lcd', 'cover', 3.5, 19.5, 16.5, 28.5, 8.2, 10.2, '#ebe5cf', { tf: tilt, bias: -0.7, cell: 99 });
      box('lcd', 'cover', 6, 17, 18.3, 26.6, 10.2, 10.35, '#a99af2', { tf: tilt, bias: -4, emi: 0.08, dyn: 'lcd', text: { face: 'f', w: 130, h: 96, lines: [] } });
    }
    setLam(l) { this.lam = l; this.draw(); }
    toggle(what, on) {
      const flip = (cur) => on == null ? (cur ? 0 : 1) : (on ? 1 : 0);
      if (what === 'lid') this.lidT = flip(this.lidT);
      if (what === 'expl') this.explT = flip(this.explT);
      if (what === 'xray') this.xrayT = flip(this.xrayT);
      if (what === 'beam') this.beam = on == null ? !this.beam : !!on;
      if (what === 'labels') this.labels = on == null ? !this.labels : !!on;
      if (what === 'auto') this.auto = on == null ? !this.auto : !!on;
      this._kick(); return this.state();
    }
    state() { return { lid: this.lidT, expl: this.explT, xray: this.xrayT, beam: this.beam, labels: this.labels, auto: this.auto }; }
    view(name) {
      const V = { depan: [0, 8], samping: [90, 10], atas: [0, 88], belakang: [180, 14], iso: [-32, 24], kompartemen: [-8, 55] };
      const v = V[name] || V.iso; this._anim = { from: [this.yaw, this.pitch], to: v, t: 0 }; this._kick();
    }
    zoom(f) { this.dist = Math.max(55, Math.min(260, this.dist * f)); this.draw(); }
    reset() { this.dist = 84; this.view('iso'); this.sel = null; if (this.opt.onPick) this.opt.onPick(null); }
    _kick() { if (!this._running) { this._running = true; this._last = performance.now(); requestAnimationFrame(this._loop); } }
    _loop(now) {
      const dt = Math.min(0.05, (now - this._last) / 1000); this._last = now;
      const ease = (cur, tgt, sp) => { const d = tgt - cur; return Math.abs(d) < 0.002 ? tgt : cur + d * Math.min(1, dt * sp); };
      this.lid = ease(this.lid, this.lidT, 5); this.expl = ease(this.expl, this.explT, 4); this.xray = ease(this.xray, this.xrayT, 5);
      if (this._anim) {
        const a = this._anim; a.t = Math.min(1, a.t + dt * 2.2); const k = a.t < 0.5 ? 2 * a.t * a.t : 1 - Math.pow(-2 * a.t + 2, 2) / 2;
        let dy = a.to[0] - a.from[0]; while (dy > 180) dy -= 360; while (dy < -180) dy += 360;
        this.yaw = a.from[0] + dy * k; this.pitch = a.from[1] + (a.to[1] - a.from[1]) * k; if (a.t >= 1) this._anim = null;
      }
      if (this.auto && !this._drag) this.yaw += dt * 18;
      this.draw();
      const busy = this.lid !== this.lidT || this.expl !== this.explT || this.xray !== this.xrayT || this._anim || (this.auto && !this._drag);
      if (busy) requestAnimationFrame(this._loop); else this._running = false;
    }
    _xf(p, part) {
      let q = p;
      if (part.tf) for (const [ax, piv, deg] of part.tf) q = ax === 'x' ? rotX(q, piv, deg * D2R) : rotY(q, piv, deg * D2R);
      if (part.grp === 'lid') { q = rotX(q, [0, 15, -2.2], -102 * this.lid * D2R); q = [q[0], q[1] + 16 * this.expl, q[2]]; }
      if (part.grp === 'cover') q = [q[0], q[1] + 16 * this.expl, q[2]];
      return q;
    }
    _cam(p) {
      const t = this.target; let x = p[0] - t[0], y = p[1] - t[1] - 7 * this.expl, z = p[2] - t[2];
      const cy = Math.cos(-this.yaw * D2R), sy = Math.sin(-this.yaw * D2R);
      const x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
      const cp = Math.cos(this.pitch * D2R), sp = Math.sin(this.pitch * D2R);
      return [x1, y * cp - z1 * sp, y * sp + z1 * cp];
    }
    _proj(c) { const zc = this.dist * (1 + 0.28 * this.expl) - c[2]; const f = this._f; return [this._cx + f * c[0] / zc, this._cy - f * c[1] / zc, zc]; }
    draw() {
      const cv = this.cv, ctx = this.ctx;
      const rect = cv.getBoundingClientRect(); const W = Math.max(240, rect.width || cv.width), H = Math.max(200, rect.height || cv.height);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this._cx = W / 2; this._cy = H / 2 + 10; this._f = Math.min(W, H * 1.45);
      const g = ctx.createRadialGradient(W * 0.5, H * 0.35, 20, W * 0.5, H * 0.5, Math.max(W, H));
      g.addColorStop(0, '#ffffff'); g.addColorStop(1, '#e8eaf7');
      ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
      // lantai ubin laboratorium
      ctx.strokeStyle = 'rgba(20,18,58,.06)'; ctx.lineWidth = 1;
      for (let i = -60; i <= 60; i += 12) {
        const a = this._proj(this._cam([i, -1.4, -60])), b = this._proj(this._cam([i, -1.4, 60]));
        const c = this._proj(this._cam([-60, -1.4, i])), d = this._proj(this._cam([60, -1.4, i]));
        if (a[2] > 1 && b[2] > 1) { ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
        if (c[2] > 1 && d[2] > 1) { ctx.beginPath(); ctx.moveTo(c[0], c[1]); ctx.lineTo(d[0], d[1]); ctx.stroke(); }
      }
      const sh = [[-25, -26.5], [25, -26.5], [25, 26.5], [-25, 26.5]].map(p => this._proj(this._cam([p[0] + 1.5, -1.4, p[1] + 1.5])));
      ctx.fillStyle = 'rgba(20,18,58,.12)'; ctx.beginPath(); sh.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.closePath(); ctx.fill();
      const light = norm([0.45, 0.85, 0.55]);
      const camPos = [0, 0, this.dist];
      const lampUV = this.lam < this.lampSwitch;
      const items = [];
      const coverA = 1 - 0.82 * this.xray, wallA = 1 - 0.75 * this.xray;
      const closed = this.xray < 0.04 && this.expl < 0.04;
      for (const part of this.parts) {
        const wv = part.v.map(p => this._xf(p, part));
        const cvs = wv.map(p => this._cam(p));
        const n = norm(cross(sub(cvs[1], cvs[0]), sub(cvs[3], cvs[0])));
        const cen = [(cvs[0][0] + cvs[2][0]) / 2, (cvs[0][1] + cvs[2][1]) / 2, (cvs[0][2] + cvs[2][2]) / 2];
        let alpha = part.alpha;
        if (part.grp === 'cover' || part.grp === 'lid') alpha *= coverA;
        if (part.grp === 'walls' || part.id === 'body') alpha *= wallA;
        if (alpha < 0.02) continue;
        if (part.grp === 'inner' && closed && !(this.lid > 0.05 && COMPART.has(part.id))) continue;
        const facing = dot(n, sub(camPos, cen)) > 0;
        if (!facing && alpha > 0.97) continue;
        const wn = norm(cross(sub(wv[1], wv[0]), sub(wv[3], wv[0])));
        const k = 0.5 + 0.55 * Math.max(0, dot(wn, light));
        let col = part.col, emi = part.emi;
        if (part.dyn === 'd2') { emi = lampUV ? 0.45 : 0.0; col = lampUV ? '#e4dcff' : '#9a96b5'; }
        if (part.dyn === 'wi') { emi = !lampUV ? 0.45 : 0.0; col = !lampUV ? '#ffe7a0' : '#a89c74'; }
        if (part.dyn === 'glow') { col = !lampUV && this.status !== 'OFF' ? '#f0873a' : '#bfa868'; emi = !lampUV && this.status !== 'OFF' ? 0.22 : 0; }
        if (part.dyn === 'cuvS') col = this.solColor;
        if (this.sel && part.id === this.sel) emi += 0.18;
        items.push({ part, pts: cvs.map(c => this._proj(c)), depth: this.dist - cen[2] + part.bias + (part.grp === 'inner' ? 0.35 : 0) + (facing ? 0 : 0.02), fill: shade(col, k, emi), alpha, facing });
      }
      if (this.beam) {
        const y = 9.5, by = this.lid > 0.5 || this.xray > 0.5 || this.expl > 0.5;
        const src = lampUV ? [19.1, y, -19.8] : [19.1, y, -13];
        const path = [[src, [15.3, y, -16]], [[15.3, y, -16], [11.2, y, -15]], [[11.2, y, -15], [3.5, y, -21]], [[3.5, y, -21], [3, y, -9.2]], [[3, y, -9.2], [3, y, 3]], [[3, y, 3], [3, y, 13]], [[3, y, 3], [-8.5, y, 3]], [[-8.5, y, 3], [-19.6, y, 3]], [[3, y, 13], [-8.5, y, 13]], [[-8.5, y, 13], [-19.6, y, 13]]];
        path.forEach(([a, b], si) => {
          for (let kk = 0; kk < 4; kk++) {
            const t0 = kk / 4, t1 = (kk + 1) / 4;
            const A = [a[0] + (b[0] - a[0]) * t0, a[1], a[2] + (b[2] - a[2]) * t0], B = [a[0] + (b[0] - a[0]) * t1, a[1], a[2] + (b[2] - a[2]) * t1];
            const ca = this._cam(A), cb = this._cam(B);
            if (closed && !(this.lid > 0.05 && si >= 6)) continue;
            const atten = si === 9 ? Math.max(0.12, Math.pow(10, -Math.max(0, Math.min(3, this.abs)))) : 1;
            items.push({ beam: true, pa: this._proj(ca), pb: this._proj(cb), depth: this.dist - (ca[2] + cb[2]) / 2 - 0.3, color: si <= 2 ? '#fffbe8' : lamColor(this.lam), uv: si > 2 && this.lam < 380, atten, show: by });
          }
        });
      }
      items.sort((a, b) => b.depth - a.depth);
      this._drawn = [];
      ctx.lineJoin = 'round';
      for (const it of items) {
        if (it.beam) {
          ctx.save(); ctx.globalAlpha = 0.28 * it.atten * (it.show ? 1 : 0.55); ctx.strokeStyle = it.color; ctx.lineWidth = 7; ctx.lineCap = 'round';
          ctx.beginPath(); ctx.moveTo(it.pa[0], it.pa[1]); ctx.lineTo(it.pb[0], it.pb[1]); ctx.stroke();
          ctx.globalAlpha = 0.95 * it.atten; ctx.lineWidth = 2.2; if (it.uv) ctx.setLineDash([5, 4]);
          ctx.beginPath(); ctx.moveTo(it.pa[0], it.pa[1]); ctx.lineTo(it.pb[0], it.pb[1]); ctx.stroke(); ctx.restore();
          continue;
        }
        const p = it.pts; ctx.globalAlpha = it.alpha;
        ctx.beginPath(); ctx.moveTo(p[0][0], p[0][1]); for (let i = 1; i < 4; i++) ctx.lineTo(p[i][0], p[i][1]); ctx.closePath();
        ctx.fillStyle = it.fill; ctx.fill();
        if (it.alpha > 0.97) { ctx.strokeStyle = it.fill; ctx.lineWidth = 1.1; ctx.stroke(); }
        if (this.sel && it.part.id === this.sel) { ctx.strokeStyle = '#7c3aed'; ctx.lineWidth = 1.2; ctx.stroke(); }
        ctx.globalAlpha = 1;
        if (it.part.text && it.facing) this._text(it.part, p);
        if (it.facing && it.alpha > 0.3) this._drawn.push({ id: it.part.id, p });
      }
      if (this.labels) this._labels(ctx);
    }
    _text(part, p) {
      const ctx = this.ctx, t = part.text;
      const TL = p[3], TR = p[2], BL = p[0];
      ctx.save();
      const dpr = this.cv.width / Math.max(1, this.cv.getBoundingClientRect().width || this.cv.width);
      ctx.setTransform((TR[0] - TL[0]) / t.w * dpr, (TR[1] - TL[1]) / t.w * dpr, (BL[0] - TL[0]) / t.h * dpr, (BL[1] - TL[1]) / t.h * dpr, TL[0] * dpr, TL[1] * dpr);
      if (part.dyn === 'lcd') {
        ctx.fillStyle = 'rgba(255,255,255,.85)'; [16, 44, 72].forEach(yy => ctx.fillRect(8, yy - 6, 24, 5));
        ctx.fillStyle = '#1b1248'; ctx.font = '500 17px "JetBrains Mono", monospace'; ctx.textAlign = 'right';
        ctx.fillText(this.lam.toFixed(1) + 'nm', 124, 22);
        ctx.fillText((this.status === 'OFF' ? '-----' : (this.abs > 3 ? 'OVER' : this.abs.toFixed(3))) + ' ABS', 124, 50);
        ctx.font = '500 13px "JetBrains Mono", monospace'; ctx.textAlign = 'left'; ctx.fillText(this.lam < this.lampSwitch ? 'D2' : 'WI', 40, 77);
        ctx.textAlign = 'right'; ctx.fillStyle = '#4a3fa0'; ctx.font = '500 10px "JetBrains Mono", monospace'; ctx.fillText(this.status, 124, 90);
      } else {
        ctx.fillStyle = t.color || '#222';
        t.lines.forEach(([s, x, y, al], i) => { ctx.font = i === 0 ? '700 15px Arial' : (i === 1 ? 'italic 9px Arial' : '6.5px Arial'); if (i === 2) ctx.fillStyle = '#c0392b'; ctx.textAlign = al || 'left'; ctx.fillText(s, x, y); });
      }
      ctx.restore();
    }
    _labels(ctx) {
      const A = [['d2', [19, 12, -20]], ['wi', [19, 11, -13]], ['mirror', [15.3, 12, -16]], ['slit1', [11.2, 13.5, -15]], ['grating', [3.5, 13, -21]], ['slit2', [3, 13.5, -9.2]], ['split', [3, 12.5, 3]], ['holderR', [-8.5, 13, 3]], ['holderS', [-8.5, 13, 13]], ['detS', [-20.6, 12, 13]], ['detR', [-20.6, 12, 3]], ['pcb', [-17, 5, -14]], ['lcd', [11.5, 29, 6]], ['keypad', [12, 17, 22]], ['lid', [-8.5, 17, 9]], ['card', [23.3, 12, 6]]];
      ctx.font = '600 11px Inter, Arial';
      const placed = [];
      const L = A.map(([id, p0]) => { const part = this.parts.find(q => q.id === id); const p = part ? this._xf(p0, part) : p0; return { id, s: this._proj(this._cam(p)) }; }).sort((a, b) => a.s[1] - b.s[1]);
      for (const { id, s } of L) {
        const name = INFO[id][0].replace(/ \(.*\)/, '');
        const w = ctx.measureText(name).width + 12; const x = s[0]; let y = s[1] - 16;
        for (let k = 0; k < 8 && placed.some(r => Math.abs(r.x - x) < (r.w + w) / 2 + 2 && Math.abs(r.y - y) < 17); k++) y -= 17;
        placed.push({ x, y, w });
        ctx.strokeStyle = 'rgba(20,18,58,.5)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(s[0], s[1]); ctx.lineTo(x, y + 4); ctx.stroke();
        ctx.fillStyle = id === this.sel ? '#4f46e5' : 'rgba(255,255,255,.95)';
        ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x - w / 2, y - 11, w, 16, 8) : ctx.rect(x - w / 2, y - 11, w, 16); ctx.fill();
        ctx.fillStyle = id === this.sel ? '#fff' : '#14123a'; ctx.textAlign = 'center'; ctx.fillText(name, x, y + 1);
        ctx.fillStyle = '#7c3aed'; ctx.beginPath(); ctx.arc(s[0], s[1], 2.5, 0, 7); ctx.fill();
      }
    }
    pickAt(x, y) {
      const inPoly = (p, px, py) => { let c = false; for (let i = 0, j = 3; i < 4; j = i++) { if (((p[i][1] > py) !== (p[j][1] > py)) && (px < (p[j][0] - p[i][0]) * (py - p[i][1]) / (p[j][1] - p[i][1]) + p[i][0])) c = !c; } return c; };
      for (let i = this._drawn.length - 1; i >= 0; i--) if (inPoly(this._drawn[i].p, x, y)) return this._drawn[i].id;
      return null;
    }
    _bind() {
      const cv = this.cv; cv.style.touchAction = 'none';
      let pts = new Map(), start = null, pinch = null;
      cv.addEventListener('pointerdown', e => { cv.setPointerCapture(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]); start = { x: e.clientX, y: e.clientY, yaw: this.yaw, pitch: this.pitch, moved: false }; this._drag = true; if (pts.size === 2) { const [a, b] = [...pts.values()]; pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), dist: this.dist }; } });
      cv.addEventListener('pointermove', e => {
        if (!pts.has(e.pointerId)) return; pts.set(e.pointerId, [e.clientX, e.clientY]);
        if (pts.size === 2 && pinch) { const [a, b] = [...pts.values()]; const d = Math.hypot(a[0] - b[0], a[1] - b[1]); this.dist = Math.max(55, Math.min(260, pinch.dist * pinch.d / Math.max(10, d))); start.moved = true; this.draw(); return; }
        if (!start) return; const dx = e.clientX - start.x, dy = e.clientY - start.y;
        if (Math.abs(dx) + Math.abs(dy) > 4) start.moved = true;
        this.yaw = start.yaw - dx * 0.45; this.pitch = Math.max(-10, Math.min(89, start.pitch + dy * 0.35)); this._anim = null; this.draw();
      });
      const up = e => {
        if (!pts.has(e.pointerId)) return; pts.delete(e.pointerId); if (pts.size < 2) pinch = null;
        if (pts.size === 0) {
          this._drag = false;
          if (start && !start.moved) { const r = cv.getBoundingClientRect(); const id = this.pickAt(e.clientX - r.left, e.clientY - r.top); this.sel = id; this.draw(); if (this.opt.onPick) this.opt.onPick(id ? { id, title: INFO[id][0], text: INFO[id][1] } : null); }
          start = null; if (this.auto) this._kick();
        }
      };
      cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
      cv.addEventListener('wheel', e => { e.preventDefault(); this.zoom(e.deltaY > 0 ? 1.08 : 1 / 1.08); }, { passive: false });
      cv.tabIndex = 0;
      cv.addEventListener('keydown', e => {
        const k = e.key; if (k === 'ArrowLeft') this.yaw -= 6; else if (k === 'ArrowRight') this.yaw += 6; else if (k === 'ArrowUp') this.pitch = Math.min(89, this.pitch + 5); else if (k === 'ArrowDown') this.pitch = Math.max(-10, this.pitch - 5); else if (k === '+' || k === '=') this.zoom(0.9); else if (k === '-') this.zoom(1.1); else return;
        e.preventDefault(); this.draw();
      });
    }
  }
  SL.Model3D = Model3D; SL.lamColor = lamColor;
})(typeof window !== 'undefined' ? window : globalThis);
</script>
<script>
var INIT = Object.assign({lam: 550, abs: 0, status: "READY", solColor: "#8e2c8f", lampSwitch: 340.8, xray: true, lid: false, compact: false, auto: false}, /*__INIT__*/null || {});
(function(){
  var SL = window.SL, $ = function(s){return document.querySelector(s)};
  if (INIT.compact) $('#wrap').classList.add('compact');
  var m = new SL.Model3D($('#c'), { onPick: function(i){ $('#ti').textContent = i ? i.title : 'Komponen'; $('#info').textContent = i ? i.text : 'Klik komponen untuk penjelasan.'; } });
  m.lampSwitch = INIT.lampSwitch; m.abs = INIT.abs; m.status = INIT.status; m.solColor = INIT.solColor;
  function sync(){ var st = m.state(); document.querySelectorAll('[data-t]').forEach(function(b){ b.setAttribute('aria-pressed', String(!!st[b.dataset.t])); }); }
  function setLam(l){ m.setLam(l); $('#lv').textContent = l; $('#lam').value = l; $('#sw').style.background = SL.lamColor(l); $('#lamp').textContent = (l < m.lampSwitch ? 'Lampu D2 aktif' : 'Lampu WI aktif') + ' · ' + (l < 400 ? 'UV' : l <= 780 ? 'sinar tampak' : 'NIR'); }
  m.toggle('beam', true); if (INIT.xray) m.toggle('xray', true); if (INIT.lid) m.toggle('lid', true); if (INIT.auto) m.toggle('auto', true);
  setLam(INIT.lam); sync();
  document.addEventListener('click', function(e){ var b = e.target.closest('button'); if(!b) return;
    if (b.dataset.t){ m.toggle(b.dataset.t); sync(); }
    if (b.dataset.v){ m.view(b.dataset.v); document.querySelectorAll('.views button').forEach(function(x){ x.classList.toggle('on', x===b); }); }
    if (b.dataset.z) m.zoom(+b.dataset.z); if (b.id==='rs') m.reset(); });
  $('#lam').addEventListener('input', function(e){ setLam(+e.target.value); });
})();
</script></body></html>
'''


# =============================================================================
# BAGIAN 4B — PRAKTIKUM VIRTUAL 3D (meja lab, kabel, UPS, CPU, monitor, tisu, kuvet)
# =============================================================================
LAB_HTML = r'''<!doctype html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--fg:#14123a;--muted:#5b6386;--line:#e1e4f2;--acc:#4f46e5;--ok:#0d9488;--bad:#e11d48}
*{box-sizing:border-box}html,body{margin:0;background:transparent;color:var(--fg);font:13px/1.45 Inter,system-ui,sans-serif}
.top{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:10px}
.stage{position:relative;height:640px;border-radius:16px;overflow:hidden;border:1px solid var(--line);background:#eef0f8}
canvas#c{width:100%;height:100%;display:block;cursor:grab;touch-action:none}
.views{position:absolute;left:8px;top:8px;display:flex;flex-wrap:wrap;gap:3px;background:rgba(255,255,255,.9);padding:4px;border-radius:10px;max-width:calc(100% - 150px)}
.views button{border:0;background:none;color:var(--muted);font:500 12px Inter,sans-serif;padding:4px 8px;border-radius:7px;cursor:pointer}
.views button.on{background:var(--acc);color:#fff}
.tools{position:absolute;left:8px;bottom:8px;display:flex;gap:4px}
.tools button{border:1px solid var(--line);background:rgba(255,255,255,.92);border-radius:8px;padding:4px 9px;cursor:pointer;font:500 12px Inter,sans-serif}
.tools button[aria-pressed=true]{background:#eef0ff;border-color:var(--acc);color:#3f37c9}
.hud{position:absolute;right:8px;top:8px;background:rgba(20,18,58,.88);color:#fff;border-radius:12px;padding:8px 12px;font:600 11px 'JetBrains Mono',monospace;text-align:right;min-width:120px}
.hud .sc{font-size:22px;line-height:1.1}.hud .tm{color:#a5b4fc}
.toast{position:absolute;left:50%;bottom:14px;transform:translateX(-50%);max-width:86%;padding:9px 14px;border-radius:12px;font-weight:500;box-shadow:0 10px 26px -10px rgba(0,0,0,.45);opacity:0;transition:opacity .25s;pointer-events:none;text-align:center}
.toast.show{opacity:1}.toast.info{background:#14123a;color:#fff}.toast.ok{background:#0d9488;color:#fff}.toast.err{background:#e11d48;color:#fff}.toast.warn{background:#f59e0b;color:#1f1300}
.side{display:flex;flex-direction:column;gap:8px;height:640px;overflow:auto;padding-right:2px}
.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:10px}
.card h4{margin:0 0 7px;font:600 11px 'JetBrains Mono',monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.chips{display:flex;flex-wrap:wrap;gap:4px}
.chip{font:500 11px 'JetBrains Mono',monospace;padding:3px 8px;border-radius:99px;background:#f1f2f8;color:#6b7194}
.chip.on{background:#dff7f1;color:#0b7a6e}.chip.mid{background:#fff4d6;color:#9a6a00}
.next{background:linear-gradient(135deg,#eef0ff,#f5efff);border-color:#d9d4ff}.next b{color:#3f37c9}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:6px}
button.a{font:500 12px Inter,sans-serif;border:1px solid var(--line);background:#fff;color:var(--fg);border-radius:9px;padding:7px 8px;cursor:pointer;text-align:left}
button.a:hover{border-color:var(--acc)}button.a:disabled{opacity:.45;cursor:not-allowed}
button.a.hl{border-color:#7c3aed;box-shadow:0 0 0 2px rgba(124,58,237,.35);background:#faf8ff}
select,input[type=number]{width:100%;font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:9px;background:#fff;color:var(--fg)}
.seg{display:flex;gap:4px;margin-bottom:6px}.seg label{flex:1;text-align:center;border:1px solid var(--line);border-radius:9px;padding:5px;cursor:pointer;font-size:12px}
.seg input{display:none}.seg label:has(input:checked){border-color:var(--acc);background:#eef0ff;color:#3f37c9;font-weight:600}
.bottom{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:10px;margin-top:10px}
.pcwrap{background:#1b1d22;border-radius:16px;padding:10px 10px 14px;box-shadow:inset 0 0 0 2px #2c2f36}
.pchead{color:#9aa1b5;font:500 11px 'JetBrains Mono',monospace;letter-spacing:.1em;margin:0 0 6px 4px;display:flex;justify-content:space-between}
canvas#pc{width:100%;display:block;border-radius:6px;background:#000;cursor:pointer}
textarea{width:100%;height:118px;font:12px 'JetBrains Mono',monospace;border:1px solid var(--line);border-radius:9px;padding:6px;resize:vertical}
.log{font:11.5px 'JetBrains Mono',monospace;color:#4a5072;max-height:190px;overflow:auto;white-space:pre-wrap}
.log .e{color:#be123c}.log .o{color:#0b7a6e}.log .w{color:#9a6a00}
.row{display:flex;gap:6px;align-items:center}.hint{color:var(--muted);font-size:12px}
@media (max-width:900px){.top,.bottom{grid-template-columns:1fr}.side{height:auto}.stage{height:440px}}
</style></head><body>
<div class="top">
  <div class="stage"><canvas id="c" aria-label="Meja praktikum 3D"></canvas>
    <div class="views">
      <button data-v="meja" class="on">Meja</button><button data-v="alat">Alat</button><button data-v="kompartemen">Kompartemen</button>
      <button data-v="kerja">Rak & tisu</button><button data-v="komputer">Komputer</button><button data-v="ups">UPS</button><button data-v="belakang">Kabel</button>
    </div>
    <div class="hud"><div>SKOR PROSEDUR</div><div class="sc" id="score">100</div><div class="tm" id="clock">00:00</div></div>
    <div class="tools"><button data-t="xray" aria-pressed="false">Tembus pandang</button><button data-t="autox" aria-pressed="true">Tembus saat mengukur</button><button data-t="sound" aria-pressed="true">Suara</button></div>
    <div class="toast" id="toast"></div>
  </div>
  <div class="side">
    <div class="card"><h4>Status</h4><div class="chips" id="chips"></div></div>
    <div class="card next"><h4>Langkah berikutnya</h4><div id="next"></div></div>
    <div class="card"><h4>A · Persiapan alat</h4><div class="grid2">
      <button class="a" data-act="cables">Sambungkan kabel</button><button class="a" data-act="ups">Nyalakan UPS</button>
      <button class="a" data-act="pc">Nyalakan komputer</button><button class="a" data-act="inst">Nyalakan UV-1700</button>
      <button class="a" data-act="skipwarm">Lewati pemanasan</button><button class="a" data-act="software">Buka software</button>
      <button class="a" data-act="connect">Hubungkan PC–alat</button><button class="a" data-act="shutdown">Matikan semua</button></div></div>
    <div class="card"><h4>B · Kuvet</h4>
      <div class="seg"><label><input type="radio" name="cv" value="R" checked><span>Kuvet referensi</span></label><label><input type="radio" name="cv" value="S"><span>Kuvet sampel</span></label></div>
      <select id="sol"></select>
      <div class="grid2" style="margin-top:6px">
      <button class="a" data-act="fill">Bilas & isi kuvet</button><button class="a" data-act="wipe">Lap dengan tisu</button>
      <button class="a" data-act="lid">Buka/tutup kompartemen</button><button class="a" data-act="insert">Masukkan ke alat</button>
      <button class="a" data-act="remove">Keluarkan kuvet</button><button class="a" data-act="empty">Buang isi kuvet</button></div></div>
    <div class="card"><h4>C · Pengukuran</h4>
      <div class="row"><input type="number" id="lam" min="190" max="1100" step="0.5"><button class="a" data-act="goto" style="white-space:nowrap">GOTO WL</button></div>
      <div class="grid2" style="margin-top:6px">
      <button class="a" data-act="autozero">Auto zero</button><button class="a" data-act="baseline">Koreksi baseline</button>
      <button class="a" data-act="measure">Ukur (fotometri)</button><button class="a" data-act="scan">Scan spektrum</button></div>
      <div class="hint" style="margin-top:6px" id="scaninfo"></div></div>
    <div class="card"><h4 id="ti">Info komponen</h4><div class="hint" id="info">Seret untuk memutar, roda/cubit untuk zoom, klik objek untuk penjelasan.</div></div>
  </div>
</div>
<div class="bottom">
  <div class="pcwrap"><div class="pchead"><span>LAYAR KOMPUTER · PERANGKAT LUNAK INSTRUMEN (SIMULASI)</span><span id="pcst">MATI</span></div><canvas id="pc" width="1040" height="600"></canvas></div>
  <div class="side" style="height:auto">
    <div class="card"><h4>Data untuk Kurva Kalibrasi</h4><textarea id="out" readonly></textarea>
      <div class="grid2" style="margin-top:6px"><button class="a" data-act="copy">Salin data</button><button class="a" data-act="csv">Unduh CSV</button></div>
      <div class="hint" style="margin-top:6px">Tempel teks ini di kotak "Impor hasil praktikum" di bawah aplikasi untuk mengirim ke Kurva Kalibrasi.</div></div>
    <div class="card"><h4>Catatan kerja</h4><div class="log" id="log"></div></div>
  </div>
</div>
<script>
(function () {
'use strict';
const INIT = Object.assign({ seed: 12345, hidden: 2.4, dec: 'koma', unit: 'mg/L',
  analyte: { nama: 'Fe(II)–1,10-fenantrolin', rumus: '[Fe(phen)3]2+', pelarut: 'Bufer asetat', lam_ref: 510, k: 0.2, warna: '#c0392b',
             bands: [[514, 1.0, 22], [478, .3, 15], [400, .08, 30], [265, 1.6, 18]], std: [0.5, 1, 2, 3, 4, 5] } }, /*__LAB__*/null || {});
const AN = INIT.analyte, D2R = Math.PI / 180, DEC = INIT.dec === 'koma';
const $ = s => document.querySelector(s), $$ = s => Array.from(document.querySelectorAll(s));
const fx = (x, d) => (x == null || !isFinite(x)) ? '—' : (DEC ? x.toFixed(d).replace('.', ',') : x.toFixed(d));
let seed = (INIT.seed >>> 0) || 1;
function rnd() { seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }
function gauss() { let u = 0, v = 0; while (!u) u = rnd(); while (!v) v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function absorptivity(l) { let s = 0; for (const [c, h, w] of AN.bands) s += h * Math.exp(-((l - c) ** 2) / (2 * w * w)); return AN.k * s; }
function lampE(l) { let e = l < 340.8 ? 0.15 + 0.85 * Math.exp(-((l - 240) ** 2) / 9800) : 0.08 + 0.92 * Math.exp(-((l - 850) ** 2) / 180000); if (l < 200) e *= Math.max(0.08, (l - 185) / 15); return Math.max(e, 0.02); }
function instOff(l) { return 0.014 + 1.6e-5 * (l - 500) + 0.006 * Math.sin(l / 37) + (l < 340.8 ? 0.0035 : 0); }
function hex(c) { c = c.replace('#', ''); if (c.length === 3) c = c.split('').map(x => x + x).join(''); const n = parseInt(c, 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
function shade(col, k, emi) { const [r, g, b] = hex(col); const f = v => Math.max(0, Math.min(255, Math.round(v * k + (emi || 0) * 255))); return 'rgb(' + f(r) + ',' + f(g) + ',' + f(b) + ')'; }
function mix(c1, c2, t) { const a = hex(c1), b = hex(c2); return '#' + a.map((v, i) => Math.round(v + (b[i] - v) * t).toString(16).padStart(2, '0')).join(''); }
function lamColor(l) {
  if (l < 380) return '#a99cf0'; if (l > 780) return '#b0302a';
  let r = 0, g = 0, b = 0;
  if (l < 440) { r = -(l - 440) / 60; b = 1; } else if (l < 490) { g = (l - 440) / 50; b = 1; } else if (l < 510) { g = 1; b = -(l - 510) / 20; }
  else if (l < 580) { r = (l - 510) / 70; g = 1; } else if (l < 645) { r = 1; g = -(l - 645) / 65; } else r = 1;
  const f = l < 420 ? 0.45 + 0.55 * (l - 380) / 40 : l > 700 ? 0.45 + 0.55 * (780 - l) / 80 : 1;
  const c = v => Math.round(255 * Math.pow(v * f, 0.8)).toString(16).padStart(2, '0'); return '#' + c(r) + c(g) + c(b);
}
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = a => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const lerp3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
function rotX(p, piv, ang) { const s = Math.sin(ang), c = Math.cos(ang); const y = p[1] - piv[1], z = p[2] - piv[2]; return [p[0], piv[1] + y * c - z * s, piv[2] + y * s + z * c]; }
function rotY(p, piv, ang) { const s = Math.sin(ang), c = Math.cos(ang); const x = p[0] - piv[0], z = p[2] - piv[2]; return [piv[0] + x * c + z * s, p[1], piv[2] - x * s + z * c]; }
function rotZ(p, piv, ang) { const s = Math.sin(ang), c = Math.cos(ang); const x = p[0] - piv[0], y = p[1] - piv[1]; return [piv[0] + x * c - y * s, piv[1] + x * s + y * c, p[2]]; }

/* ---------------- larutan ---------------- */
const SOL = { blank: { id: 'blank', label: 'Blanko (' + AN.pelarut + ')', c: 0, col: '#dff4f8' } };
AN.std.forEach((c, i) => { SOL['s' + i] = { id: 's' + i, label: 'Standar ' + (i + 1) + ' · ' + fx(c, c < 1 ? 2 : 1) + ' ' + INIT.unit, c, std: true }; });
SOL.X = { id: 'X', label: 'Sampel X (tak diketahui)', c: INIT.hidden, sample: true };
const cmax = Math.max(...AN.std, INIT.hidden);
Object.values(SOL).forEach(s => { if (s.c > 0) s.col = mix('#eef6f8', AN.warna, Math.min(1, 0.25 + 0.75 * s.c / cmax)); });
const SOL_KEYS = Object.keys(SOL);

/* ---------------- keadaan ---------------- */
const RACK = { R: [-36, 0.2, 4], S: [-36, 0.2, 10] }, CELL = { R: [-8.5, 6.6, 3], S: [-8.5, 6.6, 13] };
const ST = {
  cab: [0, 0, 0, 0, 0, 0], cabled: false, ups: false, pc: 0, inst: 'off', initLines: [], warmLeft: 0, warmSkipped: false, connected: false,
  lid: 0, lidT: 0, xray: 0, xrayT: 0, xrayUser: false, autoX: true, sound: true, lam: AN.lam_ref, lastA: 0, lcdMode: 'PHOTOMETRIC',
  cuv: { R: { where: 'rak', pos: RACK.R.slice(), content: null, level: 0, dirty: false, smudge: 0 }, S: { where: 'rak', pos: RACK.S.slice(), content: null, level: 0, dirty: false, smudge: 0 } },
  zero: null, readings: [], spectrum: null, pcTab: 'foto', score: 100, errs: new Set(), busy: false, tissue: null, beam: 0, t0: performance.now(), pcBoot: 0
};

/* ---------------- geometri ---------------- */
const P = [];
function quad(id, grp, v, col, o) { P.push({ id, grp, v, col, alpha: o.alpha == null ? 1 : o.alpha, emi: o.emi || 0, bias: o.bias || 0, tf: o.tf || null, dyn: o.dyn || null, text: o.text || null, ds: !!o.ds, bg: !!o.bg }); }
function box(id, grp, x0, x1, y0, y1, z0, z1, col, o) {
  o = o || {};
  const faces = [['f', [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]], ['b', [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]]],
    ['r', [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]]], ['l', [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]]],
    ['t', [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]]], ['u', [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]]];
  const cell = o.cell || 4.5;
  for (const [k, q] of faces) {
    if (o.skip && o.skip.includes(k)) continue;
    const fc = (o.faceCol && o.faceCol[k]) || col;
    const isT = o.text && k === o.text.face;
    const nu = isT ? 1 : Math.max(1, Math.ceil(Math.hypot(...sub(q[1], q[0])) / cell)), nv = isT ? 1 : Math.max(1, Math.ceil(Math.hypot(...sub(q[3], q[0])) / cell));
    for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
      const bl = (u, v) => lerp3(lerp3(q[0], q[1], u), lerp3(q[3], q[2], u), v);
      quad(id, grp, [bl(i / nu, j / nv), bl((i + 1) / nu, j / nv), bl((i + 1) / nu, (j + 1) / nv), bl(i / nu, (j + 1) / nv)], fc,
        Object.assign({}, o, { text: isT ? o.text : null, dyn: (o.dynFace && k !== o.dynFace) ? null : o.dyn }));
    }
  }
}
function cyl(id, grp, cx, cz, r, y0, y1, col, o, n) {
  o = Object.assign({ ds: true }, o || {}); n = n || 12;
  for (let i = 0; i < n; i++) {
    const a0 = i / n * 2 * Math.PI, a1 = (i + 1) / n * 2 * Math.PI;
    const p0 = [cx + r * Math.cos(a0), cz + r * Math.sin(a0)], p1 = [cx + r * Math.cos(a1), cz + r * Math.sin(a1)];
    quad(id, grp, [[p1[0], y0, p1[1]], [p0[0], y0, p0[1]], [p0[0], y1, p0[1]], [p1[0], y1, p1[1]]], col, o);
    if (!o.open) quad(id, grp, [[cx, y1, cz], [p1[0], y1, p1[1]], [p0[0], y1, p0[1]], [p0[0], y1, p0[1]]], o.topCol || col, o);
  }
}
const BEIGE = '#d8bf7f', BEIGE2 = '#e2cb8e', GREY = '#4a5058', DARK = '#1e2126', CREAM = '#efe9d6';
// --- ruang: dinding, meja ---
box('wall', 'room', -125, 85, -6, 75, -40, -38.5, '#f3f4f7', { bg: true, cell: 14, bias: 400 });
for (let x = -125; x <= 85; x += 15) quad('wall', 'room', [[x, -6, -38.4], [x + 0.25, -6, -38.4], [x + 0.25, 75, -38.4], [x, 75, -38.4]], '#dfe2ea', { bg: true, bias: 390 });
box('wall', 'room', -125, 85, 30, 75, -38.6, -38.4, '#f7f7f9', { bg: true, cell: 30, bias: 395, skip: ['b', 'u', 't', 'l', 'r'] });
box('bench', 'room', -125, 85, -8, -1.4, -38.5, 36, '#fbfbfc', { bg: true, cell: 15, bias: 300, faceCol: { f: '#e9ebf1' } });
for (let x = -120; x <= 85; x += 15) quad('bench', 'room', [[x, -1.38, -38.5], [x + 0.2, -1.38, -38.5], [x + 0.2, -1.38, 36], [x, -1.38, 36]], '#e3e6ee', { bg: true, bias: 290 });
for (let z = -35; z <= 36; z += 15) quad('bench', 'room', [[-125, -1.37, z], [85, -1.37, z], [85, -1.37, z + 0.2], [-125, -1.37, z + 0.2]], '#e3e6ee', { bg: true, bias: 290 });
// stop kontak dinding
box('outlet', 'room', 30, 44, 14, 22, -38.4, -37.6, '#f4f1e6', { bias: -2 });
box('outlet', 'room', 32.2, 35.8, 16.2, 19.8, -37.6, -37.4, '#d9d4c3', { bias: -3 }); box('outlet', 'room', 38.2, 41.8, 16.2, 19.8, -37.6, -37.4, '#d9d4c3', { bias: -3 });
// --- spektrofotometer (origin) ---
box('base', 'inst', -23, 23, 0, 3, -24.5, 24.5, GREY, { cell: 6 });
[[-19, 20], [19, 20], [-19, -20], [19, -20]].forEach(([x, z]) => box('base', 'inst', x - 1.6, x + 1.6, -1.4, 0, z - 1.6, z + 1.6, '#363b42', { skip: ['u'] }));
box('base', 'inst', 4, 7, 0.6, 2.2, -24.8, -24.5, '#c73a3a', { bias: -4, cell: 99 });
box('base', 'inst', 12, 18, 0.7, 2.0, -24.8, -24.5, '#2b2f35', { bias: -4, cell: 99 });
box('base', 'inst', 19, 21.5, 0.6, 2.4, -24.8, -24.5, '#15171a', { bias: -4, cell: 99 });
box('body', 'walls', -23, 23, 3, 13.5, 23.9, 24.5, BEIGE, { cell: 3 }); box('body', 'walls', -23, 23, 3, 13.5, -24.5, -23.9, BEIGE, { cell: 3 });
box('body', 'walls', -23, -22.4, 3, 13.5, -23.9, 23.9, BEIGE, { cell: 3 }); box('body', 'walls', 22.4, 23, 3, 13.5, -23.9, 23.9, BEIGE, { cell: 3 });
box('power', 'walls', -23.35, -23, 5, 8, 16, 19, '#2a2d33', { bias: -4, cell: 99, dyn: 'pwr' });
box('compartment', 'walls', -16, -1, 4, 12.5, 24.5, 24.8, '#cfb575', { bias: -4, skip: ['b'], cell: 99 });
box('plate', 'walls', 9, 20, 6.5, 10.5, 24.5, 24.7, '#f7f6f0', { bias: -4.5, skip: ['b'], cell: 99, text: { face: 'f', w: 110, h: 40, lines: [['UV-1700', 8, 18, 'left', '700 15px Arial', '#22263a'], ['PharmaSpec', 70, 18, 'left', 'italic 9px Arial', '#22263a'], ['UV-VIS SPECTROPHOTOMETER', 8, 30, 'left', '6.5px Arial', '#c0392b']] } });
for (let i = 0; i < 8; i++) box('vent', 'walls', 23, 23.25, 5.6 + i * 0.95, 6.1 + i * 0.95, -20, -11, '#bfa868', { bias: -4, cell: 99, dyn: i === 4 ? 'glow' : null });
box('card', 'walls', 23, 23.2, 11, 11.8, 1, 10, '#2b2f36', { bias: -4, cell: 99 });
box('card', 'walls', 23, 23.45, 10.9, 11.9, 11.2, 13.4, '#ebe6d6', { bias: -4.2, cell: 99 });
box('warn', 'walls', 23, 23.15, 4.6, 8.4, 4, 13, '#fafafa', { bias: -4.3, cell: 99 }); box('warn', 'walls', 23, 23.2, 7, 8.2, 7.5, 12.6, '#f6d21f', { bias: -4.5, cell: 99 });
box('pcb', 'inner', -21.8, -12, 3, 3.6, -22, -6, '#1d6b45');
box('lamphouse', 'inner', 14, 22.2, 3, 4, -23.4, -9, '#3a4048');
box('d2', 'inner', 18, 20.2, 4, 11.5, -21, -18.6, '#d9d4ff', { dyn: 'd2' }); box('wi', 'inner', 18, 20.2, 4, 10.5, -14.2, -11.8, '#fff1b8', { dyn: 'wi' });
box('mirror', 'inner', 15, 15.6, 7.5, 11.5, -18, -14, '#cfd8e6');
box('slit1', 'inner', 11, 11.4, 6, 13, -18.5, -16.4, DARK); box('slit1', 'inner', 11, 11.4, 6, 13, -15.6, -13.5, DARK);
box('mono', 'inner', -2, 10.5, 3, 4, -23.2, -9, '#2d323a'); box('mono', 'inner', -2, 10.5, 4, 12.5, -23.2, -22.7, '#3a4048', { alpha: 0.55 });
box('grating', 'inner', 1.5, 5.5, 5.5, 12.5, -21.8, -20.6, '#9fb3cf', { faceCol: { f: '#c9d6ea' } });
box('slit2', 'inner', 1.5, 2.6, 6, 13, -9.4, -9, DARK); box('slit2', 'inner', 3.4, 4.5, 6, 13, -9.4, -9, DARK);
box('split', 'inner', 2.4, 3.6, 6.5, 12, 1.4, 4.6, '#7fe6f7', { alpha: 0.7, tf: [['y', [3, 0, 3], 45]] });
box('mirror2', 'inner', 2.4, 3.6, 6.5, 12, 11.4, 14.6, '#cfd8e6', { tf: [['y', [3, 0, 13], -45]] });
box('compartment', 'comp', -15, -2, 3, 6, -2, 20, '#16181c', { cell: 4 });
box('compartment', 'comp', -15, -14.5, 6, 13.5, -2, 20, '#1d2025', { alpha: 0.9 }); box('compartment', 'comp', -2.5, -2, 6, 13.5, -2, 20, '#1d2025', { alpha: 0.9 });
box('compartment', 'comp', -14.5, -2.5, 6, 13.5, -2, -1.5, '#1d2025'); box('compartment', 'comp', -14.5, -2.5, 6, 13.5, 19.5, 20, '#1d2025');
[[-12.5, 1], [-12.5, 16.5]].forEach(([x, z]) => box('compartment', 'comp', x - 1, x + 1, 6, 6.3, z - 1, z + 1, '#0f1013'));
[['holderR', 3], ['holderS', 13]].forEach(([id, z]) => {
  box(id, 'comp', -10.6, -6.4, 6, 12.5, z - 2.1, z - 1.6, DARK); box(id, 'comp', -10.6, -6.4, 6, 12.5, z + 1.6, z + 2.1, DARK);
  box(id, 'comp', -10.6, -10.1, 6, 12.5, z - 1.6, z + 1.6, DARK, { alpha: 0.9 }); box(id, 'comp', -11.5, -5.5, 6, 6.6, z - 2.6, z + 2.6, '#101215');
});
box('detR', 'inner', -21.6, -19.6, 8, 11.5, 1.6, 4.4, '#2a3346'); box('detS', 'inner', -21.6, -19.6, 8, 11.5, 11.6, 14.4, '#2a3346');
box('cover', 'cover', -23, -15, 13.5, 15, -24.5, 24.5, BEIGE2); box('cover', 'cover', -2, 23, 13.5, 15, -24.5, 24.5, BEIGE2);
box('cover', 'cover', -15, -2, 13.5, 15, -24.5, -2, BEIGE2); box('cover', 'cover', -15, -2, 13.5, 15, 20, 24.5, BEIGE2);
for (let i = 0; i < 14; i++) box('vent', 'cover', 6 + i * 1.15, 6.55 + i * 1.15, 15, 15.25, -23.5, -15.5, '#c6ae6c');
box('lid', 'lid', -15.2, -1.8, 15, 16.6, -2.2, 20.2, '#dcc589', { cell: 4, faceCol: { u: '#111215' } });
box('keypad', 'cover', 2, 21, 15, 16.2, 13, 23.5, CREAM, { bias: -0.6 });
for (let r = 0; r < 4; r++) for (let c = 0; c < 6; c++) box('keypad', 'cover', 2.8 + c * 3.0, 4.7 + c * 3.0, 16.2, 16.55, 14.2 + r * 2.2, 15.5 + r * 2.2, c === 5 ? '#7c93c9' : '#cfd4dc', { bias: -0.8 });
box('lcd', 'cover', 9, 14, 15, 17.5, 7.5, 10, '#ddd6bd', { bias: -0.6 });
const tilt = [['x', [11.5, 16.5, 9], -18]];
box('lcd', 'cover', 3.5, 19.5, 16.5, 28.5, 8.2, 10.2, '#ebe5cf', { tf: tilt, bias: -0.7, cell: 99 });
box('lcd', 'cover', 6, 17, 18.3, 26.6, 10.2, 10.35, '#a99af2', { tf: tilt, bias: -4, dyn: 'lcd', dynFace: 'f', text: { face: 'f', w: 130, h: 96, lcd: true } });
// --- UPS ---
box('ups', 'room', 30, 48, -1.4, 22, -20, 8, '#2c2f35', { cell: 6, faceCol: { f: '#3b3f47' } });
box('ups', 'room', 33, 45, 14, 19, 8, 8.2, '#9fb5a8', { bias: -3, dyn: 'upslcd', text: { face: 'f', w: 120, h: 50, ups: true } });
box('ups', 'room', 37.5, 40.5, 8, 11, 8, 8.4, '#1a1c20', { bias: -3 });
box('ups', 'room', 38.4, 39.6, 11.6, 12.4, 8, 8.3, '#555', { bias: -3.2, dyn: 'upsled' });
box('ups', 'room', 33, 45, 1, 5, 8, 8.15, '#45484f', { bias: -3, text: { face: 'f', w: 120, h: 40, lines: [['UPS 1200 VA', 60, 26, 'center', '600 13px Arial', '#e6e6e6']] } });
[[33, 9], [40, 9], [33, 2], [40, 2]].forEach(([x, y]) => box('ups', 'room', x, x + 5, y, y + 5, -20.3, -20, '#1b1d21', { bias: -3 }));
for (let i = 0; i < 6; i++) box('ups', 'room', 48, 48.2, 6 + i * 2, 7 + i * 2, -14, 2, '#22252a', { bias: -3 });
// --- komputer: CPU, monitor, keyboard, mouse ---
box('cpu', 'room', -106, -88, -1.4, 40, -22, 14, '#1f2227', { cell: 7, faceCol: { f: '#2a2e35' } });
box('cpu', 'room', -103, -91, 30, 32, 14, 14.3, '#121417', { bias: -3 });
box('cpu', 'room', -98.5, -95.5, 34.5, 37.5, 14, 14.4, '#3a3f47', { bias: -3, dyn: 'pcled' });
for (let i = 0; i < 6; i++) box('cpu', 'room', -103, -91, 6 + i * 2.2, 6.8 + i * 2.2, 14, 14.2, '#15171a', { bias: -3 });
box('monitor', 'room', -66, -50, -1.4, 0, -21, -11, '#202328'); box('monitor', 'room', -60, -56, 0, 13, -17.5, -15.5, '#2a2e34');
box('monitor', 'room', -80, -36, 12, 40, -17, -14.4, '#17191d', { cell: 8 });
box('screen', 'room', -78.3, -37.7, 13.6, 38.4, -14.4, -14.2, '#000000', { bias: -3, dyn: 'screen', dynFace: 'f', text: { face: 'f', w: 1040, h: 600, screen: true } });
box('keyboard', 'room', -76, -44, -1.4, -0.3, 2, 12, '#e8e9ee', { cell: 6 });
for (let r = 0; r < 4; r++) for (let c = 0; c < 14; c++) box('keyboard', 'room', -75 + c * 2.15, -73.4 + c * 2.15, -0.3, 0.2, 3 + r * 2.3, 4.7 + r * 2.3, '#cfd2da', { bias: -0.5 });
box('mouse', 'room', -40, -35.5, -1.4, 0.3, 4, 10, '#e8e9ee');
// --- meja kerja: rak kuvet, kotak tisu, botol semprot, gelas buangan, labu standar ---
box('rack', 'room', -42, -30, -1.4, 1.6, 0, 14, '#3d5a80', { cell: 4 });
box('rack', 'room', -37.6, -34.4, 1.6, 1.7, 2.4, 5.6, '#11161f', { bias: -0.5 }); box('rack', 'room', -37.6, -34.4, 1.6, 1.7, 8.4, 11.6, '#11161f', { bias: -0.5 });
box('tissue', 'room', -42, -28, -1.4, 5, 17, 29, '#2f6f6a', { cell: 5, faceCol: { t: '#3a7f79' }, text: { face: 'f', w: 140, h: 64, lines: [['TISSUE', 70, 30, 'center', '700 20px Arial', '#e8f5f3'], ['lens & cuvette', 70, 48, 'center', 'italic 11px Arial', '#cde9e5']] } });
box('tissue', 'room', -37, -33, 5, 5.1, 21, 25, '#16302e', { bias: -0.4 });
box('tissue', 'room', -36.5, -33.5, 5, 10.5, 22.6, 22.9, '#ffffff', { tf: [['x', [-35, 5, 22.75], 12]], ds: true });
cyl('bottle', 'room', -50, 25, 2.6, -1.4, 9, '#f2f6f8', { alpha: 0.9 }); cyl('bottle', 'room', -50, 25, 0.9, 9, 11, '#e94b4b');
box('bottle', 'room', -50.3, -49.7, 10.5, 11.1, 25, 29, '#e94b4b');
cyl('beaker', 'room', -58, 26, 3.2, -1.4, 6.5, '#dfeff5', { alpha: 0.45, open: true }); cyl('beaker', 'room', -58, 26, 3.0, -1.3, 1.6, '#c7dce6', { alpha: 0.6 });
const FL = SOL_KEYS.map((k, i) => ({ k, x: 52 + (i % 4) * 6.5, z: 16 + Math.floor(i / 4) * 7 }));
FL.forEach(f => { cyl('flask', 'room', f.x, f.z, 2.4, -1.4, 3.5, '#eef6f8', { alpha: 0.5 }); cyl('flask', 'room', f.x, f.z, 2.1, -1.3, 2.6, SOL[f.k].col, { alpha: 0.85 });
  cyl('flask', 'room', f.x, f.z, 0.6, 3.5, 11, '#eef6f8', { alpha: 0.55 }); cyl('flask', 'room', f.x, f.z, 0.75, 11, 12.2, '#3a6fd8'); });
// --- kuvet (geometri lokal; dipindah melalui posisi) ---
['R', 'S'].forEach(k => {
  box('cuv' + k, 'cuv' + k, -1.3, 1.3, 0, 7.6, -1.3, 1.3, '#e3f4f8', { alpha: 0.38, ds: true, cell: 99, faceCol: { l: '#f4f6f6', r: '#f4f6f6' } });
  box('cuv' + k, 'liq' + k, -1.1, 1.1, 0.3, 6.2, -1.1, 1.1, '#ffffff', { alpha: 0.7, cell: 99, dyn: 'liq' + k });
  box('cuv' + k, 'cuv' + k, -1.32, 1.32, 7.6, 7.7, -1.32, 1.32, '#d5dde0', { alpha: 0.5, cell: 99 });
});
box('tw', 'tw', -1.4, 1.4, -1.8, 1.8, -0.1, 0.1, '#ffffff', { ds: true, cell: 99 });

/* ---------------- kabel ---------------- */
const Y0 = -1.1;
const CABLES = [
  { nm: 'UPS → stop kontak', col: '#1a1a1a', w: 1.0, pts: [[36, 6, -20.2], [36, 4, -27], [36, Y0, -30], [36, Y0, -36], [34, 4, -37.8], [34, 18, -37.6]] },
  { nm: 'UV-1700 → UPS', col: '#111', w: 1.0, pts: [[20, 1.5, -24.8], [20, Y0, -28], [28, Y0, -28], [35, Y0, -24], [35.5, 4.5, -20.3], [35.5, 11.5, -20.3]] },
  { nm: 'CPU → UPS', col: '#111', w: 1.0, pts: [[-97, 8, -22.2], [-97, Y0, -27], [-60, Y0, -33], [10, Y0, -33], [40, Y0, -30], [42.5, 4.5, -20.3], [42.5, 11.5, -20.3]] },
  { nm: 'Monitor → UPS', col: '#111', w: 0.9, pts: [[-58, 20, -17.1], [-58, Y0, -19], [-40, Y0, -31], [20, Y0, -31.5], [38, Y0, -29], [42.5, 2, -20.3], [42.5, 4.5, -20.3]] },
  { nm: 'Monitor → CPU (video)', col: '#2b4fae', w: 0.8, pts: [[-62, 22, -17.1], [-62, Y0, -20], [-80, Y0, -24], [-92, 14, -22.2]] },
  { nm: 'UV-1700 → CPU (RS-232C)', col: '#9aa0a8', w: 0.8, pts: [[5.5, 1.4, -24.8], [5.5, Y0, -27], [-30, Y0, -29], [-80, Y0, -27], [-94, 18, -22.2]] }
];
CABLES.forEach(c => { const p = []; for (let i = 0; i < c.pts.length - 1; i++) for (let k = 0; k < 6; k++) p.push(lerp3(c.pts[i], c.pts[i + 1], k / 6)); p.push(c.pts[c.pts.length - 1]); c.seg = p; });

const INFO = {
  wall: ['Dinding laboratorium', 'Stop kontak dinding menjadi satu-satunya sumber listrik; semua beban dilewatkan melalui UPS.'],
  bench: ['Meja kerja', 'Meja keramik putih bebas getaran; jauhkan sumber panas dan sinar matahari langsung dari instrumen.'],
  outlet: ['Stop kontak', 'Sumber listrik 220–230 V AC dengan arde. Pastikan terhubung ke arde agar tidak ada tegangan bocor pada rangka alat.'],
  ups: ['UPS 1200 VA', 'Uninterruptible Power Supply: menstabilkan tegangan dan menyediakan daya cadangan saat listrik padam sehingga pengukuran dan data tidak hilang. Nyalakan sebelum alat dan komputer.'],
  cpu: ['CPU (unit komputer)', 'Menjalankan perangkat lunak instrumen; terhubung ke UV-1700 melalui kabel RS-232C dan menerima data pengukuran.'],
  monitor: ['Monitor', 'Menampilkan perangkat lunak instrumen: tabel fotometri, kurva kalibrasi, dan spektrum.'], screen: ['Layar monitor', 'Klik tombol pada layar komputer di bawah untuk mengoperasikan perangkat lunak.'],
  keyboard: ['Papan ketik', 'Input identitas sampel dan parameter metode.'], mouse: ['Mouse', 'Navigasi perangkat lunak.'],
  rack: ['Rak kuvet', 'Tempat kuvet referensi (belakang) dan kuvet sampel (depan) saat diisi dan dilap.'],
  tissue: ['Kotak tisu lensa', 'Tisu bebas serat untuk mengelap dinding bening kuvet dari tetesan dan sidik jari sebelum dimasukkan ke alat.'],
  bottle: ['Botol semprot akuades', 'Membilas kuvet sebelum diisi larutan berikutnya.'], beaker: ['Gelas buangan', 'Menampung bilasan dan larutan bekas.'],
  flask: ['Labu takar standar', 'Larutan blanko, deret standar (warna makin pekat seiring konsentrasi), dan Sampel X.'],
  cuvR: ['Kuvet referensi', 'Kuvet kuarsa 1 cm untuk blanko; ditempatkan di sel referensi (belakang).'], cuvS: ['Kuvet sampel', 'Kuvet kuarsa 1 cm untuk standar atau sampel; ditempatkan di sel sampel (depan).'],
  lid: ['Tutup kompartemen', 'Buka untuk memasukkan kuvet; tutup rapat sebelum auto zero dan pengukuran.'], compartment: ['Kompartemen sampel', 'Ruang gelap berisi sel referensi dan sel sampel.'],
  holderR: ['Sel referensi', 'Dudukan kuvet blanko (I₀).'], holderS: ['Sel sampel', 'Dudukan kuvet sampel (I).'],
  power: ['Saklar daya UV-1700', 'Saklar di sisi kiri instrumen. Nyalakan setelah UPS aktif.'],
  lcd: ['Layar LCD UV-1700', 'Menampilkan λ, absorbansi, lampu aktif, dan status.'], keypad: ['Papan tombol alat', 'Operasi mandiri tanpa komputer.'],
  plate: ['Pelat identitas', 'UV-1700 PharmaSpec · UV-VIS SPECTROPHOTOMETER.'], body: ['Rumah instrumen', 'Rangka UV-1700.'], cover: ['Penutup atas', 'Menutupi bangku optik.'],
  base: ['Rangka dasar & port belakang', 'Port RS-232C (merah), Centronics, dan soket daya.'], vent: ['Kisi ventilasi', 'Membuang panas lampu; berpendar oranye saat lampu WI menyala.'],
  card: ['Slot IC card · CONTRAST', 'Kartu memori metode dan pengatur kontras LCD.'], warn: ['Label HOT SURFACE', 'Dinginkan lampu sebelum diganti.'],
  d2: ['Lampu D₂', 'Sumber UV.'], wi: ['Lampu WI', 'Sumber tampak–NIR.'], mirror: ['Cermin sumber', ''], slit1: ['Celah masuk', ''], mono: ['Monokromator', ''], grating: ['Kisi difraksi', ''],
  slit2: ['Celah keluar', ''], split: ['Pembagi berkas', ''], mirror2: ['Cermin pengarah', ''], detR: ['Detektor referensi', ''], detS: ['Detektor sampel', ''], pcb: ['Papan elektronik', ''], lamphouse: ['Rumah lampu', ''], tw: ['Tisu', 'Sedang mengelap kuvet.']
};

/* ---------------- kamera ---------------- */
const cv = $('#c'), ctx = cv.getContext('2d'), pcc = $('#pc'), pctx = pcc.getContext('2d');
const VIEWS = { meja: [-18, 24, 250, [-12, 6, 0]], alat: [-30, 24, 100, [0, 9, 0]], kompartemen: [-12, 58, 62, [-8, 8, 8]], kerja: [-28, 38, 80, [-40, 3, 15]],
  komputer: [-4, 8, 92, [-62, 22, -8]], ups: [-38, 18, 95, [40, 10, -4]], belakang: [205, 32, 200, [-25, 4, -25]] };
const CAM = { yaw: -18, pitch: 24, dist: 250, target: [-12, 6, 0] };
let camAnim = null, viewName = 'meja';
function setView(name) { const v = VIEWS[name]; if (!v) return; viewName = name; camAnim = { from: [CAM.yaw, CAM.pitch, CAM.dist, CAM.target.slice()], to: v, t: 0 };
  $$('.views button').forEach(b => b.classList.toggle('on', b.dataset.v === name)); dirty = true; }
function camXf(p) {
  const t = CAM.target; const x = p[0] - t[0], y = p[1] - t[1], z = p[2] - t[2];
  const cy = Math.cos(-CAM.yaw * D2R), sy = Math.sin(-CAM.yaw * D2R); const x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
  const cp = Math.cos(CAM.pitch * D2R), sp = Math.sin(CAM.pitch * D2R); return [x1, y * cp - z1 * sp, y * sp + z1 * cp];
}
let W = 800, H = 600, CX = 400, CY = 300, FOC = 800;
function proj(c) { const zc = CAM.dist - c[2]; return [CX + FOC * c[0] / zc, CY - FOC * c[1] / zc, zc]; }

/* ---------------- transformasi objek ---------------- */
function xf(p, part) {
  let q = p;
  if (part.tf) for (const [ax, piv, deg] of part.tf) q = ax === 'x' ? rotX(q, piv, deg * D2R) : ax === 'y' ? rotY(q, piv, deg * D2R) : rotZ(q, piv, deg * D2R);
  const g = part.grp;
  if (g === 'lid') q = rotX(q, [0, 15, -2.2], -102 * ST.lid * D2R);
  else if (g === 'cuvR' || g === 'cuvS' || g === 'liqR' || g === 'liqS') {
    const k = g[3]; const c = ST.cuv[k];
    if (g[0] === 'l') q = [q[0] * 0.98, 0.3 + (q[1] - 0.3) * Math.max(0.001, c.level), q[2] * 0.98];
    q = [q[0] + c.pos[0], q[1] + c.pos[1], q[2] + c.pos[2]];
  } else if (g === 'tw') {
    const t = ST.tissue; if (t) { q = rotY(q, [0, 0, 0], t.rot * D2R); q = [q[0] + t.pos[0], q[1] + t.pos[1], q[2] + t.pos[2]]; }
  }
  return q;
}
function visible(part) {
  const g = part.grp; const closed = ST.xray < 0.04;
  if (g === 'tw') return !!ST.tissue;
  if (g[0] === 'l' && g.startsWith('liq')) { const c = ST.cuv[g[3]]; if (!c.content || c.level < 0.02) return false; }
  if (g.startsWith('cuv') || g.startsWith('liq')) { const c = ST.cuv[g[3]]; if (c.where === 'sel' && closed && ST.lid < 0.05) return false; return true; }
  if (g === 'inner' && closed) return false;
  if (g === 'comp' && closed && ST.lid < 0.05) return false;
  return true;
}

/* ---------------- gambar 3D ---------------- */
const LIGHT = norm([0.45, 0.85, 0.55]);
let dirty = true, drawn = [];
function draw() {
  const r = cv.getBoundingClientRect(); W = Math.max(300, r.width || 800); H = Math.max(240, r.height || 600);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); CX = W / 2; CY = H / 2 + 10; FOC = Math.min(W, H * 1.45);
  const g = ctx.createLinearGradient(0, 0, 0, H); g.addColorStop(0, '#f6f7fb'); g.addColorStop(1, '#e4e7f1'); ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  const lampUV = ST.lam < 340.8, on = ST.inst !== 'off';
  const coverA = 1 - 0.82 * ST.xray, wallA = 1 - 0.75 * ST.xray;
  const bg = [], items = [];
  for (const part of P) {
    if (!visible(part)) continue;
    let alpha = part.alpha;
    if (part.grp === 'cover' || part.grp === 'lid') alpha *= coverA;
    if (part.grp === 'walls') alpha *= wallA;
    if (alpha < 0.02) continue;
    const wv = part.v.map(p => xf(p, part)); const cs = wv.map(camXf);
    if (cs.some(c => CAM.dist - c[2] < 4)) continue;
    const n = norm(cross(sub(cs[1], cs[0]), sub(cs[3], cs[0])));
    const cen = [(cs[0][0] + cs[2][0]) / 2, (cs[0][1] + cs[2][1]) / 2, (cs[0][2] + cs[2][2]) / 2];
    const facing = dot(n, sub([0, 0, CAM.dist], cen)) > 0;
    if (!facing && alpha > 0.97 && !part.ds) continue;
    const wn = norm(cross(sub(wv[1], wv[0]), sub(wv[3], wv[0])));
    const lk = 0.5 + 0.55 * (part.ds ? Math.abs(dot(wn, LIGHT)) : Math.max(0, dot(wn, LIGHT)));
    let col = part.col, emi = part.emi;
    switch (part.dyn) {
      case 'd2': emi = on && lampUV ? 0.45 : 0; col = on && lampUV ? '#e4dcff' : '#9a96b5'; break;
      case 'wi': emi = on && !lampUV ? 0.45 : 0; col = on && !lampUV ? '#ffe7a0' : '#a89c74'; break;
      case 'glow': col = on && !lampUV ? '#f0873a' : '#bfa868'; emi = on && !lampUV ? 0.22 : 0; break;
      case 'lcd': col = on ? '#a99af2' : '#5c5777'; emi = on ? 0.08 : 0; break;
      case 'upsled': col = ST.ups ? '#22c55e' : '#555'; emi = ST.ups ? 0.5 : 0; break;
      case 'upslcd': col = ST.ups ? '#9be3c0' : '#3c4a44'; emi = ST.ups ? 0.1 : 0; break;
      case 'pcled': col = ST.pc ? '#38bdf8' : '#3a3f47'; emi = ST.pc ? 0.5 : 0; break;
      case 'pwr': col = on ? '#3d9a5a' : '#2a2d33'; break;
      case 'liqR': case 'liqS': { const c = ST.cuv[part.dyn[3]]; col = c.content ? SOL[c.content].col : '#fff'; break; }
    }
    if (selId && part.id === selId) emi += 0.15;
    const it = { part, pts: cs.map(proj), depth: CAM.dist - cen[2] + part.bias + (part.grp === 'inner' || part.grp === 'comp' ? 0.35 : 0) + (facing ? 0 : 0.02), fill: shade(col, lk, emi), alpha, facing };
    (part.bg ? bg : items).push(it);
  }
  bg.sort((a, b) => b.depth - a.depth); paint(bg);
  // bayangan objek pada meja
  ctx.fillStyle = 'rgba(20,18,58,.10)';
  [[-25, 25, -26.5, 26.5], [29, 49, -21, 9], [-107, -87, -23, 15], [-82, -34, -22, -10], [-43, -27, -1, 30]].forEach(([x0, x1, z0, z1]) => {
    const pts = [[x0 + 1, z0 + 1], [x1 + 1.5, z0 + 1], [x1 + 1.5, z1 + 1.5], [x0 + 1, z1 + 1.5]].map(([x, z]) => proj(camXf([x, -1.36, z])));
    if (pts.some(p => p[2] < 4)) return; ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.closePath(); ctx.fill();
  });
  // kabel
  CABLES.forEach((c, ci) => {
    const prog = ST.cab[ci]; if (prog <= 0) return;
    const nseg = c.seg.length - 1, upto = prog * nseg;
    for (let i = 0; i < nseg && i < upto; i++) {
      const tEnd = Math.min(1, upto - i); const a = c.seg[i], b = lerp3(c.seg[i], c.seg[i + 1], tEnd);
      const ca = camXf(a), cb = camXf(b); if (CAM.dist - ca[2] < 4 || CAM.dist - cb[2] < 4) continue;
      items.push({ cable: true, pa: proj(ca), pb: proj(cb), depth: CAM.dist - (ca[2] + cb[2]) / 2 - 0.2, col: c.col, w: c.w, end: i === nseg - 1 && prog >= 1 });
    }
  });
  // berkas cahaya
  if (on && (ST.beam > 0.01 || ST.xray > 0.5)) {
    const y = 9.5, src = lampUV ? [19.1, y, -19.8] : [19.1, y, -13];
    const path = [[src, [15.3, y, -16]], [[15.3, y, -16], [11.2, y, -15]], [[11.2, y, -15], [3.5, y, -21]], [[3.5, y, -21], [3, y, -9.2]], [[3, y, -9.2], [3, y, 3]], [[3, y, 3], [3, y, 13]], [[3, y, 3], [-8.5, y, 3]], [[-8.5, y, 3], [-19.6, y, 3]], [[3, y, 13], [-8.5, y, 13]], [[-8.5, y, 13], [-19.6, y, 13]]];
    const att = Math.max(0.12, Math.pow(10, -Math.max(0, Math.min(3, ST.lastA || 0))));
    path.forEach(([a, b], si) => { for (let k = 0; k < 4; k++) { const A = lerp3(a, b, k / 4), B = lerp3(a, b, (k + 1) / 4); const ca = camXf(A), cb = camXf(B);
      items.push({ beam: true, pa: proj(ca), pb: proj(cb), depth: CAM.dist - (ca[2] + cb[2]) / 2 - 0.3, col: si <= 2 ? '#fffbe8' : lamColor(ST.lam), att: si === 9 ? att : 1, a: Math.max(ST.beam, 0.5) }); } });
  }
  items.sort((a, b) => b.depth - a.depth); drawn = []; paint(items);
}
function paint(list) {
  ctx.lineJoin = 'round';
  for (const it of list) {
    if (it.cable) {
      ctx.save(); const wpx = Math.max(1.2, it.w * FOC / Math.max(10, it.pa[2]) * 0.9);
      ctx.strokeStyle = it.col; ctx.lineWidth = wpx; ctx.lineCap = 'round'; ctx.beginPath(); ctx.moveTo(it.pa[0], it.pa[1]); ctx.lineTo(it.pb[0], it.pb[1]); ctx.stroke();
      if (it.end) { ctx.fillStyle = '#2b2b2b'; ctx.beginPath(); ctx.arc(it.pb[0], it.pb[1], wpx * 1.3, 0, 7); ctx.fill(); }
      ctx.restore(); continue;
    }
    if (it.beam) {
      ctx.save(); ctx.globalAlpha = 0.3 * it.att * it.a; ctx.strokeStyle = it.col; ctx.lineWidth = 7; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(it.pa[0], it.pa[1]); ctx.lineTo(it.pb[0], it.pb[1]); ctx.stroke();
      ctx.globalAlpha = 0.95 * it.att * it.a; ctx.lineWidth = 2.2; if (ST.lam < 380) ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(it.pa[0], it.pa[1]); ctx.lineTo(it.pb[0], it.pb[1]); ctx.stroke(); ctx.restore(); continue;
    }
    const p = it.pts; ctx.globalAlpha = it.alpha;
    ctx.beginPath(); ctx.moveTo(p[0][0], p[0][1]); for (let i = 1; i < 4; i++) ctx.lineTo(p[i][0], p[i][1]); ctx.closePath();
    ctx.fillStyle = it.fill; ctx.fill();
    if (it.alpha > 0.97) { ctx.strokeStyle = it.fill; ctx.lineWidth = 1; ctx.stroke(); }
    if (selId && it.part.id === selId) { ctx.strokeStyle = '#7c3aed'; ctx.lineWidth = 1.2; ctx.stroke(); }
    ctx.globalAlpha = 1;
    if (it.part.text && it.facing) faceText(it.part, p);
    if (it.facing && it.alpha > 0.3 && !it.part.bg) drawn.push({ id: it.part.id, p });
  }
}
function faceText(part, p) {
  const t = part.text, TL = p[3], TR = p[2], BL = p[0];
  const dpr = cv.width / Math.max(1, W);
  ctx.save();
  ctx.setTransform((TR[0] - TL[0]) / t.w * dpr, (TR[1] - TL[1]) / t.w * dpr, (BL[0] - TL[0]) / t.h * dpr, (BL[1] - TL[1]) / t.h * dpr, TL[0] * dpr, TL[1] * dpr);
  if (t.screen) { ctx.drawImage(pcc, 0, 0, t.w, t.h); }
  else if (t.lcd) drawLCD(ctx);
  else if (t.ups) { if (ST.ups) { ctx.fillStyle = '#0b3a26'; ctx.font = '600 15px "JetBrains Mono",monospace'; ctx.textAlign = 'center'; ctx.fillText('IN 221V  OUT 220V', 60, 22); ctx.fillText('LOAD ' + (12 + (ST.inst !== 'off' ? 18 : 0) + (ST.pc ? 22 : 0)) + '%', 60, 40); } }
  else for (const [s, x, y, al, font, col] of t.lines) { ctx.font = font; ctx.fillStyle = col; ctx.textAlign = al; ctx.fillText(s, x, y); }
  ctx.restore();
}
function drawLCD(c) {
  if (ST.inst === 'off') return;
  c.fillStyle = '#1b1248';
  if (ST.inst === 'init') {
    c.font = '500 8.5px "JetBrains Mono",monospace'; c.textAlign = 'left';
    ST.initLines.slice(-9).forEach((s, i) => c.fillText(s, 6, 11 + i * 9.6)); return;
  }
  c.fillStyle = 'rgba(255,255,255,.85)'; [16, 44, 72].forEach(yy => c.fillRect(8, yy - 6, 24, 5));
  c.fillStyle = '#1b1248'; c.font = '500 17px "JetBrains Mono",monospace'; c.textAlign = 'right';
  c.fillText(ST.lam.toFixed(1) + 'nm', 124, 22);
  c.fillText((ST.lastA > 3 ? 'OVER' : ST.lastA.toFixed(3)) + ' ABS', 124, 50);
  c.font = '500 13px "JetBrains Mono",monospace'; c.textAlign = 'left'; c.fillText(ST.lam < 340.8 ? 'D2' : 'WI', 40, 77);
  c.textAlign = 'right'; c.fillStyle = '#4a3fa0'; c.font = '500 9.5px "JetBrains Mono",monospace';
  c.fillText(ST.inst === 'warm' ? 'WARM-UP ' + mmss(ST.warmLeft) : (ST.connected ? 'PC CONTROL' : 'READY'), 124, 90);
}
const mmss = s => String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(Math.floor(s % 60)).padStart(2, '0');

/* ---------------- layar komputer ---------------- */
let pcBtns = [];
function pcButton(c, x, y, w, h, label, act, on) {
  c.fillStyle = on === false ? '#e5e7eb' : '#f3f4f6'; c.strokeStyle = '#9ca3af'; c.lineWidth = 1; c.fillRect(x, y, w, h); c.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  c.fillStyle = on === false ? '#9ca3af' : '#111827'; c.font = '500 13px Inter,Arial'; c.textAlign = 'center'; c.fillText(label, x + w / 2, y + h / 2 + 4.5);
  if (act) pcBtns.push({ x, y, w, h, act });
}
function linreg(x, y) {
  const n = x.length; if (n < 2) return null; const mx = x.reduce((a, b) => a + b) / n, my = y.reduce((a, b) => a + b) / n;
  let sxx = 0, sxy = 0, syy = 0; for (let i = 0; i < n; i++) { sxx += (x[i] - mx) ** 2; sxy += (x[i] - mx) * (y[i] - my); syy += (y[i] - my) ** 2; }
  if (!(sxx > 0)) return null; const a = sxy / sxx, b = my - a * mx; return { a, b, r2: syy > 0 ? (sxy * sxy) / (sxx * syy) : 1 };
}
function calData() {
  const last = [...ST.readings].reverse().find(r => r.std); if (!last) return null;
  const lam = last.lam; const std = {}; ST.readings.filter(r => r.std && Math.abs(r.lam - lam) < 0.05).forEach(r => { (std[r.c] = std[r.c] || []).push(r.A); });
  const xs = Object.keys(std).map(Number).sort((a, b) => a - b); const ys = xs.map(c => std[c].reduce((a, b) => a + b) / std[c].length);
  const reg = linreg(xs, ys); const smp = ST.readings.filter(r => r.sample && Math.abs(r.lam - lam) < 0.05).map(r => r.A);
  return { lam, xs, ys, reg, smp };
}
function drawPC() {
  const c = pctx, Wp = pcc.width, Hp = pcc.height; pcBtns = []; c.setTransform(1, 0, 0, 1, 0, 0);
  $('#pcst').textContent = !ST.pc ? 'MATI' : ST.pc === 1 ? 'MEMULAI' : ST.connected ? 'TERHUBUNG KE UV-1700' : 'SIAP';
  if (!ST.pc) { c.fillStyle = '#030405'; c.fillRect(0, 0, Wp, Hp); c.fillStyle = '#1f2937'; c.font = '500 16px Inter,Arial'; c.textAlign = 'center'; c.fillText(ST.ups ? 'Komputer mati — tekan "Nyalakan komputer"' : 'Tidak ada daya', Wp / 2, Hp / 2); return; }
  if (ST.pc === 1) {
    c.fillStyle = '#0b1020'; c.fillRect(0, 0, Wp, Hp); c.fillStyle = '#c7d2fe'; c.font = '600 26px Inter,Arial'; c.textAlign = 'center'; c.fillText('Memulai sistem…', Wp / 2, Hp / 2 - 20);
    c.fillStyle = '#1e293b'; c.fillRect(Wp / 2 - 160, Hp / 2 + 6, 320, 8); c.fillStyle = '#6366f1'; c.fillRect(Wp / 2 - 160, Hp / 2 + 6, 320 * ST.pcBoot, 8); return;
  }
  const gr = c.createLinearGradient(0, 0, Wp, Hp); gr.addColorStop(0, '#1e1b4b'); gr.addColorStop(1, '#0f766e'); c.fillStyle = gr; c.fillRect(0, 0, Wp, Hp);
  c.fillStyle = 'rgba(10,12,20,.85)'; c.fillRect(0, Hp - 34, Wp, 34); c.fillStyle = '#e5e7eb'; c.font = '500 13px Inter,Arial'; c.textAlign = 'right';
  const d = new Date(); c.fillText(String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0'), Wp - 14, Hp - 12);
  if (ST.pc === 2) {
    c.fillStyle = 'rgba(255,255,255,.14)'; c.fillRect(30, 30, 92, 92); c.fillStyle = '#a5b4fc'; c.fillRect(52, 44, 48, 40); c.fillStyle = '#fff'; c.font = '600 12px Inter,Arial'; c.textAlign = 'center'; c.fillText('Spectra', 76, 102); c.fillText('Control', 76, 116);
    pcBtns.push({ x: 30, y: 30, w: 92, h: 92, act: 'software' }); c.fillStyle = 'rgba(255,255,255,.8)'; c.font = '500 14px Inter,Arial'; c.fillText('Klik ikon untuk membuka perangkat lunak instrumen', Wp / 2, Hp / 2); return;
  }
  // jendela perangkat lunak
  const X = 16, Y = 14, WW = Wp - 32, HH = Hp - 62;
  c.fillStyle = '#ffffff'; c.fillRect(X, Y, WW, HH); c.fillStyle = '#1f3a8a'; c.fillRect(X, Y, WW, 28);
  c.fillStyle = '#fff'; c.font = '600 13px Inter,Arial'; c.textAlign = 'left'; c.fillText('Spectra Control — UV-1700 (simulasi)   ·   ' + AN.nama, X + 10, Y + 19);
  c.fillStyle = '#f3f4f6'; c.fillRect(X, Y + 28, WW, 40);
  pcButton(c, X + 8, Y + 34, 96, 28, ST.connected ? 'Terhubung' : 'Connect', 'connect', !ST.connected);
  pcButton(c, X + 110, Y + 34, 96, 28, 'Auto Zero', 'autozero', ST.connected); pcButton(c, X + 212, Y + 34, 96, 28, 'Baseline', 'baseline', ST.connected);
  pcButton(c, X + 314, Y + 34, 96, 28, 'Start', 'measure', ST.connected); pcButton(c, X + 416, Y + 34, 96, 28, 'Scan', 'scan', ST.connected);
  pcButton(c, X + 560, Y + 34, 110, 28, (ST.pcTab === 'foto' ? '■ ' : '') + 'Fotometri', 'tabfoto'); pcButton(c, X + 676, Y + 34, 110, 28, (ST.pcTab === 'scan' ? '■ ' : '') + 'Spektrum', 'tabscan');
  // status bar
  c.fillStyle = '#e5e7eb'; c.fillRect(X, Y + HH - 24, WW, 24); c.fillStyle = '#374151'; c.font = '500 12px "JetBrains Mono",monospace'; c.textAlign = 'left';
  const zs = ST.zero ? (ST.zero.type === 'az' ? 'AZ @' + fx(ST.zero.lam, 1) + ' nm' : 'Baseline ' + ST.zero.lo + '–' + ST.zero.hi + ' nm') : 'belum auto zero';
  c.fillText((ST.connected ? '● Terhubung' : '○ Tidak terhubung') + '   |   λ ' + fx(ST.lam, 1) + ' nm   |   ' + (ST.lam < 340.8 ? 'D2' : 'WI') + '   |   ' + zs + '   |   ' + (ST.lid > 0.5 ? 'KOMPARTEMEN TERBUKA' : 'kompartemen tertutup'), X + 10, Y + HH - 8);
  if (!ST.connected) { c.fillStyle = '#6b7280'; c.font = '500 16px Inter,Arial'; c.textAlign = 'center'; c.fillText(ST.inst === 'off' ? 'Instrumen belum dinyalakan.' : 'Tekan Connect untuk menghubungkan perangkat lunak dengan UV-1700.', X + WW / 2, Y + HH / 2); return; }
  const top = Y + 78, bot = Y + HH - 32;
  if (ST.pcTab === 'foto') {
    // tabel
    const tx = X + 10, tw = 470; c.fillStyle = '#eef2ff'; c.fillRect(tx, top, tw, 24); c.fillStyle = '#1e1b4b'; c.font = '600 12px Inter,Arial'; c.textAlign = 'left';
    [['No', 8], ['Larutan', 40], ['C (' + INIT.unit + ')', 220], ['λ (nm)', 310], ['Abs', 380], ['%T', 430]].forEach(([s, x]) => c.fillText(s, tx + x, top + 16));
    const rows = ST.readings.slice(-14); c.font = '12px "JetBrains Mono",monospace';
    rows.forEach((r, i) => { const yy = top + 24 + i * 21; if (i % 2) { c.fillStyle = '#f9fafb'; c.fillRect(tx, yy, tw, 21); }
      c.fillStyle = r.flag ? '#b91c1c' : '#111827'; c.fillText(String(r.no), tx + 8, yy + 15); c.fillText(r.label.slice(0, 24), tx + 40, yy + 15);
      c.fillText(r.std ? fx(r.c, 2) : (r.sample ? '?' : '0'), tx + 220, yy + 15); c.fillText(fx(r.lam, 1), tx + 310, yy + 15); c.fillText(fx(r.A, 4), tx + 380, yy + 15); c.fillText(fx(100 * Math.pow(10, -r.A), 1), tx + 430, yy + 15); });
    if (!rows.length) { c.fillStyle = '#9ca3af'; c.font = '500 13px Inter,Arial'; c.fillText('Belum ada data. Tekan Start untuk mengukur.', tx + 10, top + 50); }
    // grafik kalibrasi
    const gx = X + 500, gy = top, gw = WW - 520, gh = bot - top - 70; chartFrame(c, gx, gy, gw, gh, 'Kurva kalibrasi', 'C (' + INIT.unit + ')', 'Abs');
    const cd = calData();
    if (cd && cd.xs.length) {
      const xmax = Math.max(...cd.xs, 0.01) * 1.15, ymax = Math.max(...cd.ys, ...(cd.smp.length ? cd.smp : [0]), 0.05) * 1.2;
      const PX = x => gx + 40 + (gw - 52) * x / xmax, PY = y => gy + gh - 28 - (gh - 52) * y / ymax;
      if (cd.reg) { c.strokeStyle = '#7c3aed'; c.lineWidth = 2; c.beginPath(); c.moveTo(PX(0), PY(cd.reg.b)); c.lineTo(PX(xmax), PY(cd.reg.a * xmax + cd.reg.b)); c.stroke(); }
      c.fillStyle = '#4f46e5'; cd.xs.forEach((x, i) => { c.beginPath(); c.arc(PX(x), PY(cd.ys[i]), 5, 0, 7); c.fill(); });
      const ax = cd.smp.length ? cd.smp.reduce((a, b) => a + b) / cd.smp.length : null;
      if (cd.reg && ax != null) { const cx = (ax - cd.reg.b) / cd.reg.a; c.fillStyle = '#e11d48'; c.beginPath(); c.moveTo(PX(cx), PY(ax) - 7); c.lineTo(PX(cx) + 7, PY(ax)); c.lineTo(PX(cx), PY(ax) + 7); c.lineTo(PX(cx) - 7, PY(ax)); c.fill(); }
      c.fillStyle = '#111827'; c.font = '500 13px "JetBrains Mono",monospace'; c.textAlign = 'left';
      const ly = gy + gh + 18;
      c.fillText('λ = ' + fx(cd.lam, 1) + ' nm   ·   ' + cd.xs.length + ' standar', gx, ly);
      if (cd.reg) { c.fillText('y = ' + fx(cd.reg.a, 4) + 'x ' + (cd.reg.b < 0 ? '− ' : '+ ') + fx(Math.abs(cd.reg.b), 4) + '    R² = ' + fx(cd.reg.r2, 4), gx, ly + 20);
        if (ax != null) c.fillText('Sampel X: Ā = ' + fx(ax, 4) + ' (n=' + cd.smp.length + ')  →  C = ' + fx((ax - cd.reg.b) / cd.reg.a, 3) + ' ' + INIT.unit, gx, ly + 40); }
    } else { c.fillStyle = '#9ca3af'; c.font = '500 13px Inter,Arial'; c.textAlign = 'center'; c.fillText('Ukur larutan standar untuk membentuk kurva', gx + gw / 2, gy + gh / 2); }
  } else {
    const gx = X + 10, gy = top, gw = WW - 20, gh = bot - top - 10; chartFrame(c, gx, gy, gw, gh, 'Spektrum absorbansi', 'λ (nm)', 'Abs');
    const sp = ST.spectrum;
    if (sp && sp.l.length > 1) {
      const lo = sp.lo, hi = sp.hi, ymax = Math.max(0.1, ...sp.a) * 1.15, ymin = Math.min(0, ...sp.a);
      const PX = l => gx + 46 + (gw - 60) * (l - lo) / (hi - lo), PY = a => gy + gh - 28 - (gh - 52) * (a - ymin) / (ymax - ymin);
      for (let l = Math.max(380, lo); l < Math.min(780, hi); l += 5) { c.fillStyle = lamColor(l + 2.5); c.fillRect(PX(l), gy + gh - 30, Math.max(1, PX(l + 5) - PX(l)), 4); }
      c.strokeStyle = '#4f46e5'; c.lineWidth = 2; c.beginPath(); sp.l.forEach((l, i) => { const X1 = PX(l), Y1 = PY(sp.a[i]); i ? c.lineTo(X1, Y1) : c.moveTo(X1, Y1); }); c.stroke();
      c.fillStyle = '#374151'; c.font = '11px Inter,Arial'; c.textAlign = 'center';
      for (let l = Math.ceil(lo / 50) * 50; l <= hi; l += 50) c.fillText(String(l), PX(l), gy + gh - 12);
      if (sp.done) { let im = 0; sp.a.forEach((a, i) => { if (a > sp.a[im]) im = i; }); c.strokeStyle = '#0d9488'; c.setLineDash([5, 4]); c.beginPath(); c.moveTo(PX(sp.l[im]), gy + 30); c.lineTo(PX(sp.l[im]), gy + gh - 30); c.stroke(); c.setLineDash([]);
        c.fillStyle = '#0d9488'; c.font = '600 13px "JetBrains Mono",monospace'; c.fillText('λmaks ' + fx(sp.l[im], 1) + ' nm · A = ' + fx(sp.a[im], 4), PX(sp.l[im]), gy + 26); sp.lmax = sp.l[im]; }
      if (!sp.zeroed) { c.fillStyle = '#b91c1c'; c.font = '500 12px Inter,Arial'; c.textAlign = 'right'; c.fillText('Tanpa koreksi baseline', gx + gw - 10, gy + 18); }
    } else { c.fillStyle = '#9ca3af'; c.font = '500 13px Inter,Arial'; c.textAlign = 'center'; c.fillText('Tekan Scan untuk memindai spektrum ' + scanRange().join('–') + ' nm', gx + gw / 2, gy + gh / 2); }
  }
}
function chartFrame(c, x, y, w, h, title, xl, yl) {
  c.strokeStyle = '#d1d5db'; c.lineWidth = 1; c.strokeRect(x + 0.5, y + 0.5, w, h); c.fillStyle = '#111827'; c.font = '600 13px Inter,Arial'; c.textAlign = 'left'; c.fillText(title, x + 10, y + 18);
  c.strokeStyle = '#9ca3af'; c.beginPath(); c.moveTo(x + 40, y + 24); c.lineTo(x + 40, y + h - 28); c.lineTo(x + w - 10, y + h - 28); c.stroke();
  c.fillStyle = '#6b7280'; c.font = '11px Inter,Arial'; c.textAlign = 'right'; c.fillText(xl, x + w - 12, y + h - 4); c.save(); c.translate(x + 12, y + h / 2); c.rotate(-Math.PI / 2); c.textAlign = 'center'; c.fillText(yl, 0, 0); c.restore();
}

/* ---------------- animasi & utilitas ---------------- */
const tweens = [];
function tween(dur, fn) { return new Promise(res => tweens.push({ t0: performance.now(), dur: dur * 1000, fn, res })); }
const ease = t => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
const wait = s => tween(s, () => { });
let audio = null;
function beep(f, d) { if (!ST.sound) return; try { audio = audio || new (window.AudioContext || window.webkitAudioContext)(); const o = audio.createOscillator(), g = audio.createGain(); o.frequency.value = f || 880; g.gain.value = 0.05; o.connect(g); g.connect(audio.destination); o.start(); o.stop(audio.currentTime + (d || 0.12)); } catch (e) { } }
let toastT = 0;
function toast(msg, kind) { const t = $('#toast'); t.textContent = msg; t.className = 'toast show ' + (kind || 'info'); clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove('show'), 3600); }
function log(msg, kind) { const el = $('#log'); const d = document.createElement('div'); if (kind) d.className = kind; d.textContent = mmss((performance.now() - ST.t0) / 1000) + '  ' + msg; el.insertBefore(d, el.firstChild); }
function mistake(code, msg) { if (!ST.errs.has(code)) { ST.errs.add(code); ST.score = Math.max(0, ST.score - 5); log('Kesalahan prosedur (−5): ' + msg, 'e'); } toast(msg, 'err'); beep(220, 0.25); sync(); }
function ok(msg) { toast(msg, 'ok'); log(msg, 'o'); }
const ready = () => ST.inst === 'warm' || ST.inst === 'ready';
const sel = () => ($$('input[name=cv]').find(i => i.checked) || {}).value || 'S';
function moveCuv(k, pts, durEach) { const c = ST.cuv[k]; let chain = Promise.resolve(); for (let i = 0; i < pts.length; i++) { chain = chain.then(() => { const from = c.pos.slice(), to = pts[i]; return tween(durEach, t => { c.pos = lerp3(from, to, ease(t)); }); }); } return chain; }
function scanRange() { const lo = AN.lam_ref < 320 ? 200 : Math.max(190, Math.round((AN.lam_ref - 150) / 10) * 10); return [lo, Math.min(1100, lo + 320)]; }

/* ---------------- pengukuran ---------------- */
function cellA(k, l) { const c = ST.cuv[k]; if (c.where !== 'sel') return 0; let a = 0.045 + c.smudge; if (!c.content) return a + 0.035; return a + 0.003 + SOL[c.content].c * absorptivity(l); }
function rawA(l, noisy) {
  let A = cellA('S', l) - cellA('R', l) + instOff(l);
  const t = (performance.now() - ST.t0) / 1000;
  if (ST.lid > 0.5) A -= 0.05 + 0.06 * rnd();
  A += ST.warmSkipped ? 0.006 * Math.sin(t / 7) + 0.002 * gauss() : 0.0004 * Math.sin(t / 30);
  if (noisy !== false) { const T = Math.pow(10, -A); A += gauss() * 0.4343 * 1.5e-4 / Math.max(T * lampE(l), 1e-4); }
  const s = l < 230 ? 4e-4 : 1e-4; const T = (Math.pow(10, -A) + s) / (1 + s); return Math.min(4, Math.max(-0.5, -Math.log10(T)));
}
function zeroOff(l) { const z = ST.zero; if (!z) return null; if (z.type === 'az' && Math.abs(z.lam - l) < 0.05) return z.off; if (z.type === 'bl' && l >= z.lo && l <= z.hi) return z.offs[Math.round(l) - z.lo]; return null; }
function readAt(l) { const r = rawA(l); const z = zeroOff(l); return { A: z == null ? r : r - z, zeroed: z != null }; }
function checksBeforeRead() {
  if (!ready()) { toast(ST.inst === 'off' ? 'Instrumen belum dinyalakan.' : 'Tunggu inisialisasi selesai.', 'warn'); return false; }
  if (ST.lid > 0.5) mistake('lidopen', 'Kompartemen masih terbuka: cahaya ruangan masuk ke detektor. Tutup sebelum mengukur.');
  if (ST.inst === 'warm' && !ST.warmSkipped) mistake('nowarm', 'Lampu belum stabil (pemanasan belum selesai); pembacaan dapat bergeser.');
  return true;
}

/* ---------------- aksi ---------------- */
const ACT = {
  async cables() {
    if (ST.cabled) return toast('Semua kabel sudah tersambung.');
    setView('belakang'); log('Menyambungkan kabel daya dan data…');
    for (let i = 0; i < CABLES.length; i++) { await tween(0.7, t => { ST.cab[i] = t; }); beep(600, 0.05); log('Tersambung: ' + CABLES[i].nm); }
    ST.cabled = true; ok('Kabel daya (UPS → stop kontak; alat, CPU, monitor → UPS), kabel video, dan kabel data RS-232C tersambung.');
    await wait(0.6); setView('meja');
  },
  async ups() {
    if (!ST.cab[0]) return mistake('upsnocable', 'UPS belum tersambung ke stop kontak. Sambungkan kabel terlebih dahulu.');
    if (ST.ups) return toast('UPS sudah menyala.');
    setView('ups'); await wait(0.6); ST.ups = true; beep(1200, 0.15); await wait(0.25); beep(1200, 0.15); ok('UPS menyala: tegangan keluaran stabil 220 V.'); await wait(0.6); setView('meja');
  },
  async pc() {
    if (!ST.ups || !ST.cab[2]) return mistake('pcnoups', 'Komputer harus mendapat daya dari UPS yang sudah menyala dan kabelnya tersambung.');
    if (ST.pc) return toast('Komputer sudah menyala.');
    setView('komputer'); ST.pc = 1; ST.pcBoot = 0; log('Komputer dinyalakan; sistem memulai.');
    await tween(2.5, t => { ST.pcBoot = t; }); ST.pc = 2; ok('Komputer siap. Buka perangkat lunak instrumen.');
  },
  async inst() {
    if (ST.inst !== 'off') return toast('Instrumen sudah menyala.');
    if (!ST.ups || !ST.cab[1]) return mistake('instnoups', 'UV-1700 belum mendapat daya: sambungkan kabel dan nyalakan UPS terlebih dahulu.');
    if (ST.cuv.R.where === 'sel' || ST.cuv.S.where === 'sel') mistake('initcuv', 'Kompartemen harus kosong saat inisialisasi (pemeriksaan energi berkas).');
    setView('alat'); ST.inst = 'init'; ST.initLines = []; beep(900, 0.1); log('UV-1700 dinyalakan; inisialisasi berjalan.');
    const L = ['UV-1700 INITIALIZE', 'LSI INITIALIZE    OK', 'ROM CHECK         OK', 'RAM CHECK         OK', 'FILTER INIT       OK', 'SLIT INIT         OK', 'D2 LAMP ON        OK', 'WI LAMP ON        OK', 'WL ORIGIN 656.1   OK', 'ENERGY CHECK      OK'];
    for (const s of L) { ST.initLines.push(s); dirty = true; await wait(0.45); }
    beep(1400, 0.2); ST.inst = 'warm'; ST.warmLeft = 15 * 60; ST.lam = AN.lam_ref; ST.lastA = 0;
    ok('Inisialisasi selesai. Lampu dipanaskan 15 menit (dipercepat) agar intensitas stabil.');
  },
  skipwarm() {
    if (ST.inst !== 'warm') return toast(ST.inst === 'ready' ? 'Pemanasan sudah selesai.' : 'Nyalakan instrumen terlebih dahulu.', 'warn');
    ST.warmSkipped = true; ST.inst = 'ready'; ST.warmLeft = 0; mistake('skipwarm', 'Pemanasan dilewati: drift baseline meningkat selama pengukuran.');
  },
  software() { if (ST.pc < 2) return toast('Nyalakan komputer terlebih dahulu.', 'warn'); if (ST.pc === 3) return toast('Perangkat lunak sudah terbuka.'); ST.pc = 3; ok('Perangkat lunak instrumen dibuka.'); setView('komputer'); },
  async connect() {
    if (ST.pc < 3) return toast('Buka perangkat lunak di komputer terlebih dahulu.', 'warn');
    if (ST.connected) return toast('Sudah terhubung.');
    if (!ST.cab[5]) return mistake('nodata', 'Kabel data RS-232C belum tersambung.');
    if (!ready()) return mistake('connearly', 'Hubungkan setelah inisialisasi UV-1700 selesai.');
    log('Menghubungkan perangkat lunak ke UV-1700…'); await wait(1.4); ST.connected = true; beep(1000, 0.1); ok('Komunikasi PC–UV-1700 terhubung. Hasil pengukuran akan tampil di layar komputer.');
  },
  async fill() {
    const k = sel(), c = ST.cuv[k], sid = $('#sol').value;
    if (c.where !== 'rak') return toast('Keluarkan kuvet dari alat dahulu sebelum mengisi.', 'warn');
    setView('kerja'); const prev = c.content;
    if (prev) await tween(0.6, t => { c.level = 1 - t; });
    log('Kuvet ' + (k === 'R' ? 'referensi' : 'sampel') + ' dibilas akuades lalu dibilas dan diisi ' + SOL[sid].label + '.');
    c.content = sid; await tween(1.1, t => { c.level = 0.85 * ease(t); }); c.dirty = true; c.smudge = 0;
    if (k === 'R' && sid !== 'blank') toast('Perhatian: kuvet referensi biasanya diisi blanko.', 'warn'); else toast('Kuvet terisi. Lap dinding beningnya dengan tisu sebelum dimasukkan.', 'info');
  },
  async empty() { const k = sel(), c = ST.cuv[k]; if (c.where !== 'rak') return toast('Keluarkan kuvet dahulu.', 'warn'); if (!c.content) return toast('Kuvet sudah kosong.'); setView('kerja'); await tween(0.6, t => { c.level = 0.85 * (1 - t); }); c.content = null; c.level = 0; log('Isi kuvet dibuang ke gelas buangan.'); },
  async wipe() {
    const k = sel(), c = ST.cuv[k];
    if (c.where !== 'rak') return toast('Kuvet harus di luar alat untuk dilap.', 'warn');
    setView('kerja'); const p = c.pos; ST.tissue = { pos: [-35, 9, 23], rot: 0 };
    const near = [p[0] + 1.6, p[1] + 5, p[2]];
    await tween(0.7, t => { ST.tissue.pos = lerp3([-35, 9, 23], near, ease(t)); ST.tissue.rot = 90 * t; });
    await tween(1.6, t => { ST.tissue.pos = [near[0], p[1] + 4 + 2.8 * Math.sin(t * Math.PI * 6), near[2]]; });
    await tween(0.6, t => { ST.tissue.pos = lerp3(ST.tissue.pos, [-58, 7, 26], ease(t)); });
    ST.tissue = null; c.dirty = false; ok('Kuvet ' + (k === 'R' ? 'referensi' : 'sampel') + ' dilap dengan tisu: dinding bening bebas tetesan dan sidik jari.');
  },
  async lid() { const to = ST.lid > 0.5 ? 0 : 1; if (to && viewName === 'meja') setView('kompartemen'); await tween(0.7, t => { ST.lid = to ? t : 1 - t; }); ST.lid = to; log(to ? 'Tutup kompartemen dibuka.' : 'Tutup kompartemen ditutup.'); },
  async insert() {
    const k = sel(), c = ST.cuv[k];
    if (c.where === 'sel') return toast('Kuvet ini sudah di dalam alat.');
    if (ST.lid < 0.5) return mistake('lidclosed', 'Buka tutup kompartemen sebelum memasukkan kuvet.');
    if (!c.content) toast('Kuvet masih kosong (berisi udara).', 'warn');
    if (c.dirty) { mistake('dirty' + k, 'Kuvet dimasukkan tanpa dilap: tetesan/sidik jari menghamburkan cahaya dan menaikkan absorbansi.'); c.smudge = 0.012 + 0.02 * rnd(); }
    setView('kompartemen'); const r = RACK[k], d = CELL[k];
    await moveCuv(k, [[r[0], r[1] + 22, r[2]], [d[0], d[1] + 18, d[2]], d], 0.75); c.where = 'sel'; beep(700, 0.05);
    log('Kuvet ' + (k === 'R' ? 'referensi → sel referensi (belakang)' : 'sampel → sel sampel (depan)') + '; sisi bening menghadap berkas.');
  },
  async remove() {
    const k = sel(), c = ST.cuv[k];
    if (c.where !== 'sel') return toast('Kuvet ini tidak sedang di dalam alat.');
    if (ST.lid < 0.5) return mistake('lidclosed2', 'Buka tutup kompartemen untuk mengeluarkan kuvet.');
    setView('kompartemen'); const r = RACK[k], d = CELL[k];
    await moveCuv(k, [[d[0], d[1] + 18, d[2]], [r[0], r[1] + 22, r[2]], r], 0.75); c.where = 'rak'; c.smudge = 0; log('Kuvet dikeluarkan ke rak.');
  },
  goto() { const v = parseFloat(String($('#lam').value).replace(',', '.')); if (!(v >= 190 && v <= 1100)) return toast('λ harus 190–1100 nm.', 'warn'); if (ST.inst === 'off') return toast('Instrumen mati.', 'warn');
    ST.lam = Math.round(v * 10) / 10; ST.lastA = ready() ? readAt(ST.lam).A : 0; beep(800, 0.05); log('GOTO WL ' + fx(ST.lam, 1) + ' nm.'); },
  async autozero() {
    if (!checksBeforeRead()) return;
    const R = ST.cuv.R, S = ST.cuv.S;
    if (R.where !== 'sel') mistake('azref', 'Sel referensi kosong: masukkan kuvet blanko sebelum auto zero.');
    if (S.where === 'sel' && S.content && S.content !== 'blank') mistake('aznonblank', 'Auto zero harus dilakukan dengan blanko di sel sampel, bukan larutan standar/sampel.');
    setView('alat'); await beamPulse(0.9);
    ST.zero = { type: 'az', lam: ST.lam, off: rawA(ST.lam, false) }; ST.lastA = 0; beep(1100, 0.1); ok('AUTO ZERO pada ' + fx(ST.lam, 1) + ' nm: absorbansi blanko = 0,000.');
  },
  async baseline() {
    if (!checksBeforeRead()) return;
    const R = ST.cuv.R, S = ST.cuv.S;
    if (R.where !== 'sel' || S.where !== 'sel' || R.content !== 'blank' || S.content !== 'blank') mistake('blnotblank', 'Koreksi baseline memerlukan blanko di kedua sel.');
    const [lo, hi] = scanRange(); setView('alat'); ST.xrayT = ST.autoX ? 1 : ST.xrayT; log('Koreksi baseline ' + lo + '–' + hi + ' nm…');
    const offs = []; for (let l = lo; l <= hi; l++) offs.push(rawA(l, false));
    await tween(2.4, t => { ST.lam = hi - (hi - lo) * t; ST.beam = 1; }); ST.beam = 0; ST.lam = AN.lam_ref; if (!ST.xrayUser) ST.xrayT = 0;
    ST.zero = { type: 'bl', lo, hi, offs }; ST.lastA = 0; beep(1100, 0.1); ok('Baseline ' + lo + '–' + hi + ' nm tersimpan.');
  },
  async measure() {
    if (!checksBeforeRead()) return;
    const S = ST.cuv.S; if (S.where !== 'sel') return mistake('nosample', 'Sel sampel kosong: masukkan kuvet sampel.');
    if (zeroOff(ST.lam) == null) mistake('nozero', 'Belum auto zero/baseline pada λ ini: pembacaan masih mengandung offset instrumen dan blanko.');
    setView('alat'); await beamPulse(1.0);
    let s = 0; for (let i = 0; i < 5; i++) s += readAt(ST.lam).A; const A = s / 5; ST.lastA = A; beep(1300, 0.08);
    const sol = S.content ? SOL[S.content] : { label: 'Kuvet kosong', c: 0 };
    if (ST.connected) {
      ST.readings.push({ no: ST.readings.length + 1, label: sol.label.replace(/ · .*/, ''), c: sol.c, std: !!sol.std, sample: !!sol.sample, lam: ST.lam, A, flag: S.smudge > 0 || ST.lid > 0.5 || zeroOff(ST.lam) == null });
      ST.pcTab = 'foto'; updateOut(); ok(sol.label + ': A = ' + fx(A, 4) + ' pada ' + fx(ST.lam, 1) + ' nm (tercatat di komputer).');
    } else { toast('A = ' + fx(A, 4) + ' (hanya di LCD; hubungkan PC agar tercatat).', 'warn'); log('Pembacaan ' + fx(A, 4) + ' tidak tercatat di komputer.', 'w'); }
  },
  async scan() {
    if (!ST.connected) return toast('Pemindaian spektrum dijalankan dari komputer: hubungkan PC terlebih dahulu.', 'warn');
    if (!checksBeforeRead()) return;
    const S = ST.cuv.S; if (S.where !== 'sel') return mistake('nosample', 'Sel sampel kosong: masukkan kuvet sampel.');
    const [lo, hi] = scanRange(); const zeroed = ST.zero && ST.zero.type === 'bl' && ST.zero.lo <= lo && ST.zero.hi >= hi;
    if (!zeroed) mistake('noblscan', 'Pemindaian tanpa koreksi baseline: spektrum mengandung serapan blanko dan offset instrumen.');
    ST.pcTab = 'scan'; ST.spectrum = { lo, hi, l: [], a: [], done: false, zeroed, name: S.content ? SOL[S.content].label : 'kosong' }; setView('alat');
    ST.xrayT = ST.autoX ? 1 : ST.xrayT; let li = hi;
    await tween(4.0, t => { const target = hi - (hi - lo) * t; while (li >= target) { const r = readAt(li); ST.spectrum.l.unshift(li); ST.spectrum.a.unshift(r.A); li -= 1; } ST.lam = Math.max(lo, target); ST.lastA = ST.spectrum.a[0] || 0; ST.beam = 1; });
    ST.beam = 0; if (!ST.xrayUser) ST.xrayT = 0; ST.spectrum.done = true; drawPC();
    const lm = ST.spectrum.lmax; ST.lam = lm || AN.lam_ref; $('#lam').value = ST.lam; ST.lastA = readAt(ST.lam).A; beep(1300, 0.1);
    ok('Scan selesai: λmaks ≈ ' + fx(lm, 1) + ' nm. Gunakan GOTO WL ke λmaks untuk pengukuran kuantitatif.');
  },
  tabfoto() { ST.pcTab = 'foto'; }, tabscan() { ST.pcTab = 'scan'; },
  async shutdown() {
    if (ST.cuv.R.where === 'sel' || ST.cuv.S.where === 'sel') toast('Keluarkan dan bersihkan kuvet setelah praktikum.', 'warn');
    ST.connected = false; ST.pc = 0; ST.inst = 'off'; ST.beam = 0; ST.zero = null; await wait(0.4); ST.ups = false; log('Perangkat lunak ditutup, UV-1700 dan komputer dimatikan, lalu UPS.'); ok('Semua perangkat dimatikan dengan urutan benar.');
  },
  copy() { const t = $('#out'); t.removeAttribute('readonly'); t.select(); let okc = false; try { okc = document.execCommand('copy'); } catch (e) { } t.setAttribute('readonly', '');
    if (navigator.clipboard) navigator.clipboard.writeText(t.value).then(() => toast('Data disalin.', 'ok'), () => toast(okc ? 'Data disalin.' : 'Pilih teks lalu tekan Ctrl+C.', okc ? 'ok' : 'warn')); else toast(okc ? 'Data disalin.' : 'Pilih teks lalu tekan Ctrl+C.', okc ? 'ok' : 'warn'); },
  csv() { const rows = ['No;Larutan;C (' + INIT.unit + ');Lambda (nm);Abs;%T'].concat(ST.readings.map(r => [r.no, r.label, r.std ? fx(r.c, 3) : (r.sample ? '' : '0'), fx(r.lam, 1), fx(r.A, 4), fx(100 * Math.pow(10, -r.A), 2)].join(';')));
    if (ST.spectrum && ST.spectrum.done) { rows.push(''); rows.push('Spektrum ' + ST.spectrum.name); rows.push('Lambda (nm);Abs'); ST.spectrum.l.forEach((l, i) => rows.push(fx(l, 1) + ';' + fx(ST.spectrum.a[i], 4))); }
    try { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([rows.join('\n')], { type: 'text/csv' })); a.download = 'praktikum-uvvis.csv'; document.body.appendChild(a); a.click(); a.remove(); } catch (e) { toast('Unduhan diblokir browser; gunakan Salin data.', 'warn'); } }
};
async function beamPulse(s) { ST.xrayT = ST.autoX ? 1 : ST.xrayT; await tween(s, t => { ST.beam = Math.sin(Math.PI * t); }); ST.beam = 0; if (!ST.xrayUser) ST.xrayT = 0; }
function updateOut() {
  const cd = calData(); let s = '#LAMBDA ' + fx(cd ? cd.lam : ST.lam, 1) + '\n#STANDAR\n';
  if (cd) cd.xs.forEach((x, i) => { s += fx(x, 3) + ';' + fx(cd.ys[i], 4) + '\n'; });
  s += '#SAMPEL\n' + (cd && cd.smp.length ? cd.smp.map(a => fx(a, 4)).join('; ') : '') + '\n'; $('#out').value = s;
}

/* ---------------- panduan langkah ---------------- */
function nextStep() {
  const R = ST.cuv.R, S = ST.cuv.S;
  if (!ST.cabled) return ['cables', 'Sambungkan kabel daya ke UPS dan stop kontak, kabel video, serta kabel data RS-232C.'];
  if (!ST.ups) return ['ups', 'Nyalakan UPS terlebih dahulu.'];
  if (!ST.pc) return ['pc', 'Nyalakan komputer.'];
  if (ST.inst === 'off') return ['inst', 'Nyalakan UV-1700 (kompartemen kosong saat inisialisasi).'];
  if (ST.inst === 'init') return [null, 'Tunggu inisialisasi selesai…'];
  if (ST.pc === 2) return ['software', 'Buka perangkat lunak instrumen di komputer.'];
  if (!ST.connected) return ['connect', 'Hubungkan perangkat lunak dengan UV-1700.'];
  if (R.content !== 'blank') return ['fill', 'Pilih Kuvet referensi, pilih Blanko, lalu Bilas & isi.'];
  if (R.where === 'rak' && R.dirty) return ['wipe', 'Lap kuvet referensi dengan tisu.'];
  if (!S.content && S.where === 'rak') return ['fill', 'Pilih Kuvet sampel dan isi dengan Blanko untuk auto zero/baseline.'];
  if (S.where === 'rak' && S.dirty) return ['wipe', 'Lap kuvet sampel dengan tisu.'];
  if ((R.where === 'rak' || S.where === 'rak') && ST.lid < 0.5) return ['lid', 'Buka tutup kompartemen.'];
  if (R.where === 'rak') return ['insert', 'Pilih Kuvet referensi lalu Masukkan ke alat.'];
  if (S.where === 'rak') return ['insert', 'Pilih Kuvet sampel lalu Masukkan ke alat.'];
  if (ST.lid > 0.5) return ['lid', 'Tutup kompartemen sebelum mengukur.'];
  if (ST.inst === 'warm') return [null, 'Lampu masih pemanasan (' + mmss(ST.warmLeft) + '). Tunggu, atau lewati (mengurangi skor).'];
  if (!ST.zero && S.content === 'blank') return ['baseline', 'Lakukan koreksi baseline (untuk scan) atau GOTO λ lalu auto zero.'];
  if (!ST.spectrum && ST.zero && ST.zero.type === 'bl' && S.content === 'blank') return ['remove', 'Keluarkan kuvet sampel, isi dengan standar tengah, lap, masukkan, lalu Scan untuk mencari λmaks.'];
  if (zeroOff(ST.lam) == null && S.content === 'blank') return ['autozero', 'Auto zero pada λ kerja dengan blanko.'];
  const done = new Set(ST.readings.filter(r => r.std).map(r => r.c)); const left = AN.std.filter(c => !done.has(c));
  if (left.length) return ['measure', 'Ukur deret standar berurutan (sisa ' + left.length + '): keluarkan kuvet sampel, ganti isi, lap, masukkan, tutup, Ukur.'];
  if (!ST.readings.some(r => r.sample)) return ['measure', 'Ukur Sampel X tiga kali (triplo).'];
  return ['copy', 'Data lengkap. Salin data dan kirim ke Kurva Kalibrasi; matikan alat setelah selesai.'];
}
function sync() {
  const R = ST.cuv.R, S = ST.cuv.S, ch = [];
  const chip = (t, s) => ch.push('<span class="chip ' + (s === 2 ? 'on' : s === 1 ? 'mid' : '') + '">' + t + '</span>');
  chip('Kabel', ST.cabled ? 2 : (ST.cab.some(x => x > 0) ? 1 : 0)); chip('UPS', ST.ups ? 2 : 0); chip('PC', ST.pc === 3 ? 2 : ST.pc ? 1 : 0);
  chip('UV-1700 ' + (ST.inst === 'warm' ? mmss(ST.warmLeft) : ST.inst.toUpperCase()), ST.inst === 'ready' ? 2 : ST.inst === 'off' ? 0 : 1); chip('Link PC', ST.connected ? 2 : 0);
  chip('Tutup ' + (ST.lid > 0.5 ? 'terbuka' : 'tertutup'), ST.lid > 0.5 ? 1 : 2);
  chip('Ref: ' + (R.content ? SOL[R.content].label.replace(/ ·.*| \(.*/, '') : 'kosong') + ' · ' + (R.where === 'sel' ? 'di alat' : 'rak') + (R.dirty ? ' · belum dilap' : ''), R.where === 'sel' ? 2 : 1);
  chip('Sampel: ' + (S.content ? SOL[S.content].label.replace(/ \(.*/, '') : 'kosong') + ' · ' + (S.where === 'sel' ? 'di alat' : 'rak') + (S.dirty ? ' · belum dilap' : ''), S.where === 'sel' ? 2 : 1);
  chip(ST.zero ? (ST.zero.type === 'az' ? 'AZ ' + fx(ST.zero.lam, 1) : 'Baseline') : 'Belum zero', ST.zero ? 2 : 0);
  $('#chips').innerHTML = ch.join('');
  const [act, txt] = nextStep(); $('#next').innerHTML = '<b>' + txt + '</b>';
  $$('button.a').forEach(b => { b.classList.toggle('hl', b.dataset.act === act); b.disabled = ST.busy && !['copy', 'csv'].includes(b.dataset.act); });
  $('#score').textContent = ST.score; const [lo, hi] = scanRange(); $('#scaninfo').textContent = 'Rentang scan/baseline: ' + lo + '–' + hi + ' nm · λ referensi metode ' + AN.lam_ref + ' nm.';
  dirty = true;
}
async function run(name) {
  const f = ACT[name]; if (!f || ST.busy) return;
  ST.busy = true; sync();
  try { await f(); } catch (e) { console.error(e); toast('Galat: ' + e.message, 'err'); }
  ST.busy = false; sync(); drawPC();
}

/* ---------------- loop ---------------- */
let last = performance.now(), lastPC = 0;
function loop(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  for (let i = tweens.length - 1; i >= 0; i--) { const tw = tweens[i]; const t = Math.min(1, (now - tw.t0) / tw.dur); tw.fn(t); dirty = true; if (t >= 1) { tweens.splice(i, 1); tw.res(); } }
  const e = (cur, tgt, sp) => { const d = tgt - cur; return Math.abs(d) < 0.002 ? tgt : cur + d * Math.min(1, dt * sp); };
  const nx = e(ST.xray, ST.xrayT, 5); if (nx !== ST.xray) { ST.xray = nx; dirty = true; }
  if (camAnim) { const a = camAnim; a.t = Math.min(1, a.t + dt * 1.6); const k = ease(a.t); let dy = a.to[0] - a.from[0]; while (dy > 180) dy -= 360; while (dy < -180) dy += 360;
    CAM.yaw = a.from[0] + dy * k; CAM.pitch = a.from[1] + (a.to[1] - a.from[1]) * k; CAM.dist = a.from[2] + (a.to[2] - a.from[2]) * k; CAM.target = lerp3(a.from[3], a.to[3], k); if (a.t >= 1) camAnim = null; dirty = true; }
  if (ST.inst === 'warm' && !ST.warmSkipped) { ST.warmLeft -= dt * 90; if (ST.warmLeft <= 0) { ST.warmLeft = 0; ST.inst = 'ready'; beep(1400, 0.2); ok('Pemanasan selesai: lampu stabil, instrumen READY.'); sync(); } dirty = true; }
  if (now - lastPC > 250) { lastPC = now; $('#clock').textContent = mmss((now - ST.t0) / 1000); if (ST.inst === 'warm') sync(); drawPC(); dirty = true; }
  if (dirty) { dirty = false; draw(); }
  requestAnimationFrame(loop);
}

/* ---------------- input ---------------- */
let selId = null;
function pickAt(x, y) {
  const inPoly = (p, px, py) => { let c = false; for (let i = 0, j = 3; i < 4; j = i++) { if (((p[i][1] > py) !== (p[j][1] > py)) && (px < (p[j][0] - p[i][0]) * (py - p[i][1]) / (p[j][1] - p[i][1]) + p[i][0])) c = !c; } return c; };
  for (let i = drawn.length - 1; i >= 0; i--) if (inPoly(drawn[i].p, x, y)) return drawn[i].id; return null;
}
(function bind() {
  let pts = new Map(), start = null, pinch = null;
  cv.addEventListener('pointerdown', e => { try { cv.setPointerCapture(e.pointerId); } catch (_) { } pts.set(e.pointerId, [e.clientX, e.clientY]); start = { x: e.clientX, y: e.clientY, yaw: CAM.yaw, pitch: CAM.pitch, moved: false }; if (pts.size === 2) { const [a, b] = [...pts.values()]; pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), dist: CAM.dist }; } });
  cv.addEventListener('pointermove', e => {
    if (!pts.has(e.pointerId)) return; pts.set(e.pointerId, [e.clientX, e.clientY]);
    if (pts.size === 2 && pinch) { const [a, b] = [...pts.values()]; CAM.dist = Math.max(40, Math.min(420, pinch.dist * pinch.d / Math.max(10, Math.hypot(a[0] - b[0], a[1] - b[1])))); start.moved = true; dirty = true; return; }
    if (!start) return; const dx = e.clientX - start.x, dy = e.clientY - start.y; if (Math.abs(dx) + Math.abs(dy) > 4) start.moved = true;
    CAM.yaw = start.yaw - dx * 0.4; CAM.pitch = Math.max(-5, Math.min(89, start.pitch + dy * 0.3)); camAnim = null; dirty = true;
  });
  const up = e => { if (!pts.has(e.pointerId)) return; pts.delete(e.pointerId); if (pts.size < 2) pinch = null;
    if (pts.size === 0) { if (start && !start.moved) { const r = cv.getBoundingClientRect(); const id = pickAt(e.clientX - r.left, e.clientY - r.top); selId = id; const inf = id && INFO[id];
      $('#ti').textContent = inf ? inf[0] : 'Info komponen'; $('#info').textContent = inf ? (inf[1] || 'Komponen bangku optik.') : 'Klik objek untuk penjelasan.'; dirty = true; } start = null; } };
  cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
  cv.addEventListener('wheel', e => { e.preventDefault(); CAM.dist = Math.max(40, Math.min(420, CAM.dist * (e.deltaY > 0 ? 1.08 : 1 / 1.08))); camAnim = null; dirty = true; }, { passive: false });
  pcc.addEventListener('click', e => { const r = pcc.getBoundingClientRect(); const x = (e.clientX - r.left) * pcc.width / r.width, y = (e.clientY - r.top) * pcc.height / r.height;
    const b = pcBtns.find(b => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h); if (b) { if (b.act.startsWith('tab')) { ACT[b.act](); drawPC(); } else run(b.act); } });
  document.addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return;
    if (b.dataset.act) run(b.dataset.act);
    if (b.dataset.v) setView(b.dataset.v);
    if (b.dataset.t) { const k = b.dataset.t, on = b.getAttribute('aria-pressed') !== 'true'; b.setAttribute('aria-pressed', String(on));
      if (k === 'xray') { ST.xrayUser = on; ST.xrayT = on ? 1 : 0; } if (k === 'autox') ST.autoX = on; if (k === 'sound') ST.sound = on; } });
  const s = $('#sol'); SOL_KEYS.forEach(k => { const o = document.createElement('option'); o.value = k; o.textContent = SOL[k].label; s.appendChild(o); });
  $('#lam').value = AN.lam_ref;
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(() => { dirty = true; }).observe(cv.parentElement);
})();
window.LAB = { ST, ACT, run, nextStep, readAt };
log('Praktikum dimulai: ' + AN.nama + ' (' + AN.rumus + '). Ikuti kotak "Langkah berikutnya".');
updateOut(); sync(); drawPC(); requestAnimationFrame(loop);
})();
</script></body></html>
'''

# =============================================================================
# BAGIAN 5 — APLIKASI STREAMLIT
# =============================================================================
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="SPECTRA LAB 2.0", page_icon=":material/biotech:", layout="wide", initial_sidebar_state="expanded")


def _ver(v: str) -> tuple:
    return tuple(int(p) for p in re.findall(r"\d+", v)[:2])


# lebar penuh: API width="stretch" (Streamlit ≥ 1.50) atau use_container_width (versi lama)
WIDE = {"width": "stretch"} if _ver(st.__version__) >= (1, 50) else {"use_container_width": True}
PAL = ["#4f46e5", "#0d9488", "#e11d48", "#d4a017", "#7c3aed", "#0284c7"]
PCFG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 3}}

_tpl = go.layout.Template(pio.templates["plotly_white"])
_tpl.layout.update(font=dict(family="Inter, system-ui, sans-serif", size=12, color="#14123a"), colorway=PAL,
                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
                   xaxis=dict(gridcolor="#eceef8", linecolor="#c9cde6", zeroline=False, ticks="outside", tickcolor="#c9cde6"),
                   yaxis=dict(gridcolor="#eceef8", linecolor="#c9cde6", zeroline=False, ticks="outside", tickcolor="#c9cde6"),
                   hoverlabel=dict(bgcolor="#14123a", font_color="#ffffff", bordercolor="#14123a"))
pio.templates["spectralab"] = _tpl

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--ink:#14123a;--muted:#5b6386;--line:#e1e4f2;--ind:#4f46e5;--vio:#7c3aed;--teal:#0d9488;--amb:#d4a017;--rose:#e11d48}
html, body, .stApp, .stMarkdown, p, li, label { font-family:'Inter', system-ui, sans-serif; }
h1, h2, h3, h4 { font-family:'Space Grotesk','Inter',sans-serif !important; letter-spacing:-.01em; color:var(--ink); }
.stApp { background: radial-gradient(1100px 480px at 100% -8%, #ede8ff 0%, rgba(237,232,255,0) 60%), radial-gradient(900px 420px at -8% 0%, #e3f4ff 0%, rgba(227,244,255,0) 55%), #f7f8fc; }
.block-container { padding-top: 1.6rem; max-width: 1400px; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#14123a 0%,#1e1a5c 55%,#2d2172 100%); }
section[data-testid="stSidebar"] * { color:#e6e4ff !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] * { color:#14123a !important; }
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"] { background: rgba(255,255,255,.12); border-radius: 10px; }
.stButton > button, .stDownloadButton > button { width:100%; border-radius:11px; font-weight:600; border:1px solid var(--line); transition: all .15s ease; }
.stButton > button:hover { border-color: var(--ind); color: var(--ind); transform: translateY(-1px); }
.stButton > button[kind="primary"] { background: linear-gradient(135deg,#4f46e5,#7c3aed); border:0; color:#fff; box-shadow: 0 6px 18px -8px rgba(79,70,229,.7); }
.stButton > button[kind="primary"]:hover { color:#fff; filter: brightness(1.06); }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; background: rgba(255,255,255,.75); }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 6px 14px; }
.ph { margin: 0 0 14px; }
.ph .eb { font: 600 .7rem 'JetBrains Mono', monospace; letter-spacing: .14em; text-transform: uppercase; color: var(--vio); }
.ph h1 { margin: 2px 0 4px; font-size: 2rem; }
.ph p { color: var(--muted); margin: 0; max-width: 900px; }
.topbar { display:flex; flex-wrap:wrap; gap:6px; margin-bottom: 4px; }
.pill { display:inline-flex; align-items:center; gap:6px; font: 500 .72rem 'JetBrains Mono', monospace; padding: 4px 10px; border-radius: 99px; background:#eef0ff; color:#3f37c9; border:1px solid #dfe2ff; }
.pill .d { width:7px; height:7px; border-radius:50%; background: currentColor; }
.pill.sim { background:#fff6dd; color:#9a6a00; border-color:#f6e2a8; } .pill.ok { background:#dff7f1; color:#0b7a6e; border-color:#bfece0; } .pill.bad { background:#ffe6ea; color:#be123c; border-color:#fecdd6; }
.hero { position:relative; overflow:hidden; border-radius: 22px; padding: 28px 30px 30px; color:#ece9ff;
  background: radial-gradient(500px 220px at 85% 10%, rgba(124,58,237,.55), rgba(124,58,237,0)), linear-gradient(135deg,#14123a 0%,#251d6b 55%,#3b1d86 100%);
  box-shadow: 0 24px 48px -24px rgba(37,29,107,.65); }
.hero:after { content:""; position:absolute; left:0; right:0; bottom:0; height:6px; background: linear-gradient(90deg,#7b3fe4,#2f5bff,#00b3d6,#22c55e,#facc15,#f97316,#ef4444); }
.hero h1 { color:#fff !important; font-size: 2.6rem; margin: 6px 0 8px; letter-spacing: .02em; }
.hero .eb { font: 600 .72rem 'JetBrains Mono', monospace; letter-spacing:.16em; color:#a5b4fc; text-transform: uppercase; }
.hero p { color:#d4d2f6; max-width: 640px; margin: 0 0 14px; }
.hero svg { position:absolute; right:-10px; bottom:8px; opacity:.55; }
.kpi-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 8px 0 14px; }
.kpi { position:relative; overflow:hidden; background:#fff; border:1px solid var(--line); border-radius: 16px; padding: 12px 14px 12px 18px; box-shadow: 0 1px 2px rgba(20,18,58,.04); }
.kpi:before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background: var(--ind); }
.kpi.teal:before{background:var(--teal)} .kpi.amber:before{background:var(--amb)} .kpi.rose:before{background:var(--rose)} .kpi.violet:before{background:var(--vio)}
.kpi .kl { font: 600 .66rem 'JetBrains Mono', monospace; letter-spacing:.09em; text-transform:uppercase; color: var(--muted); }
.kpi .kv { font: 600 1.35rem 'Space Grotesk', sans-serif; color: var(--ink); margin-top: 3px; line-height: 1.2; word-break: break-word; }
.kpi .ks { font-size: .76rem; color: var(--muted); margin-top: 2px; }
.lcd-bezel { background: repeating-linear-gradient(90deg,#e6dfc4 0 2px,#eee8d2 2px 4px); border:1px solid #d3caa6; border-radius: 14px; padding: 12px; box-shadow: inset 0 1px 0 #fff; }
.lcd { background: linear-gradient(160deg,#c6b9ff 0%,#a596f4 100%); border-radius: 6px; padding: 10px 14px; color:#1b1248; font-family:'JetBrains Mono', monospace; box-shadow: inset 0 0 0 2px #6d60c8, inset 0 3px 14px rgba(0,0,0,.22); }
.lcd .lr { display:flex; align-items:center; justify-content:space-between; gap: 10px; }
.lcd .bar { width: 34px; height: 7px; background: rgba(255,255,255,.88); border-radius: 1px; flex: none; }
.lcd .big { font-size: 1.55rem; letter-spacing: .06em; } .lcd small { font-size: .7rem; margin-left: 4px; }
.lcd .lf { border-top: 1px dashed rgba(27,18,72,.3); margin-top: 6px; padding-top: 4px; font-size: .7rem; color:#3b2f8f; display:flex; justify-content:space-between; }
.lcd-caption { display:flex; justify-content:space-between; font: 500 .62rem 'JetBrains Mono', monospace; color:#8a8061; margin-top: 6px; letter-spacing: .08em; }
.eq { font: 500 1.05rem 'JetBrains Mono', monospace; background: linear-gradient(135deg,#eef0ff,#f5efff); border: 1px solid #e0dcff; border-radius: 12px; padding: 10px 14px; color:#2b2177; }
.banner { background: linear-gradient(135deg,#fff8e6,#fff1d1); border:1px solid #f4dd9c; border-radius: 14px; padding: 10px 14px; font-size: .88rem; color:#5b4300; margin-bottom: 10px; }
.banner.info { background: linear-gradient(135deg,#eef6ff,#eef0ff); border-color:#d6defd; color:#28306b; }
.step { display:flex; gap: 12px; align-items:flex-start; padding: 12px 14px; border:1px solid var(--line); border-radius: 14px; background:#fff; margin-bottom: 8px; }
.step .dot { width: 30px; height: 30px; border-radius: 50%; display:grid; place-items:center; font: 600 .8rem 'JetBrains Mono', monospace; background:#eef0ff; color: var(--ind); flex: none; }
.step.done .dot { background: var(--teal); color:#fff; } .step.cur { border-color: var(--ind); box-shadow: 0 0 0 3px rgba(79,70,229,.12); }
.step .t { font-weight: 600; color: var(--ink); } .step .s { color: var(--muted); font-size: .86rem; }
.feat { background:#fff; border:1px solid var(--line); border-radius: 16px; padding: 14px 16px; height: 100%; }
.feat .ic { font: 600 .72rem 'JetBrains Mono', monospace; color: var(--vio); letter-spacing:.1em; }
.feat .h { font: 600 1rem 'Space Grotesk', sans-serif; color: var(--ink); margin: 4px 0; }
.feat .b { color: var(--muted); font-size: .85rem; }
.verdict { border-radius: 14px; padding: 10px 14px; font-weight: 500; margin: 6px 0; }
.verdict.ok { background:#e3f8f2; color:#0b6b60; border:1px solid #bfece0; } .verdict.bad { background:#ffe9ed; color:#a8123a; border:1px solid #fecdd6; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def _empty_cal() -> pd.DataFrame:
    return pd.DataFrame({"C": pd.Series(dtype=float), "A": pd.Series(dtype=float), "Pakai": pd.Series(dtype=bool)})


def _defaults() -> dict:
    return {
        "settings": {"dec": "koma", "mode": "latihan",
                     "crit": {"r2_min": 0.995, "rsd_max": 2.0, "rec_min": 90.0, "rec_max": 110.0, "std_res_max": 2.0, "a_min": 0.1, "a_max": 1.0}},
        "inst": {"power": "off", "baseline": None, "lamp_switch": 340.8, "cuvette": "kuarsa", "path": 1.0, "blank": "pelarut",
                 "analyte": "kmno4", "solution": "standar", "conc": 30.0, "custom": 25.0, "matrix": 0.0, "start": 400.0, "end": 700.0,
                 "interval": 1.0, "speed": "sedang", "lam": 550.0, "autozero": [], "last_A": 0.0},
        "seed": int(np.random.default_rng().integers(1, 1_000_000)), "counter": 1,
        "spectra": [], "last_scan": None,
        "cal_base": _empty_cal(), "cal_ver": 0,
        "cal_src": "pengguna", "cal_name": "", "cal_unit": "mg/L", "cal_lambda": None, "cal_zero": False, "cal_analyte": None,
        "samples": [],
        "valid": {"reps": "", "kind": "konsentrasi", "level": "", "level_unit": "mg/L", "rec_mode": "spike", "cs": "", "css": "", "cadd": "", "cm": "", "cr": ""},
        "photo": [], "kin": {"A0": 1.2, "Ainf": 0.02, "k": 0.012, "dur": 300.0, "dt": 5.0, "noise": 0.003, "lam": 590.0, "data": [], "source": "simulasi", "n_init": 6, "ainf_fit": ""},
        "sa": {"df": pd.DataFrame({"C ditambahkan": pd.Series(dtype=float), "A": pd.Series(dtype=float)}), "ver": 0, "src": "pengguna",
               "analyte": None, "sens": 0.85, "fp": 1.0, "unit": "mg/L"},
        "mix": {"cx": 20.0, "cy": 50.0, "lam1": 350.0, "lam2": 545.0, "data": None, "mode": "simulasi",
                "manual": {"aX1": "", "aY1": "", "aX2": "", "aY2": "", "A1": "", "A2": ""}},
        "pq": {"wl_off": 0.15, "wl_tol": 0.5, "ph_bias": 0.3, "sl_pct": 0.04, "sbw": 1.0, "res_min": 1.5, "results": {}},
        "report": {"judul": "Penetapan Kadar dengan Spektrofotometri UV-Vis", "praktikan": "", "nim": "", "kelas": "", "dosen": "", "sampel": "", "tanggal": date.today().isoformat(), "catatan": "", "pdf": None, "generated": False},
        "history": [], "log": [],
    }


if "S" not in st.session_state:
    st.session_state.S = _defaults()
S = st.session_state.S
for _k, _v in _defaults().items():           # kompatibilitas proyek lama
    S.setdefault(_k, _v)
I = S["inst"]
CR = S["settings"]["crit"]


def fmt(x, d=4):
    return fnum(x, d, S["settings"]["dec"])


def sig(x, s=4):
    return fsig(x, s, S["settings"]["dec"])


def rng():
    S["counter"] += 1
    return np.random.default_rng(S["seed"] * 7919 + S["counter"])


def log(msg):
    S["log"].insert(0, f"{datetime.now():%H.%M.%S}  {msg}"); S["log"] = S["log"][:80]


def num_in(text, default=float("nan")):
    v = parse_number(text)
    return default if math.isnan(v) else v


def an():
    return ANALYTES[I["analyte"]]


def sol_conc():
    if I["solution"] == "blanko":
        return 0.0
    if I["solution"] == "sampelX":
        return hidden_conc(I["analyte"], S["seed"])
    return I["custom"] if I["solution"] == "bebas" else I["conc"]


def sol_label():
    a = an()
    if I["solution"] == "blanko":
        return f"Blanko ({a['pelarut']})"
    if I["solution"] == "sampelX":
        return f"{a['rumus']} Sampel X"
    c = sol_conc()
    return f"{a['rumus']} {fmt(c, 2 if c < 1 else 1)} mg/L"


def baseline_valid(lo, hi):
    b = I["baseline"]
    return bool(b) and b["cuvette"] == I["cuvette"] and b["blank"] == I["blank"] and b["analyte"] == I["analyte"] and min(lo, hi) >= b["start"] and max(lo, hi) <= b["end"]


def az_at(lam):
    return any(abs(z["lam"] - lam) < 0.05 and z["cuvette"] == I["cuvette"] and z["blank"] == I["blank"] for z in I["autozero"])


def read_at(lam, conc=None, analyte=None):
    akey = analyte or I["analyte"]
    corrected = baseline_valid(lam, lam) or az_at(lam)
    m = measure(lam, analyte_id=akey, conc=sol_conc() if conc is None else conc, path=I["path"], cuvette=I["cuvette"], blank=I["blank"],
                baseline_done=corrected, speed="lambat", lamp_switch=I["lamp_switch"], matrix=0 if I["solution"] == "blanko" else I["matrix"], rng=rng())
    return float(m["A"][0]), corrected, bool(m["over"][0])


def cal_points() -> pd.DataFrame:
    df = S.get("cal_cur", S["cal_base"])
    if df is None or len(df) == 0:
        return _empty_cal()
    df = df.copy()
    df["C"] = pd.to_numeric(df["C"], errors="coerce"); df["A"] = pd.to_numeric(df["A"], errors="coerce")
    df = df[df["C"].notna() & df["A"].notna()]
    return df[df["Pakai"].fillna(True).astype(bool)]


def cal_reg():
    p = cal_points()
    if len(p) < 2:
        return None
    return linreg(p["C"].astype(float).values, p["A"].astype(float).values, S["cal_zero"])


def set_calibration(df, src, name=None, unit=None, lam=None, analyte=None):
    S["cal_base"] = df.reset_index(drop=True); S["cal_ver"] += 1; S["cal_src"] = src; S.pop("cal_cur", None)
    if name is not None: S["cal_name"] = name
    if unit is not None: S["cal_unit"] = unit
    if lam is not None: S["cal_lambda"] = lam
    if analyte is not None: S["cal_analyte"] = analyte


def df_same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if a.shape != b.shape:
        return False
    nrm = lambda d: d.reset_index(drop=True).astype(object).where(d.reset_index(drop=True).notna(), "").astype(str).values
    return bool((nrm(a) == nrm(b)).all())


def sample_fp(s):
    if s["dil_mode"] == "manual":
        return s["fp_manual"] if s["fp_manual"] > 0 else float("nan")
    return dilution_factor([(r["V pipet (mL)"], r["V labu (mL)"]) for r in s["steps"]])


def sample_res(s, reg):
    try:
        reps = parse_list(s["reps"])
    except ValueError:
        return None
    if not reg or not reg.ok or not reps:
        return None
    return sample_result(reg, reps, fp=sample_fp(s), mass_g=num_in(s["mass"], None) if s["mass"] else None,
                         vol_ml=num_in(s["vol"], None) if s["vol"] else None, unit=S["cal_unit"],
                         a_min=CR["a_min"], a_max=CR["a_max"], rsd_max=CR["rsd_max"])


def new_sample(name, reps="", src="pengguna", hidden=None):
    S["counter"] += 1
    return {"id": S["counter"], "name": name, "reps": reps, "src": src, "dil_mode": "langkah", "steps": [], "fp_manual": 1.0, "mass": "", "vol": "", "hidden": hidden}


# ---------------------------------------------------------------------------
# Komponen UI
# ---------------------------------------------------------------------------
def html(s: str):
    st.markdown(s, unsafe_allow_html=True)


def header(eyebrow: str, title: str, desc: str = ""):
    html(f'<div class="ph"><div class="eb">{eyebrow}</div><h1>{title}</h1>' + (f"<p>{desc}</p>" if desc else "") + "</div>")


def kpis(items):
    """items: (label, nilai, keterangan, nada) — nada: indigo|teal|amber|rose|violet."""
    cells = "".join(f'<div class="kpi {t}"><div class="kl">{l}</div><div class="kv">{v}</div><div class="ks">{s}</div></div>' for l, v, s, t in items)
    html(f'<div class="kpi-grid">{cells}</div>')


def verdict(ok: bool, text: str):
    html(f'<div class="verdict {"ok" if ok else "bad"}">{"LULUS" if ok else "TIDAK LULUS"} · {text}</div>')


def pill(text, kind=""):
    return f'<span class="pill {kind}"><span class="d"></span>{text}</span>'


def src_pill(src):
    lab = {"simulasi": "DATA SIMULASI", "contoh": "DATA CONTOH", "pengguna": "INPUT PENGGUNA", "impor": "IMPOR CSV", "campuran": "CAMPURAN"}.get(src, str(src).upper())
    return pill(lab, "sim" if src in ("simulasi", "contoh", "campuran") else "ok")


def show(fig):
    st.plotly_chart(fig, config=PCFG, **WIDE)


def table(df, **kw):
    st.dataframe(df, hide_index=True, **WIDE, **kw)


def lcd(lam=None, A=None, mode="PHOTOMETRIC"):
    lam = I["lam"] if lam is None else lam
    A = I["last_A"] if A is None else A
    on = I["power"] == "ready"
    a_txt = ("OVER" if A > 3 else f"{A:.3f}") if on else "-----"
    t_txt = f"{float(abs_to_t(A)):.1f}" if on else "----"
    lamp = "D2" if lam < I["lamp_switch"] else "WI"
    html(f'<div class="lcd-bezel"><div class="lcd">'
         f'<div class="lr"><span class="bar"></span><span class="big">{lam:.1f}<small>nm</small></span></div>'
         f'<div class="lr"><span class="bar"></span><span class="big">{a_txt}<small>ABS</small></span></div>'
         f'<div class="lr"><span class="bar"></span><span>{lamp}</span><span>{t_txt} %T</span></div>'
         f'<div class="lf"><span>{mode}</span><span>{"READY" if on else I["power"].upper()}</span></div></div>'
         f'<div class="lcd-caption"><span>UV-1700 PharmaSpec</span><span style="color:{lam_color(lam)}">● {region(lam)}</span></div></div>')


def fig_base(xt, yt, height=380):
    fig = go.Figure()
    fig.update_layout(template="spectralab", height=height, margin=dict(l=60, r=20, t=34, b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"), hovermode="x unified",
                      separators=",." if S["settings"]["dec"] == "koma" else ".,")
    fig.update_xaxes(title_text=xt); fig.update_yaxes(title_text=yt)
    return fig


def add_rainbow(fig, lo, hi):
    """Pita spektrum tampak di dasar plot sebagai penanda daerah warna."""
    lo, hi = max(lo, 380), min(hi, 780)
    if hi - lo < 10:
        return
    for l0 in np.arange(lo, hi, 5):
        fig.add_shape(type="rect", xref="x", yref="paper", x0=l0, x1=min(l0 + 5, hi), y0=0, y1=0.028,
                      fillcolor=lam_color(l0 + 2.5), line_width=0, layer="below", opacity=0.9)


def model3d(height=560, lid=False, xray=True, compact=False, auto=False):
    init = {"lam": float(I["lam"]), "abs": float(I["last_A"]) if I["power"] == "ready" else 0.0,
            "status": "READY" if I["power"] == "ready" else "OFF", "solColor": "#d9f3f7" if I["solution"] == "blanko" else an()["warna"],
            "lampSwitch": float(I["lamp_switch"]), "xray": xray, "lid": lid, "compact": compact, "auto": auto}
    components.html(MODEL3D_HTML.replace("/*__INIT__*/null", json.dumps(init)), height=height, scrolling=False)


def csv_bytes(df):
    k = S["settings"]["dec"] == "koma"
    return df.to_csv(index=False, sep=";" if k else ",", decimal="," if k else ".")


# ---------------------------------------------------------------------------
# Halaman: Beranda
# ---------------------------------------------------------------------------
def page_home():
    reg = cal_reg()
    c1, c2 = st.columns([1.1, 1], gap="large")
    with c1:
        html("""<div class="hero"><div class="eb">Politeknik AKA Bogor · Analisis Spektrofotometri</div>
<h1>SPECTRA LAB 2.0</h1>
<p>Ruang kerja analitik UV-Vis: simulasi instrumen double-beam bergaya UV-1700 PharmaSpec, analisis spektrum, kalibrasi Lambert–Beer dengan uji Mandel, adisi standar, analisis multikomponen, kualifikasi kinerja alat, dan laporan PDF otomatis.</p>
<span class="pill sim"><span class="d"></span>MODE SIMULASI</span> <span class="pill"><span class="d"></span>190–1100 nm</span> <span class="pill"><span class="d"></span>Double-beam</span>
<svg width="300" height="90" viewBox="0 0 300 90"><defs><linearGradient id="rb" x1="0" x2="1"><stop offset="0" stop-color="#7b3fe4"/><stop offset=".3" stop-color="#2f5bff"/><stop offset=".5" stop-color="#22c55e"/><stop offset=".7" stop-color="#facc15"/><stop offset="1" stop-color="#ef4444"/></linearGradient></defs>
<path d="M0 80 C 40 78, 60 70, 80 50 S 110 8, 130 10 S 150 40, 165 38 S 185 16, 200 20 S 230 70, 300 78" fill="none" stroke="url(#rb)" stroke-width="3"/></svg></div>""")
        kpis([("Spektrum", str(len(S["spectra"])), "tersimpan di pustaka", "indigo"),
              ("Titik kalibrasi", str(len(cal_points())), S["cal_name"] or "belum ada", "violet"),
              ("R²", fmt(reg.r2, 4) if reg and reg.ok else "—", "linearitas kurva", "teal" if reg and reg.ok and reg.r2 >= CR["r2_min"] else "amber"),
              ("λ kerja", f"{fmt(S['cal_lambda'], 1)} nm" if S["cal_lambda"] else "—", "pengukuran kuantitatif", "amber")])
        f1, f2, f3 = st.columns(3)
        for col, (ic, h, b) in zip((f1, f2, f3), [("01 · INSTRUMEN", "Simulasi UV-1700", "Inisialisasi, baseline, auto zero, pemindaian, dan fotometri dengan derau dan sinar sesat realistis."),
                                                 ("02 · ANALISIS", "Kalibrasi & matriks", "Regresi dengan SK 95 %, uji Mandel, adisi standar, dan multikomponen Vierordt/CLS."),
                                                 ("03 · MUTU", "Validasi & PQ", "Presisi, Grubbs, Horwitz, recovery, serta kualifikasi kinerja Ph. Eur. 2.2.25.")]):
            col.markdown(f'<div class="feat"><div class="ic">{ic}</div><div class="h">{h}</div><div class="b">{b}</div></div>', unsafe_allow_html=True)
    with c2:
        model3d(height=430, xray=False, compact=True, auto=True)
        st.caption("Model 3D skematis mengikuti tata letak unit UV-1700 PharmaSpec di laboratorium; bukan gambar resmi pabrikan.")
    if S["spectra"]:
        sp = next((s for s in S["spectra"] if s["id"] == S["last_scan"]), S["spectra"][-1])
        fig = fig_base("Panjang gelombang (nm)", "Absorbansi", 300)
        fig.add_trace(go.Scatter(x=sp["lam"], y=sp["A"], mode="lines", name=sp["name"], line=dict(color=PAL[0], width=2.2), fill="tozeroy", fillcolor="rgba(79,70,229,.08)"))
        add_rainbow(fig, min(sp["lam"]), max(sp["lam"]))
        st.subheader("Spektrum terakhir"); show(fig)
    else:
        html('<div class="banner info">Coba <b>Praktikum Virtual 3D</b> untuk merasakan praktik langsung di meja laboratorium. Mulai dari <b>Panduan Praktikum</b> untuk alur langkah demi langkah, atau buka <b>Simulasi Instrumen</b> untuk menyalakan alat virtual.</div>')


# ---------------------------------------------------------------------------
# Halaman: Model 3D
# ---------------------------------------------------------------------------
def page_model3d():
    header("Instrumen", "Model 3D UV-1700", "Seret untuk memutar, roda/cubit untuk zoom, klik komponen untuk penjelasan. Gunakan Buka tutup, Tembus pandang, Terurai, dan Jalur cahaya. Warna berkas mengikuti λ instrumen virtual; kisi ventilasi kanan berpendar oranye saat lampu WI aktif.")
    model3d(height=640, xray=True)
    st.subheader("Fungsi komponen utama")
    cols = st.columns(2)
    for i, (k, v) in enumerate(COMPONENTS.items()):
        cols[i % 2].markdown(f'<div class="feat" style="margin-bottom:10px"><div class="h">{k}</div><div class="b">{v}</div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Halaman: Simulasi Instrumen
# ---------------------------------------------------------------------------
def page_instrument():
    header("Instrumen", "Simulasi Instrumen", "Spektrum dihasilkan model (pita Gauss, derau fotometrik σA = 0,434·σT/T, sinar sesat, serapan kuvet/pelarut) dengan absorptivitas mendekati literatur.")
    left, right = st.columns([1, 2], gap="large")
    lo, hi = min(I["start"], I["end"]), max(I["start"], I["end"])
    with left:
        with st.container(border=True):
            st.markdown("**1 · Daya & inisialisasi**")
            if I["power"] != "ready":
                if st.button("Nyalakan instrumen", type="primary"):
                    with st.status("Inisialisasi UV-1700…", expanded=True) as stt:
                        for chk in ["Memori & ROM", "Filter & celah", "Lampu D2 menyala", "Lampu WI menyala", "Titik nol λ (656,1 nm)", "Energi berkas"]:
                            st.write("✓ " + chk); time.sleep(0.2)
                        stt.update(label="Inisialisasi selesai", state="complete")
                    I["power"] = "ready"; log("Inisialisasi selesai. Alat nyata: panaskan lampu 15–30 menit."); st.rerun()
            else:
                html(pill("Lampu D₂ & WI menyala", "ok"))
                if st.button("Matikan instrumen"):
                    I.update(power="off", baseline=None, autozero=[]); log("Instrumen dimatikan."); st.rerun()
        with st.container(border=True):
            st.markdown("**2 · Larutan & kuvet**")
            keys = list(ANALYTES)
            new_an = st.selectbox("Analit (model simulasi)", keys, index=keys.index(I["analyte"]), format_func=lambda k: f"{ANALYTES[k]['nama']} · λref {ANALYTES[k]['lam_ref']} nm")
            if new_an != I["analyte"]:
                a = ANALYTES[new_an]; I.update(analyte=new_an, conc=float(a["std"][len(a["std"]) // 2]), lam=float(a["lam_ref"]))
                I["start"], I["end"] = (200.0, 400.0) if a["lam_ref"] < 320 else (float(max(190, round((a["lam_ref"] - 150) / 10) * 10)), float(min(1100, round((a["lam_ref"] + 170) / 10) * 10)))
                st.rerun()
            sols = ["blanko", "standar", "sampelX", "bebas"]
            I["solution"] = st.selectbox("Isi sel sampel", sols, index=sols.index(I["solution"]), format_func=lambda s: {"blanko": "Blanko", "standar": "Larutan standar", "sampelX": "Sampel X (tak diketahui)", "bebas": "Konsentrasi bebas"}[s])
            if I["solution"] == "standar":
                stds = [float(x) for x in an()["std"]]
                I["conc"] = st.selectbox("Konsentrasi standar (mg/L)", stds, index=stds.index(I["conc"]) if I["conc"] in stds else len(stds) // 2)
            elif I["solution"] == "bebas":
                I["custom"] = st.number_input("Konsentrasi (mg/L)", min_value=0.0, value=float(I["custom"]), step=0.1)
            bl = ["pelarut", "reagen", "udara"]
            I["blank"] = st.selectbox("Sel referensi (blanko)", bl, index=bl.index(I["blank"]), format_func=lambda b: {"pelarut": "Pelarut", "reagen": "Blanko reagen", "udara": "Udara"}[b])
            cv = list(CUVETTES)
            I["cuvette"] = st.selectbox("Kuvet", cv, index=cv.index(I["cuvette"]), format_func=lambda k: f"{CUVETTES[k]['nama']} (≥{CUVETTES[k]['cutoff']} nm)")
            I["path"] = st.select_slider("Tebal kuvet b (cm)", options=[0.5, 1.0, 2.0, 5.0], value=I["path"])
            I["matrix"] = st.slider("Kekeruhan / serapan matriks (A)", 0.0, 0.15, float(I["matrix"]), 0.005)
            st.caption(f"**{an()['rumus']}** dalam {an()['pelarut']}. {an()['catatan']}")
        with st.container(border=True):
            st.markdown("**3 · Parameter pemindaian**")
            c_a, c_b = st.columns(2)
            I["start"] = c_a.number_input("λ awal (nm)", 190.0, 1100.0, float(I["start"]), 10.0)
            I["end"] = c_b.number_input("λ akhir (nm)", 190.0, 1100.0, float(I["end"]), 10.0)
            iv = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
            I["interval"] = c_a.selectbox("Interval (nm)", iv, index=iv.index(I["interval"]))
            spd = ["cepat", "sedang", "lambat"]
            I["speed"] = c_b.selectbox("Kecepatan", spd, index=spd.index(I["speed"]))
            I["lamp_switch"] = st.number_input("Pergantian lampu D2→WI (nm)", 295.0, 364.0, float(I["lamp_switch"]), 0.1)
            lo, hi = min(I["start"], I["end"]), max(I["start"], I["end"])
            st.caption(f"Jumlah titik: {int(round((hi - lo) / I['interval'])) + 1}")
            if lo < CUVETTES[I["cuvette"]]["cutoff"]:
                st.error(f"Kuvet {CUVETTES[I['cuvette']]['nama']} menyerap kuat di bawah ±{CUVETTES[I['cuvette']]['cutoff']} nm; gunakan kuarsa untuk UV.")
            if I["blank"] == "udara":
                st.warning("Referensi udara tidak mengoreksi serapan kuvet dan pelarut.")
            if an()["reagent"] and I["blank"] != "reagen":
                st.warning("Metode ini memakai reagen; gunakan blanko reagen.")
            if I["baseline"] and not baseline_valid(lo, hi):
                st.warning("Rentang atau konfigurasi berubah sejak koreksi baseline; ulangi baseline.")
    with right:
        a1, a2 = st.columns([1, 1.4])
        ready = I["power"] == "ready"
        with a1:
            lcd(mode="SPEKTRUM")
            gw = st.number_input("GOTO WL (nm)", 190.0, 1100.0, float(I["lam"]), 0.5)
            k1, k2 = st.columns(2)
            if k1.button("GOTO WL"):
                I["lam"] = gw
                if ready:
                    I["last_A"] = read_at(gw)[0]
                st.rerun()
            if k2.button("AUTO ZERO", disabled=not ready):
                I["autozero"] = [{"lam": I["lam"], "cuvette": I["cuvette"], "blank": I["blank"]}]; I["last_A"] = 0.0
                log(f"AUTO ZERO pada {fmt(I['lam'], 1)} nm."); st.rerun()
        with a2:
            model3d(height=300, lid=True, xray=False, compact=True)
        b1, b2 = st.columns(2)
        if b1.button("Koreksi baseline", disabled=not ready):
            bar = st.progress(0.0, "Baseline: blanko di kedua sel…")
            for k in range(11):
                bar.progress(k / 10); time.sleep(0.05)
            I["baseline"] = {"start": lo, "end": hi, "cuvette": I["cuvette"], "blank": I["blank"], "analyte": I["analyte"], "time": f"{datetime.now():%H.%M}"}
            I["autozero"] = []; log(f"Baseline {lo:.0f}–{hi:.0f} nm tersimpan."); st.rerun()
        scan = b2.button("Mulai scan", type="primary", disabled=not ready)
        if not ready:
            st.caption("Nyalakan instrumen untuk mengaktifkan tombol.")
        chart = st.empty()
        if scan:
            ok = baseline_valid(lo, hi)
            sim = simulate_scan(lo, hi, I["interval"], analyte_id=I["analyte"], conc=sol_conc(), path=I["path"], cuvette=I["cuvette"],
                                blank=I["blank"], baseline_done=ok, speed=I["speed"], lamp_switch=I["lamp_switch"],
                                matrix=0 if I["solution"] == "blanko" else I["matrix"], rng=rng())
            n = len(sim["lam"])
            for k in range(1, 13):
                m = max(2, int(n * k / 12))
                fig = fig_base("Panjang gelombang λ (nm)", "Absorbansi (A)")
                fig.add_trace(go.Scatter(x=sim["lam"][:m], y=sim["A"][:m], mode="lines", name=sol_label(), line=dict(color=PAL[0], width=2)))
                fig.update_xaxes(range=[lo, hi]); add_rainbow(fig, lo, hi)
                chart.plotly_chart(fig, config=PCFG, **WIDE); time.sleep(0.06)
            S["counter"] += 1
            spec = {"id": S["counter"], "name": sol_label(), "source": "simulasi", "lam": sim["lam"].tolist(), "A": sim["A"].round(5).tolist(),
                    "time": f"{datetime.now():%d/%m %H.%M}", "params": {"analyte": I["analyte"], "baseline": ok, "interval": I["interval"], "speed": I["speed"], "start": lo, "end": hi},
                    "flags": {k: sim[k] for k in ("over", "low_energy", "uv_cutoff")}}
            S["spectra"].append(spec); S["last_scan"] = spec["id"]; S["spectra"] = S["spectra"][-40:]
            log(f"Scan {hi:.0f}→{lo:.0f} nm: {sol_label()}{'' if ok else ' (tanpa baseline berlaku)'}."); st.rerun()
        sp = next((s for s in S["spectra"] if s["id"] == S["last_scan"]), None)
        if sp:
            peaks = find_peaks(sp["lam"], sp["A"])
            fig = fig_base("Panjang gelombang λ (nm)", "Absorbansi (A)")
            fig.add_trace(go.Scatter(x=sp["lam"], y=sp["A"], mode="lines", name=sp["name"], line=dict(color=PAL[0], width=2.2), fill="tozeroy", fillcolor="rgba(79,70,229,.08)"))
            for i, p in enumerate(peaks[:3]):
                fig.add_vline(x=p["lambda"], line_dash="dash", line_color=PAL[4] if i == 0 else PAL[3],
                              annotation_text=("λmaks " if i == 0 else "") + fmt(p["lambda"], 1) + " nm", annotation_position="top")
            add_rainbow(fig, min(sp["lam"]), max(sp["lam"]))
            chart.plotly_chart(fig, config=PCFG, **WIDE)
            html(src_pill("simulasi"))
            r1, r2 = st.columns(2)
            with r1:
                if peaks:
                    p0 = peaks[0]
                    kpis([("λmaks", f"{fmt(p0['lambda'], 1)} nm", region(p0["lambda"]), "violet"), ("A puncak", fmt(p0["abs"], 4), f"%T {fmt(float(abs_to_t(p0['abs'])), 2)}", "indigo"), ("FWHM", f"{fmt(p0['fwhm'], 1)} nm", "lebar pita", "teal")])
                    if st.button(f"Gunakan {fmt(p0['lambda'], 1)} nm untuk kuantitatif"):
                        I["lam"] = round(p0["lambda"], 1); S["cal_lambda"] = I["lam"]; S["cal_analyte"] = sp["params"]["analyte"]
                        S["cal_name"] = S["cal_name"] or ANALYTES[sp["params"]["analyte"]]["nama"]; st.toast(f"λ kerja = {fmt(I['lam'], 1)} nm"); st.rerun()
                else:
                    st.info("Tidak ada puncak yang memenuhi prominensi 0,01 A.")
                if not sp["params"]["baseline"]:
                    st.warning("Diambil tanpa koreksi baseline yang berlaku (offset ±0,01–0,02 A).")
                if sp["flags"]["over"]:
                    st.error("Sebagian titik di luar rentang fotometrik −0,5…3,0 A.")
                if sp["flags"]["low_energy"] or sp["flags"]["uv_cutoff"]:
                    st.warning("Energi sangat rendah di sebagian rentang; data di daerah itu tidak dapat dipercaya.")
            with r2:
                lam_r = st.number_input("Baca pada λ (nm)", 190.0, 1100.0, float(round(peaks[0]["lambda"], 1)) if peaks else 500.0, 0.5)
                a_r = interp_y(sp["lam"], sp["A"], lam_r)
                html(f'<div class="eq">A({fmt(lam_r, 1)} nm) = {fmt(a_r, 4)} · %T = {fmt(float(abs_to_t(a_r)), 2)} %</div>')
        with st.container(border=True):
            st.markdown(f"**Kuantitatif pada λ kerja = {fmt(I['lam'], 1)} nm**")
            q1, q2 = st.columns(2)
            if q1.button("Ukur deret standar → Kurva Kalibrasi", disabled=not ready):
                a = an(); r = rng(); rows = []; corr = True
                for c in a["std"]:
                    A, corr, _ = read_at(I["lam"], conc=prep_conc(c, r))
                    rows.append({"C": float(c), "A": round(A, 4), "Pakai": True})
                set_calibration(pd.DataFrame(rows), "simulasi", name=a["nama"], unit="mg/L", lam=I["lam"], analyte=I["analyte"])
                log(f"Deret standar diukur pada {fmt(I['lam'], 1)} nm" + ("" if corr else " TANPA auto zero/baseline") + ".")
                st.toast("Deret standar terkirim ke Kurva Kalibrasi")
            if q2.button("Ukur Sampel X (triplo)", disabled=not ready):
                hx = hidden_conc(I["analyte"], S["seed"])
                reps = [read_at(I["lam"], conc=hx)[0] for _ in range(3)]
                S["samples"].append(new_sample(f"Sampel X {an()['rumus']}", "; ".join(fmt(v, 4) for v in reps), "simulasi", I["analyte"]))
                log("Sampel X diukur triplo."); st.toast("Sampel X ditambahkan ke Kurva Kalibrasi")
        with st.expander("Log instrumen"):
            st.code("\n".join(S["log"]) or "Belum ada aktivitas.", language=None)


# ---------------------------------------------------------------------------
# Halaman: Praktikum Virtual 3D
# ---------------------------------------------------------------------------
def parse_lab_result(txt: str) -> dict:
    """Format keluaran praktikum: #LAMBDA, #STANDAR (C;A per baris), #SAMPEL (A; A; A)."""
    sec, std, smp, lam = None, [], [], None
    for line in (txt or "").splitlines():
        l = line.strip()
        if not l:
            continue
        up = l.upper()
        if up.startswith("#LAMBDA"):
            lam = parse_number(l.split()[-1]); continue
        if up.startswith("#STANDAR"):
            sec = "s"; continue
        if up.startswith("#SAMPEL"):
            sec = "x"; continue
        if l.startswith("#"):
            continue
        if sec == "s":
            parts = [q for q in re.split(r"[;\t]", l) if q.strip()]
            if len(parts) >= 2:
                c_, a_ = parse_number(parts[0]), parse_number(parts[1])
                if not (math.isnan(c_) or math.isnan(a_)):
                    std.append((c_, a_))
        elif sec == "x":
            try:
                smp += parse_list(l)
            except ValueError:
                pass
    return {"lam": lam, "std": std, "smp": smp}


def page_lab():
    header("Instrumen", "Praktikum Virtual 3D",
           "Simulasi praktik langsung di meja laboratorium: sambungkan kabel daya ke UPS dan stop kontak, nyalakan UPS, komputer (CPU dan monitor), "
           "dan UV-1700; bilas, isi, dan lap kuvet dengan tisu; masukkan kuvet ke kompartemen; lakukan baseline, auto zero, scan, dan pengukuran. "
           "Hasil tampil di layar komputer, dan setiap kesalahan prosedur mengurangi skor.")
    keys = list(ANALYTES)
    akey = st.selectbox("Percobaan (analit)", keys, index=keys.index(S.get("lab_an", "fephen")), format_func=lambda k: f"{ANALYTES[k]['nama']} · λref {ANALYTES[k]['lam_ref']} nm", key="lab_an_sel")
    if akey != S.get("lab_an"):
        S["lab_an"] = akey
    a = ANALYTES[akey]
    html('<div class="banner info">Mengganti analit atau memuat ulang halaman akan mengulang praktikum dari awal. '
         'Salin data dari panel <b>Data untuk Kurva Kalibrasi</b> sebelum berpindah. Kotak <b>Langkah berikutnya</b> menuntun urutan kerja yang benar.</div>')
    init = {"seed": int(S["seed"]) % 2147483647, "hidden": hidden_conc(akey, S["seed"]), "dec": S["settings"]["dec"], "unit": "mg/L",
            "analyte": {"nama": a["nama"], "rumus": a["rumus"], "pelarut": a["pelarut"], "lam_ref": a["lam_ref"], "k": float(a["k"]), "warna": a["warna"],
                        "bands": [list(b) for b in a["bands"]], "std": [float(x) for x in a["std"]]}}
    components.html(LAB_HTML.replace("/*__LAB__*/null", json.dumps(init)), height=1320, scrolling=True)
    with st.container(border=True):
        st.markdown("**Impor hasil praktikum ke Kurva Kalibrasi**")
        txt = st.text_area("Tempel teks dari tombol Salin data", key="lab_paste", height=150,
                           placeholder="#LAMBDA 510,0\n#STANDAR\n0,500;0,1012\n…\n#SAMPEL\n0,4891; 0,4895; 0,4893")
        if st.button("Impor hasil praktikum", type="primary"):
            res = parse_lab_result(txt)
            if len(res["std"]) < 2:
                st.error("Data standar tidak ditemukan; pastikan teks diawali #STANDAR dan berisi pasangan C;A.")
            else:
                set_calibration(pd.DataFrame({"C": [c_ for c_, _ in res["std"]], "A": [a_ for _, a_ in res["std"]], "Pakai": True}), "simulasi",
                                name=a["nama"] + " (praktikum virtual)", unit="mg/L", lam=res["lam"], analyte=akey)
                if res["smp"]:
                    S["samples"] = [s_ for s_ in S["samples"] if s_.get("src") != "praktikum"] + [
                        new_sample(f"Sampel X {a['rumus']} (praktikum)", "; ".join(fmt(v, 4) for v in res["smp"]), "simulasi", akey)]
                log(f"Hasil praktikum virtual diimpor: {len(res['std'])} standar, {len(res['smp'])} pembacaan sampel.")
                st.success(f"{len(res['std'])} titik standar dan {len(res['smp'])} pembacaan sampel dikirim. Buka halaman Kurva Kalibrasi untuk evaluasi.")


# ---------------------------------------------------------------------------
# Halaman: Fotometri
# ---------------------------------------------------------------------------
def page_photometric():
    header("Instrumen", "Mode Fotometri", "Pembacaan titik tunggal pada hingga tiga panjang gelombang dengan ulangan, setara mode PHOTOMETRIC pada panel UV-1700.")
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        html(f"Larutan aktif: <b>{sol_label()}</b> · kuvet {CUVETTES[I['cuvette']]['nama']} · referensi {I['blank']}<br>{pill('SIMULASI', 'sim')}")
        st.caption("Ubah larutan, kuvet, dan blanko di halaman Simulasi Instrumen.")
        lams_txt = st.text_input("Panjang gelombang (maks 3, pisahkan ;)", fmt(I["lam"], 1))
        reps = st.select_slider("Ulangan per λ", [1, 2, 3, 5, 7, 10], value=3)
        ready = I["power"] == "ready"
        try:
            lams = parse_list(lams_txt)[:3]
        except ValueError:
            lams = []; st.error("Isi λ berupa angka, pisahkan dengan ;")
        a, b = st.columns(2)
        if a.button("Auto zero (blanko)", disabled=not ready or not lams):
            I["autozero"] = [{"lam": l, "cuvette": I["cuvette"], "blank": I["blank"]} for l in lams]; I["last_A"] = 0.0; st.toast("Auto zero selesai")
        if b.button("Ukur", type="primary", disabled=not ready or not lams):
            for l in lams:
                for _ in range(reps):
                    A, corr, _o = read_at(l)
                    S["photo"].append({"Waktu": f"{datetime.now():%H.%M.%S}", "Larutan": sol_label(), "λ (nm)": l, "A": round(A, 4), "%T": round(float(abs_to_t(A)), 2), "Terkoreksi": corr})
                    I["lam"], I["last_A"] = l, A
        if not ready:
            st.caption("Nyalakan instrumen di Simulasi Instrumen.")
        lcd(mode="PHOTOMETRIC")
        st.markdown("**Konversi A ↔ %T**")
        cA = st.number_input("Absorbansi", -0.5, 4.0, 0.5, 0.01)
        html(f'<div class="eq">%T = 100 × 10<sup>−A</sup> = {fmt(float(abs_to_t(cA)), 3)} %</div>')
    with right:
        if not S["photo"]:
            st.info("Belum ada pembacaan. Lakukan auto zero dengan blanko, lalu tekan Ukur."); return
        df = pd.DataFrame(S["photo"])
        if not df["Terkoreksi"].all():
            st.warning("Sebagian pembacaan tanpa auto zero/baseline pada λ tersebut (mengandung offset instrumen).")
        grp = df.groupby(["Larutan", "λ (nm)"])["A"].agg(n="count", rerata="mean", s="std").reset_index()
        grp["RSD %"] = (grp["s"] / grp["rerata"] * 100).round(2); grp["%T rerata"] = abs_to_t(grp["rerata"]).round(2)
        table(grp.round(5))
        table(df.iloc[::-1], height=320)
        c1, c2 = st.columns(2)
        c1.download_button("Unduh CSV", csv_bytes(df), "fotometri-spectralab.csv", "text/csv")
        if c2.button("Hapus semua"):
            S["photo"] = []; st.rerun()


# ---------------------------------------------------------------------------
# Halaman: Kinetika
# ---------------------------------------------------------------------------
def kin_result():
    k = S["kin"]; d = k["data"]
    if len(d) < 3:
        return None
    t = np.array([p[0] for p in d]); A = np.array([p[1] for p in d])
    ainf = num_in(k["ainf_fit"]) if k["ainf_fit"] else (k["Ainf"] if k["source"] == "simulasi" else A[-1])
    return {"t": t, "A": A, "ir": initial_rate(t, A, k["n_init"]), "fo": first_order_fit(t, A, ainf), "avg": (A[-1] - A[0]) / (t[-1] - t[0]), "ainf": ainf}


def page_kinetics():
    header("Instrumen", "Kinetika", "Mode time-scan: A(t) = A∞ + (A₀ − A∞)·e<sup>−kt</sup> (orde satu semu, model generik). Untuk data nyata pilih Data pengguna.")
    k = S["kin"]
    left, right = st.columns([1, 1.7], gap="large")
    with left:
        src = st.radio("Sumber data", ["simulasi", "pengguna"], index=0 if k["source"] == "simulasi" else 1, horizontal=True, format_func=lambda s: "Simulasi" if s == "simulasi" else "Data pengguna")
        if src != k["source"]:
            k["source"] = src; k["data"] = []
        run = False
        speed = 30
        if src == "simulasi":
            pre = st.selectbox("Skenario", ["Pemudaran warna (orde satu semu)", "Pembentukan kompleks berwarna", "Atur sendiri"])
            if pre.startswith("Pemudaran"): k.update(A0=1.2, Ainf=0.02, k=0.012, lam=590.0)
            elif pre.startswith("Pembentukan"): k.update(A0=0.02, Ainf=0.85, k=0.02, lam=510.0)
            custom = pre.startswith("Atur")
            a, b = st.columns(2)
            k["A0"] = a.number_input("A₀", 0.0, 3.0, float(k["A0"]), 0.01, disabled=not custom)
            k["Ainf"] = b.number_input("A∞", 0.0, 3.0, float(k["Ainf"]), 0.01, disabled=not custom)
            k["k"] = a.number_input("k (s⁻¹)", 0.0001, 1.0, float(k["k"]), 0.001, format="%.4f", disabled=not custom)
            k["lam"] = b.number_input("λ (nm)", 190.0, 1100.0, float(k["lam"]), 1.0)
            k["dur"] = a.number_input("Durasi (s)", 10.0, 3600.0, float(k["dur"]), 10.0)
            k["dt"] = b.number_input("Interval (s)", 0.5, 120.0, float(k["dt"]), 0.5)
            k["noise"] = a.number_input("Derau σ (A)", 0.0, 0.05, float(k["noise"]), 0.001, format="%.3f")
            speed = b.selectbox("Kecepatan tayang", [10, 30, 100], index=1, format_func=lambda x: f"{x}×")
            run = st.button("Mulai", type="primary", disabled=I["power"] != "ready")
            if I["power"] != "ready":
                st.caption("Nyalakan instrumen di Simulasi Instrumen.")
            if st.button("Atur ulang"):
                k["data"] = []; st.rerun()
        else:
            txt = st.text_area("Tempel data: waktu (s) dan absorbansi per baris", height=180, placeholder="0;0,812\n30;0,744\n60;0,688")
            if st.button("Muat data") and txt.strip():
                res = parse_csv(txt)
                if res["ok"]:
                    d = sorted([(t, a_) for t, a_ in zip(res["cols"][0], res["cols"][1]) if not (math.isnan(t) or math.isnan(a_))])
                    k["data"] = d if len(d) >= 3 else []; st.toast(f"{len(d)} titik dimuat")
                else:
                    st.error(res["reason"])
        k["n_init"] = int(st.number_input("Titik untuk laju awal", 3, 50, int(k["n_init"])))
        k["ainf_fit"] = st.text_input("A∞ untuk ln-plot (kosong = otomatis)", k["ainf_fit"])
    with right:
        ph = st.empty()

        def draw():
            d = k["data"]; fig = fig_base("Waktu (s)", "Absorbansi", 360)
            if d:
                fig.add_trace(go.Scatter(x=[p[0] for p in d], y=[p[1] for p in d], mode="lines+markers", name="A(t)", line=dict(color=PAL[0]), marker=dict(size=5)))
            if k["source"] == "simulasi":
                fig.update_xaxes(range=[0, k["dur"]])
            ph.plotly_chart(fig, config=PCFG, **WIDE)
        if run:
            k["data"] = []; gen = np.random.default_rng(S["seed"] + S["counter"]); n = int(k["dur"] // k["dt"]) + 1
            for i in range(n):
                t = i * k["dt"]; k["data"].append((t, round(float(kinetic_model(k["A0"], k["Ainf"], k["k"], t) + gen.normal(0, k["noise"])), 4)))
                if i % max(1, n // 40) == 0 or i == n - 1:
                    draw(); time.sleep(min(0.12, k["dt"] / speed))
            log(f"Kinetika selesai: {n} titik pada {fmt(k['lam'], 1)} nm.")
        draw()
        r = kin_result()
        if not r:
            st.info("Minimal 3 titik data diperlukan."); return
        html(src_pill(k["source"]))
        items = []
        if r["ir"]: items.append(("Laju awal dA/dt", sig(r["ir"]["rate"], 3) + " s⁻¹", f"{r['ir']['n']} titik · R² {fmt(r['ir']['r2'], 4)}", "indigo"))
        items.append(("Laju rata-rata", sig(r["avg"], 3) + " s⁻¹", "ΔA/Δt", "violet"))
        if r["fo"]:
            items += [("k teramati", sig(r["fo"]["k"], 3) + " s⁻¹", f"A∞ = {fmt(r['ainf'], 4)}", "teal"), ("t½", sig(r["fo"]["half_life"], 3) + " s", f"R² ln-plot {fmt(r['fo']['r2'], 4)}", "amber")]
        kpis(items)
        if r["fo"]:
            if r["fo"]["r2"] < 0.98:
                st.warning("ln-plot kurang linear; data mungkin tidak orde satu atau A∞ belum tepat.")
            fo = r["fo"]; fig = fig_base("Waktu (s)", "ln|A − A∞|", 260); fig.update_layout(hovermode="closest")
            fig.add_trace(go.Scatter(x=fo["t"], y=fo["lny"], mode="markers", name="data", marker=dict(color=PAL[2], size=6)))
            fig.add_trace(go.Scatter(x=fo["t"], y=fo["lnA0"] - fo["k"] * fo["t"], mode="lines", name="regresi", line=dict(color=PAL[1])))
            show(fig)
        st.download_button("Unduh CSV", csv_bytes(pd.DataFrame(k["data"], columns=["Waktu (s)", "Absorbansi"])), "kinetika-spectralab.csv", "text/csv")


# ---------------------------------------------------------------------------
# Halaman: Analisis Spektrum
# ---------------------------------------------------------------------------
def page_spectrum():
    header("Analisis", "Analisis Spektrum", "Pustaka spektrum: tumpang-tindih, pemulusan Savitzky–Golay, normalisasi, turunan pertama, deteksi puncak, dan ekspor CSV.")
    left, right = st.columns([1, 2.2], gap="large")
    act, proc, ymode, prom, minh = [], "Asli", "Absorbansi", 0.01, 0.02
    with left:
        up = st.file_uploader("Impor CSV (kolom 1 = λ nm, berikutnya = A)", type=["csv", "txt", "tsv"])
        if up is not None and st.button("Tambahkan ke pustaka"):
            res = parse_csv(up.getvalue().decode("utf-8", errors="replace"))
            if not res["ok"]:
                st.error(res["reason"])
            else:
                for c in range(1, len(res["cols"])):
                    lam, A = res["cols"][0], res["cols"][c]; m = ~(np.isnan(lam) | np.isnan(A))
                    if m.sum() < 5:
                        continue
                    o = np.argsort(lam[m]); S["counter"] += 1
                    nm = res["header"][c] if res["header"] and c < len(res["header"]) else f"{up.name} #{c}"
                    S["spectra"].append({"id": S["counter"], "name": nm, "source": "impor", "lam": lam[m][o].tolist(), "A": A[m][o].tolist(), "time": f"{datetime.now():%d/%m %H.%M}", "params": {"baseline": True}, "flags": {}})
                st.toast("Spektrum diimpor"); st.rerun()
        if S["settings"]["mode"] == "latihan" and st.button("Muat 3 spektrum contoh KMnO₄"):
            for k, c in enumerate([10, 30, 50]):
                sim = simulate_scan(400, 700, 1, analyte_id="kmno4", conc=c, rng=np.random.default_rng(100 + k)); S["counter"] += 1
                S["spectra"].append({"id": S["counter"], "name": f"Contoh KMnO₄ {c} mg/L", "source": "contoh", "lam": sim["lam"][::-1].tolist(), "A": sim["A"][::-1].tolist(), "time": "", "params": {"analyte": "kmno4", "baseline": True}, "flags": {}})
            st.rerun()
        if not S["spectra"]:
            st.info("Pustaka kosong. Pindai di Simulasi Instrumen atau impor CSV.")
            return
        names = {s["id"]: f"{s['name']} [{s['source']}]" for s in S["spectra"]}
        act = st.multiselect("Spektrum ditampilkan", list(names), default=[S["spectra"][-1]["id"]], format_func=names.get)
        ymode = st.radio("Sumbu Y", ["Absorbansi", "%T"], horizontal=True)
        proc = st.radio("Pengolahan", ["Asli", "Dihaluskan (SG-7)", "Normalisasi", "Turunan-1"], horizontal=True)
        prom = st.number_input("Prominensi min (A)", 0.0, 1.0, 0.01, 0.005, format="%.3f")
        minh = st.number_input("Tinggi min (A)", 0.0, 3.0, 0.02, 0.01, format="%.3f")
        if st.button("Hapus spektrum terpilih", disabled=not act):
            S["spectra"] = [s for s in S["spectra"] if s["id"] not in act]; st.rerun()
    with right:
        sel = [s for s in S["spectra"] if s["id"] in act]
        fig = fig_base("Panjang gelombang λ (nm)", {"Turunan-1": "dA/dλ (nm⁻¹)", "Normalisasi": "A / Amaks"}.get(proc, "Transmitansi (%T)" if ymode == "%T" else "Absorbansi (A)"), 440)
        lo_all, hi_all = 1100, 190
        for i, s in enumerate(sel):
            x = np.asarray(s["lam"]); y = np.asarray(s["A"])
            lo_all, hi_all = min(lo_all, x.min()), max(hi_all, x.max())
            if proc.startswith("Dihaluskan"): y = smooth(y, 7)
            if proc == "Normalisasi": y = y / np.nanmax(y)
            if proc == "Turunan-1": y = np.gradient(smooth(np.asarray(s["A"]), 9), x)
            elif ymode == "%T" and proc != "Normalisasi": y = abs_to_t(y)
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=s["name"], line=dict(color=PAL[i % len(PAL)], width=2)))
        if sel:
            add_rainbow(fig, lo_all, hi_all)
        show(fig)
        st.caption("Seret untuk memperbesar, klik dua kali untuk kembali; arahkan kursor untuk membaca nilai.")
        if sel:
            lam_r = st.number_input("Baca nilai pada λ (nm)", 190.0, 1100.0, 525.0, 0.5)
            table(pd.DataFrame([{"Spektrum": s["name"], "A": interp_y(s["lam"], s["A"], lam_r), "%T": float(abs_to_t(interp_y(s["lam"], s["A"], lam_r)))} for s in sel]))
            tgt = sel[-1]; pk = find_peaks(tgt["lam"], tgt["A"], prom, minh)
            st.markdown(f"**Puncak terdeteksi — {tgt['name']}**")
            table(pd.DataFrame([{"λ (nm)": round(p["lambda"], 1), "A": round(p["abs"], 4), "%T": round(float(abs_to_t(p["abs"])), 2), "Prominensi": round(p["prominence"], 4), "FWHM (nm)": round(p["fwhm"], 1)} for p in pk[:8]]))
            same = len({len(s["lam"]) for s in sel}) == 1 and all(np.allclose(s["lam"], sel[0]["lam"]) for s in sel)
            if same:
                df = pd.DataFrame({"Panjang gelombang (nm)": sel[0]["lam"], **{s["name"] + " (A)": s["A"] for s in sel}})
            else:
                df = pd.concat([pd.DataFrame({"Spektrum": s["name"], "Panjang gelombang (nm)": s["lam"], "Absorbansi": s["A"]}) for s in sel])
            with st.expander("Tabel data"):
                table(df, height=300)
            st.download_button("Unduh CSV", csv_bytes(df), "spektrum-spectralab.csv", "text/csv")


# ---------------------------------------------------------------------------
# Halaman: Kurva Kalibrasi
# ---------------------------------------------------------------------------
def page_calibration():
    header("Analisis", "Kurva Kalibrasi", "Regresi kuadrat terkecil y = ax + b dengan pita kepercayaan 95 %, uji Mandel, LOD/LOQ, dan prediksi invers sampel (Miller & Miller).")
    c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1.2])
    S["cal_name"] = c1.text_input("Analit / metode", S["cal_name"], placeholder="mis. Fe dengan 1,10-fenantrolin")
    units = ["mg/L", "µg/mL", "ppm", "µg/L", "mmol/L", "mol/L", "%b/v"]
    S["cal_unit"] = c2.selectbox("Satuan", units, index=units.index(S["cal_unit"]))
    lam_txt = c3.text_input("λ ukur (nm)", fmt(S["cal_lambda"], 1) if S["cal_lambda"] else "")
    S["cal_lambda"] = num_in(lam_txt, None) if lam_txt else None
    S["cal_zero"] = c4.checkbox("Paksa melalui nol", S["cal_zero"], help="Hanya dibenarkan bila intersep tidak berbeda nyata dari nol.")
    L, Rr = st.columns([1, 1.6], gap="large")
    with L:
        html("<b>Larutan standar</b> " + src_pill(S["cal_src"]))
        edited = st.data_editor(S["cal_base"], key=f"cal_{S['cal_ver']}", num_rows="dynamic", hide_index=True, **WIDE,
                                column_config={"C": st.column_config.NumberColumn(f"C ({S['cal_unit']})", format="%.4f"),
                                               "A": st.column_config.NumberColumn("Absorbansi", format="%.4f"),
                                               "Pakai": st.column_config.CheckboxColumn("Pakai", default=True)})
        S["cal_cur"] = edited
        if not df_same(edited, S["cal_base"]) and S["cal_src"] in ("simulasi", "contoh"):
            S["cal_src"] = "campuran"
        paste = st.text_area("Tempel dua kolom dari Excel (C lalu A)", height=90, placeholder="0,5\t0,101\n1,0\t0,198")
        b1, b2, b3 = st.columns(3)
        if b1.button("Masukkan") and paste.strip():
            res = parse_csv(paste)
            if res["ok"]:
                add = pd.DataFrame({"C": res["cols"][0], "A": res["cols"][1], "Pakai": True}).dropna()
                set_calibration(pd.concat([cal_points(), add], ignore_index=True), "pengguna"); st.rerun()
            else:
                st.error(res["reason"])
        if S["settings"]["mode"] == "latihan" and b2.button("Data contoh"):
            set_calibration(pd.DataFrame({"C": [0.5, 1, 2, 3, 4, 5], "A": [0.102, 0.197, 0.401, 0.598, 0.792, 1.001], "Pakai": True}), "contoh",
                            name="Fe(II)–1,10-fenantrolin (data contoh)", unit="mg/L", lam=510.0, analyte="fephen")
            S["samples"] = [dict(new_sample("Air sumur A (contoh)", "0,452; 0,448; 0,455", "contoh"), steps=[{"V pipet (mL)": 10.0, "V labu (mL)": 50.0}])]
            st.rerun()
        if b3.button("Kosongkan"):
            set_calibration(_empty_cal(), "pengguna"); st.rerun()
        up = st.file_uploader("Impor CSV standar (C; A)", type=["csv", "txt"], key="calcsv")
        if up is not None and st.button("Impor"):
            res = parse_csv(up.getvalue().decode("utf-8", errors="replace"))
            if res["ok"]:
                set_calibration(pd.DataFrame({"C": res["cols"][0], "A": res["cols"][1], "Pakai": True}).dropna(), "impor"); st.rerun()
            else:
                st.error(res["reason"])
    reg = cal_reg()
    pts = cal_points()
    with Rr:
        tabs = st.tabs(["Kurva", "Residual", "Uji Mandel"])
        with tabs[0]:
            fig = fig_base(f"Konsentrasi ({S['cal_unit']})", "Absorbansi (A)", 380)
            fig.update_layout(hovermode="closest")
            if reg and reg.ok:
                xs = np.linspace(min(0, reg.xmin), reg.xmax * 1.05, 80); yh = reg.slope * xs + reg.intercept; bd = reg.band(xs)
                fig.add_trace(go.Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[yh + bd, (yh - bd)[::-1]], fill="toself", fillcolor="rgba(124,58,237,.12)", line=dict(width=0), name="SK 95 % garis", hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=xs, y=yh, mode="lines", name="Regresi", line=dict(color=PAL[4], width=2.2)))
            fig.add_trace(go.Scatter(x=pts["C"], y=pts["A"], mode="markers", name="Standar", marker=dict(color=PAL[0], size=11, line=dict(color="white", width=1.5))))
            if reg and reg.ok:
                for s in S["samples"]:
                    r = sample_res(s, reg)
                    if r:
                        fig.add_trace(go.Scatter(x=[r["c_meas"]], y=[r["A_mean"]], mode="markers+text", name=s["name"], text=[s["name"]], textposition="top right",
                                                 error_x=dict(type="data", array=[r["ci"]], color=PAL[2]), marker=dict(symbol="diamond", size=13, color=PAL[2])))
            show(fig)
        with tabs[1]:
            if reg and reg.ok:
                fig = fig_base(f"Konsentrasi ({S['cal_unit']})", "Residual terstandar", 300); fig.update_layout(hovermode="closest")
                fig.add_trace(go.Bar(x=pts["C"], y=reg.std_residuals, marker_color=[PAL[2] if abs(v) > CR["std_res_max"] else PAL[0] for v in reg.std_residuals], name="e/s"))
                for sgn in (1, -1):
                    fig.add_hline(y=sgn * CR["std_res_max"], line_dash="dot", line_color="#8a91b5")
                show(fig)
        md = mandel(pts["C"].values, pts["A"].values) if reg and reg.ok and not S["cal_zero"] else None
        with tabs[2]:
            if md:
                xs = np.linspace(reg.xmin, reg.xmax, 80)
                fig = fig_base(f"Konsentrasi ({S['cal_unit']})", "Absorbansi (A)", 300); fig.update_layout(hovermode="closest")
                fig.add_trace(go.Scatter(x=pts["C"], y=pts["A"], mode="markers", name="Standar", marker=dict(color=PAL[0], size=9)))
                fig.add_trace(go.Scatter(x=xs, y=reg.slope * xs + reg.intercept, mode="lines", name="Linear", line=dict(color=PAL[4])))
                fig.add_trace(go.Scatter(x=xs, y=np.polyval(md["quad"], xs), mode="lines", name="Kuadratik", line=dict(color=PAL[3], dash="dash")))
                show(fig)
                kpis([("F Mandel", fmt(md["F"], 2), "DS² / s²(kuadratik)", "indigo"), ("F kritis", fmt(md["Fcrit"], 2), f"F(1; {md['n'] - 3}; 0,99)", "violet"),
                      ("s linear", sig(md["s_lin"], 3), "s_y/x", "teal"), ("s kuadratik", sig(md["s_quad"], 3), "", "amber")])
                verdict(md["linear_ok"], "model linear memadai" if md["linear_ok"] else "kelengkungan signifikan")
            else:
                st.info("Uji Mandel memerlukan ≥ 5 titik dan model dengan intersep.")
        if not reg or not reg.ok:
            st.info(reg.reason if reg else "Minimal 3 titik valid diperlukan untuk regresi.")
        else:
            html(f'<div class="eq">{reg.equation()}   ·   R² = {fmt(reg.r2, 5)}</div>')
            kpis([("R²", fmt(reg.r2, 5), "memenuhi" if reg.r2 >= CR["r2_min"] else "di bawah kriteria", "teal" if reg.r2 >= CR["r2_min"] else "rose"),
                  ("Kemiringan a", sig(reg.slope), f"± {sig(reg.ci_slope, 2)} (SK 95 %)", "indigo"),
                  ("Intersep b", "0 (dipaksa)" if reg.through_zero else sig(reg.intercept, 3), "" if reg.through_zero else f"± {sig(reg.ci_intercept, 2)}", "violet"),
                  ("s y/x", sig(reg.sxy, 3), f"n = {reg.n}", "amber"),
                  ("LOD", f"{sig(reg.lod, 3)}", S["cal_unit"], "teal"), ("LOQ", f"{sig(reg.loq, 3)}", S["cal_unit"], "teal")])
            with st.container(border=True):
                st.markdown("**Interpretasi otomatis**")
                for p in calib_interpretation(reg, S["cal_unit"], CR["r2_min"], CR["std_res_max"], S["settings"]["dec"], md):
                    st.markdown("- " + p)
    page_samples(reg)


def page_samples(reg):
    st.subheader("Analisis kuantitatif sampel")
    t1, t2 = st.columns(2)
    if t1.button("+ Tambah sampel"):
        S["samples"].append(new_sample(f"Sampel {len(S['samples']) + 1}")); st.rerun()
    if t2.button("Ukur Sampel X (simulator)", disabled=I["power"] != "ready"):
        akey = S["cal_analyte"] or I["analyte"]; lam = S["cal_lambda"] or ANALYTES[akey]["lam_ref"]
        reps = [read_at(lam, conc=hidden_conc(akey, S["seed"]), analyte=akey)[0] for _ in range(3)]
        S["samples"].append(new_sample(f"Sampel X {ANALYTES[akey]['rumus']}", "; ".join(fmt(v, 4) for v in reps), "simulasi", akey)); st.rerun()
    for idx, s in enumerate(list(S["samples"])):
        with st.expander(f"{s['name']}", expanded=True):
            a, b = st.columns([1, 1.1], gap="large")
            with a:
                s["name"] = st.text_input("Nama sampel", s["name"], key=f"n{s['id']}")
                s["reps"] = st.text_input("Absorbansi ulangan (pisahkan ;)", s["reps"], key=f"r{s['id']}", placeholder="0,412; 0,415; 0,409")
                s["dil_mode"] = st.radio("Pengenceran", ["langkah", "manual"], index=0 if s["dil_mode"] == "langkah" else 1, horizontal=True, key=f"dm{s['id']}",
                                         format_func=lambda x: "Langkah pengenceran" if x == "langkah" else "FP manual")
                if s["dil_mode"] == "manual":
                    s["fp_manual"] = st.number_input("Faktor pengenceran", min_value=0.0001, value=float(s["fp_manual"]), key=f"fp{s['id']}")
                else:
                    st.caption("Tiap baris: V pipet larutan dimasukkan ke labu V labu. FP = Π(V labu / V pipet).")
                    steps = st.data_editor(pd.DataFrame(s["steps"], columns=["V pipet (mL)", "V labu (mL)"]), num_rows="dynamic", hide_index=True, key=f"st{s['id']}", **WIDE)
                    s["steps"] = steps.dropna().to_dict("records")
                m1, m2 = st.columns(2)
                s["mass"] = m1.text_input("Massa sampel padat (g)", s["mass"], key=f"m{s['id']}")
                s["vol"] = m2.text_input("Volume larutan awal (mL)", s["vol"], key=f"v{s['id']}")
                if st.button("Hapus sampel", key=f"del{s['id']}"):
                    S["samples"].pop(idx); st.rerun()
            with b:
                html(src_pill(s["src"]))
                r = sample_res(s, reg)
                if not r:
                    st.info("Isi absorbansi (angka dipisah ;) dan pastikan regresi valid.")
                    continue
                u = S["cal_unit"]
                kpis([("Ā (n = %d)" % r["n"], fmt(r["A_mean"], 4), f"RSD {fmt(r['rsd'], 2)} %" if r["n"] > 1 else "satu pembacaan", "indigo"),
                      ("C terukur", sig(r["c_meas"]), f"± {sig(r['ci'], 2)} {u}", "violet"), ("FP", sig(r["fp"]), "pengenceran", "amber")])
                html(f'<div class="eq">C sampel asal = {sig(r["c_final"])} ± {sig(r["ci_final"], 2)} {u}' + (f'<br>Kadar sampel padat = {sig(r["kadar"])} mg/kg' if r["mass_ok"] else "") + "</div>")
                for fl in r["flags"]:
                    st.warning(fl)
                if not r["flags"]:
                    st.success("Absorbansi sampel di dalam rentang kalibrasi dan daerah kerja.")
                with st.expander("Langkah perhitungan"):
                    st.latex(r"x = \frac{\bar y - b}{a} = \frac{%s - (%s)}{%s} = %s" % (fmt(r["A_mean"], 4), sig(reg.intercept, 4), sig(reg.slope, 5), sig(r["c_meas"], 5)))
                    st.latex(r"s_{x_0} = \frac{s_{y/x}}{a}\sqrt{\frac{1}{m}+\frac{1}{n}+\frac{(\bar y_0-\bar y)^2}{a^2\sum(x_i-\bar x)^2}} = %s" % sig(r["sx0"], 3))
                    st.latex(r"C_{asal} = x \times FP = %s \times %s = %s\ \mathrm{%s}" % (sig(r["c_meas"], 5), sig(r["fp"], 4), sig(r["c_final"], 5), u.replace("µ", r"\mu ").replace("%", r"\%")))
                if s.get("hidden") and S["settings"]["mode"] == "latihan" and st.button("Ungkap nilai sebenarnya (cek diri)", key=f"rv{s['id']}"):
                    tv = hidden_conc(s["hidden"], S["seed"])
                    st.info(f"Nilai model: {sig(tv, 3)} mg/L · perolehan kembali {fmt(r['c_final'] / tv * 100, 1)} %")


# ---------------------------------------------------------------------------
# Halaman: Adisi Standar
# ---------------------------------------------------------------------------
def page_stdadd():
    header("Analisis", "Adisi Standar", "Kalibrasi di dalam matriks sampel: aliquot sampel ditambah standar bertingkat dan diencerkan ke volume sama. C_x = b/a diperoleh dari ekstrapolasi ke A = 0, sehingga efek matriks multiplikatif (perubahan kemiringan) terkompensasi.")
    sa = S["sa"]
    L, Rr = st.columns([1, 1.6], gap="large")
    with L:
        with st.container(border=True):
            st.markdown("**Simulasi seri adisi**")
            keys = list(ANALYTES)
            akey = st.selectbox("Analit", keys, index=keys.index(sa["analyte"] or I["analyte"]), format_func=lambda k: ANALYTES[k]["nama"], key="sa_an")
            sa["sens"] = st.slider("Kepekaan dalam matriks relatif terhadap pelarut (×)", 0.60, 1.20, float(sa["sens"]), 0.01,
                                   help="< 1: supresi (mis. kompleksasi ion pengganggu); > 1: peningkatan.")
            if st.button("Ukur seri adisi (5 tingkat)", type="primary"):
                a = ANALYTES[akey]; c0 = hidden_conc(akey, S["seed"]) / 2; r = rng(); ka = float(absorptivity(akey, a["lam_ref"]))
                adds = [0.0] + [float(x) / 2 for x in a["std"][:4]]
                rows = [{"C ditambahkan": x, "A": round(float((c0 + prep_conc(x, r) if x else c0) * ka * sa["sens"] + r.normal(0, 0.0025)), 4)} for x in adds]
                sa.update(df=pd.DataFrame(rows), src="simulasi", analyte=akey, unit="mg/L", truth=c0, ka=ka); sa["ver"] += 1
                log(f"Seri adisi standar {a['rumus']} diukur ({len(adds)} tingkat)."); st.rerun()
        html("<b>Data seri adisi</b> " + src_pill(sa["src"]))
        ed = st.data_editor(sa["df"], key=f"sa_{sa['ver']}", num_rows="dynamic", hide_index=True, **WIDE,
                            column_config={"C ditambahkan": st.column_config.NumberColumn(f"C ditambahkan ({sa['unit']})", format="%.4f"),
                                           "A": st.column_config.NumberColumn("Absorbansi", format="%.4f")})
        if not df_same(ed, sa["df"]) and sa["src"] == "simulasi":
            sa["src"] = "campuran"
        sa["cur"] = ed
        sa["fp"] = st.number_input("Faktor pengenceran aliquot sampel (V labu / V aliquot)", 0.0001, 1e6, float(sa["fp"]), 1.0)
        if st.button("Kosongkan data adisi"):
            sa.update(df=pd.DataFrame({"C ditambahkan": pd.Series(dtype=float), "A": pd.Series(dtype=float)}), src="pengguna"); sa.pop("truth", None); sa["ver"] += 1; st.rerun()
    with Rr:
        d = ed.copy()
        d = d[pd.to_numeric(d["C ditambahkan"], errors="coerce").notna() & pd.to_numeric(d["A"], errors="coerce").notna()]
        x = d["C ditambahkan"].astype(float).values; y = d["A"].astype(float).values
        res = std_addition(x, y) if len(x) >= 3 else None
        if not res:
            st.info("Masukkan ≥ 3 titik (termasuk sampel tanpa adisi) atau jalankan simulasi."); return
        rg = res["reg"]
        fig = fig_base(f"C ditambahkan ({sa['unit']})", "Absorbansi (A)", 380); fig.update_layout(hovermode="closest")
        xs = np.linspace(-res["cx"] * 1.15, x.max() * 1.05, 80)
        fig.add_trace(go.Scatter(x=xs, y=rg.slope * xs + rg.intercept, mode="lines", name="Regresi (diekstrapolasi)", line=dict(color=PAL[4], width=2.2)))
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Seri adisi", marker=dict(color=PAL[0], size=11, line=dict(color="white", width=1.5))))
        fig.add_trace(go.Scatter(x=[-res["cx"]], y=[0], mode="markers+text", name="−C_x", text=[f"−{sig(res['cx'], 3)}"], textposition="bottom center",
                                 marker=dict(symbol="x", size=14, color=PAL[2]), error_x=dict(type="data", array=[res["ci"]], color=PAL[2])))
        fig.add_hline(y=0, line_color="#9aa1c2", line_width=1); fig.add_vline(x=0, line_color="#9aa1c2", line_width=1)
        show(fig)
        html(f'<div class="eq">{rg.equation()}  ·  C_x = b/a = {sig(rg.intercept, 4)} / {sig(rg.slope, 4)}</div>')
        kpis([("C larutan ukur", sig(res["cx"]), f"± {sig(res['ci'], 2)} {sa['unit']} (SK 95 %)", "violet"),
              ("C sampel asal", sig(res["cx"] * sa["fp"]), f"± {sig(res['ci'] * sa['fp'], 2)} {sa['unit']}", "indigo"),
              ("R²", fmt(rg.r2, 5), f"n = {rg.n}", "teal" if rg.r2 >= CR["r2_min"] else "rose"),
              ("RSD hasil", fmt(res["s"] / res["cx"] * 100, 2) + " %", "s(C_x)/C_x", "amber")])
        if sa.get("truth") and sa["src"] == "simulasi":
            ext = rg.intercept / sa["ka"]
            with st.container(border=True):
                st.markdown("**Perbandingan dengan kalibrasi eksternal (pelarut murni)**")
                table(pd.DataFrame({"Pendekatan": ["Nilai model (benar)", "Adisi standar", "Kalibrasi eksternal"],
                                    f"C ({sa['unit']})": [sig(sa["truth"]), sig(res["cx"]), sig(ext)],
                                    "Recovery (%)": ["100", fmt(res["cx"] / sa["truth"] * 100, 1), fmt(ext / sa["truth"] * 100, 1)]}))
                st.caption(f"Dengan kepekaan matriks {fmt(sa['sens'], 2)}×, kalibrasi eksternal menghasilkan bias proporsional ≈ {fmt((sa['sens'] - 1) * 100, 0)} %, "
                           "sedangkan adisi standar tidak terpengaruh karena kemiringan diukur di matriks yang sama. Adisi standar tidak mengoreksi serapan latar aditif; gunakan blanko sampel untuk itu.")
        if rg.n < 4:
            st.warning("Ketidakpastian ekstrapolasi besar; gunakan ≥ 4 tingkat adisi dengan rentang hingga 2–3 × C_x.")


# ---------------------------------------------------------------------------
# Halaman: Multikomponen
# ---------------------------------------------------------------------------
MIX_X, MIX_Y = "kmno4", "k2cr2o7"


def page_multi():
    header("Analisis", "Analisis Multikomponen", "Penetapan simultan MnO₄⁻ dan Cr₂O₇²⁻: persamaan Vierordt pada dua panjang gelombang dan kuadrat terkecil klasik (CLS) pada spektrum penuh.")
    mx = S["mix"]
    L, Rr = st.columns([1, 1.6], gap="large")
    with L:
        mx["mode"] = st.radio("Sumber data", ["simulasi", "manual"], index=0 if mx["mode"] == "simulasi" else 1, horizontal=True,
                              format_func=lambda m: "Simulasi" if m == "simulasi" else "Input manual")
        if mx["mode"] == "simulasi":
            with st.container(border=True):
                mx["cx"] = st.slider("C KMnO₄ sebenarnya (mg/L)", 0.0, 50.0, float(mx["cx"]), 0.5)
                mx["cy"] = st.slider("C K₂Cr₂O₇ sebenarnya (mg/L)", 0.0, 100.0, float(mx["cy"]), 1.0)
                a, b = st.columns(2)
                mx["lam1"] = a.number_input("λ₁ (nm)", 300.0, 700.0, float(mx["lam1"]), 1.0)
                mx["lam2"] = b.number_input("λ₂ (nm)", 300.0, 700.0, float(mx["lam2"]), 1.0)
                if st.button("Ukur standar murni & campuran", type="primary"):
                    lam = np.arange(300.0, 700.01, 1.0); r = rng()
                    mx["data"] = {"lam": lam, "X": mix_spectrum(lam, {MIX_X: 30.0}, rng=r), "Y": mix_spectrum(lam, {MIX_Y: 60.0}, rng=r),
                                  "M": mix_spectrum(lam, {MIX_X: mx["cx"], MIX_Y: mx["cy"]}, rng=r), "cx": mx["cx"], "cy": mx["cy"]}
                    log("Spektrum standar murni (KMnO₄ 30, K₂Cr₂O₇ 60 mg/L) dan campuran diukur."); st.rerun()
                st.caption("Standar murni: KMnO₄ 30 mg/L dan K₂Cr₂O₇ 60 mg/L, kuvet kuarsa 1 cm.")
        else:
            m = mx["manual"]
            st.caption("Absorptivitas a dalam A·L·mg⁻¹ (b = 1 cm); X = komponen 1, Y = komponen 2.")
            a, b = st.columns(2)
            for key, lab, col in [("aX1", "a_X pada λ₁", a), ("aY1", "a_Y pada λ₁", b), ("aX2", "a_X pada λ₂", a), ("aY2", "a_Y pada λ₂", b), ("A1", "A campuran λ₁", a), ("A2", "A campuran λ₂", b)]:
                m[key] = col.text_input(lab, m[key], key=f"mx_{key}")
    with Rr:
        if mx["mode"] == "manual":
            m = mx["manual"]; v = [num_in(m[k]) for k in ("aX1", "aY1", "aX2", "aY2", "A1", "A2")]
            if any(math.isnan(z) for z in v):
                st.info("Lengkapi enam nilai numerik."); return
            res = vierordt(*v)
            if not res:
                st.error("Matriks absorptivitas singular; pilih λ lain."); return
            kpis([("C komponen X", sig(res["cx"]), "mg/L", "violet"), ("C komponen Y", sig(res["cy"]), "mg/L", "indigo"),
                  ("Bilangan kondisi", fmt(res["cond"], 1), "makin kecil makin andal", "teal" if res["cond"] < 10 else "amber")])
            return
        d = mx["data"]
        if not d:
            st.info("Tekan Ukur standar murni & campuran."); return
        lam = np.asarray(d["lam"]); kX = np.asarray(d["X"]) / 30.0; kY = np.asarray(d["Y"]) / 60.0; M = np.asarray(d["M"])
        fig = fig_base("Panjang gelombang λ (nm)", "Absorbansi (A)", 380)
        fig.add_trace(go.Scatter(x=lam, y=d["X"], mode="lines", name="KMnO₄ 30 mg/L", line=dict(color="#8e2c8f", width=1.8)))
        fig.add_trace(go.Scatter(x=lam, y=d["Y"], mode="lines", name="K₂Cr₂O₇ 60 mg/L", line=dict(color="#d98a00", width=1.8)))
        fig.add_trace(go.Scatter(x=lam, y=M, mode="lines", name="Campuran", line=dict(color=PAL[0], width=2.6)))
        w = (lam >= 320) & (lam <= 650)
        cl = cls_fit(lam[w], M[w], [kX[w], kY[w]])
        fig.add_trace(go.Scatter(x=lam[w], y=cl["fit"], mode="lines", name="Fit CLS", line=dict(color=PAL[2], dash="dot", width=2)))
        for l0 in (mx["lam1"], mx["lam2"]):
            fig.add_vline(x=l0, line_dash="dash", line_color="#8a91b5", annotation_text=f"{fmt(l0, 0)} nm")
        add_rainbow(fig, 300, 700)
        show(fig)
        g1 = lambda arr, l0: float(np.interp(l0, lam, arr))
        vr = vierordt(g1(kX, mx["lam1"]), g1(kY, mx["lam1"]), g1(kX, mx["lam2"]), g1(kY, mx["lam2"]), g1(M, mx["lam1"]), g1(M, mx["lam2"]))
        rows = [["KMnO₄", sig(d["cx"]), sig(vr["cx"]) if vr else "—", fmt(vr["cx"] / d["cx"] * 100, 1) if vr and d["cx"] else "—", sig(cl["c"][0]), fmt(cl["c"][0] / d["cx"] * 100, 1) if d["cx"] else "—"],
                ["K₂Cr₂O₇", sig(d["cy"]), sig(vr["cy"]) if vr else "—", fmt(vr["cy"] / d["cy"] * 100, 1) if vr and d["cy"] else "—", sig(cl["c"][1]), fmt(cl["c"][1] / d["cy"] * 100, 1) if d["cy"] else "—"]]
        table(pd.DataFrame(rows, columns=["Komponen", "Benar (mg/L)", "Vierordt", "R Vierordt (%)", "CLS", "R CLS (%)"]))
        kpis([("Bilangan kondisi", fmt(vr["cond"], 1) if vr else "—", f"λ₁ = {fmt(mx['lam1'], 0)}, λ₂ = {fmt(mx['lam2'], 0)} nm", "teal" if vr and vr["cond"] < 10 else "amber"),
              ("RMSE fit CLS", sig(cl["rmse"], 2), "A, 320–650 nm", "violet"), ("Titik spektrum", str(int(w.sum())), "dipakai CLS", "indigo")])
        st.caption("Vierordt hanya memakai dua titik sehingga sensitif terhadap derau dan pemilihan λ; CLS memanfaatkan seluruh spektrum (redundansi) dan menekan derau acak. "
                   "Keduanya mengasumsikan aditivitas absorbansi dan tidak adanya interaksi kimia antarkomponen.")


# ---------------------------------------------------------------------------
# Halaman: Validasi Data
# ---------------------------------------------------------------------------
def precision():
    v = S["valid"]
    try:
        vals = parse_list(v["reps"])
    except ValueError:
        return None
    if len(vals) < 2:
        return None
    d = describe(vals); gr = grubbs(vals)
    hw = None
    if v["kind"] == "konsentrasi":
        lvl = num_in(v["level"], d["mean"]) if v["level"] else d["mean"]
        fac = {"mg/L": 1e-6, "µg/mL": 1e-6, "ppm": 1e-6, "µg/L": 1e-9, "%b/v": 1e-2}.get(v["level_unit"])
        if fac and lvl > 0:
            pr = horwitz(lvl * fac); hw = {"level": lvl, "prsd": pr, "ref": 0.67 * pr, "horrat": d["rsd"] / pr}
    return {"vals": vals, "d": d, "g": gr, "pass": d["rsd"] <= CR["rsd_max"], "hw": hw}


def recovery():
    v = S["valid"]
    if v["rec_mode"] == "spike":
        cs, css, cad = num_in(v["cs"]), num_in(v["css"]), num_in(v["cadd"])
        if any(math.isnan(x) for x in (cs, css, cad)) or cad <= 0:
            return None
        Rv = recovery_spike(cs, css, cad)
        return {"mode": "spike", "R": Rv, "pass": CR["rec_min"] <= Rv <= CR["rec_max"], "txt": f"({sig(css)} − {sig(cs)}) / {sig(cad)} × 100"}
    cm, cr = num_in(v["cm"]), num_in(v["cr"])
    if math.isnan(cm) or not cr > 0:
        return None
    o = recovery_reference(cm, cr)
    return {"mode": "acuan", "R": o["R"], "bias": o["bias"], "rel": o["rel_bias"], "pass": CR["rec_min"] <= o["R"] <= CR["rec_max"], "txt": f"{sig(cm)} / {sig(cr)} × 100"}


def page_validation():
    header("Mutu", "Validasi Data", "Presisi (RSD, uji Grubbs, rasio Horwitz), akurasi (% recovery spike atau CRM), dan diagnostik residual kalibrasi.")
    with st.container(border=True):
        st.markdown("**Kriteria penerimaan (ditetapkan pengguna).** Tidak ada batas universal; sesuaikan dengan SOP, metode baku, atau instruksi dosen.")
        c = st.columns(7)
        CR["r2_min"] = c[0].number_input("R² min", 0.0, 1.0, CR["r2_min"], 0.001, format="%.3f")
        CR["rsd_max"] = c[1].number_input("RSD maks (%)", 0.0, 100.0, CR["rsd_max"], 0.5)
        CR["rec_min"] = c[2].number_input("Recovery min (%)", 0.0, 200.0, CR["rec_min"], 1.0)
        CR["rec_max"] = c[3].number_input("Recovery maks (%)", 0.0, 200.0, CR["rec_max"], 1.0)
        CR["std_res_max"] = c[4].number_input("|e/s| maks", 0.5, 5.0, CR["std_res_max"], 0.5)
        CR["a_min"] = c[5].number_input("A min kerja", 0.0, 3.0, CR["a_min"], 0.05)
        CR["a_max"] = c[6].number_input("A maks kerja", 0.0, 3.0, CR["a_max"], 0.05)
    v = S["valid"]
    p1, p2 = st.columns(2, gap="large")
    with p1:
        st.subheader("Presisi (ulangan)")
        v["kind"] = st.radio("Jenis data", ["konsentrasi", "absorbansi"], index=0 if v["kind"] == "konsentrasi" else 1, horizontal=True)
        v["reps"] = st.text_area("Nilai ulangan (pisahkan ;)", v["reps"], placeholder="2,251; 2,238; 2,264; 2,247; 2,255; 2,243")
        if v["kind"] == "konsentrasi":
            a, b = st.columns(2)
            ulist = ["mg/L", "µg/mL", "ppm", "µg/L", "%b/v"]
            v["level"] = a.text_input("Kadar acuan Horwitz (opsional)", v["level"], placeholder="= rata-rata")
            v["level_unit"] = b.selectbox("Satuan", ulist, index=ulist.index(v["level_unit"]) if v["level_unit"] in ulist else 0)
        pr = precision()
        if pr:
            d = pr["d"]
            kpis([("n", str(d["n"]), f"{sig(d['min'])} – {sig(d['max'])}", "indigo"), ("Rata-rata", sig(d["mean"], 5), f"± {sig(d['ci95'], 3)} (SK 95 %)", "violet"),
                  ("s", sig(d["sd"], 3), "simpangan baku", "amber"), ("RSD", fmt(d["rsd"], 2) + " %", f"batas {fmt(CR['rsd_max'], 2)} %", "teal" if pr["pass"] else "rose")])
            verdict(pr["pass"], f"keterulangan RSD {fmt(d['rsd'], 2)} % {'≤' if pr['pass'] else '>'} {fmt(CR['rsd_max'], 2)} %")
            if pr["g"]["applicable"]:
                (st.warning if pr["g"]["outlier"] else st.info)(f"Uji Grubbs (α = 0,05): G = {fmt(pr['g']['G'], 3)} vs G kritis {fmt(pr['g']['Gcrit'], 3)} → " + (f"nilai {sig(pr['g']['value'])} terindikasi pencilan." if pr["g"]["outlier"] else "tidak ada pencilan."))
            if pr["hw"]:
                h = pr["hw"]; st.caption(f"Horwitz pada {sig(h['level'], 3)} {v['level_unit']}: PRSD_R = {fmt(h['prsd'], 2)} %; 0,67 × PRSD_R = {fmt(h['ref'], 2)} %; HorRat = {fmt(h['horrat'], 2)}. Pembanding, bukan pengganti kriteria SOP.")
            fig = fig_base("Ulangan ke-", "Nilai", 220); fig.update_layout(hovermode="closest", showlegend=False)
            fig.add_hrect(y0=d["mean"] - 2 * d["sd"], y1=d["mean"] + 2 * d["sd"], fillcolor="rgba(13,148,136,.08)", line_width=0)
            fig.add_hline(y=d["mean"], line_color=PAL[1])
            fig.add_trace(go.Scatter(x=list(range(1, d["n"] + 1)), y=pr["vals"], mode="markers+lines", marker=dict(size=9, color=PAL[0]), line=dict(color="#c9cde6")))
            show(fig)
        else:
            st.info("Masukkan minimal 2 nilai numerik.")
    with p2:
        st.subheader("Akurasi (% recovery)")
        v["rec_mode"] = st.radio("Metode", ["spike", "acuan"], index=0 if v["rec_mode"] == "spike" else 1, horizontal=True, format_func=lambda m: "Adisi standar (spike)" if m == "spike" else "Nilai acuan / CRM")
        if v["rec_mode"] == "spike":
            a, b, c3 = st.columns(3)
            v["cs"] = a.text_input("C sampel", v["cs"]); v["css"] = b.text_input("C sampel + spike", v["css"]); v["cadd"] = c3.text_input("C ditambahkan", v["cadd"])
        else:
            a, b = st.columns(2)
            v["cm"] = a.text_input("C terukur", v["cm"]); v["cr"] = b.text_input("Nilai acuan", v["cr"])
        rc = recovery()
        if rc:
            html(f'<div class="eq">%R = {rc["txt"]} = <b>{fmt(rc["R"], 2)} %</b></div>')
            if rc["mode"] == "acuan":
                st.caption(f"Bias = {sig(rc['bias'], 3)} · bias relatif = {fmt(rc['rel'], 2)} %")
            verdict(rc["pass"], f"recovery {fmt(rc['R'], 2)} % terhadap rentang {fmt(CR['rec_min'], 0)}–{fmt(CR['rec_max'], 0)} %")
            fig = go.Figure(go.Indicator(mode="gauge+number", value=rc["R"], number={"suffix": " %", "valueformat": ".1f"},
                                         gauge={"axis": {"range": [60, 140]}, "bar": {"color": PAL[0]},
                                                "steps": [{"range": [CR["rec_min"], CR["rec_max"]], "color": "rgba(13,148,136,.18)"}]}))
            fig.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
            show(fig)
        else:
            st.info("Lengkapi nilai untuk menghitung % recovery.")
    st.subheader("Residual regresi kalibrasi")
    reg = cal_reg()
    if reg and reg.ok:
        pts = cal_points()
        fig = fig_base(f"Konsentrasi ({S['cal_unit']})", "Residual (A)", 300); fig.update_layout(hovermode="closest")
        fig.add_trace(go.Scatter(x=pts["C"], y=reg.residuals, mode="markers", name="Residual", marker=dict(size=10, color=PAL[2])))
        fig.add_hline(y=0, line_color=PAL[1])
        for sgn in (1, -1):
            fig.add_hline(y=sgn * CR["std_res_max"] * reg.sxy, line_dash="dot", line_color="#8a91b5")
        a, b = st.columns([1.5, 1])
        with a:
            show(fig)
        with b:
            table(pd.DataFrame({"C": pts["C"].values, "A": pts["A"].values, "Â": reg.yhat.round(4), "e": reg.residuals.round(4), "e/s": reg.std_residuals.round(2)}))
        runs = residual_runs(reg.residuals)
        st.caption(f"{runs} kelompok tanda (runs) dari {reg.n} titik" + ("; pola sistematis → kemungkinan kelengkungan." if reg.n >= 5 and runs <= 2 else "; tidak tampak pola sistematis yang jelas."))
    else:
        st.info("Belum ada regresi kalibrasi.")


# ---------------------------------------------------------------------------
# Halaman: Kualifikasi Kinerja (PQ)
# ---------------------------------------------------------------------------
def page_pq():
    header("Mutu", "Kualifikasi Kinerja Instrumen", "Simulasi uji kinerja menurut Ph. Eur. 2.2.25 / USP ⟨857⟩: akurasi panjang gelombang, akurasi fotometrik, sinar sesat, dan resolusi. Parameter penyimpangan instrumen dapat diatur untuk mempelajari dampaknya.")
    pq = S["pq"]; res = pq["results"]
    tabs = st.tabs(["Akurasi λ", "Akurasi fotometrik", "Sinar sesat", "Resolusi", "Ringkasan"])
    with tabs[0]:
        a, b = st.columns([1, 2], gap="large")
        with a:
            pq["wl_off"] = st.slider("Pergeseran kalibrasi λ instrumen (nm)", -2.0, 2.0, float(pq["wl_off"]), 0.05)
            pq["wl_tol"] = st.number_input("Toleransi internal laboratorium (± nm)", 0.1, 3.0, float(pq["wl_tol"]), 0.1)
            if st.button("Jalankan uji akurasi λ", type="primary"):
                res["wl"] = pq_wavelength(pq["wl_off"], rng()); log("Uji akurasi panjang gelombang dijalankan.")
        with b:
            if res.get("wl"):
                rows = res["wl"]
                ok_pe = all(abs(r["delta"]) <= r["lim_pe"] for r in rows); ok_in = all(abs(r["delta"]) <= pq["wl_tol"] for r in rows)
                fig = fig_base("Garis acuan (nm)", "Δλ = terukur − acuan (nm)", 300); fig.update_layout(hovermode="closest", showlegend=False)
                fig.add_trace(go.Bar(x=[f"{r['ref']:.2f}" for r in rows], y=[r["delta"] for r in rows], marker_color=[PAL[1] if abs(r["delta"]) <= pq["wl_tol"] else PAL[2] for r in rows]))
                for s_ in (1, -1):
                    fig.add_hline(y=s_ * pq["wl_tol"], line_dash="dot", line_color="#8a91b5")
                fig.update_xaxes(type="category"); show(fig)
                table(pd.DataFrame([{"Sumber": r["sumber"], "Acuan (nm)": fmt(r["ref"], 2), "Terukur (nm)": fmt(r["meas"], 2), "Δλ (nm)": fmt(r["delta"], 2), "Batas Ph. Eur. (± nm)": fmt(r["lim_pe"], 0)} for r in rows]))
                verdict(ok_pe, "kriteria Ph. Eur. (±1 nm UV, ±3 nm Vis)"); verdict(ok_in, f"toleransi internal ±{fmt(pq['wl_tol'], 1)} nm")
                res["wl_pass"] = ok_pe and ok_in
            else:
                st.info("Tekan Jalankan uji akurasi λ.")
    with tabs[1]:
        a, b = st.columns([1, 2], gap="large")
        with a:
            pq["ph_bias"] = st.slider("Bias fotometrik instrumen (%)", -3.0, 3.0, float(pq["ph_bias"]), 0.1)
            st.caption("K₂Cr₂O₇ ≈ 60 mg/L dalam H₂SO₄ 0,005 M, kuvet 1 cm, blanko H₂SO₄ 0,005 M.")
            if st.button("Jalankan uji fotometrik", type="primary"):
                res["ph"] = pq_photometric(pq["ph_bias"], rng()); log("Uji akurasi fotometrik dijalankan.")
        with b:
            if res.get("ph"):
                rows = res["ph"]
                fig = fig_base("λ (nm)", "A(1 %, 1 cm)", 300); fig.update_layout(hovermode="closest")
                fig.add_trace(go.Scatter(x=[r["lam"] for r in rows], y=[r["a1"] for r in rows], mode="markers", name="Terukur", marker=dict(size=12, color=PAL[0]),
                                         error_y=dict(type="data", symmetric=False, array=[r["hi"] - r["a1"] for r in rows], arrayminus=[r["a1"] - r["lo"] for r in rows], color="rgba(91,99,134,.35)", thickness=6, width=0)))
                fig.add_trace(go.Scatter(x=[r["lam"] for r in rows], y=[r["nom"] for r in rows], mode="markers", name="Nominal", marker=dict(symbol="line-ew-open", size=18, color=PAL[1])))
                show(fig)
                table(pd.DataFrame([{"λ (nm)": r["lam"], "A terukur": fmt(r["A"], 4), "A(1 %, 1 cm)": fmt(r["a1"], 1), "Rentang Ph. Eur.": f"{fmt(r['lo'], 1)}–{fmt(r['hi'], 1)}", "Status": "lulus" if r["pass"] else "gagal"} for r in rows]))
                res["ph_pass"] = all(r["pass"] for r in rows)
                verdict(res["ph_pass"], "akurasi fotometrik pada empat panjang gelombang")
            else:
                st.info("Tekan Jalankan uji fotometrik.")
    with tabs[2]:
        a, b = st.columns([1, 2], gap="large")
        with a:
            pq["sl_pct"] = st.slider("Sinar sesat instrumen (% T)", 0.001, 2.0, float(pq["sl_pct"]), 0.001, format="%.3f")
            if st.button("Jalankan uji sinar sesat", type="primary"):
                res["sl"] = pq_stray(pq["sl_pct"], rng()); log("Uji sinar sesat KCl 198 nm dijalankan.")
        with b:
            s_ = np.logspace(-3, np.log10(2), 80)
            fig = fig_base("Sinar sesat (% T, skala log)", "A terukur KCl 12 g/L, 198 nm", 300); fig.update_layout(hovermode="closest")
            fig.add_trace(go.Scatter(x=s_, y=-np.log10(10 ** -5.5 + s_ / 100), mode="lines", name="A_obs = −log(10⁻ᴬ + s)", line=dict(color=PAL[4])))
            fig.add_hline(y=2.0, line_dash="dash", line_color=PAL[2], annotation_text="batas A ≥ 2,0")
            if res.get("sl") is not None:
                fig.add_trace(go.Scatter(x=[pq["sl_pct"]], y=[res["sl"]], mode="markers", name="Hasil", marker=dict(size=13, color=PAL[0])))
            fig.update_xaxes(type="log"); show(fig)
            if res.get("sl") is not None:
                res["sl_pass"] = res["sl"] >= 2.0
                verdict(res["sl_pass"], f"A(198 nm) = {fmt(res['sl'], 2)}; sinar sesat ≈ {fmt(100 * 10 ** -res['sl'], 3)} %")
                st.caption("Sinar sesat membatasi A maksimum yang dapat diukur secara linear dan menimbulkan penyimpangan negatif Lambert–Beer pada absorbansi tinggi, terutama di ujung UV tempat energi lampu D₂ rendah.")
    with tabs[3]:
        a, b = st.columns([1, 2], gap="large")
        with a:
            pq["sbw"] = st.slider("Lebar pita spektral SBW (nm)", 0.2, 5.0, float(pq["sbw"]), 0.1, help="UV-1700: 1 nm (tetap).")
            pq["res_min"] = st.number_input("Rasio minimum A₂₆₉/A₂₆₆", 1.0, 3.0, float(pq["res_min"]), 0.1)
            if st.button("Jalankan uji resolusi", type="primary"):
                res["res"] = toluene_ratio(pq["sbw"]); res["res_sbw"] = pq["sbw"]; log("Uji resolusi toluena dijalankan.")
        with b:
            fig = fig_base("λ (nm)", "Absorbansi", 300)
            for sb, col in [(0.5, "#c9cde6"), (2.0, "#e7b8c3"), (pq["sbw"], PAL[0])]:
                lam_t, At = toluene_spectrum(sb)
                fig.add_trace(go.Scatter(x=lam_t, y=At, mode="lines", name=f"SBW {fmt(sb, 1)} nm", line=dict(color=col, width=2.4 if sb == pq["sbw"] else 1.4)))
            fig.add_vrect(x0=265.5, x1=266.5, fillcolor="rgba(225,29,72,.08)", line_width=0)
            fig.add_vrect(x0=268.5, x1=269.5, fillcolor="rgba(13,148,136,.10)", line_width=0)
            fig.update_xaxes(range=[245, 275]); show(fig)
            if res.get("res"):
                rr = res["res"]; res["res_pass"] = rr["ratio"] >= pq["res_min"]
                kpis([("A maks", fmt(rr["Amax"], 4), f"{fmt(rr['lam_max'], 1)} nm", "teal"), ("A min", fmt(rr["Amin"], 4), f"{fmt(rr['lam_min'], 1)} nm", "rose"),
                      ("Rasio", fmt(rr["ratio"], 2), f"SBW {fmt(res['res_sbw'], 1)} nm", "indigo")])
                verdict(res["res_pass"], f"rasio {fmt(rr['ratio'], 2)} terhadap batas {fmt(pq['res_min'], 1)}")
                st.caption("Model spektrum toluena disederhanakan (lima pita vibronik); konvolusi fungsi celah dilakukan pada domain transmitansi.")
    with tabs[4]:
        items = [("Akurasi λ", res.get("wl_pass")), ("Akurasi fotometrik", res.get("ph_pass")), ("Sinar sesat", res.get("sl_pass")), ("Resolusi", res.get("res_pass"))]
        kpis([(n, "—" if v is None else ("LULUS" if v else "GAGAL"), "belum diuji" if v is None else "", "indigo" if v is None else ("teal" if v else "rose")) for n, v in items])
        done = [v for _, v in items if v is not None]
        if len(done) == 4:
            verdict(all(done), "instrumen layak dipakai untuk analisis kuantitatif" if all(done) else "lakukan perawatan/kalibrasi sebelum analisis kuantitatif")
        else:
            st.info(f"{len(done)}/4 uji selesai.")


# ---------------------------------------------------------------------------
# Halaman: Pembelajaran & Kuis
# ---------------------------------------------------------------------------
def page_learn():
    header("Belajar", "Pembelajaran & Kuis", "Materi inti spektrofotometri UV-Vis dengan simulasi interaktif dan kuis evaluasi diri.")
    tabs = st.tabs(["Prinsip kerja", "Lambert–Beer", "A dan %T", "λmaks", "Ringbom", "Blanko", "Faktor", "Kualifikasi", "Kesalahan umum", "Kuis"])
    with tabs[0]:
        st.markdown(LESSONS["Prinsip kerja"]); model3d(height=480, xray=True)
    with tabs[1]:
        st.latex(r"A = \varepsilon \cdot b \cdot C = -\log T")
        a, b, c = st.columns(3)
        eps = a.slider("ε (L·mol⁻¹·cm⁻¹)", 1000, 50000, 11100, 500); bb = b.slider("b (cm)", 0.1, 5.0, 1.0, 0.1); cc = c.slider("C (µmol/L)", 0, 150, 40)
        stray = st.slider("Sinar sesat (% T) — tampilkan penyimpangan", 0.0, 2.0, 0.0, 0.05)
        A = eps * bb * cc * 1e-6
        html(f'<div class="eq">A = {eps} × {fmt(bb, 1)} × {fmt(cc * 1e-6, 6)} = <b>{fmt(A, 3)}</b> · %T = {fmt(float(abs_to_t(A)), 1)} %</div>')
        xs = np.arange(0, 301, 5); At = eps * bb * xs * 1e-6; s = stray / 100
        fig = fig_base("C (µmol/L)", "Absorbansi", 300)
        fig.add_trace(go.Scatter(x=xs, y=At, mode="lines", name="A = εbC (ideal)", line=dict(color=PAL[0])))
        if s > 0:
            fig.add_trace(go.Scatter(x=xs, y=-np.log10((10 ** -At + s) / (1 + s)), mode="lines", name="dengan sinar sesat", line=dict(color=PAL[2], dash="dash")))
        fig.add_trace(go.Scatter(x=[cc], y=[A], mode="markers", name="titik", marker=dict(size=12, color=PAL[3])))
        show(fig)
        st.caption("Nilai ε awal ≈ kompleks Fe(II)–fenantrolin pada 510 nm. Berlaku untuk radiasi monokromatis dan larutan encer.")
    with tabs[2]:
        A2 = np.linspace(0.04, 2, 100); T = 10 ** -A2; err = np.abs(0.4343 * 0.005 / (T * np.log10(T))) * 100
        a, b = st.columns(2)
        with a:
            fig = fig_base("Absorbansi", "%T", 280); fig.add_trace(go.Scatter(x=A2, y=abs_to_t(A2), mode="lines", name="%T = 100·10⁻ᴬ", line=dict(color=PAL[0]))); show(fig)
        with b:
            fig = fig_base("Absorbansi", "Galat relatif C (%)", 280); fig.add_trace(go.Scatter(x=A2, y=err, mode="lines", name="ΔT = 0,5 %", line=dict(color=PAL[2])))
            fig.add_vline(x=0.434, line_dash="dash", line_color=PAL[1], annotation_text="A = 0,434"); fig.add_vrect(x0=0.2, x1=0.8, fillcolor=PAL[1], opacity=0.10, line_width=0)
            show(fig)
        st.markdown("A = 1 → 10 % cahaya diteruskan; A = 2 → 1 %; A = 3 → 0,1 % (batas atas UV-1700). Galat relatif minimum pada A ≈ 0,434, dasar anjuran daerah kerja ±0,2–0,8.")
    with tabs[3]:
        Lm = st.slider("λ pengukuran (nm)", 440, 600, 510)
        lam = np.arange(400, 621); y = 2 * absorptivity("fephen", lam)
        fig = fig_base("λ (nm)", "A (Fe 2 mg/L)", 280); fig.add_trace(go.Scatter(x=lam, y=y, mode="lines", name="Fe(II)–fenantrolin", line=dict(color=PAL[0]))); fig.add_vline(x=Lm, line_color=PAL[2])
        add_rainbow(fig, 400, 620); show(fig)
        a_ = float(absorptivity("fephen", Lm)); d_ = float(absorptivity("fephen", Lm + 1) - absorptivity("fephen", Lm - 1)) / 2
        kpis([("Absorptivitas a", fmt(a_, 4), "L·mg⁻¹·cm⁻¹", "indigo"), ("Kepekaan relatif", fmt(a_ / float(y.max() / 2) * 100, 1) + " %", "terhadap maksimum", "violet"),
              ("|dA/dλ| / A", fmt(abs(d_ / a_) * 100, 2) + " %/nm", "sensitivitas terhadap galat λ", "amber")])
    with tabs[4]:
        st.markdown("Plot Ringbom, (1 − T) terhadap log C, menunjukkan rentang konsentrasi dengan galat relatif terkecil: bagian paling curam dan paling linear kurva.")
        eps2 = st.slider("ε (L·mol⁻¹·cm⁻¹) ", 1000, 50000, 11100, 500, key="rb_eps")
        lc = np.linspace(-7, -3, 200); Tt = 10 ** -(eps2 * 10 ** lc)
        fig = fig_base("log C (mol/L)", "1 − T", 300); fig.add_trace(go.Scatter(x=lc, y=1 - Tt, mode="lines", name="Ringbom", line=dict(color=PAL[4], width=2.4)))
        i0 = int(np.argmax(np.gradient(1 - Tt, lc)))
        fig.add_vline(x=lc[i0], line_dash="dash", line_color=PAL[1], annotation_text="kemiringan maks (A = 0,434)")
        show(fig)
    with tabs[5]:
        st.markdown(LESSONS["Fungsi blanko"])
    with tabs[6]:
        st.markdown(LESSONS["Faktor yang memengaruhi"])
    with tabs[7]:
        st.markdown(LESSONS["Kualifikasi kinerja"])
    with tabs[8]:
        st.markdown(LESSONS["Kesalahan umum"])
    with tabs[9]:
        right = 0; answered = 0
        for i, (q, opts, ans, exp) in enumerate(QUIZ):
            with st.container(border=True):
                st.markdown(f"**{i + 1}. {q}**")
                choice = st.radio(f"Jawaban {i + 1}", list(range(len(opts))), index=None, key=f"quiz{i}", format_func=lambda j, o=opts: f"{chr(65 + j)}. {o[j]}", label_visibility="collapsed")
                if choice is not None:
                    answered += 1
                    (st.success if choice == ans else st.error)(("Benar. " if choice == ans else "Kurang tepat. ") + exp)
                    right += choice == ans
        kpis([("Skor", f"{right}/{len(QUIZ)}", f"{answered} dijawab", "teal" if right >= 0.8 * len(QUIZ) else "indigo"),
              ("Persentase", fmt(right / len(QUIZ) * 100, 0) + " %", "", "violet")])


# ---------------------------------------------------------------------------
# Halaman: Panduan Praktikum
# ---------------------------------------------------------------------------
def page_guide():
    header("Belajar", "Panduan Praktikum", "Alur kerja terpandu; status setiap langkah diperbarui otomatis dari aktivitas Anda di aplikasi.")
    exps = {"fephen": ("Penetapan besi dengan 1,10-fenantrolin", "reagen", (400, 650)), "kmno4": ("Penetapan kalium permanganat", "pelarut", (400, 700)), "crdpc": ("Penetapan Cr(VI) metode DPC", "reagen", (400, 700))}
    key = st.selectbox("Percobaan", list(exps), format_func=lambda k_: exps[k_][0])
    nama, blank, rng_ = exps[key]; reg = cal_reg()
    if st.button("Terapkan pengaturan percobaan ke instrumen", type="primary"):
        a = ANALYTES[key]; I.update(analyte=key, blank=blank, cuvette="kuarsa", start=float(rng_[0]), end=float(rng_[1]), solution="standar", conc=float(a["std"][len(a["std"]) // 2]), lam=float(a["lam_ref"])); st.toast("Diterapkan")
    pqr = S["pq"]["results"]
    steps = [
        ("Nyalakan instrumen", "Inisialisasi; alat nyata dipanaskan 15–30 menit.", I["power"] == "ready"),
        ("Periksa kinerja alat (opsional)", "Akurasi λ dan fotometrik di halaman Kualifikasi Kinerja.", bool(pqr.get("wl") or pqr.get("ph"))),
        ("Siapkan larutan & kuvet", f"Analit {ANALYTES[key]['rumus']}, kuvet kuarsa, referensi {blank}.", I["analyte"] == key and I["blank"] == blank),
        ("Koreksi baseline", f"Blanko di kedua sel, {rng_[0]}–{rng_[1]} nm.", bool(I["baseline"]) and I["baseline"]["analyte"] == key),
        ("Pindai spektrum standar", "Tentukan λmaks.", any(s["params"].get("analyte") == key and s["source"] == "simulasi" for s in S["spectra"])),
        ("Ukur deret standar", "Minimal 5 titik pada λmaks.", len(cal_points()) >= 5),
        ("Evaluasi kurva", f"R² ≥ {fmt(CR['r2_min'], 3)}, residual tanpa pola, uji Mandel.", bool(reg and reg.ok and reg.r2 >= CR["r2_min"])),
        ("Ukur sampel triplo", "Hitung kadar dengan faktor pengenceran.", any((lambda r: bool(r) and r["n"] >= 3)(sample_res(s, reg)) for s in S["samples"]) if reg else False),
        ("Evaluasi presisi & akurasi", "RSD ulangan dan % recovery.", bool(precision() or recovery())),
        ("Susun laporan", "Lengkapi identitas, unduh PDF.", S["report"]["generated"]),
    ]
    done = sum(s[2] for s in steps); st.progress(done / len(steps), f"Kemajuan {done}/{len(steps)}")
    cur = next((i for i, s in enumerate(steps) if not s[2]), None)
    out = ""
    for i, (t, d, ok) in enumerate(steps):
        cls = "done" if ok else ("cur" if i == cur else "")
        out += f'<div class="step {cls}"><div class="dot">{"✓" if ok else i + 1}</div><div><div class="t">{t}</div><div class="s">{d}</div></div></div>'
    html(out)


# ---------------------------------------------------------------------------
# Halaman: Laporan
# ---------------------------------------------------------------------------
def report_model() -> dict:
    rp = S["report"]; reg = cal_reg(); blocks = []; srcs = set(); u = S["cal_unit"]; nfig = 1
    blocks += [{"t": "h", "text": "1. Identitas"}, {"t": "kv", "rows": [("Judul", rp["judul"] or "—"), ("Praktikan / NIM", (rp["praktikan"] or "—") + (f" / {rp['nim']}" if rp["nim"] else "")), ("Kelas", rp["kelas"] or "—"), ("Dosen / asisten", rp["dosen"] or "—"), ("Sampel", rp["sampel"] or "—"), ("Tanggal analisis", rp["tanggal"]), ("Mode kerja", S["settings"]["mode"])]}]
    blocks += [{"t": "h", "text": "2. Instrumen & parameter"}, {"t": "kv", "rows": [("Instrumen acuan", "Shimadzu UV-1700 PharmaSpec (double-beam, 190–1100 nm). Pembacaan berasal dari instrumen virtual."), ("λ pengukuran", f"{fmt(S['cal_lambda'], 1)} nm" if S["cal_lambda"] else "—"), ("Kuvet / tebal", f"{CUVETTES[I['cuvette']]['nama']} / {fmt(I['path'], 1)} cm"), ("Referensi", I["blank"]), ("Koreksi baseline", f"{I['baseline']['start']:.0f}–{I['baseline']['end']:.0f} nm" if I["baseline"] else "tidak tercatat")]}]
    n = 3
    pqr = S["pq"]["results"]
    pq_rows = [(nm, "lulus" if pqr[k] else "gagal") for nm, k in [("Akurasi λ", "wl_pass"), ("Akurasi fotometrik", "ph_pass"), ("Sinar sesat", "sl_pass"), ("Resolusi", "res_pass")] if pqr.get(k) is not None]
    if pq_rows:
        blocks += [{"t": "h", "text": f"{n}. Kualifikasi kinerja instrumen"}, {"t": "kv", "rows": pq_rows}]; n += 1; srcs.add("simulasi")
    sp = next((s for s in S["spectra"] if s["id"] == S["last_scan"]), S["spectra"][-1] if S["spectra"] else None)
    if sp:
        srcs.add(sp["source"]); pk = find_peaks(sp["lam"], sp["A"])
        blocks += [{"t": "h", "text": f"{n}. Spektrum absorbansi"}, {"t": "img", "png": spectrum_png(sp["lam"], sp["A"], sp["name"], pk[0]["lambda"] if pk else None), "cap": f"Gambar {nfig}. Spektrum {sp['name']} ({sp['source']})."},
                   {"t": "kv", "rows": [("λmaks", f"{fmt(pk[0]['lambda'], 1)} nm (A = {fmt(pk[0]['abs'], 4)})" if pk else "tidak terdeteksi")]}]; n += 1; nfig += 1
    sres = []; md = None
    if reg and reg.ok:
        pts = cal_points(); srcs.add(S["cal_src"])
        md = mandel(pts["C"].values, pts["A"].values) if not S["cal_zero"] else None
        sres = [(s, sample_res(s, reg)) for s in S["samples"]]; sres = [(s, r) for s, r in sres if r]
        blocks += [{"t": "h", "text": f"{n}. Kurva kalibrasi"},
                   {"t": "table", "head": ["No", f"C ({u})", "A", "Â", "Residual"], "rows": [[i + 1, sig(c_), fmt(a_, 4), fmt(reg.yhat[i], 4), fmt(reg.residuals[i], 4)] for i, (c_, a_) in enumerate(zip(pts["C"], pts["A"]))]},
                   {"t": "img", "png": calib_png(pts["C"].values, pts["A"].values, reg, [(s["name"], r["c_meas"], r["A_mean"]) for s, r in sres], u), "cap": f"Gambar {nfig}. Kurva kalibrasi {S['cal_name']} dengan pita kepercayaan 95 %."},
                   {"t": "kv", "rows": [("Persamaan", reg.equation()), ("r / R²", f"{fmt(reg.r, 5)} / {fmt(reg.r2, 5)}"), ("s y/x", sig(reg.sxy, 3)), ("LOD / LOQ", f"{sig(reg.lod, 3)} / {sig(reg.loq, 3)} {u}"), ("Rentang", f"{sig(reg.xmin, 3)}–{sig(reg.xmax, 3)} {u}")] + ([("Uji Mandel", f"F = {fmt(md['F'], 2)}; F kritis = {fmt(md['Fcrit'], 2)}: {'linear diterima' if md['linear_ok'] else 'kelengkungan signifikan'}")] if md else [])}]
        n += 1; nfig += 1
        if sres:
            for s, _ in sres: srcs.add(s["src"])
            blocks += [{"t": "h", "text": f"{n}. Hasil penetapan kadar"}, {"t": "table", "head": ["Sampel", "n", "A rerata", "RSD %", "C terukur", "FP", "C asal ± SK95%", "Kadar mg/kg"],
                       "rows": [[s["name"], r["n"], fmt(r["A_mean"], 4), fmt(r["rsd"], 2) if r["n"] > 1 else "-", sig(r["c_meas"]), sig(r["fp"]), f"{sig(r['c_final'])} ± {sig(r['ci_final'], 2)} {u}", sig(r["kadar"]) if r["mass_ok"] else "-"] for s, r in sres]}]
            n += 1
            for s, r in sres:
                if r["flags"]: blocks.append({"t": "note", "text": s["name"] + ": " + " ".join(r["flags"])})
    sa = S["sa"]; sa_res = None
    if "cur" in sa and len(sa["cur"]) >= 3:
        d = sa["cur"].dropna()
        if len(d) >= 3:
            sa_res = std_addition(d["C ditambahkan"].astype(float).values, d["A"].astype(float).values)
    if sa_res:
        srcs.add(sa["src"])
        blocks += [{"t": "h", "text": f"{n}. Adisi standar"}, {"t": "img", "png": stdadd_png(d["C ditambahkan"].astype(float).values, d["A"].astype(float).values, sa_res, sa["unit"]), "cap": f"Gambar {nfig}. Kurva adisi standar; tanda silang = −C_x."},
                   {"t": "kv", "rows": [("Persamaan", sa_res["reg"].equation()), ("C larutan ukur", f"{sig(sa_res['cx'])} ± {sig(sa_res['ci'], 2)} {sa['unit']}"), ("C sampel asal", f"{sig(sa_res['cx'] * sa['fp'])} ± {sig(sa_res['ci'] * sa['fp'], 2)} {sa['unit']} (FP {sig(sa['fp'])})")]}]
        n += 1; nfig += 1
    pr, rc = precision(), recovery()
    rows = []
    if reg and reg.ok: rows.append(("Linearitas", f"R² = {fmt(reg.r2, 4)} (≥ {fmt(CR['r2_min'], 3)}): {'memenuhi' if reg.r2 >= CR['r2_min'] else 'tidak memenuhi'}"))
    if pr: rows.append(("Presisi", f"n = {pr['d']['n']}, RSD = {fmt(pr['d']['rsd'], 2)} % (≤ {fmt(CR['rsd_max'], 2)} %): {'memenuhi' if pr['pass'] else 'tidak memenuhi'}"))
    if rc: rows.append(("Akurasi", f"% recovery = {fmt(rc['R'], 2)} % ({fmt(CR['rec_min'], 0)}–{fmt(CR['rec_max'], 0)} %): {'memenuhi' if rc['pass'] else 'tidak memenuhi'}"))
    if rows: blocks += [{"t": "h", "text": "Evaluasi mutu data"}, {"t": "kv", "rows": rows}]
    interp = calib_interpretation(reg, u, CR["r2_min"], CR["std_res_max"], S["settings"]["dec"], md) if reg and reg.ok else []
    concl = [f"Konsentrasi {S['cal_name'] or 'analit'} dalam {s['name']} sebesar {sig(r['c_final'])} ± {sig(r['ci_final'], 2)} {u} (SK 95 %, n = {r['n']})" + (f", setara {sig(r['kadar'])} mg/kg sampel." if r["mass_ok"] else ".") for s, r in sres]
    if sa_res: concl.append(f"Metode adisi standar memberikan {sig(sa_res['cx'] * sa['fp'])} ± {sig(sa_res['ci'] * sa['fp'], 2)} {sa['unit']} dalam sampel asal.")
    if reg and reg.ok: concl.append(f"Kurva kalibrasi {'memenuhi' if reg.r2 >= CR['r2_min'] else 'belum memenuhi'} kriteria linearitas (R² = {fmt(reg.r2, 4)}).")
    if pr: concl.append(f"Presisi {'memenuhi' if pr['pass'] else 'belum memenuhi'} kriteria (RSD {fmt(pr['d']['rsd'], 2)} %).")
    if rc: concl.append(f"Akurasi {'memenuhi' if rc['pass'] else 'belum memenuhi'} kriteria (% recovery {fmt(rc['R'], 2)} %).")
    if not concl: concl.append("Data belum cukup untuk menarik kesimpulan kuantitatif.")
    simulated = bool(srcs & {"simulasi", "contoh", "campuran"})
    if simulated: concl.append("Sebagian atau seluruh data berasal dari model simulasi/data contoh; hasil ini latihan, bukan hasil pengujian laboratorium.")
    if interp: blocks += [{"t": "h", "text": "Interpretasi"}] + [{"t": "p", "text": p} for p in interp]
    blocks += [{"t": "h", "text": "Kesimpulan"}] + [{"t": "p", "text": p} for p in concl]
    if rp["catatan"]: blocks += [{"t": "h", "text": "Catatan praktikan"}, {"t": "p", "text": rp["catatan"]}]
    blocks.append({"t": "note", "text": "Sumber data: " + (", ".join(sorted(srcs)) or "tidak ada") + ". Dibuat dengan SPECTRA LAB 2.0; aplikasi tidak terhubung langsung ke instrumen."})
    return {"blocks": blocks, "simulated": simulated, "judul": rp["judul"], "tanggal": rp["tanggal"], "praktikan": rp["praktikan"]}


def page_report():
    header("Dokumen", "Laporan", "Laporan PDF tersusun otomatis dari data yang tersedia: identitas, parameter instrumen, kualifikasi kinerja, spektrum, kalibrasi, hasil sampel, adisi standar, evaluasi mutu, interpretasi, dan kesimpulan.")
    rp = S["report"]
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        with st.container(border=True):
            for k_, lab in [("judul", "Judul percobaan"), ("praktikan", "Nama praktikan"), ("nim", "NIM"), ("kelas", "Kelas"), ("dosen", "Dosen / asisten"), ("sampel", "Identitas sampel")]:
                rp[k_] = st.text_input(lab, rp[k_])
            rp["tanggal"] = st.date_input("Tanggal analisis", date.fromisoformat(rp["tanggal"])).isoformat()
            rp["catatan"] = st.text_area("Catatan praktikan (opsional)", rp["catatan"])
        if st.button("Buat laporan PDF", type="primary"):
            with st.spinner("Menyusun laporan…"):
                rp["pdf"] = build_pdf(report_model()); rp["generated"] = True
        if rp["pdf"]:
            st.download_button("Unduh laporan PDF", rp["pdf"], f"laporan-uvvis-{rp['nim'] or 'spectralab'}.pdf", "application/pdf")
        skip = ("cal_base", "cal_cur", "history", "sa", "mix", "pq")
        proj = {k_: v_ for k_, v_ in S.items() if k_ not in skip}
        proj["report"] = {k_: v_ for k_, v_ in rp.items() if k_ != "pdf"}; proj["cal_points"] = S["cal_base"].to_dict("records")
        proj["sa_points"] = S["sa"].get("cur", S["sa"]["df"]).to_dict("records")
        st.download_button("Ekspor proyek (.json)", json.dumps(proj, default=str, ensure_ascii=False), "proyek-spectralab.json", "application/json")
        up = st.file_uploader("Impor proyek (.json)", type=["json"])
        if up is not None and st.button("Muat proyek"):
            try:
                o = json.loads(up.getvalue())
                new = _defaults()
                for k_, v_ in o.items():
                    if k_ in new and k_ not in skip:
                        new[k_] = v_
                new["cal_base"] = pd.DataFrame(o.get("cal_points", []), columns=["C", "A", "Pakai"])
                new["sa"]["df"] = pd.DataFrame(o.get("sa_points", []), columns=["C ditambahkan", "A"])
                new["report"]["pdf"] = None
                st.session_state.S = new; st.rerun()
            except Exception as e:
                st.error(f"Impor gagal: {e}")
    with right:
        st.markdown("**Pratinjau isi laporan**")
        with st.container(border=True):
            for b in report_model()["blocks"]:
                if b["t"] == "h": st.markdown(f"#### {b['text']}")
                elif b["t"] == "p": st.markdown(b["text"])
                elif b["t"] == "note": st.caption(b["text"])
                elif b["t"] == "kv": table(pd.DataFrame(b["rows"], columns=["Parameter", "Nilai"]))
                elif b["t"] == "table": table(pd.DataFrame(b["rows"], columns=b["head"]))
                elif b["t"] == "img": st.image(b["png"], caption=b["cap"])


def page_history():
    header("Dokumen", "Riwayat", "Cuplikan keadaan kerja selama sesi. Untuk penyimpanan permanen, unduh berkas JSON atau gunakan Ekspor proyek pada halaman Laporan.")
    if st.button("Simpan keadaan sekarang", type="primary"):
        reg = cal_reg()
        S["history"].insert(0, {"waktu": f"{datetime.now():%d/%m/%Y %H.%M}", "judul": S["report"]["judul"], "analit": S["cal_name"], "R²": round(reg.r2, 5) if reg and reg.ok else None,
                                "titik": len(cal_points()), "hasil": "; ".join(f"{s['name']}: {sig(r['c_final'])} {S['cal_unit']}" for s in S["samples"] for r in [sample_res(s, reg)] if r) if reg else "",
                                "cal": S["cal_base"].to_dict("records"), "samples": json.loads(json.dumps(S["samples"], default=str))})
        st.toast("Tersimpan")
    if not S["history"]:
        st.info("Belum ada riwayat."); return
    table(pd.DataFrame([{k_: v_ for k_, v_ in h.items() if k_ not in ("cal", "samples")} for h in S["history"]]))
    i = st.selectbox("Pilih entri", range(len(S["history"])), format_func=lambda j: f"{S['history'][j]['waktu']} · {S['history'][j]['judul']}")
    c1, c2 = st.columns(2)
    if c1.button("Muat ke kurva kalibrasi"):
        h = S["history"][i]; set_calibration(pd.DataFrame(h["cal"], columns=["C", "A", "Pakai"]), "pengguna"); S["samples"] = h["samples"]; st.toast("Dimuat")
    c2.download_button("Unduh entri (.json)", json.dumps(S["history"][i], default=str, ensure_ascii=False), "riwayat-spectralab.json", "application/json")


# ---------------------------------------------------------------------------
# Navigasi & bilah samping
# ---------------------------------------------------------------------------
PAGES = {
    "Ruang kerja": [st.Page(page_home, title="Beranda", icon=":material/space_dashboard:", url_path="beranda", default=True)],
    "Instrumen": [st.Page(page_lab, title="Praktikum Virtual 3D", icon=":material/science:", url_path="praktikum-3d"),
                  st.Page(page_model3d, title="Model 3D UV-1700", icon=":material/view_in_ar:", url_path="model-3d"),
                  st.Page(page_instrument, title="Simulasi Instrumen", icon=":material/settings_input_component:", url_path="instrumen"),
                  st.Page(page_photometric, title="Fotometri", icon=":material/light_mode:", url_path="fotometri"),
                  st.Page(page_kinetics, title="Kinetika", icon=":material/timeline:", url_path="kinetika")],
    "Analisis": [st.Page(page_spectrum, title="Analisis Spektrum", icon=":material/ssid_chart:", url_path="spektrum"),
                 st.Page(page_calibration, title="Kurva Kalibrasi", icon=":material/show_chart:", url_path="kalibrasi"),
                 st.Page(page_stdadd, title="Adisi Standar", icon=":material/add_chart:", url_path="adisi-standar"),
                 st.Page(page_multi, title="Multikomponen", icon=":material/stacked_line_chart:", url_path="multikomponen")],
    "Mutu": [st.Page(page_validation, title="Validasi Data", icon=":material/fact_check:", url_path="validasi"),
             st.Page(page_pq, title="Kualifikasi Kinerja", icon=":material/verified:", url_path="kualifikasi")],
    "Belajar": [st.Page(page_learn, title="Pembelajaran & Kuis", icon=":material/school:", url_path="belajar"),
                st.Page(page_guide, title="Panduan Praktikum", icon=":material/checklist:", url_path="panduan")],
    "Dokumen": [st.Page(page_report, title="Laporan", icon=":material/picture_as_pdf:", url_path="laporan"),
                st.Page(page_history, title="Riwayat", icon=":material/history:", url_path="riwayat")],
}
nav = st.navigation(PAGES)

with st.sidebar:
    html("<div style='padding:4px 0 2px'><div style='font:700 1.25rem Space Grotesk,sans-serif;letter-spacing:.04em'>SPECTRA LAB 2.0</div>"
         "<div style='font:500 .7rem JetBrains Mono,monospace;letter-spacing:.12em;opacity:.7'>UV-VIS ANALYTICAL WORKBENCH</div>"
         "<div style='height:4px;margin-top:8px;border-radius:4px;background:linear-gradient(90deg,#7b3fe4,#2f5bff,#00b3d6,#22c55e,#facc15,#f97316,#ef4444)'></div></div>")
    S["settings"]["mode"] = st.radio("Mode kerja", ["latihan", "analisis"], index=0 if S["settings"]["mode"] == "latihan" else 1,
                                     format_func=lambda m: "Latihan (data contoh)" if m == "latihan" else "Analisis (data saya)")
    S["settings"]["dec"] = st.radio("Pemisah desimal", ["koma", "titik"], index=0 if S["settings"]["dec"] == "koma" else 1,
                                    format_func=lambda d: "Koma (0,512)" if d == "koma" else "Titik (0.512)", horizontal=True)
    st.caption("Politeknik AKA Bogor · Analisis Kimia. Instrumen acuan: Shimadzu UV-1700 PharmaSpec. Aplikasi ini simulator edukatif dan tidak terhubung ke alat.")

_bl = I["baseline"]
html('<div class="topbar">' + pill("SIMULASI", "sim") +
     pill("Instrumen siap" if I["power"] == "ready" else "Instrumen mati", "ok" if I["power"] == "ready" else "bad") +
     pill(f"Baseline {int(_bl['start'])}–{int(_bl['end'])} nm" if _bl else "Baseline belum", "ok" if _bl else "sim") +
     pill(f"λ {fmt(I['lam'], 1)} nm · {'D2' if I['lam'] < I['lamp_switch'] else 'WI'}") +
     pill(f"{ANALYTES[I['analyte']]['rumus']} · {CUVETTES[I['cuvette']]['nama']}") + "</div>")

nav.run()
