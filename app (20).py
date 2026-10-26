"""
SHIMADZU UV-1700 — UV-Vis Analytical Workbench (Python / Streamlit, SATU BERKAS)
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
                            title="Laporan Analisis Spektrofotometri UV-Vis", author=model.get("praktikan") or "SHIMADZU UV-1700")
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
        canvas.setFillColor(colors.HexColor("#a5b4fc")); canvas.setFont(F, 8.5); canvas.drawString(18 * mm, A4[1] - 11 * mm, "SHIMADZU UV-1700  ·  UV-Vis Analytical Workbench")
        canvas.setFillColor(colors.white); canvas.setFont(FB, 15); canvas.drawString(18 * mm, A4[1] - 19 * mm, "LAPORAN ANALISIS SPEKTROFOTOMETRI UV-VIS")
        canvas.setFillColor(colors.HexColor("#c7c9ef")); canvas.setFont(F, 9); canvas.drawString(18 * mm, A4[1] - 26 * mm, (model.get("judul") or "") + ("  ·  " + model["tanggal"] if model.get("tanggal") else ""))
        canvas.setStrokeColor(colors.HexColor("#e1e4f2")); canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
        canvas.setFillColor(colors.HexColor(MUTED)); canvas.setFont(F, 7.5)
        canvas.drawString(18 * mm, 9 * mm, f"SHIMADZU UV-1700 · dibuat {stamp}"); canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Halaman {doc_.page}")
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
.wrap.tour{grid-template-columns:minmax(0,1fr) 330px}
.chips{display:flex;flex-wrap:wrap;gap:4px}.chips button{font-size:11.5px;padding:3px 8px;border-radius:99px;color:var(--muted)}.chips button.on{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}
.cmpi{margin-top:8px}
.wrap.slave,.wrap.slave.tour{grid-template-columns:1fr!important;padding:0}.wrap.slave .side,.wrap.slave .views,.wrap.slave .tools,.wrap.slave .badge{display:none!important}.wrap.slave .stage{border:0;border-radius:0}
.tourcard{display:none}.wrap.tour .tourcard{display:block}.wrap.tour .plain{display:none}
.steps{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px}
.steps button{width:28px;height:28px;padding:0;border-radius:50%;font:600 12px 'JetBrains Mono',monospace;color:var(--muted)}
.steps button.on{background:var(--acc);border-color:var(--acc);color:#fff}.steps button.seen{background:#e8ecff}
.tt{font:700 15px 'Space Grotesk',Inter,sans-serif;margin:0 0 4px;color:#1e1b4b}.tx{font-size:12.5px;color:#2b2f55;margin:0 0 8px}
.kv{display:grid;grid-template-columns:84px 1fr;gap:6px 10px;font-size:12.5px;color:#2b2f55}.kv b{color:#4f46e5;font-size:11px;line-height:1.35}
.eq{font:500 12.5px 'JetBrains Mono',monospace;background:linear-gradient(135deg,#eef0ff,#f5efff);border:1px solid #e0dcff;border-radius:10px;padding:7px 10px;color:#2b2177;margin-bottom:8px}
.nav2{display:flex;gap:6px}.nav2 button{flex:1;font-weight:600}.nav2 .pri{background:var(--acc);border-color:var(--acc);color:#fff}
.meter{display:grid;grid-template-columns:auto 1fr auto;gap:4px 8px;align-items:center;font:500 11px 'JetBrains Mono',monospace;color:var(--muted)}
.meter .b{height:9px;border-radius:5px;background:#eceef9;overflow:hidden}.meter .b i{display:block;height:100%;border-radius:5px}
.meter b{color:var(--fg)}
.rg{margin:6px 0 2px;font-size:12px;display:flex;justify-content:space-between}
@media (max-width:760px){.wrap,.wrap.tour{grid-template-columns:1fr}.stage{height:380px}}
</style></head><body>
<div class="wrap" id="wrap"><div class="stage"><canvas id="c" aria-label="Model 3D spektrofotometer UV-1700"></canvas>
<div class="badge" id="badge">UV-1700 · MODEL 3D</div>
<div class="tools"><button data-z="0.85" title="Perbesar">+</button><button data-z="1.18" title="Perkecil">−</button><button id="rs" title="Tampilan awal">⟲</button></div>
<div class="views"><button data-v="iso" class="on">Isometrik</button><button data-v="depan">Depan</button><button data-v="samping">Kanan</button><button data-v="belakang">Belakang</button><button data-v="atas">Atas</button><button data-v="kompartemen">Kompartemen</button></div></div>
<div class="side"><div class="card" id="cmp"><h4>Pilih komponen · kamera langsung zoom</h4><div class="chips" id="chips"></div><div class="cmpi" id="cmpi"><div class="hint">Pilih komponen atau klik dua kali pada bagian di model.</div></div></div>
<div class="card tourcard"><div class="steps" id="steps"></div><div class="tt" id="tt"></div><div class="tx" id="tx"></div><div class="nav2"><button id="prev">◀ Sebelumnya</button><button id="next" class="pri">Berikutnya ▶</button></div></div>
<div class="card tourcard"><h4>Animasi jalur cahaya</h4><div class="hint" id="lstage" style="min-height:34px;color:#2b2f55;font-size:12.5px">Tekan tombol untuk melihat perjalanan cahaya dari lampu sampai nilai absorbansi (contoh: C = 2,5 mg/L).</div>
<div class="meter" style="margin:6px 0"><span>I₀</span><span class="b"><i id="lb0" style="background:#4f46e5;width:0"></i></span><b id="lv0">0</b><span>I</span><span class="b"><i id="lb1" style="background:#0d9488;width:0"></i></span><b id="lv1">0</b><span>T</span><span></span><b id="lvT">–</b><span>A</span><span></span><b id="lvA">–</b></div>
<div class="nav2"><button id="lplay" class="pri">▶ Putar animasi cahaya</button></div></div>
<div class="card tourcard" style="display:none!important"><h4>Eksperimen: λ dan konsentrasi</h4>
<div class="rg"><span>λ = <b id="lv2"></b> nm</span><span id="lamp2" class="hint"></span></div><input type="range" id="lam2" min="190" max="1100" step="1">
<div class="rg"><span>C = <b id="cv2"></b> mg/L</span><span class="hint">ε·b = 0,20 L mg⁻¹ (b = 1 cm)</span></div><input type="range" id="conc" min="0" max="10" step="0.1" value="2.5">
<div class="meter" style="margin-top:6px"><span>I₀</span><span class="b"><i id="b0" style="background:#4f46e5;width:100%"></i></span><b id="v0">100</b><span>I</span><span class="b"><i id="b1" style="background:#0d9488"></i></span><b id="v1"></b><span>%T</span><span class="b"><i id="b2" style="background:#d4a017"></i></span><b id="v2"></b><span>A</span><span class="b"><i id="b3" style="background:#e11d48"></i></span><b id="v3"></b></div></div>
<div class="card tourcard"><h4>Tampilan</h4><div class="tg"><button data-t="top" aria-pressed="false">Pisahkan bagian atas</button><button data-t="side" aria-pressed="false">Pisahkan bagian samping</button><button data-t="xray" aria-pressed="false">Tembus pandang</button><button data-t="lid" aria-pressed="false">Buka tutup (masukkan kuvet)</button><button data-t="shell" aria-pressed="false">Pasang cangkang</button><button data-t="pan" aria-pressed="false">Mode geser</button></div><div class="hint" style="margin-top:6px">Seret = putar · Shift+seret atau Mode geser = geser · roda/cubit = zoom (bisa masuk ke dalam).</div></div>
<div class="card plain"><h4>Tampilan</h4><div class="tg">
<button data-t="shell" aria-pressed="false">Pasang cangkang</button><button data-t="pan" aria-pressed="false">Mode geser</button><button data-t="lid" aria-pressed="false">Buka tutup</button><button data-t="xray" aria-pressed="false">Tembus pandang</button><button data-t="expl" aria-pressed="false">Terurai</button><button data-t="beam" aria-pressed="false">Jalur cahaya</button><button data-t="labels" aria-pressed="false">Label</button><button data-t="auto" aria-pressed="false">Putar otomatis</button></div></div>
<div class="card plain"><h4><span class="sw" id="sw"></span>λ = <span id="lv"></span> nm</h4><input type="range" id="lam" min="190" max="1100" step="1"><div class="hint" id="lamp"></div></div>
<div class="card plain"><h4 id="ti">Komponen</h4><div class="hint" id="info">Seret untuk memutar, roda/cubit untuk zoom, klik komponen untuk penjelasan.</div></div>
<div class="hint plain">Model skematis disederhanakan; proporsi mengikuti unit UV-1700 PharmaSpec di laboratorium, bukan gambar teknik pabrikan.</div></div></div>
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
  function rotZ(p, piv, ang) { const s = Math.sin(ang), c = Math.cos(ang); const x = p[0] - piv[0], y = p[1] - piv[1]; return [piv[0] + x * c - y * s, piv[1] + x * s + y * c, p[2]]; }
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
    console: ['Panel kendali', 'Rumah LCD dan papan tombol pada bagian depan atas; cangkang lain dilepas pada tampilan terbuka.'],
    warn: ['Label peringatan', 'WARNING — HOT SURFACE: dinginkan lampu hingga suhu ruang sebelum penggantian.']
  };
  const COMPART = new Set(['compartment', 'holderR', 'holderS', 'cuvR', 'cuvS']);

  class Model3D {
    constructor(canvas, opt) {
      this.cv = canvas; this.ctx = canvas.getContext('2d'); this.opt = opt || {};
      this.yaw = -32; this.pitch = 24; this.dist = 54; this.target = [0, 9, 0];
      this.lid = 0; this.lidT = 0; this.expl = 0; this.explT = 0; this.topX = 0; this.topT = 0; this.sideX = 0; this.sideT = 0; this.xray = 0; this.xrayT = 0;
      this.beam = true; this.labels = false; this.auto = false; this.sel = null; this.hl = new Set(); this.beamLimit = 99; this.flow = false; this.tour = false; this.ph = null; this.dispAbs = null; this.onLight = null; this._dots = [];
      this.lam = 550; this.lampSwitch = 340.8; this.abs = 0; this.status = 'READY'; this.solColor = '#8e2c8f';
      this.shell = 0; this.shellT = 0; this.fanAng = 0; this.fanW = 1; this.panMode = false; this.compSel = false; this._vis = true;
      this._build(); this._bind(); this._loop = this._loop.bind(this);
      this._ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(() => this.draw()) : null;
      if (this._ro) this._ro.observe(canvas.parentElement || canvas);
      if (typeof IntersectionObserver !== 'undefined') new IntersectionObserver(es => { this._vis = es[0].isIntersecting; if (this._vis) this._kick(); }).observe(canvas);
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
            P.push({ id, grp, col: fc, alpha: o.alpha == null ? 1 : o.alpha, emi: o.emi || 0, bias: o.bias || 0, tf: o.tf || null, ex: o.ex || null, key: k,
              v: [bl(i / nu, j / nv), bl((i + 1) / nu, j / nv), bl((i + 1) / nu, (j + 1) / nv), bl(i / nu, (j + 1) / nv)],
              text: (o.text && k === o.text.face && nu === 1 && nv === 1) ? o.text : null, dyn: o.dyn || null });
          }
        }
      };

      const basis = a => { const w = norm(a); const h = Math.abs(w[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0]; const u = norm(cross(h, w)); const v = cross(w, u); return [u, v, w]; };
      const pushQ = (id, grp, v, col, o) => P.push({ id, grp, col, alpha: o.alpha == null ? 1 : o.alpha, emi: o.emi || 0, bias: o.bias || 0, tf: o.tf || null, ex: o.ex || null, key: 'q', v, text: null, dyn: o.dyn || null, ds: !!o.ds, fan: o.fan || null });
      // benda putar (lathe): profil [[jari-jari, tinggi], ...] diputar mengelilingi sumbu ax dari titik p0
      const lathe = (id, grp, p0, ax, prof, col, o, n) => {
        o = o || {}; n = n || 14; const [u, v, w] = basis(ax);
        const pt = (r, h, i) => { const a = i / n * 2 * Math.PI, c = Math.cos(a), s = Math.sin(a); return [p0[0] + r * (u[0] * c + v[0] * s) + h * w[0], p0[1] + r * (u[1] * c + v[1] * s) + h * w[1], p0[2] + r * (u[2] * c + v[2] * s) + h * w[2]]; };
        for (let k = 0; k < prof.length - 1; k++) for (let i = 0; i < n; i++) {
          const [r0, h0] = prof[k], [r1, h1] = prof[k + 1]; const cc = (o.segCols && o.segCols[k]) || col;
          const q = [pt(r0, h0, i), pt(r0, h0, i + 1), pt(r1, h1, i + 1), pt(r1, h1, i)];
          pushQ(id, grp, r0 === 0 ? [q[3], q[0], q[1], q[2]] : q, cc, o);
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
      box('body', 'walls', -23, 23, 3, 13.5, 23.9, 24.5, BEIGE, Object.assign({ ex: [0, 0, 17] }, wallO));
      box('body', 'walls', -23, 23, 3, 13.5, -24.5, -23.9, BEIGE, Object.assign({ ex: [0, 0, -17] }, wallO));
      box('body', 'walls', -23, -22.4, 3, 13.5, -23.9, 23.9, BEIGE, Object.assign({ ex: [-19, 0, 0] }, wallO));
      box('body', 'walls', 22.4, 23, 3, 13.5, -23.9, 23.9, BEIGE, Object.assign({ ex: [19, 0, 0] }, wallO));
      // panel depan kompartemen (pintu krem lebih terang) dan pelat identitas
      box('compartment', 'walls', -16, -1, 4, 12.5, 24.5, 24.8, '#cfb575', { bias: -4, skip: ['b'], cell: 99, ex: [0, 0, 17] });
      box('plate', 'walls', 9, 20, 6.5, 10.5, 24.5, 24.7, '#f7f6f0', { bias: -4.5, skip: ['b'], cell: 99, ex: [0, 0, 17],
        text: { face: 'f', w: 110, h: 40, lines: [['UV-1700', 8, 18, 'left'], ['PharmaSpec', 70, 18, 'left'], ['UV-VIS SPECTROPHOTOMETER', 8, 30, 'left']], color: '#22263a', font: '600 12px Arial' } });
      // sisi kanan: kisi ventilasi (satu kisi berpendar oranye saat WI menyala), slot IC card, EJECT, CONTRAST, label peringatan
      for (let i = 0; i < 8; i++) box('vent', 'walls', 23, 23.25, 5.6 + i * 0.95, 6.1 + i * 0.95, -20, -11, '#bfa868', { bias: -4, cell: 99, ex: [19, 0, 0], dyn: i === 4 ? 'glow' : null });
      box('card', 'walls', 23, 23.2, 11, 11.8, 1, 10, '#2b2f36', { bias: -4, cell: 99, ex: [19, 0, 0] });
      box('card', 'walls', 23, 23.35, 11.05, 11.75, -0.6, 0.8, '#f2efe4', { bias: -4.2, cell: 99, ex: [19, 0, 0] });
      box('card', 'walls', 23, 23.45, 10.9, 11.9, 11.2, 13.4, '#ebe6d6', { bias: -4.2, cell: 99, ex: [19, 0, 0] });
      box('warn', 'walls', 23, 23.15, 4.6, 8.4, 4, 13, '#fafafa', { bias: -4.3, cell: 99, ex: [19, 0, 0] });
      box('warn', 'walls', 23, 23.2, 7, 8.2, 7.5, 12.6, '#f6d21f', { bias: -4.5, cell: 99, ex: [19, 0, 0] });
      // bangku optik
      box('pcb', 'inner', -21.8, -12, 3, 3.6, -22, -6, '#1d6b45');
      [[-20, -19], [-17, -15], [-14, -10]].forEach(([a, b], k) => box('pcb', 'inner', a, b, 3.6, 4.4, -20 + k * 4, -18 + k * 4, '#15181d'));

      // kipas pendingin aksial: rangka, hub, tujuh bilah melengkung yang berputar
      const FC = [-18.3, 6, -22.7];
      box('fan', 'inner', -21.3, -20.6, 3, 9, -23.6, -21.8, '#3a4048'); box('fan', 'inner', -16, -15.3, 3, 9, -23.6, -21.8, '#3a4048');
      box('fan', 'inner', -20.6, -16, 3, 3.7, -23.6, -21.8, '#3a4048'); box('fan', 'inner', -20.6, -16, 8.3, 9, -23.6, -21.8, '#3a4048');
      box('fan', 'inner', -21.3, -15.3, 3, 9, -23.6, -23.4, '#22252a', { cell: 99 });
      lathe('fan', 'inner', [FC[0], FC[1], -23.4], [0, 0, 1], [[0, 0], [1.0, 0], [1.0, 1.4], [0, 1.4]], '#16181c', { segCols: ['#16181c', '#16181c', '#3a3f47'] }, 14);
      for (let b = 0; b < 7; b++) {
        const th = b * 2 * Math.PI / 7, pp = (r, ang, z) => [FC[0] + r * Math.cos(ang), FC[1] + r * Math.sin(ang), FC[2] + z];
        pushQ('fan', 'inner', [pp(1.0, th, -0.3), pp(2.7, th + 0.22, -0.3), pp(2.7, th + 0.72, 0.3), pp(1.0, th + 0.55, 0.3)], b % 2 ? '#2c3037' : '#3b4048', { ds: true, fan: FC });
      }
      // catu daya: trafo berlilitan tembaga
      lathe('fan', 'inner', [-7.5, 3, -16.5], [0, 1, 0], [[0, 0], [2.6, 0], [2.6, 3.8], [0, 3.8]], '#2b2d33', { segCols: ['#2b2d33', '#2b2d33', '#aab0b8'] }, 16);
      lathe('fan', 'inner', [-7.5, 4.0, -16.5], [0, 1, 0], [[2.72, 0], [2.72, 1.7]], '#b87333', { ds: true }, 16);

      [[-14, -8.5], [-16.2, -8.5], [-18.4, -8.5], [-20.4, -10.5]].forEach(([x, z]) => lathe('pcb', 'inner', [x, 3.6, z], [0, 1, 0], [[0, 0], [0.75, 0], [0.75, 1.8], [0, 1.8]], '#1c3f8f', { segCols: ['#1c3f8f', '#1c3f8f', '#c9ccd2'] }, 10));
      box('lamphouse', 'inner', 14, 22.2, 3, 4, -23.4, -9, '#3a4048');
      box('lamphouse', 'inner', 21.6, 22.2, 4, 12, -23.4, -9, '#4a515b', { alpha: 0.9 });

      // lampu deuterium: soket, bohlam kuarsa berkubah, anoda, jendela pancar
      lathe('d2', 'inner', [19.1, 4, -19.8], [0, 1, 0], [[0, 0], [1.6, 0], [1.6, 1.4], [1.0, 1.4], [1.0, 2.0]], '#3a4048', {}, 16);
      lathe('d2', 'inner', [19.1, 6, -19.8], [0, 1, 0], [[1.25, 0], [1.25, 3.9], [1.0, 5.0], [0.55, 5.7], [0, 6.0]], '#d9d4ff', { alpha: 0.62, dyn: 'd2' }, 16);
      lathe('d2', 'inner', [19.1, 7.2, -19.8], [0, 1, 0], [[0, 0], [0.7, 0], [0.7, 3.4], [0, 3.4]], '#2e3238', {}, 10);
      lathe('d2', 'inner', [18.55, 9.5, -19.8], [-1, 0, 0], [[0, 0], [0.38, 0], [0.38, 0.45], [0, 0.45]], '#ffffff', { dyn: 'd2', bias: -0.5 }, 10);
      // lampu wolfram-iodin: kapsul halogen horizontal, filamen, penyangga
      lathe('wi', 'inner', [19.1, 4, -13], [0, 1, 0], [[0, 0], [1.0, 0], [1.0, 3.2], [0.5, 3.2], [0.5, 5.0], [0, 5.0]], '#4a515b', {}, 12);
      lathe('wi', 'inner', [17.6, 9.5, -13], [1, 0, 0], [[0, 0], [0.7, 0], [0.95, 0.4], [0.95, 2.6], [0.7, 3.0], [0, 3.0]], '#fff1b8', { alpha: 0.5, dyn: 'wi' }, 14);
      lathe('wi', 'inner', [17.9, 9.5, -13], [1, 0, 0], [[0, 0], [0.28, 0], [0.28, 2.4], [0, 2.4]], '#fff1b8', { dyn: 'wi' }, 8);
      lathe('mirror', 'inner', [15.05, 9.5, -16], [1, 0, 0], [[0, 0], [2.3, 0], [2.3, 0.5], [0, 0.5]], '#cfd8e6', { ds: true, tf: [['y', [15.3, 0, -16], -25]], segCols: ['#7b8794', '#aeb8c6', '#eef3fa'] }, 20);
      lathe('mirror', 'inner', [15.3, 3.2, -16], [0, 1, 0], [[0, 0], [1.4, 0], [1.4, 2.6], [0.4, 2.6], [0.4, 5.6], [0, 5.6]], '#2d3138', {}, 12);
      box('slit1', 'inner', 11, 11.4, 6, 13, -18.5, -16.4, DARK); box('slit1', 'inner', 11, 11.4, 6, 13, -15.6, -13.5, DARK);
      box('mono', 'inner', -2, 10.5, 3, 4, -23.2, -9, '#2d323a');
      box('mono', 'inner', -2, 10.5, 4, 12.5, -23.2, -22.7, '#3a4048', { alpha: 0.55 });
      box('mono', 'inner', -2, -1.5, 4, 12.5, -22.7, -9, '#3a4048', { alpha: 0.55 });

      { // kisi difraksi cekung: segmen-segmen busur beriridesensi
        const GC = [3.5, -14.2], GR = 7, GR2 = 8.1, y0 = 5.5, y1 = 12.5, NA = 8, cols = ['#c9d6ea', '#d9c9ea', '#c9eadb', '#eadcc9'];
        const A = (R, a, y) => [GC[0] + R * Math.sin(a * D2R), y, GC[1] - R * Math.cos(a * D2R)];
        for (let i = 0; i < NA; i++) {
          const a0 = -24 + 48 * i / NA, a1 = -24 + 48 * (i + 1) / NA;
          pushQ('grating', 'inner', [A(GR, a0, y0), A(GR, a1, y0), A(GR, a1, y1), A(GR, a0, y1)], cols[i % 4], {});
          pushQ('grating', 'inner', [A(GR2, a1, y0), A(GR2, a0, y0), A(GR2, a0, y1), A(GR2, a1, y1)], '#6d7a8c', {});
          pushQ('grating', 'inner', [A(GR, a0, y1), A(GR, a1, y1), A(GR2, a1, y1), A(GR2, a0, y1)], '#8a97aa', {});
          pushQ('grating', 'inner', [A(GR, a0, y0), A(GR2, a0, y0), A(GR2, a1, y0), A(GR, a1, y0)], '#55606f', {});
        }
        lathe('grating', 'inner', [3.5, 3, -21.7], [0, 1, 0], [[0, 0], [1.6, 0], [1.6, 2.5], [0, 2.5]], '#3a4048', {}, 12);
      }
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

      [['detR', 3], ['detS', 13]].forEach(([id, z]) => {
        lathe(id, 'inner', [-21.8, 9.5, z], [1, 0, 0], [[0, 0], [1.25, 0], [1.25, 1.9], [1.7, 1.9], [1.7, 2.2], [0, 2.2]], '#aeb4bd', {}, 14);
        lathe(id, 'inner', [-19.58, 9.5, z], [1, 0, 0], [[0, 0], [0.85, 0], [0.85, 0.05], [0, 0.05]], '#1e3a8a', { emi: 0.08, bias: -0.4 }, 12);
        [-0.6, 0, 0.6].forEach(dz => lathe(id, 'inner', [-22.8, 9.5, z + dz], [1, 0, 0], [[0, 0], [0.12, 0], [0.12, 1.0], [0, 1.0]], '#c8a24a', {}, 6));
        box(id, 'inner', -21.0, -20.2, 3, 8.2, z - 0.7, z + 0.7, '#2a3346');
      });
      const covO = { cell: 4.5 };
      box('cover', 'cover', -23, -15, 13.5, 15, -24.5, 24.5, BEIGE2, covO);
      box('cover', 'cover', -2, 23, 13.5, 15, -24.5, 7, BEIGE2, covO); box('cover', 'cover', -2, 1.5, 13.5, 15, 7, 24.5, BEIGE2, covO); box('cover', 'cover', 22.4, 23, 13.5, 15, 7, 24.5, BEIGE2, covO);
      box('cover', 'cover', -15, -2, 13.5, 15, -24.5, -2, BEIGE2, covO);
      box('cover', 'cover', -15, -2, 13.5, 15, 20, 24.5, BEIGE2, covO);
      for (let i = 0; i < 14; i++) box('vent', 'cover', 6 + i * 1.15, 6.55 + i * 1.15, 15, 15.25, -23.5, -15.5, '#c6ae6c');
      box('lid', 'lid', -15.2, -1.8, 15, 16.6, -2.2, 20.2, '#dcc589', { cell: 4, faceCol: { u: '#111215' } });
      box('console', 'panel', 1.5, 22.4, 9, 15, 7, 24.5, BEIGE, { cell: 4.5 }); box('console', 'panel', 6, 17, 9, 16.5, 7.4, 10.4, BEIGE, { cell: 4.5 });
      box('keypad', 'panel', 2, 21, 15, 16.2, 13, 23.5, CREAM, { bias: -0.6 });
      const keys = [];
      for (let r = 0; r < 4; r++) for (let c = 0; c < 6; c++) keys.push([2.8 + c * 3.0, 14.2 + r * 2.2]);
      keys.forEach(([x, z], k) => box('keypad', 'panel', x, x + 1.9, 16.2, 16.55, z, z + 1.3, k % 6 === 5 ? '#7c93c9' : '#cfd4dc', { bias: -0.8 }));
      box('lcd', 'panel', 9, 14, 15, 17.5, 7.5, 10, '#ddd6bd', { bias: -0.6 });
      const tilt = [['x', [11.5, 16.5, 9], -18]];
      box('lcd', 'panel', 3.5, 19.5, 16.5, 28.5, 8.2, 10.2, '#ebe5cf', { tf: tilt, bias: -0.7, cell: 99 });
      box('lcd', 'panel', 6, 17, 18.3, 26.6, 10.2, 10.35, '#a99af2', { tf: tilt, bias: -4, emi: 0.08, dyn: 'lcd', text: { face: 'f', w: 130, h: 96, lines: [] } });
    }
    setLam(l) { this.lam = l; this.draw(); }
    toggle(what, on) {
      const flip = (cur) => on == null ? (cur ? 0 : 1) : (on ? 1 : 0);
      if (what === 'lid') this.lidT = flip(this.lidT);
      if (what === 'expl') { var nv = flip(this.explT); this.topT = nv; this.sideT = nv; }
      if (what === 'top') this.topT = flip(this.topT);
      if (what === 'side') this.sideT = flip(this.sideT);
      if (what === 'xray') this.xrayT = flip(this.xrayT);
      if (what === 'shell') this.shellT = flip(this.shellT);
      if (what === 'pan') this.panMode = on == null ? !this.panMode : !!on;
      if (what === 'beam') this.beam = on == null ? !this.beam : !!on;
      if (what === 'labels') this.labels = on == null ? !this.labels : !!on;
      if (what === 'auto') this.auto = on == null ? !this.auto : !!on;
      this._kick(); return this.state();
    }
    state() { return { lid: this.lidT, expl: this.explT, top: this.topT, side: this.sideT, xray: this.xrayT, shell: this.shellT, pan: this.panMode, beam: this.beam, labels: this.labels, auto: this.auto }; }
    view(name) {
      const V = { depan: [0, 8], samping: [90, 10], atas: [0, 88], belakang: [180, 14], iso: [-32, 24], kompartemen: [-8, 55] };
      const v = V[name] || V.iso; this._anim = { from: [this.yaw, this.pitch], to: v, t: 0 }; this._kick();
    }
    zoom(f) { this.dist = Math.max(14, Math.min(260, this.dist * f)); this.draw(); }
    focus(v) { this._anim = { from: [this.yaw, this.pitch, this.dist, this.target.slice()], to: v, t: 0, full: true }; this._kick(); }
    reset() { this.sel = null; this.compSel = false; if (!this.tour) this.hl = new Set(); this.focus([-32, 24, 54, [0, 9, 0]]); if (this.opt.onPick) this.opt.onPick(null); }
    _kick() { if (!this._running) { this._running = true; this._last = performance.now(); requestAnimationFrame(this._loop); } }
    _loop(now) {
      const dt = Math.min(0.05, (now - this._last) / 1000); this._last = now;
      const ease = (cur, tgt, sp) => { const d = tgt - cur; return Math.abs(d) < 0.002 ? tgt : cur + d * Math.min(1, dt * sp); };
      this.shell = ease(this.shell, this.shellT, 4); this.fanW = ease(this.fanW, this.status === 'OFF' ? 0 : 1, 1.5); this.fanAng += this.fanW * dt * 9;
      this.lid = ease(this.lid, this.lidT, 5); this.topX = ease(this.topX, this.topT, 4); this.sideX = ease(this.sideX, this.sideT, 4); this.explT = Math.max(this.topT, this.sideT); this.expl = Math.max(this.topX, this.sideX); this.xray = ease(this.xray, this.xrayT, 5);
      if (this._anim) {
        const a = this._anim; a.t = Math.min(1, a.t + dt * 2.2); const k = a.t < 0.5 ? 2 * a.t * a.t : 1 - Math.pow(-2 * a.t + 2, 2) / 2;
        let dy = a.to[0] - a.from[0]; while (dy > 180) dy -= 360; while (dy < -180) dy += 360;
        this.yaw = a.from[0] + dy * k; this.pitch = a.from[1] + (a.to[1] - a.from[1]) * k;
        if (a.full) { this.dist = a.from[2] + (a.to[2] - a.from[2]) * k; this.target = [0, 1, 2].map(i => a.from[3][i] + (a.to[3][i] - a.from[3][i]) * k); }
        if (a.t >= 1) this._anim = null;
      }
      if (this.ph) this._phStep(dt);
      if (this.auto && !this._drag) this.yaw += dt * 18;
      if ((this.flow || this.fanW > 0.02) && !this._anim && !this._drag && now - (this._ld || 0) < 30) { requestAnimationFrame(this._loop); return; }
      this._ld = now; this.draw();
      const busy = this.shell !== this.shellT || (this.fanW > 0.02 && this._vis) || this.lid !== this.lidT || this.expl !== this.explT || this.topX !== this.topT || this.sideX !== this.sideT || this.xray !== this.xrayT || this._anim || (this.auto && !this._drag) || (this.flow && this.beam) || this.ph;
      if (busy) requestAnimationFrame(this._loop); else this._running = false;
    }
    _phPath() {
      const y = 9.5, lampUV = this.lam < this.lampSwitch, src = lampUV ? [19.1, y, -19.8] : [19.1, y, -13];
      return { com: [src, [15.3, y, -16], [11.2, y, -15], [3.5, y, -21], [3, y, -9.2], [3, y, 3]], R: [[3, y, 3], [-8.5, y, 3], [-19.6, y, 3]], S: [[3, y, 3], [3, y, 13], [-8.5, y, 13], [-19.6, y, 13]] };
    }
    lightStart() {
      const P = this._phPath(), len = pl => pl.slice(1).reduce((a, p, i) => a + Math.hypot(p[0] - pl[i][0], p[2] - pl[i][2]), 0);
      this.ph = { t: 0, N: 30, gap: 0.22, speed: 12, T: Math.pow(10, -Math.max(0, this.abs)), P, Lc: len(P.com), LR: len(P.R), LS: len(P.S), LCell: len(P.S.slice(0, 3)), arrR: 0, arrS: 0, cnt: {}, stage: 0 };
      const EL = [[-20.6, 12, 8], [-17, 6, -6], [-17, 5, -14], [6, 6, -14], [11.5, 26, 6]]; this.ph.EL = EL; this.ph.Le = EL.slice(1).reduce((q, p, i) => q + Math.hypot(p[0] - EL[i][0], p[1] - EL[i][1], p[2] - EL[i][2]), 0); this.ph.lead = P.com[0];
      this.dispAbs = 0; this.xrayT = 1; this.lidT = 0; this._anim = null; this._kick();
    }
    _phAt(pl, d) {
      let acc = 0;
      for (let i = 1; i < pl.length; i++) { const L = Math.hypot(pl[i][0] - pl[i - 1][0], pl[i][2] - pl[i - 1][2]); if (d <= acc + L || i === pl.length - 1) { const u = Math.max(0, Math.min(1, (d - acc) / L)); return [pl[i - 1][0] + (pl[i][0] - pl[i - 1][0]) * u, pl[i - 1][1], pl[i - 1][2] + (pl[i][2] - pl[i - 1][2]) * u]; } acc += L; }
      return pl[pl.length - 1];
    }
    _phStep(dt) {
      const h = this.ph; h.t += dt; this._dots = [];
      const col = this.lam < 380 ? '#c4b5fd' : lamColor(this.lam), pre = '#fffbe8';
      const cum = [0]; for (let i = 1; i < h.P.com.length; i++) cum.push(cum[i - 1] + Math.hypot(h.P.com[i][0] - h.P.com[i - 1][0], h.P.com[i][2] - h.P.com[i - 1][2]));
      let lead = h.lead || h.P.com[0];
      for (let k = 0; k < h.N; k++) {
        const d = (h.t - k * h.gap) * h.speed; if (d < 0) continue;
        const absorbed = (((k + 0.5) * 0.6180339887) % 1) >= h.T;
        if (d <= h.Lc) { const seg = cum.findIndex((c, i) => i > 0 && d <= c), p = this._phAt(h.P.com, d); this._dots.push({ p, c: seg <= 2 ? pre : col, a: 1 }); if (k === 0) { lead = p; h.stage = Math.max(h.stage, Math.max(0, seg - 1)); } continue; }
        const e = d - h.Lc;
        if (k === 0) { h.stage = Math.max(h.stage, 5); lead = this._phAt(h.P.R, Math.min(e, h.LR)); }
        if (e <= h.LR + 2) { if (e <= h.LR) this._dots.push({ p: this._phAt(h.P.R, e), c: col, a: 1 }); else if (!h.cnt['r' + k]) { h.cnt['r' + k] = 1; h.arrR++; } }
        if (e <= h.LS + 2) {
          if (absorbed && e > h.LCell - 2.5) { const f = Math.max(0, 1 - (e - (h.LCell - 2.5)) / 2.5); if (f > 0) this._dots.push({ p: this._phAt(h.P.S, Math.min(e, h.LCell)), c: '#f97316', a: f }); }
          else if (e <= h.LS) this._dots.push({ p: this._phAt(h.P.S, e), c: col, a: 1 });
          else if (!h.cnt['s' + k]) { h.cnt['s' + k] = 1; h.arrS++; }
        }
      }
      h.lead = lead;
      if (h.arrR + h.arrS > 0) h.stage = Math.max(h.stage, 6);
      const pend = (h.N - 1) * h.gap + (h.Lc + h.LS + 2) / h.speed + 0.2;
      if (h.t >= pend) {
        if (h.e0 == null) h.e0 = h.t;
        const te = h.t - h.e0, EL = h.EL; h.stage = 7;
        for (let k = 0; k < 6; k++) { const u = te * 0.34 - k * 0.07; if (u < 0 || u > 1) continue; const p = this._phAt3(EL, u * h.Le); this._dots.push({ p, c: '#22d3ee', a: 1 }); if (k === 0) h.lead = p; }
        if (te > 3.2) { h.stage = 8; h.lead = [11.5, 26, 6]; const q = Math.min(1, (te - 3.2) / 1.2); this.dispAbs = this.abs * q * (2 - q); }
        if (te > 4.6) { this.dispAbs = this.abs; h.done = true; }
      }
      if (!this._drag) {
        const G = { 0: [-34, 40, 70], 1: [-30, 42, 64], 2: [-18, 44, 62], 3: [-8, 46, 62], 4: [6, 50, 66], 5: [12, 56, 72], 6: [20, 50, 68], 7: [-14, 34, 74], 8: [0, 14, 52] }[h.stage];
        const r = Math.min(1, dt * 2.6), L = h.lead, T = this.target;
        this.target = [T[0] + (L[0] - T[0]) * r, T[1] + (L[1] - T[1]) * r, T[2] + (L[2] - T[2]) * r];
        this.yaw += (G[0] - this.yaw) * r * 0.7; this.pitch += (G[1] - this.pitch) * r * 0.7; this.dist += (G[2] - this.dist) * r * 0.7;
      }
      if (this.onLight) this.onLight({ stage: h.stage, arrR: h.arrR, arrS: h.arrS, N: h.N, done: !!h.done });
      if (h.done) { this.ph = null; this._dots = []; }
    }
    _phAt3(pl, d) {
      let acc = 0;
      for (let i = 1; i < pl.length; i++) { const L = Math.hypot(pl[i][0] - pl[i - 1][0], pl[i][1] - pl[i - 1][1], pl[i][2] - pl[i - 1][2]); if (d <= acc + L || i === pl.length - 1) { const u = Math.max(0, Math.min(1, (d - acc) / L)); return [0, 1, 2].map(j => pl[i - 1][j] + (pl[i][j] - pl[i - 1][j]) * u); } acc += L; }
      return pl[pl.length - 1];
    }
    _xf(p, part) {
      let q = p;
      if (part.tf) for (const [ax, piv, deg] of part.tf) q = ax === 'x' ? rotX(q, piv, deg * D2R) : rotY(q, piv, deg * D2R);
      if (part.grp === 'lid') { q = rotX(q, [0, 15, -2.2], -102 * this.lid * D2R); q = [q[0], q[1] + 16 * this.topX, q[2]]; }
      if (part.grp === 'cover') q = [q[0], q[1] + 16 * this.topX, q[2]];
      if (part.grp === 'panel') q = [q[0], q[1] + 16 * this.topX, q[2]];
      if (part.fan) q = rotZ(q, part.fan, this.fanAng);
      if (part.id === 'cuvR' || part.id === 'cuvS') q = [q[0], q[1] + 9 * this.lid, q[2]];
      if (part.ex) q = [q[0] + part.ex[0] * this.sideX, q[1] + part.ex[1] * this.sideX, q[2] + part.ex[2] * this.sideX];
      if (part.id === 'grating') q = rotY(q, [3.5, 0, -21.2], (this.lam - 600) * 0.012 * D2R);
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
      const coverA = 1 - 0.82 * this.xray, wallA = 1 - 0.75 * this.xray, shl = this.shell;
      const closed = shl > 0.5 && this.xray < 0.04 && this.expl < 0.04;
      for (const part of this.parts) {
        const wv = part.v.map(p => this._xf(p, part));
        const cvs = wv.map(p => this._cam(p));
        const n = norm(cross(sub(cvs[1], cvs[0]), sub(cvs[3], cvs[0])));
        const cen = [(cvs[0][0] + cvs[2][0]) / 2, (cvs[0][1] + cvs[2][1]) / 2, (cvs[0][2] + cvs[2][2]) / 2];
        let alpha = part.alpha;
        if (part.grp === 'cover' || part.grp === 'lid') alpha *= coverA * shl;
        if (part.grp === 'panel') alpha *= coverA;
        if (part.grp === 'walls' || part.id === 'body') alpha *= wallA * shl;
        if (alpha < 0.02) continue;
        if (part.grp === 'inner' && closed && !(this.lid > 0.05 && COMPART.has(part.id))) continue;
        const facing = dot(n, sub(camPos, cen)) > 0;
        if (!facing && alpha > 0.97 && !part.ds) continue;
        if (cvs.some(c => this.dist * (1 + 0.28 * this.expl) - c[2] < 2.5)) continue;
        const wn = norm(cross(sub(wv[1], wv[0]), sub(wv[3], wv[0])));
        let k = 0.5 + 0.55 * (part.ds ? Math.abs(dot(wn, light)) : Math.max(0, dot(wn, light)));
        if (this.tour && part.grp === 'inner' && !this.hl.has(part.id)) k *= 0.62;
        let col = part.col, emi = part.emi;
        if (part.dyn === 'd2') { emi = lampUV ? 0.45 : 0.0; col = lampUV ? '#e4dcff' : '#9a96b5'; }
        if (part.dyn === 'wi') { emi = !lampUV ? 0.45 : 0.0; col = !lampUV ? '#ffe7a0' : '#a89c74'; }
        if (part.dyn === 'glow') { col = !lampUV && this.status !== 'OFF' ? '#f0873a' : '#bfa868'; emi = !lampUV && this.status !== 'OFF' ? 0.22 : 0; }
        if (part.dyn === 'cuvS') col = this.solColor;
        if (this.sel && part.id === this.sel) emi += 0.18;
        if (this.hl.has(part.id)) emi += 0.2;
        items.push({ part, pts: cvs.map(c => this._proj(c)), depth: this.dist - cen[2] + part.bias + (part.grp === 'inner' ? 0.35 : 0) + (facing ? 0 : 0.02), fill: shade(col, k, emi), alpha, facing });
      }
      if (this.beam) {
        const y = 9.5, by = this.lid > 0.5 || this.xray > 0.5 || this.expl > 0.5 || shl < 0.5;
        const src = lampUV ? [19.1, y, -19.8] : [19.1, y, -13];
        const path = [[src, [15.3, y, -16]], [[15.3, y, -16], [11.2, y, -15]], [[11.2, y, -15], [3.5, y, -21]], [[3.5, y, -21], [3, y, -9.2]], [[3, y, -9.2], [3, y, 3]], [[3, y, 3], [3, y, 13]], [[3, y, 3], [-8.5, y, 3]], [[-8.5, y, 3], [-19.6, y, 3]], [[3, y, 13], [-8.5, y, 13]], [[-8.5, y, 13], [-19.6, y, 13]]];
        path.forEach(([a, b], si) => {
          for (let kk = 0; kk < 4; kk++) {
            const t0 = kk / 4, t1 = (kk + 1) / 4;
            const A = [a[0] + (b[0] - a[0]) * t0, a[1], a[2] + (b[2] - a[2]) * t0], B = [a[0] + (b[0] - a[0]) * t1, a[1], a[2] + (b[2] - a[2]) * t1];
            const ca = this._cam(A), cb = this._cam(B);
            if (closed && !(this.lid > 0.05 && si >= 6)) continue;
            const atten = (si === 9 ? Math.max(0.12, Math.pow(10, -Math.max(0, Math.min(3, this.abs)))) : 1) * (si > this.beamLimit ? 0.12 : 1);
            items.push({ beam: true, pa: this._proj(ca), pb: this._proj(cb), depth: this.dist - (ca[2] + cb[2]) / 2 - 0.3, color: si <= 2 ? '#fffbe8' : lamColor(this.lam), uv: si > 2 && this.lam < 380, atten, show: by });
          }
        });
      }
      if (this.ph) for (const d of this._dots) { const c = this._cam(d.p), pr = this._proj(c); items.push({ dot: true, pa: pr, depth: this.dist - c[2] - 0.6, col: d.c, a: d.a, r: Math.max(2.5, 0.5 * this._f / pr[2]) }); }
      items.sort((a, b) => b.depth - a.depth);
      this._drawn = [];
      ctx.lineJoin = 'round';
      for (const it of items) {
        if (it.beam) {
          ctx.save(); ctx.globalAlpha = 0.28 * it.atten * (it.show ? 1 : 0.55); ctx.strokeStyle = it.color; ctx.lineWidth = 7; ctx.lineCap = 'round';
          ctx.beginPath(); ctx.moveTo(it.pa[0], it.pa[1]); ctx.lineTo(it.pb[0], it.pb[1]); ctx.stroke();
          ctx.globalAlpha = 0.95 * it.atten; ctx.lineWidth = 2.2; if (it.uv) ctx.setLineDash([5, 4]); if (this.flow) { ctx.setLineDash([7, 6]); ctx.lineDashOffset = -(performance.now() / 35) % 13; }
          ctx.beginPath(); ctx.moveTo(it.pa[0], it.pa[1]); ctx.lineTo(it.pb[0], it.pb[1]); ctx.stroke(); ctx.restore();
          continue;
        }
        if (it.dot) { ctx.save(); ctx.fillStyle = it.col; ctx.globalAlpha = 0.3 * it.a; ctx.beginPath(); ctx.arc(it.pa[0], it.pa[1], it.r * 2.3, 0, 6.2832); ctx.fill(); ctx.globalAlpha = it.a; ctx.beginPath(); ctx.arc(it.pa[0], it.pa[1], it.r, 0, 6.2832); ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.stroke(); ctx.restore(); continue; }
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
        ctx.fillText((this.status === 'OFF' ? '-----' : (function (a) { return a > 3 ? 'OVER' : a.toFixed(3); })(this.dispAbs == null ? this.abs : this.dispAbs)) + ' ABS', 124, 50);
        ctx.font = '500 13px "JetBrains Mono", monospace'; ctx.textAlign = 'left'; ctx.fillText(this.lam < this.lampSwitch ? 'D2' : 'WI', 40, 77);
        ctx.textAlign = 'right'; ctx.fillStyle = '#4a3fa0'; ctx.font = '500 10px "JetBrains Mono", monospace'; ctx.fillText(this.status, 124, 90);
      } else {
        ctx.fillStyle = t.color || '#222';
        t.lines.forEach(([s, x, y, al], i) => { ctx.font = i === 0 ? '700 15px Arial' : (i === 1 ? 'italic 9px Arial' : '6.5px Arial'); if (i === 2) ctx.fillStyle = '#c0392b'; ctx.textAlign = al || 'left'; ctx.fillText(s, x, y); });
      }
      ctx.restore();
    }
    _labels(ctx) {
      const A0 = [['mono', [4, 12.6, -22.5]], ['lamphouse', [16, 4.5, -22]], ['d2', [19, 12, -20]], ['fan', [-18, 9.5, -22.5]], ['wi', [19, 11, -13]], ['mirror', [15.3, 12, -16]], ['slit1', [11.2, 13.5, -15]], ['grating', [3.5, 13, -21]], ['slit2', [3, 13.5, -9.2]], ['split', [3, 12.5, 3]], ['holderR', [-8.5, 13, 3]], ['holderS', [-8.5, 13, 13]], ['detS', [-20.6, 12, 13]], ['detR', [-20.6, 12, 3]], ['pcb', [-17, 5, -14]], ['lcd', [11.5, 29, 6]], ['keypad', [12, 17, 22]], ['lid', [-8.5, 17, 9]], ['card', [23.3, 12, 6]]];
      const A = (this.tour || this.compSel) ? A0.filter(([id]) => this.hl.has(id)) : A0.filter(([id]) => id !== 'mono' && id !== 'lamphouse');
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
      cv.addEventListener('pointerdown', e => { cv.setPointerCapture(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]); start = { x: e.clientX, y: e.clientY, lx: e.clientX, ly: e.clientY, yaw: this.yaw, pitch: this.pitch, moved: false, pan: e.shiftKey || e.button === 2 || this.panMode }; this._drag = true; if (pts.size === 2) { const [a, b] = [...pts.values()]; pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), dist: this.dist }; } });
      cv.addEventListener('pointermove', e => {
        if (!pts.has(e.pointerId)) return; pts.set(e.pointerId, [e.clientX, e.clientY]);
        if (pts.size === 2 && pinch) { const [a, b] = [...pts.values()]; const d = Math.hypot(a[0] - b[0], a[1] - b[1]); this.dist = Math.max(14, Math.min(260, pinch.dist * pinch.d / Math.max(10, d))); start.moved = true; this.draw(); return; }
        if (!start) return; const dx = e.clientX - start.x, dy = e.clientY - start.y;
        if (Math.abs(dx) + Math.abs(dy) > 4) start.moved = true;
        if (start.pan) { const ddx = e.clientX - start.lx, ddy = e.clientY - start.ly; start.lx = e.clientX; start.ly = e.clientY; const sc = this.dist / this._f * 1.1, cy = Math.cos(-this.yaw * D2R), sy = Math.sin(-this.yaw * D2R), cp = Math.cos(this.pitch * D2R), sp = Math.sin(this.pitch * D2R);
          const R = [cy, 0, sy], U = [sy * sp, cp, -cy * sp]; this.target = [0, 1, 2].map(k => this.target[k] - R[k] * ddx * sc + U[k] * ddy * sc); this._anim = null; this.draw(); return; }
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
      cv.addEventListener('contextmenu', e => e.preventDefault());
      cv.addEventListener('dblclick', e => { const r = cv.getBoundingClientRect(); const id = this.pickAt(e.clientX - r.left, e.clientY - r.top); if (id && this.opt.onDbl) this.opt.onDbl(id); });
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
var INIT = Object.assign({tour: false, lam: 550, abs: 0, status: "READY", solColor: "#8e2c8f", lampSwitch: 340.8, xray: true, lid: false, compact: false, auto: false}, /*__INIT__*/null || {});
(function(){
  var SL = window.SL, $ = function(s){return document.querySelector(s)};
  INIT.xray = false;
  if (INIT.compact) $('#wrap').classList.add('compact');
  var m = new SL.Model3D($('#c'), { onPick: function(i){ $('#ti').textContent = i ? i.title : 'Komponen'; $('#info').textContent = i ? i.text : 'Klik komponen untuk penjelasan.'; } });
  m.lampSwitch = INIT.lampSwitch; m.abs = INIT.abs; m.status = INIT.status; m.solColor = INIT.solColor;
  function sync(){ var st = m.state(); document.querySelectorAll('[data-t]').forEach(function(b){ b.setAttribute('aria-pressed', String(!!st[b.dataset.t])); }); }
  function setLam(l){ m.setLam(l); $('#lv').textContent = l; $('#lam').value = l; if ($('#lv2')) { $('#lv2').textContent = l; $('#lam2').value = l; $('#lamp2').textContent = (l < m.lampSwitch ? 'Lampu D₂ · ' : 'Lampu WI · ') + (l < 400 ? 'UV' : l <= 780 ? 'tampak' : 'NIR'); } $('#sw').style.background = SL.lamColor(l); $('#lamp').textContent = (l < m.lampSwitch ? 'Lampu D2 aktif' : 'Lampu WI aktif') + ' · ' + (l < 400 ? 'UV' : l <= 780 ? 'sinar tampak' : 'NIR'); }
  var TOUR = [
    { t: 'Lampu D₂ dan WI', ids: ['d2', 'wi', 'lamphouse'], lim: 0, v: [-38, 28, 120, [17, 8, -16]],
      f: 'Menyediakan radiasi kontinu UV (D₂) dan tampak–inframerah dekat (WI).', p: 'Pelepasan listrik pada gas deuterium dan pijaran filamen wolfram menghasilkan spektrum kontinu.', m: 'Lampu deuterium berjendela kuarsa dan lampu wolfram-iodin (halogen).', h: 'Intensitas lampu menentukan derau, dan lampu berganti otomatis di 340,8 nm.' },
    { t: 'Cermin pemilih dan celah masuk', ids: ['mirror', 'slit1'], lim: 1, v: [-30, 28, 111, [13, 9, -16]],
      f: 'Memilih berkas lampu aktif dan membentuk berkas sempit menuju monokromator.', p: 'Pemantulan oleh cermin dan pembatasan berkas oleh celah.', m: 'Cermin berlapis logam reflektif dan celah logam tetap.', h: 'Menentukan energi yang masuk dan lebar pita spektral.' },
    { t: 'Kisi difraksi (monokromator)', ids: ['grating', 'mono'], lim: 2, v: [-20, 28, 111, [5, 9, -17]],
      f: 'Memisahkan cahaya polikromatik menjadi panjang gelombang tunggal.', p: 'Difraksi menurut nλ = d (sin α + sin β); sudut kisi menentukan λ.', m: 'Kisi difraksi holografik dalam rumah monokromator gelap.', h: 'Sudut kisi menetapkan λ pengukuran dan menekan sinar sesat.' },
    { t: 'Celah keluar', ids: ['slit2'], lim: 3, v: [-26, 28, 105, [3, 9, -10]],
      f: 'Meneruskan pita sempit di sekitar λ terpilih.', p: 'Seleksi spasial berkas terdispersi pada bidang fokus.', m: 'Celah logam tetap.', h: 'Menentukan resolusi spektral (SBW) dan jumlah cahaya ke detektor.' },
    { t: 'Pembagi berkas', ids: ['split', 'mirror2'], lim: 5, v: [-46, 28, 102, [3, 9, 8]],
      f: 'Membagi berkas menjadi berkas referensi dan berkas sampel.', p: 'Pemantulan dan transmisi parsial pada permukaan semi-reflektif (double-beam).', m: 'Cermin pembagi berkas semi-reflektif dan cermin pengarah.', h: 'Mengoreksi fluktuasi lampu sehingga baseline stabil.' },
    { t: 'Kompartemen sampel dan kuvet', ids: ['holderR', 'holderS', 'cuvR', 'cuvS'], lim: 9, v: [-16, 28, 99, [-8, 9, 8]],
      f: 'Menempatkan blanko dan sampel pada jalur berkas.', p: 'Analit menyerap radiasi menurut hukum Lambert–Beer, A = ε · b · C.', m: 'Kuvet kuarsa jalur 1 cm pada dudukan hitam.', h: 'Sumber utama sinyal: A sebanding dengan konsentrasi.' },
    { t: 'Detektor fotodioda', ids: ['detR', 'detS'], lim: 9, v: [-32, 28, 108, [-18, 9, 8]],
      f: 'Mengubah intensitas cahaya menjadi arus listrik.', p: 'Efek fotolistrik pada sambungan p–n semikonduktor.', m: 'Fotodioda silikon (referensi dan sampel).', h: 'Mengukur I₀ dan I untuk memperoleh T = I/I₀.' },
    { t: 'Pengolahan sinyal', ids: ['pcb', 'fan'], lim: 9, v: [-30, 28, 135, [-16, 4, -12]],
      f: 'Menguatkan, mendigitalkan, dan mengoreksi sinyal detektor.', p: 'Penguat → ADC → A = −log₁₀ T dikurangi baseline.', m: 'Papan elektronik (PCB) bermikroprosesor.', h: 'Menghasilkan absorbansi terkoreksi yang dibaca atau dikirim ke PC.' },
    { t: 'Layar dan keluaran', ids: ['lcd', 'keypad'], lim: 9, v: [-24, 28, 138, [12, 18, 10]],
      f: 'Menampilkan λ, absorbansi atau %T, dan lampu aktif.', p: 'Pengendali LCD menampilkan data digital; PC menerima data lewat RS-232C.', m: 'Layar LCD dan papan tombol membran.', h: 'Hasil akhir yang dipakai untuk kurva kalibrasi.' }
  ];
  var EPSB = 0.2, cur = 0, seen = {};
  var COMP = [
    { n: 'Lampu deuterium (D₂)', ids: ['d2'], v: [-38, 24],
      f: 'Sumber radiasi kontinu daerah ultraviolet (±190–350 nm).', p: 'Busur listrik mengeksitasi molekul D₂ menjadi D₂* yang berdisosiasi; selisih energi dipancarkan sebagai kontinum UV.', m: 'Bohlam berjendela kuarsa (transparan UV) berisi D₂ bertekanan rendah; katoda berpijar dan anoda berlubang.', h: 'Dipakai bila λ di bawah titik pergantian lampu (±340,8 nm); membatasi batas bawah UV.' },
    { n: 'Lampu wolfram-iodin (WI)', ids: ['wi'], v: [-38, 24],
      f: 'Sumber radiasi kontinu daerah tampak dan inframerah dekat (±340–1100 nm).', p: 'Filamen wolfram berpijar (±3000 K) memancarkan radiasi benda hitam; siklus halogen mengembalikan wolfram ke filamen.', m: 'Filamen W dalam kapsul kuarsa berisi gas iodin/halogen.', h: 'Dipakai untuk pengukuran zat berwarna (λ > 340,8 nm); intensitas di UV rendah.' },
    { n: 'Cermin pemilih sumber', ids: ['mirror'], v: [-30, 26],
      f: 'Memilih berkas dari D₂ atau WI sesuai panjang gelombang terpilih.', p: 'Pemantulan; motor memutar cermin ke posisi lampu yang dikehendaki.', m: 'Kaca berlapis aluminium/dielektrik pada poros motor stepper.', h: 'Menentukan lampu aktif dan stabilitas intensitas awal.' },
    { n: 'Celah masuk', ids: ['slit1'], v: [-30, 26],
      f: 'Membatasi berkas yang masuk ke monokromator.', p: 'Pembatasan spasial berkas; bersama celah keluar menentukan lebar pita spektral.', m: 'Pelat logam hitam dengan bukaan sempit.', h: 'Lebar celah mengatur resolusi spektral dan energi cahaya.' },
    { n: 'Kisi difraksi', ids: ['grating'], v: [-20, 26],
      f: 'Menyebarkan cahaya polikromatik menjadi spektrum.', p: 'Difraksi: nλ = d (sin α + sin β); sudut kisi menentukan λ keluaran.', m: 'Kisi holografik cekung berlapis aluminium pada penyangga yang diputar motor.', h: 'Menetapkan panjang gelombang pengukuran dan resolusi.' },
    { n: 'Celah keluar', ids: ['slit2'], v: [-26, 26],
      f: 'Meloloskan pita sempit di sekitar λ terpilih.', p: 'Seleksi spasial pada bidang fokus spektrum.', m: 'Pelat logam hitam dengan bukaan sempit.', h: 'Menentukan kemurnian spektral dan intensitas ke detektor.' },
    { n: 'Pembagi berkas', ids: ['split', 'mirror2'], v: [-46, 28],
      f: 'Membagi berkas menjadi berkas referensi dan berkas sampel (double-beam).', p: 'Pemantulan dan transmisi parsial pada permukaan semi-reflektif.', m: 'Pelat kuarsa semi-reflektif dan cermin pengarah.', h: 'Mengoreksi fluktuasi lampu sehingga baseline stabil.' },
    { n: 'Kuvet dan kompartemen', ids: ['holderR', 'holderS', 'cuvR', 'cuvS'], v: [-16, 34],
      f: 'Menempatkan blanko dan sampel pada jalur berkas.', p: 'Absorpsi menurut hukum Lambert–Beer: A = ε · b · C.', m: 'Kuvet kuarsa jalur 1 cm pada dudukan hitam.', h: 'Sumber utama sinyal; A sebanding dengan konsentrasi.' },
    { n: 'Detektor referensi', ids: ['detR'], v: [-70, 26],
      f: 'Mengukur intensitas berkas referensi (I₀).', p: 'Efek fotolistrik pada sambungan p–n silikon: foton menghasilkan arus.', m: 'Fotodioda Si dalam kemasan logam (TO-can) berjendela.', h: 'Penyebut pada T = I/I₀.' },
    { n: 'Detektor sampel', ids: ['detS'], v: [-70, 26],
      f: 'Mengukur intensitas berkas yang melewati sampel (I).', p: 'Efek fotolistrik pada sambungan p–n silikon.', m: 'Fotodioda Si dalam kemasan logam (TO-can) berjendela.', h: 'Pembilang pada T = I/I₀; A = log(I₀/I).' },
    { n: 'Papan elektronik', ids: ['pcb'], v: [-30, 34],
      f: 'Menguatkan, mendigitalkan, dan mengolah sinyal detektor.', p: 'Penguat → ADC → mikroprosesor menghitung %T dan A = −log₁₀ T.', m: 'PCB bermikroprosesor, IC, dan kapasitor elektrolit.', h: 'Menghasilkan nilai absorbansi terkoreksi.' },
    { n: 'Kipas dan catu daya', ids: ['fan'], v: [-18, 24],
      f: 'Mendinginkan lampu/elektronik dan menyuplai daya 230 V AC.', p: 'Kipas aksial memindahkan udara; trafo menurunkan tegangan.', m: 'Kipas tujuh bilah, trafo berlilitan tembaga.', h: 'Suhu lampu stabil menjaga baseline; gunakan UPS/stabilizer.' },
    { n: 'Layar LCD dan tombol', ids: ['lcd', 'keypad', 'console'], v: [-24, 30],
      f: 'Menampilkan λ, absorbansi atau %T, dan lampu aktif; menerima perintah.', p: 'Pengendali LCD menampilkan data digital; PC lewat RS-232C.', m: 'LCD dan papan tombol membran.', h: 'Antarmuka hasil ukur di alat.' }
  ];
  function compBounds(ids) {
    var mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
    m.parts.forEach(function (pt) { if (ids.indexOf(pt.id) < 0) return; pt.v.forEach(function (q) { var w = m._xf(q, pt); for (var k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], w[k]); mx[k] = Math.max(mx[k], w[k]); } }); });
    return { c: [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2], s: Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]) };
  }
  function pickComp(k) {
    var c = COMP[k], b = compBounds(c.ids), d = Math.max(20, Math.min(95, b.s * 2.6 + 12));
    m.hl = new Set(c.ids); m.sel = null; m.labels = true; m.compSel = true; m.beamLimit = 99; m.focus([c.v[0], c.v[1], d, b.c]);
    document.querySelectorAll('#chips button').forEach(function (x, i) { x.classList.toggle('on', i === k); });
    $('#cmpi').innerHTML = '<div class="tt">' + c.n + '</div><div class="kv"><b>Fungsi</b><span>' + c.f + '</span></div>';
  }
  (function () {
    var box = $('#chips'); if (!box) return;
    COMP.forEach(function (c, k) { var b = document.createElement('button'); b.textContent = c.n.replace(/ \(.*\)/, ''); b.addEventListener('click', function () { pickComp(k); }); box.appendChild(b); });
    var r = document.createElement('button'); r.textContent = '⟲ Tampilan awal'; r.addEventListener('click', function () { m.reset(); document.querySelectorAll('#chips button').forEach(function (x) { x.classList.remove('on'); }); $('#cmpi').innerHTML = '<div class="hint">Pilih komponen atau klik dua kali pada bagian di model.</div>'; }); box.appendChild(r);
    m.opt.onDbl = function (id) { for (var k = 0; k < COMP.length; k++) if (COMP[k].ids.indexOf(id) >= 0) { pickComp(k); return; } };
  })();

  function mixc(a, b, t) { var p = function (h) { h = h.replace('#', ''); return [0, 2, 4].map(function (i) { return parseInt(h.substr(i, 2), 16); }); }, x = p(a), y = p(b); return '#' + x.map(function (v, i) { return Math.round(v + (y[i] - v) * t).toString(16).padStart(2, '0'); }).join(''); }
  function fm(v, d) { return v.toFixed(d).replace('.', ','); }
  function meter() {
    var c = +$('#conc').value, A = EPSB * c, T = Math.pow(10, -A), I = 100 * T;
    m.abs = A; m.solColor = mixc('#eaf6f8', '#8e2c8f', Math.min(1, c / 10)); $('#cv2').textContent = fm(c, 1);
    $('#v1').textContent = fm(I, 1); $('#v2').textContent = fm(100 * T, 1) + ' %'; $('#v3').textContent = fm(A, 3);
    $('#b1').style.width = I + '%'; $('#b2').style.width = (100 * T) + '%'; $('#b3').style.width = Math.min(100, A / 2 * 100) + '%'; m._kick();
  }
  function goStep(i) {
    cur = Math.max(0, Math.min(TOUR.length - 1, i)); var s = TOUR[cur]; seen[cur] = 1;
    m.hl = new Set(s.ids); m.beamLimit = s.lim; m.focus(INIT.slave ? [s.v[0], Math.min(s.v[1], 26), s.v[2] * 0.6, s.v[3]] : s.v); m.sel = null;
    $('#tt').textContent = (cur + 1) + '. ' + s.t; $('#tx').innerHTML = '<div class="kv"><b>Fungsi</b><span>' + s.f + '</span></div>';
    document.querySelectorAll('#steps button').forEach(function (b, k) { b.classList.toggle('on', k === cur); b.classList.toggle('seen', k !== cur && !!seen[k]); });
    $('#prev').disabled = cur === 0; $('#next').textContent = cur === TOUR.length - 1 ? 'Ulangi tur ↺' : 'Berikutnya ▶';
    m.labels = true; sync();
  }
  if (INIT.tour) {
    $('#wrap').classList.add('tour'); $('#badge').textContent = 'BAGIAN DALAM UV-1700 · PRINSIP KERJA'; m.tour = true; m.flow = true;
    var sb = $('#steps'); TOUR.forEach(function (_, k) { var b = document.createElement('button'); b.textContent = k + 1; b.title = TOUR[k].t; b.addEventListener('click', function () { goStep(k); }); sb.appendChild(b); });
    $('#prev').addEventListener('click', function () { goStep(cur - 1); }); $('#next').addEventListener('click', function () { goStep(cur === TOUR.length - 1 ? 0 : cur + 1); });
    $('#conc').addEventListener('input', meter); $('#lam2').addEventListener('input', function (e) { setLam(+e.target.value); });
    INIT.xray = false;
  }
  var LST = ['1. Lampu D₂/WI memancarkan radiasi kontinu; cermin memilih sumber.', '2. Celah masuk membentuk berkas sempit menuju monokromator.', '3. Kisi difraksi menguraikan cahaya menjadi spektrum.', '4. Celah keluar meloloskan satu panjang gelombang (monokromatik).', '5. Berkas menuju pembagi berkas.', '6. Pembagi membelah berkas: referensi melewati blanko, sampel melewati kuvet sampel dan sebagian diserap (titik oranye).', '7. Fotodioda menghitung cahaya yang tiba: I₀ (referensi) dan I (sampel).', '8. Sinyal listrik dari fotodioda menuju PCB dan LCD.', '9. Rangkaian elektronik menghitung A = log(I₀/I); nilai tampil di layar LCD.'];
  if (INIT.tour) {
    m.onLight = function (s) {
      $('#lstage').textContent = LST[s.stage]; $('#lb0').style.width = (s.arrR / s.N * 100) + '%'; $('#lb1').style.width = (s.arrS / s.N * 100) + '%';
      $('#lv0').textContent = s.arrR; $('#lv1').textContent = s.arrS;
      if (s.done) { var A = m.abs, T = Math.pow(10, -A); $('#lvT').textContent = fm(T * 100, 1) + ' %'; $('#lvA').textContent = fm(A, 3) + '  (A = log I₀/I)'; $('#lplay').textContent = '↺ Ulangi animasi'; }
    };
    $('#lplay').addEventListener('click', function () {
      m.hl = new Set(['d2', 'wi', 'mirror', 'slit1', 'grating', 'slit2', 'split', 'holderR', 'holderS', 'detR', 'detS', 'pcb', 'lcd']); m.beamLimit = 9; m.labels = false; m.sel = null;
      $('#lv0').textContent = 0; $('#lv1').textContent = 0; $('#lb0').style.width = '0'; $('#lb1').style.width = '0'; $('#lvT').textContent = '–'; $('#lvA').textContent = '–';
      $('#lplay').textContent = '… animasi berjalan'; m.lightStart(); sync();
    });
  }
  if (INIT.slave) {
    $('#wrap').classList.add('slave');
    window.addEventListener('message', function (e) { var d = e.data; if (!d || d.src !== 'uv1700') return;
      if (d.cmd === 'tour') goStep(d.k); else if (d.cmd === 'cmp') pickComp(d.k); else if (d.cmd === 'lam') setLam(d.v); else if (d.cmd === 'abs') { m.abs = d.v; m._kick(); } else if (d.cmd === 'sol') { m.solColor = d.v; m._kick(); } });
  }
  m.toggle('beam', true); if (INIT.shell) m.toggle('shell', true); if (INIT.xray) m.toggle('xray', true); if (INIT.lid) m.toggle('lid', true); if (INIT.auto) m.toggle('auto', true);
  setLam(INIT.lam); sync(); if (INIT.tour) { meter(); goStep(0); }
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
.stage{position:relative;height:575px;border-radius:16px;overflow:hidden;border:1px solid var(--line);background:#eef0f8}
canvas#c{width:100%;height:100%;display:block;cursor:grab;touch-action:none}
.views{position:absolute;left:8px;top:8px;display:flex;flex-wrap:wrap;gap:3px;background:rgba(255,255,255,.92);padding:4px;border-radius:10px;max-width:calc(100% - 170px)}
.views button{border:0;background:none;color:var(--muted);font:500 12px Inter,sans-serif;padding:4px 8px;border-radius:7px;cursor:pointer}
.views button.on{background:var(--acc);color:#fff}
.tools{position:absolute;left:8px;bottom:8px;display:flex;gap:4px;flex-wrap:wrap}
.tools button{border:1px solid var(--line);background:rgba(255,255,255,.94);border-radius:8px;padding:4px 9px;cursor:pointer;font:500 12px Inter,sans-serif}
.tools button[aria-pressed=true]{background:#eef0ff;border-color:var(--acc);color:#3f37c9}
.hud{display:none;position:absolute;right:8px;top:8px;background:rgba(20,18,58,.9);color:#fff;border-radius:12px;padding:8px 12px;font:600 11px 'JetBrains Mono',monospace;text-align:right;min-width:150px}
.hud .sc{font-size:22px;line-height:1.1}.hud .tm{color:#a5b4fc}
.bar{height:5px;background:rgba(255,255,255,.2);border-radius:3px;margin-top:5px;overflow:hidden}.bar i{display:block;height:100%;background:linear-gradient(90deg,#22c55e,#a3e635);width:0}
.ikbar{position:absolute;left:50%;top:58px;transform:translateX(-50%);z-index:6;display:none;gap:6px;align-items:center;background:rgba(20,18,58,.88);padding:6px 8px;border-radius:12px;color:#e0e7ff;font:500 12px 'JetBrains Mono',monospace}.ikbar.show{display:flex}.ikbar button{border:0;border-radius:8px;padding:6px 10px;background:#fff;color:#1e1b4b;font:600 12px Inter,sans-serif;cursor:pointer}
.ikread{position:absolute;inset:0;z-index:9;background:rgba(20,18,58,.86);display:none;flex-direction:column;align-items:center;padding:8px;gap:8px}.ikread.show{display:flex}.ikbar2{display:flex;gap:6px}.ikbar2 button{border:0;border-radius:8px;padding:6px 12px;background:#fff;color:#1e1b4b;font:600 12px Inter,sans-serif;cursor:pointer}.ikpgs{flex:1;min-height:0;display:flex;gap:8px;justify-content:center;width:100%}.ikpgs canvas{height:100%;max-width:49%;object-fit:contain;background:#fff;border-radius:3px}
.toast{z-index:7;position:absolute;left:50%;top:50px;transform:translateX(-50%);max-width:86%;padding:9px 14px;border-radius:12px;font-weight:500;box-shadow:0 10px 26px -10px rgba(0,0,0,.45);opacity:0;transition:opacity .25s;pointer-events:none;text-align:center}
.toast.show{opacity:1}.toast.info{background:#14123a;color:#fff}.toast.ok{background:#0d9488;color:#fff}.toast.err{background:#e11d48;color:#fff}.toast.warn{background:#f59e0b;color:#1f1300}
.modal{position:absolute;inset:0;background:rgba(20,18,58,.45);display:none;align-items:center;justify-content:center;z-index:8}
.modal.show{display:flex}
.mcard{background:#fff;border-radius:18px;padding:20px 22px;width:min(520px,92%);box-shadow:0 30px 60px -20px rgba(0,0,0,.5);max-height:94%;overflow:auto}
.mcard h2{margin:0 0 6px;font:700 20px 'Space Grotesk',Inter,sans-serif}.mcard p{margin:4px 0 10px;color:var(--muted)}
.mcard ol{padding-left:18px;margin:6px 0 12px}.mcard li{margin:4px 0}
.mcard label{display:block;font-size:12px;color:var(--muted);margin:8px 0 3px}
.mcard input,.mcard textarea{width:100%;font:inherit;padding:7px 9px;border:1px solid var(--line);border-radius:9px}
.mcard .row3{display:grid;grid-template-columns:2fr 1fr 1fr;gap:8px}
.big{font:600 14px Inter,sans-serif;border:0;border-radius:11px;padding:10px 16px;background:linear-gradient(135deg,#4f46e5,#7c3aed);color:#fff;cursor:pointer;width:100%;margin-top:12px}
.side{display:flex;flex-direction:column;gap:8px;height:575px;overflow:auto;padding-right:2px}
.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:10px}
.card h4{margin:0 0 7px;font:600 11px 'JetBrains Mono',monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);display:flex;justify-content:space-between;align-items:center}
.chips{display:flex;flex-wrap:wrap;gap:4px}
.chips button.chip{border:0;cursor:pointer}.chips button.chip:hover{background:#e0e3f7}
.chip{font:500 11px 'JetBrains Mono',monospace;padding:3px 8px;border-radius:99px;background:#f1f2f8;color:#6b7194}
.chip.on{background:#dff7f1;color:#0b7a6e}.chip.mid{background:#fff4d6;color:#9a6a00}
.now{background:linear-gradient(135deg,#eef0ff,#f5efff);border-color:#d9d4ff}
.now .sec{font:600 11px 'JetBrains Mono',monospace;color:#7c3aed;letter-spacing:.06em}
.now .tx{font-weight:600;font-size:14px;margin:4px 0 8px;color:#1e1b4b}
.now .why{color:var(--muted);font-size:12px;margin-bottom:8px}
.go{font:600 13px Inter,sans-serif;border:0;border-radius:10px;padding:9px 12px;background:linear-gradient(135deg,#4f46e5,#7c3aed);color:#fff;cursor:pointer;width:100%}
.go:disabled{opacity:.5;cursor:wait}
.ik{max-height:none}.ik .grp{font:600 11px 'JetBrains Mono',monospace;color:#3f37c9;margin:10px 0 4px;letter-spacing:.05em}
.ik .it{display:flex;gap:8px;padding:5px 6px;border-radius:8px;font-size:12px;align-items:flex-start}
.ik .it .d{flex:none;width:20px;height:20px;border-radius:50%;display:grid;place-items:center;font:600 10px 'JetBrains Mono',monospace;background:#eef0ff;color:#4f46e5}
.ik .it.done{color:#6b7280}.ik .it.done .d{background:#0d9488;color:#fff}
.ik .it.cur{background:#f5f3ff;box-shadow:inset 0 0 0 1px #c4b5fd;color:#1e1b4b;font-weight:600}
.ik .rk{font:500 9px 'JetBrains Mono',monospace;color:#9a6a00;background:#fff4d6;border-radius:5px;padding:1px 5px;margin-left:4px}
details.man summary{cursor:pointer;font:600 11px 'JetBrains Mono',monospace;letter-spacing:.08em;color:var(--muted);text-transform:uppercase}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:6px}
button.a{font:500 12px Inter,sans-serif;border:1px solid var(--line);background:#fff;color:var(--fg);border-radius:9px;padding:7px 8px;cursor:pointer;text-align:left}
button.a:hover{border-color:var(--acc)}button.a:disabled{opacity:.45;cursor:not-allowed}
select,input[type=number],input[type=text]{width:100%;font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:9px;background:#fff;color:var(--fg)}
.seg{display:flex;gap:4px;margin:6px 0}.seg label{flex:1;text-align:center;border:1px solid var(--line);border-radius:9px;padding:5px;cursor:pointer;font-size:12px}
.seg input{display:none}.seg label:has(input:checked){border-color:var(--acc);background:#eef0ff;color:#3f37c9;font-weight:600}
.bottom{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:10px}
.pip{z-index:6;position:absolute;right:8px;bottom:8px;width:35%;min-width:250px;max-width:460px;background:#111214;border-radius:12px;padding:5px 6px 7px;box-shadow:0 14px 34px -10px rgba(0,0,0,.65),inset 0 0 0 2px #2a2c31;transition:width .25s}
.pip.big{width:70%;max-width:none}.pip.hide{display:none}
.pip .ph{display:flex;justify-content:space-between;align-items:center;color:#9aa1b5;font:500 10px 'JetBrains Mono',monospace;letter-spacing:.1em;padding:0 2px 4px}
.pip .ph button{border:0;background:#2a2c31;color:#cfd3e4;border-radius:6px;padding:2px 7px;font:500 10px 'JetBrains Mono',monospace;cursor:pointer;margin-left:4px}
.pip canvas{width:100%;display:block;border-radius:4px;background:#000;cursor:pointer}
.prog{height:6px;border-radius:4px;background:#e6e8f5;overflow:hidden;margin:2px 0 8px}.prog i{display:block;height:100%;background:linear-gradient(90deg,#4f46e5,#22c55e);width:0;transition:width .3s}
.pcwrap{background:#111214;border-radius:16px;padding:10px 10px 14px;box-shadow:inset 0 0 0 2px #26282d}
.pchead{color:#9aa1b5;font:500 11px 'JetBrains Mono',monospace;letter-spacing:.1em;margin:0 0 6px 4px;display:flex;justify-content:space-between}
canvas#pc{width:100%;display:block;border-radius:4px;background:#000;cursor:pointer}
textarea{width:100%;height:110px;font:12px 'JetBrains Mono',monospace;border:1px solid var(--line);border-radius:9px;padding:6px;resize:vertical}
.log{font:11.5px 'JetBrains Mono',monospace;color:#4a5072;max-height:170px;overflow:auto;white-space:pre-wrap}
.log .e{color:#be123c}.log .o{color:#0b7a6e}.log .w{color:#9a6a00}
.prints{display:flex;gap:6px;flex-wrap:wrap}.prints a{display:block;border:1px solid var(--line);border-radius:6px;overflow:hidden;width:72px;text-decoration:none;color:var(--muted);font-size:10px;text-align:center}
.prints img{width:100%;display:block}
.row{display:flex;gap:6px;align-items:center}.hint{color:var(--muted);font-size:12px}
@media (max-width:980px){.top,.bottom{grid-template-columns:1fr}.side{height:auto}.stage{height:520px}.pip{width:62%}}
.ctl{display:flex;gap:5px;margin:7px 0}
.ctl button{flex:1;min-width:0;border:1px solid var(--line);background:#fff;border-radius:9px;padding:8px 3px;font:600 12px Inter,sans-serif;cursor:pointer;color:var(--fg);white-space:nowrap}
.ctl button.pri{background:linear-gradient(135deg,#4f46e5,#7c3aed);color:#fff;border-color:transparent;flex:1.4}
.ctl button:disabled{opacity:.45;cursor:not-allowed}
.ctl button.on{background:#eef0ff;border-color:var(--acc);color:#3f37c9}
.ctl .lb{flex:none;align-self:center;font:600 10.5px 'JetBrains Mono',monospace;color:var(--muted);letter-spacing:.06em;text-transform:uppercase;margin-right:2px}
.sub{position:absolute;left:50%;bottom:46px;transform:translateX(-50%);max-width:58%;background:rgba(20,18,58,.88);color:#fff;border-radius:10px;padding:6px 12px;font:500 12.5px/1.35 Inter,sans-serif;z-index:2;pointer-events:none;text-align:center}
.sub:empty{display:none}
.inner{position:absolute;inset:0;background:#f4f5fb;z-index:4;opacity:0;pointer-events:none;transition:opacity .4s}
.inner.show{opacity:1;pointer-events:auto}
.inner iframe{position:absolute;left:0;right:0;top:46px;bottom:0;width:100%;height:calc(100% - 46px);border:0}
.inner .itop{position:absolute;left:8px;right:8px;top:8px;display:flex;justify-content:space-between;align-items:center;z-index:2}
.inner .ibadge{background:#14123a;color:#fff;font:600 11px 'JetBrains Mono',monospace;letter-spacing:.08em;padding:6px 10px;border-radius:9px}
.inner .itop button{border:1px solid var(--line);background:#fff;border-radius:8px;padding:5px 10px;font:600 12px Inter,sans-serif;cursor:pointer}
.inner .icap{position:absolute;left:12px;bottom:12px;max-width:50%;background:rgba(255,255,255,.96);border:1px solid var(--line);border-radius:12px;padding:8px 12px;z-index:2;box-shadow:0 10px 26px -14px rgba(0,0,0,.4)}
.inner .icap b{display:block;font:700 14px Inter,sans-serif;color:#1e1b4b;margin-bottom:2px}.inner .icap span{color:#4a5072;font-size:12.5px}
</style></head><body>
<div class="top">
  <div class="stage"><canvas id="c" aria-label="Meja praktikum 3D"></canvas>
    <div class="views">
      <button data-v="meja" class="on">Meja</button><button data-v="alat">UV-1700</button><button data-v="kompartemen">Kompartemen</button>
      <button data-v="isi">Labu & kuvet</button><button data-v="komputer">Komputer</button><button data-v="belakang">Kabel & UPS</button><button data-v="ik">Buku IK</button>
    </div>
    <div class="hud"><div>SKOR PROSEDUR</div><div class="sc" id="score">100</div><div class="tm" id="clock">00:00</div><div class="bar"><i id="prog"></i></div></div>
    <div class="tools"><button data-t="xray" aria-pressed="false">Tembus pandang</button><button data-t="sound" aria-pressed="true">Suara</button><button data-t="pip">Zoom monitor</button></div>
    <div class="pip" id="pip"><div class="ph"><span>MONITOR · <span id="pcst" style="color:#8b93b8">MATI</span></span><span><button data-t="pipbig" id="pipbig">Perbesar</button><button data-t="pipx">Tutup</button></span></div><canvas id="pc" width="1040" height="610"></canvas></div>
    <div class="ikbar" id="ikbar"><button data-ik="-1">◀ Halaman</button><span id="ikpg"></span><button data-ik="1">Halaman ▶</button><button data-ikr="1">Baca besar</button><button data-ik="0">Tutup buku</button></div>
    <div class="ikread" id="ikread"><div class="ikbar2"><button data-ik="-1">◀ Halaman</button><button data-ik="1">Halaman ▶</button><button data-ikr="0">Tutup ✕</button></div><div class="ikpgs"><canvas id="ikA" width="720" height="1020"></canvas><canvas id="ikB" width="720" height="1020"></canvas></div></div>
    <div class="toast" id="toast"></div>
    <div class="sub" id="sub"></div>
    <div class="inner" id="inner"><div class="itop"><span class="ibadge" id="ibadge"></span><button id="icx"></button></div><iframe id="innerf" title="UV-1700"></iframe><div class="icap"><b id="icT"></b><span id="icF"></span></div></div>
    <div class="modal" id="logm"><div class="mcard">
      <h2>Log Book Alat UV-1700</h2><p>Catat semua kejadian selama penggunaan dan tanda tangani dengan nama jelas (IK C.1).</p>
      <label>Tanggal</label><input id="lg_t" readonly><label>Nama pengguna</label><input id="lg_n">
      <label>Kejadian / kondisi alat</label><textarea id="lg_k" style="height:80px"></textarea>
      <label>Tanda tangan (ketik nama jelas)</label><input id="lg_s">
      <button class="big" id="lg_ok">Simpan ke Log Book</button></div></div>
  </div>
  <div class="side">
    <div class="card now"><div class="sec" id="nowsec">LANGKAH SEKARANG</div><div class="tx" id="nowtx"></div><div class="prog"><i id="prog2"></i></div><div class="why" id="nowwhy"></div>
      <div class="ctl"><button id="b_play" class="pri" style="padding:10px 4px;font-size:13px"></button></div>
      <div class="ctl"><button id="b_rst"></button><button id="b_prev"></button><button id="b_next"></button></div>
      <div class="ctl"><span class="lb" id="lb_spd"></span><button data-spd="0.5">0,5×</button><button data-spd="1">1×</button><button data-spd="2">2×</button><span class="lb" id="lb_lng" style="margin-left:6px"></span><button data-lng="id">ID</button><button data-lng="en">EN</button></div>
      <button class="go" id="gobtn" style="display:none">Kerjakan</button></div>
    <div class="card"><div class="sec" id="cmpT" style="font:600 11px 'JetBrains Mono',monospace;letter-spacing:.08em;color:var(--muted);margin-bottom:6px">ZOOM KOMPONEN UV-1700</div><div class="chips" id="cmps"></div></div>
    <div class="card"><details class="man"><summary>Semua langkah IK/LAB.INST/6 <span id="ikn"></span></summary><div class="ik" id="iklist" style="margin-top:6px"></div>
      <div class="hint" style="margin-top:6px"><span class="rk">REK</span> halaman 1 IK belum difoto; langkah bagian A dan B-Spektrum 1–5 disusun dari prosedur baku UVProbe dan perlu dicocokkan dengan IK asli.</div></details></div>
    <div class="card"><details class="man"><summary>Status alat <span id="ident" style="text-transform:none;letter-spacing:0"></span></summary><div class="chips" id="chips" style="margin-top:6px"></div></details></div>
    <div class="card"><details class="man"><summary>Kontrol manual (mode bebas)</summary>
      <div class="grid2" style="margin-top:8px">
        <button class="a" data-act="cables">Sambungkan kabel</button><button class="a" data-act="ups">UPS ON/OFF</button>
        <button class="a" data-act="silica">Silica gel keluar/masuk</button><button class="a" data-act="inst">Power UV-1700</button>
        <button class="a" data-act="pc">Nyalakan CPU, monitor, printer</button><button class="a" data-act="pcctrl">MODE → PC CTRL</button>
        <button class="a" data-act="software">Buka UVProbe</button><button class="a" data-act="connect">Connect</button>
        <button class="a" data-act="skipwarm">Lewati pemanasan</button><button class="a" data-act="lid">Buka/tutup kompartemen</button></div>
      <div class="seg"><label><input type="radio" name="cv" value="R"><span>Kuvet referensi</span></label><label><input type="radio" name="cv" value="S" checked><span>Kuvet sampel</span></label></div>
      <select id="sol"></select>
      <div class="grid2" style="margin-top:6px">
        <button class="a" data-act="fill">Bilas & isi kuvet</button><button class="a" data-act="wipe">Lap dengan tisu</button>
        <button class="a" data-act="insert">Masukkan ke alat</button><button class="a" data-act="remove">Keluarkan kuvet</button>
        <button class="a" data-act="empty">Buang isi kuvet</button><button class="a" data-act="dry">Keringkan (etanol, aseton)</button></div>
      <div class="row" style="margin-top:6px"><input type="number" id="lam" min="190" max="1100" step="0.5"><button class="a" data-act="goto" style="white-space:nowrap">Go To WL</button></div>
      <div class="grid2" style="margin-top:6px">
        <button class="a" data-act="baseline">Baseline</button><button class="a" data-act="autozero">Auto Zero</button>
        <button class="a" data-act="scan">Start (scan)</button><button class="a" data-act="readstd">Read Std</button>
        <button class="a" data-act="readunk">Read Unk</button><button class="a" data-act="disconnect">Disconnect</button></div>
      <div class="hint" style="margin-top:6px" id="scaninfo"></div></details></div>
    <div class="card" style="display:none"><h4 id="ti">Info objek</h4><div class="hint" id="info"></div></div>
  </div>
</div>
<div class="bottom">
  <div class="card"><h4>Hasil cetak printer</h4><div class="prints" id="prints"><span class="hint">Belum ada cetakan.</span></div></div>
  <div class="card"><h4>Data untuk Kurva Kalibrasi</h4><textarea id="out" readonly></textarea>
    <div class="grid2" style="margin-top:6px"><button class="a" data-act="copy">Salin data</button><button class="a" data-act="csv">Unduh CSV</button></div>
    <div class="hint" style="margin-top:6px">Tempel di kotak <b>Impor hasil praktikum</b> di bawah halaman.</div></div>
  <div class="card" style="display:none"><h4>Catatan kerja</h4><div class="log" id="log"></div></div>
</div>
<script>
(function () {
'use strict';
const INIT = Object.assign({ seed: 12345, hidden: 2.4, dec: 'koma', unit: 'mg/L',
  analyte: { nama: 'Fe(II)–1,10-fenantrolin', rumus: '[Fe(phen)3]2+', pelarut: 'Bufer asetat', lam_ref: 510, k: 0.2, warna: '#c0392b',
             bands: [[514, 1.0, 22], [478, .3, 15], [400, .08, 30], [265, 1.6, 18]], std: [0.5, 1, 2, 3, 4, 5] } }, /*__LAB__*/null || {});
const AN = INIT.analyte; if (AN.std.length > 5) { const n0 = AN.std.length; AN.std = [0, 1, 2, 3, 4].map(i => AN.std[Math.round(i * (n0 - 1) / 4)]); }
const  D2R = Math.PI / 180, DEC = INIT.dec === 'koma';
const NUL = { style: {}, dataset: {}, classList: { add() { }, remove() { }, toggle() { } }, value: '', textContent: '', innerHTML: '', firstChild: null, addEventListener() { }, setAttribute() { }, removeAttribute() { }, getAttribute() { return null; }, insertBefore() { }, appendChild() { }, select() { } };
const $ = s => document.querySelector(s) || NUL, $$ = s => Array.from(document.querySelectorAll(s));
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
function rotZ(p, piv, ang) { const s = Math.sin(ang), c = Math.cos(ang); const x = p[0] - piv[0], y = p[1] - piv[1]; return [piv[0] + x * c - y * s, piv[1] + x * s + y * c, p[2]]; }
function rotY(p, piv, ang) { const s = Math.sin(ang), c = Math.cos(ang); const x = p[0] - piv[0], z = p[2] - piv[2]; return [piv[0] + x * c + z * s, p[1], piv[2] - x * s + z * c]; }

/* ================= larutan & identitas ================= */
const SOL = { blank: { id: 'blank', label: 'Blanko (' + AN.pelarut + ')', short: 'Blanko', c: 0, col: '#dff4f8' } };
AN.std.forEach((c, i) => { SOL['s' + i] = { id: 's' + i, label: 'Standar ' + (i + 1) + ' · ' + fx(c, c < 1 ? 2 : 1) + ' ' + INIT.unit, short: 'Std ' + (i + 1), c, std: true, idx: i }; });
SOL.X = { id: 'X', label: 'Sampel X (tak diketahui)', short: 'Sampel X', c: INIT.hidden, sample: true };
const cmax = Math.max(...AN.std, INIT.hidden);
Object.values(SOL).forEach(s => { if (s.c > 0) s.col = mix('#eef6f8', AN.warna, Math.min(1, 0.25 + 0.75 * s.c / cmax)); });
const SOL_KEYS = Object.keys(SOL);
const ID = { nama: '', kelas: 'Kelas', kel: '2' };
const today = new Date(); const dstr = String(today.getDate()).padStart(2, '0') + '-' + String(today.getMonth() + 1).padStart(2, '0') + '-' + today.getFullYear();
const fname = () => dstr + '_' + (ID.kelas || 'Kelas').replace(/\s+/g, '') + '_' + (ID.kel || 'Kelompok').replace(/\s+/g, '');
const scanRange = () => { const lo = AN.lam_ref < 320 ? 200 : Math.max(190, Math.round((AN.lam_ref - 150) / 10) * 10); return [lo, Math.min(1100, lo + 320)]; };

/* ================= keadaan ================= */
const RACK = { R: [-31, 0.2, 4], S: [-31, 0.2, 10] }, CELL = { R: [-8.5, 6.6, 3], S: [-8.5, 6.6, 13] }, DRY = { R: [-16, -1.25, 30], S: [-11, -1.25, 30] };
const ST = {
  cab: [0, 0, 0, 0, 0, 0, 0, 0, 0], cabled: false, ups: false, pc: 0, pcBoot: 0, printer: false, inst: 'off', initLines: [], warmLeft: 0, warmSkipped: false,
  pcctrl: false, sw: false, connected: false, lid: 0, xray: 0, xrayT: 0, xrayUser: false, autoX: true, sound: true, lam: AN.lam_ref, lastA: 0, lcdFold: 0,
  silica: 'comp', silicaPos: [-8.5, 6, 8], logbook: null, ikp: 0, ikT: 0, tidy: false, paper: 0, chairZ: 66, wash: null, pour: null, stream: null,
  cuv: { R: { where: 'rak', pos: RACK.R.slice(), content: null, level: 0, dirty: false, smudge: 0, flip: 0, wet: false }, S: { where: 'rak', pos: RACK.S.slice(), content: null, level: 0, dirty: false, smudge: 0, flip: 0, wet: false } },
  zero: null, win: null, menu: null, dlg: null, busyMsg: null, ptab: 'std',
  spectrum: null, peaks: null, smethod: null, pmethod: null, stdTable: [], unkTable: [], files: [], prints: [],
  score: 100, errs: new Set(), done: new Set(), busy: false, tissue: null, beam: 0, t0: performance.now()
};

/* ================= geometri ================= */
const P = [];
function quad(id, grp, v, col, o) { P.push({ id, grp, v, col, alpha: o.alpha == null ? 1 : o.alpha, emi: o.emi || 0, bias: o.bias || 0, tf: o.tf || null, dyn: o.dyn || null, text: o.text || null, ds: !!o.ds, bg: !!o.bg, leaf: o.leaf == null ? null : o.leaf, lside: o.lside || null }); }
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
function lathe(id, grp, cx, cz, prof, col, o, n) {
  o = Object.assign({ ds: true }, o || {}); n = n || 14; const off = o.off || 0;
  const a0s = o.arc ? o.arc[0] : 0, a1s = o.arc ? o.arc[1] : 360, nn = o.arc ? Math.max(1, Math.round(n * (a1s - a0s) / 360)) : n;
  for (let k = 0; k < prof.length - 1; k++) {
    const [r0, y0] = prof[k], [r1, y1] = prof[k + 1];
    for (let i = 0; i < nn; i++) {
      const A0 = (a0s + (a1s - a0s) * i / nn + off) * D2R, A1 = (a0s + (a1s - a0s) * (i + 1) / nn + off) * D2R;
      const c0 = Math.cos(A0), s0 = Math.sin(A0), c1 = Math.cos(A1), s1 = Math.sin(A1);
      if (r0 === 0 && y0 === y1) { quad(id, grp, [[cx, y0, cz], [cx + r1 * c1, y1, cz + r1 * s1], [cx + r1 * c0, y1, cz + r1 * s0], [cx + r1 * c0, y1, cz + r1 * s0]], o.topCol || col, o); continue; }
      if (r1 === 0 && y0 === y1) { quad(id, grp, [[cx, y0, cz], [cx + r0 * c1, y0, cz + r0 * s1], [cx + r0 * c0, y0, cz + r0 * s0], [cx + r0 * c0, y0, cz + r0 * s0]], o.topCol || col, o); continue; }
      quad(id, grp, [[cx + r0 * c1, y0, cz + r0 * s1], [cx + r0 * c0, y0, cz + r0 * s0], [cx + r1 * c0, y1, cz + r1 * s0], [cx + r1 * c1, y1, cz + r1 * s1]], col, o);
    }
  }
}
function clipProf(prof, ymax, dr) {
  const out = [];
  for (let k = 0; k < prof.length; k++) { const [r, y] = prof[k]; if (y <= ymax) out.push([Math.max(0, r - dr), y]); else { const [r0, y0] = prof[k - 1]; const t = (ymax - y0) / (y - y0); out.push([Math.max(0, r0 + (r - r0) * t - dr), ymax]); break; } }
  return out;
}
const BEIGE = '#d8bf7f', BEIGE2 = '#e2cb8e', GREY = '#4a5058', DARK = '#1e2126', CREAM = '#efe9d6', Y0 = -1.1;
// ruang
box('wall', 'room', -200, 200, -80, 85, -40, -38.5, '#f6f7fb', { bg: true, cell: 40, bias: 400, emi: 0.1 });
box('bench', 'room', -130, 90, -8, -1.4, -38.5, 38, '#fbfbfc', { bg: true, cell: 15, bias: 300, faceCol: { f: '#e9ebf1' } });
for (let x = -130; x <= 90; x += 15) quad('bench', 'room', [[x, -1.38, -38.5], [x + 0.2, -1.38, -38.5], [x + 0.2, -1.38, 38], [x, -1.38, 38]], '#e3e6ee', { bg: true, bias: 290 });
for (let z = -35; z <= 38; z += 15) quad('bench', 'room', [[-130, -1.37, z], [90, -1.37, z], [90, -1.37, z + 0.2], [-130, -1.37, z + 0.2]], '#e3e6ee', { bg: true, bias: 290 });
box('outlet', 'room', 66, 80, 14, 22, -38.4, -37.6, '#f4f1e6', { bias: -2 });
box('outlet', 'room', 68.2, 71.8, 16.2, 19.8, -37.6, -37.4, '#d9d4c3', { bias: -3 }); box('outlet', 'room', 74.2, 77.8, 16.2, 19.8, -37.6, -37.4, '#d9d4c3', { bias: -3 });
// kabel perpanjangan (terminal)
box('strip', 'room', 50, 68, -1.4, 0.4, -32, -28, '#f2f0e8', { cell: 6 });
[52, 56, 60, 64].forEach(x => box('strip', 'room', x, x + 2.4, 0.4, 0.5, -31, -29, '#2b2b2b', { bias: -0.5 }));
box('strip', 'room', 66, 67.6, 0.4, 0.6, -30.6, -29.4, '#e25555', { bias: -0.6 });
// lantai ubin putih, kaki meja, kursi
box('floor', 'room', -200, 200, -82, -80, -90, 150, '#eceff4', { bg: true, cell: 40, bias: 600 });
for (let x = -200; x <= 200; x += 40) quad('floor', 'room', [[x, -79.9, -90], [x + 0.4, -79.9, -90], [x + 0.4, -79.9, 150], [x, -79.9, 150]], '#d5dae4', { bg: true, bias: 590 });
for (let z = -80; z <= 150; z += 40) quad('floor', 'room', [[-200, -79.9, z], [200, -79.9, z], [200, -79.9, z + 0.4], [-200, -79.9, z + 0.4]], '#d5dae4', { bg: true, bias: 590 });
[[-128, -36], [-128, 34], [88, -36], [88, 34]].forEach(([x, z]) => box('bench', 'room', x - 1.6, x + 1.6, -80, -8, z - 1.6, z + 1.6, '#c9ced8', { bg: true, bias: 310 }));
// ---------- UV-1700 ----------
box('base', 'inst', -23, 23, 0, 3, -24.5, 24.5, GREY, { cell: 6 });
[[-19, 20], [19, 20], [-19, -20], [19, -20]].forEach(([x, z]) => box('base', 'inst', x - 1.6, x + 1.6, -1.4, 0, z - 1.6, z + 1.6, '#363b42', { skip: ['u'] }));
box('base', 'inst', 4, 7, 0.6, 2.2, -24.8, -24.5, '#c73a3a', { bias: -4, cell: 99 }); box('base', 'inst', 12, 18, 0.7, 2.0, -24.8, -24.5, '#2b2f35', { bias: -4, cell: 99 });
box('base', 'inst', 19, 21.5, 0.6, 2.4, -24.8, -24.5, '#15171a', { bias: -4, cell: 99 });
box('body', 'walls', -23, 23, 3, 13.5, 23.9, 24.5, BEIGE, { cell: 3 }); box('body', 'walls', -23, 23, 3, 13.5, -24.5, -23.9, BEIGE, { cell: 3 });
box('body', 'walls', -23, -22.4, 3, 13.5, -23.9, 23.9, BEIGE, { cell: 3 }); box('body', 'walls', 22.4, 23, 3, 13.5, -23.9, 23.9, BEIGE, { cell: 3 });
box('power', 'walls', -23.4, -23, 5, 8.4, 15.5, 19.5, '#2a2d33', { bias: -4, cell: 99, dyn: 'pwr' });
box('compartment', 'walls', -16, -1, 4, 12.5, 24.5, 24.8, '#cfb575', { bias: -4, skip: ['b'], cell: 99 });
box('plate', 'walls', 9, 20, 6.5, 10.5, 24.5, 24.7, '#f7f6f0', { bias: -4.5, skip: ['b'], cell: 99, text: { face: 'f', w: 110, h: 40, lines: [['UV-1700', 8, 18, 'left', '700 15px Arial', '#22263a'], ['PharmaSpec', 70, 18, 'left', 'italic 9px Arial', '#22263a'], ['UV-VIS SPECTROPHOTOMETER', 8, 30, 'left', '6.5px Arial', '#c0392b']] } });
for (let i = 0; i < 8; i++) box('vent', 'walls', 23, 23.25, 5.6 + i * 0.95, 6.1 + i * 0.95, -20, -11, '#bfa868', { bias: -4, cell: 99, dyn: i === 4 ? 'glow' : null });
box('card', 'walls', 23, 23.2, 11, 11.8, 1, 10, '#2b2f36', { bias: -4, cell: 99 }); box('card', 'walls', 23, 23.45, 10.9, 11.9, 11.2, 13.4, '#ebe6d6', { bias: -4.2, cell: 99 });
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
for (let r = 0; r < 4; r++) for (let c = 0; c < 6; c++) box('keypad', 'cover', 2.8 + c * 3.0, 4.7 + c * 3.0, 16.2, 16.55, 14.2 + r * 2.2, 15.5 + r * 2.2, c === 5 ? '#7c93c9' : '#cfd4dc', { bias: -0.8, dyn: r === 0 && c === 0 ? 'modekey' : null });
box('lcd', 'cover', 9, 14, 15, 17.5, 7.5, 10, '#ddd6bd', { bias: -0.6 });
const tilt = [['x', [11.5, 16.5, 9], -18]];
box('lcd', 'lcdp', 3.5, 19.5, 16.5, 28.5, 8.2, 10.2, '#ebe5cf', { tf: tilt, bias: -0.7, cell: 99 });
box('lcd', 'lcdp', 6, 17, 18.3, 26.6, 10.2, 10.35, '#a99af2', { tf: tilt, bias: -4, dyn: 'lcd', dynFace: 'f', text: { face: 'f', w: 130, h: 96, lcd: true } });
box('silica', 'silica', -2.4, 2.4, 0, 0.8, -1.4, 1.4, '#f5f7fb', { cell: 99, faceCol: { t: '#e6ecf8' }, text: { face: 't', w: 60, h: 36, lines: [['SILICA GEL', 30, 15, 'center', '700 9px Arial', '#1d4ed8'], ['• • • • •', 30, 27, 'center', '10px Arial', '#3b82f6']] } });
// ---------- UPS ICA ----------
box('ups', 'room', 28, 44, -1.4, 17, -22, 8, '#141518', { cell: 6 });
quad('ups', 'room', [[33.6, 9.4, 8.06], [38.4, 9.4, 8.06], [43.6, 16.6, 8.06], [28.4, 16.6, 8.06]], '#c9a24a', { bias: -3 });
box('ups', 'room', 35.1, 36.9, 11.2, 13.0, 8.06, 8.5, '#0d0e10', { bias: -3.4, dyn: 'upsbtn' });
for (let i = 0; i < 5; i++) box('upsled', 'room', 35.6, 36.4, 3.4 + i * 1.15, 4.0 + i * 1.15, 8.06, 8.3, '#2b2f2b', { bias: -3.3, dyn: 'upsled' + i });
box('ups', 'room', 29, 43, -1.2, 2.2, 8.06, 8.2, '#b8923f', { bias: -3, text: { face: 'f', w: 140, h: 34, lines: [['ICA', 70, 22, 'center', '700 14px Arial', '#2b1d05']] } });
[[30.5, 9], [35.5, 9], [30.5, 3], [35.5, 3]].forEach(([x, y]) => box('ups', 'room', x, x + 4, y, y + 4, -22.3, -22, '#1b1d21', { bias: -3 }));
// ---------- komputer (CPU, monitor HP, keyboard, mouse, printer) ----------
box('cpu', 'room', -106, -88, -1.4, 40, -22, 14, '#18191c', { cell: 7 });
box('cpu', 'room', -105.2, -88.8, 1, 39, 14, 14.25, '#c9ccd2', { bias: -2.5, cell: 99 });
box('cpu', 'room', -105.2, -88.8, -1.4, 1, 14, 14.3, '#0f1012', { bias: -2.6, cell: 99 });
box('cpu', 'room', -104, -90, 33.5, 35.6, 14.25, 14.45, '#101113', { bias: -3, cell: 99 });
box('cpu', 'room', -104, -90, 26.5, 31.5, 14.25, 14.4, '#f8f8f6', { bias: -3.2, cell: 99, text: { face: 'f', w: 140, h: 50, lines: [['19.01.01.24.2005', 70, 21, 'center', '700 14px Courier New', '#111'], ['2.08.01.11.138.', 70, 41, 'center', '700 14px Courier New', '#111']] } });
box('cpu', 'room', -101, -93, 18, 19.2, 14.25, 14.4, '#111214', { bias: -3, cell: 99 });
box('cpu', 'room', -98.6, -95.4, 9, 12.2, 14.25, 14.7, '#7d838c', { bias: -3.2, cell: 99, dyn: 'pcled' });
cyl('monitor', 'room', -58, -16, 8.5, -1.4, -0.4, '#1a1b1f', { cell: 99 }, 18); cyl('monitor', 'room', -58, -16, 3.2, -0.4, 0.4, '#24262b', {}, 14);
box('monitor', 'room', -60.5, -55.5, 0.4, 11.5, -18.2, -15.4, '#1b1c20', { cell: 99 });
box('monitor', 'room', -82, -34, 11, 41.5, -17.4, -14.6, '#0e0f12', { cell: 99, skip: ['f'] });
box('monitor', 'room', -82, -34, 11, 13.2, -14.6, -14.4, '#17181c', { cell: 99, bias: -4 }); box('monitor', 'room', -82, -34, 40.2, 41.5, -14.6, -14.4, '#17181c', { cell: 99, bias: -4 });
box('monitor', 'room', -82, -81, 13.2, 40.2, -14.6, -14.4, '#17181c', { cell: 99, bias: -4 }); box('monitor', 'room', -35, -34, 13.2, 40.2, -14.6, -14.4, '#17181c', { cell: 99, bias: -4 });
box('screen', 'room', -81, -35, 13.2, 40.2, -14.6, -14.5, '#000000', { bias: -6, dyn: 'screen', dynFace: 'f', cell: 99, text: { face: 'f', w: 1040, h: 610, screen: true } });
box('monitor', 'room', -37.6, -36.6, 11.7, 12.4, -14.6, -14.3, '#222', { bias: -7, dyn: 'monled' });
box('keyboard', 'room', -84, -52, -1.4, -0.4, 0, 11, '#16171a', { cell: 6 });
for (let r = 0; r < 5; r++) for (let c = 0; c < 14; c++) box('keyboard', 'room', -83 + c * 2.2, -81.3 + c * 2.2, -0.4, 0.1, 0.8 + r * 2.0, 2.3 + r * 2.0, '#2a2c31', { bias: -0.5 });
box('mousepad', 'room', -50, -39, -1.4, -1.15, 0, 11, '#efe08a'); box('mouse', 'room', -46, -42, -1.15, 0.5, 3, 8.5, '#111214', { faceCol: { t: '#9aa0a8' } });
box('printer', 'room', -126, -108, -1.4, 8, -16, 6, '#d6d3c8', { cell: 6, faceCol: { t: '#c5c1b4' } });
box('printer', 'room', -123, -111, 4, 7, 6, 6.2, '#3f4247', { bias: -2.5 });
box('printer', 'room', -110.5, -109.5, 6, 7, 6, 6.3, '#444', { bias: -3, dyn: 'prled' });
box('paper', 'paper', -122, -112, 4.6, 4.7, 0, 12, '#ffffff', { ds: true, cell: 99 });
// ---------- meja kerja ----------
box('rack', 'room', -37, -25, -1.4, 1.6, 0, 14, '#3d5a80', { cell: 4 });
box('rack', 'room', -32.6, -29.4, 1.6, 1.7, 2.4, 5.6, '#11161f', { bias: -0.5 }); box('rack', 'room', -32.6, -29.4, 1.6, 1.7, 8.4, 11.6, '#11161f', { bias: -0.5 });
box('tissue', 'room', -38, -25, -1.4, 5, 17, 29, '#2f6f6a', { cell: 5, faceCol: { t: '#3a7f79', f: '#2c6762' } });
box('tissue', 'room', -35.6, -27.4, 0.2, 3.9, 29, 29.18, '#fbfbf8', { bias: -3, cell: 99, text: { face: 'f', w: 160, h: 74, lines: [['KOTAK TISU', 80, 28, 'center', '700 16px Arial', '#1f3a37'], ['Lab. Instrumentasi', 80, 48, 'center', '500 11px Arial', '#44605d'], ['Politeknik AKA Bogor', 80, 63, 'center', '500 9px Arial', '#44605d']] } });
box('tissue', 'room', -33.5, -29.5, 5, 5.1, 21, 25, '#16302e', { bias: -0.4 });
box('tissue', 'room', -33, -30, 5, 10.5, 22.6, 22.9, '#ffffff', { tf: [['x', [-31.5, 5, 22.75], 12]], ds: true });
[[-35, 19.6], [-30.6, 19.6]].forEach(([x, z]) => { cyl('tissue', 'room', x, z, 1.35, 5, 9.4, '#f7f9fa', { alpha: 0.95 }, 10); cyl('tissue', 'room', x, z, 0.95, 9.4, 10.6, '#ffffff', {}, 10); });
box('cuvcase', 'room', -50, -41, -1.4, 1.8, 16, 23, '#2952a3', { cell: 99, faceCol: { t: '#3563c0' }, text: { face: 't', w: 90, h: 70, lines: [['QUARTZ', 45, 30, 'center', '700 13px Arial', '#e6edff'], ['CUVETTE', 45, 46, 'center', '700 13px Arial', '#e6edff']] } });
// botol semprot akuades (LDPE putih tembus cahaya, tutup biru, nozel bengkok)
const WASH0 = [-47, -1.4, 30.5];
{ const [wx, , wz] = WASH0, by = -1.4;
  const body = [[0, 0], [2.6, 0], [3.1, 0.5], [3.3, 1.6], [3.3, 11.2], [3.15, 12.5], [2.5, 13.7], [1.6, 14.4], [1.35, 15.0]].map(([r, y]) => [r, y + by]);
  lathe('bottle', 'wash', wx, wz, body, '#eef4f8', { alpha: 0.62 }, 14);
  lathe('bottle', 'wash', wx, wz, clipProf(body.map(([r, y]) => [r, y]), by + 9.2, 0.18), '#bfe2ff', { alpha: 0.7, topCol: '#d6eeff' }, 14);
  lathe('bottle', 'wash', wx, wz, [[1.5, by + 15], [1.75, by + 15.1], [1.75, by + 16.6], [1.3, by + 16.95], [0, by + 16.95]], '#2b6fd6', {}, 14);
  lathe('bottle', 'wash', wx, wz, [[0.5, by + 16.95], [0.5, by + 19.4]], '#d8e6f2', {}, 8);
  box('bottle', 'wash', wx - 0.1, wx + 5.2, by + 19.0, by + 19.8, wz - 0.4, wz + 0.4, '#e4eef6', { cell: 99, tf: [['z', [wx, by + 19.4, wz], 40]], ds: true });
  box('bottle', 'wash', wx + 5.0, wx + 6.4, by + 19.15, by + 19.65, wz - 0.22, wz + 0.22, '#d0dde8', { cell: 99, tf: [['z', [wx, by + 19.4, wz], 40]], ds: true });
  box('bottle', 'wash', wx - 1.6, wx + 1.6, by + 3.4, by + 9.6, wz + 3.34, wz + 3.5, '#ffffff', { bias: -3, cell: 99, text: { face: 'f', w: 110, h: 130, lines: [['AKUADES', 55, 40, 'center', '700 17px Arial', '#1d4ed8'], ['H\u2082O', 55, 72, 'center', '600 22px Arial', '#1d4ed8'], ['Blanko ' + fx(0, 1) + ' ' + INIT.unit, 55, 104, 'center', '600 11px Arial', '#475569']] } });
}
const WASHTIP = [WASH0[0] + 4.9, WASH0[1] + 19.4 + 4.11, WASH0[2]];
// gelas piala berisi Sampel X (di samping botol akuades)
const BKX = -40.5, BKZ = 33.5;
cyl('beaker', 'beakerW', BKX, BKZ, 3.2, -1.4, 6.5, '#dfeff5', { alpha: 0.4, open: true }, 16); cyl('beaker', 'beakerW', BKX, BKZ, 3.0, -1.3, 1.6, '#c7dce6', { alpha: 0.55 }, 16);
cyl('beaker', 'beakerW', BKX, BKZ, 2.95, -1.2, 4.6, '#cfe6ee', { alpha: 0.5, topCol: '#e9f6fa' }, 16);
box('beaker', 'beakerW', BKX + 2.8, BKX + 4.0, 5.6, 6.5, BKZ - 0.6, BKZ + 0.6, '#dfeff5', { alpha: 0.5, cell: 99 });
box('beaker', 'beakerW', BKX - 1.6, BKX + 1.6, 0.4, 3.6, BKZ + 3.18, BKZ + 3.3, '#ffffff', { bias: -3, cell: 99, text: { face: 'f', w: 90, h: 72, lines: [['LIMBAH', 45, 40, 'center', '700 19px Arial', '#111827']] } });
box('logbook', 'room', -104, -88, -1.4, -0.4, 20, 33, '#1e3a8a', { cell: 99, faceCol: { t: '#22409a' }, text: { face: 't', w: 120, h: 100, lines: [['LOG BOOK', 60, 40, 'center', '700 15px Arial', '#e0e7ff'], ['UV-1700', 60, 60, 'center', '600 12px Arial', '#c7d2fe']] } });
box('logbook', 'room', -86, -85.4, -0.6, -0.1, 21, 31, '#111', {});

box('drymat', 'room', -20, -7, -1.4, -1.25, 26, 34, '#ffffff', { cell: 99 });
// labu takar 50 mL: alas datar, badan bulat, leher panjang bergaris tera, sumbat biru
const FLY = -1.4, FS = 0.74;
const FPROF = [[0, 0], [2.5, 0], [3.4, 0.5], [4.0, 1.6], [4.25, 3.1], [4.0, 4.7], [3.2, 6.1], [2.0, 7.3], [1.1, 8.3], [0.78, 9.4], [0.74, 15.2], [0.9, 15.7], [0.98, 16.2]].map(([r, y]) => [r * FS, FLY + y * FS]);
const FMARK = FLY + 12.9 * FS, FTOP = FLY + 16.2 * FS;
const FL = {}; const STDK = SOL_KEYS.filter(k => SOL[k].std); const FLK = STDK.concat(['X']); FLK.forEach((k, i) => { FL[k] = { x: -88 + i * 6.6, z: 26 }; });
FLK.forEach(k => { const f = FL[k], g = 'fl_' + k;
  lathe('flask', g, f.x, f.z, FPROF, '#e9f4f8', { alpha: 0.34 }, 10);
  lathe('flask', g, f.x, f.z, clipProf(FPROF, FMARK, 0.13), SOL[k].col, { alpha: 0.9, topCol: mix(SOL[k].col, '#ffffff', 0.35) }, 10);
  lathe('flask', g, f.x, f.z, [[0.78 * FS + 0.02, FMARK - 0.05], [0.78 * FS + 0.02, FMARK + 0.1]], '#1f2937', {}, 10);
  lathe('flask', 'st_' + k, f.x, f.z, [[0.5, FTOP - 0.9], [0.62, FTOP - 0.1], [0.62, FTOP + 0.35], [0.95, FTOP + 0.35], [0.95, FTOP + 1.45], [0, FTOP + 1.45]], '#2f6fe0', {}, 10); });
['R', 'S'].forEach(k => {
  box('cuv' + k, 'cuv' + k, -1.3, 1.3, 0, 7.6, -1.3, 1.3, '#e3f4f8', { alpha: 0.38, ds: true, cell: 99, faceCol: { l: '#f4f6f6', r: '#f4f6f6' } });
  box('cuv' + k, 'liq' + k, -1.1, 1.1, 0.3, 6.2, -1.1, 1.1, '#ffffff', { alpha: 0.7, cell: 99, dyn: 'liq' + k });
  box('cuv' + k, 'cuv' + k, -1.32, 1.32, 7.6, 7.7, -1.32, 1.32, '#d5dde0', { alpha: 0.5, cell: 99 });
});
box('tw', 'tw', -1.4, 1.4, -1.8, 1.8, -0.1, 0.1, '#ffffff', { ds: true, cell: 99 });

/* ================= kabel ================= */
const CABLES = [
  { nm: 'Kabel perpanjangan → stop kontak dinding', col: '#e8e6df', w: 1.1, pts: [[68, -0.5, -30], [72, Y0, -31], [72, Y0, -36.5], [72.5, 5, -37.8], [73, 18, -37.6]] },
  { nm: 'UPS → kabel perpanjangan', col: '#111', w: 1.0, pts: [[32.5, 5, -22.3], [32.5, Y0, -27], [44, Y0, -30], [53.2, Y0, -30], [53.2, 0.4, -30]] },
  { nm: 'UV-1700 → UPS', col: '#111', w: 1.0, pts: [[20, 1.5, -24.8], [20, Y0, -27.5], [33, Y0, -27], [37.5, Y0, -24], [37.5, 5, -22.3]] },
  { nm: 'CPU → UPS', col: '#111', w: 1.0, pts: [[-97, 8, -22.2], [-97, Y0, -27], [-60, Y0, -34], [20, Y0, -34.5], [32, Y0, -30], [32.5, 11, -22.3]] },
  { nm: 'Monitor → UPS', col: '#111', w: 0.9, pts: [[-58, 20, -17.1], [-58, Y0, -19], [-40, Y0, -31.5], [22, Y0, -32], [37, Y0, -28], [37.5, 11, -22.3]] },
  { nm: 'Printer → kabel perpanjangan', col: '#1d1d1d', w: 0.9, pts: [[-116, 3, -16.2], [-116, Y0, -22], [-100, Y0, -36], [57, Y0, -36], [57.2, Y0, -31], [57.2, 0.4, -30]] },
  { nm: 'Monitor → CPU (VGA)', col: '#2b4fae', w: 0.8, pts: [[-62, 22, -17.1], [-62, Y0, -20], [-80, Y0, -24], [-92, 14, -22.2]] },
  { nm: 'UV-1700 → CPU (RS-232C)', col: '#9aa0a8', w: 0.8, pts: [[5.5, 1.4, -24.8], [5.5, Y0, -27], [-30, Y0, -29], [-80, Y0, -27], [-94, 18, -22.2]] },
  { nm: 'Printer → CPU (USB)', col: '#6b7280', w: 0.7, pts: [[-112, 4, -16.2], [-106, Y0, -20], [-100, Y0, -24], [-99, 22, -22.2]] }
];
CABLES.forEach(c => { const p = []; for (let i = 0; i < c.pts.length - 1; i++) for (let k = 0; k < 6; k++) p.push(lerp3(c.pts[i], c.pts[i + 1], k / 6)); p.push(c.pts[c.pts.length - 1]); c.seg = p; });

const INFO = {
  wall: ['Dinding laboratorium', 'Stop kontak dinding menyuplai kabel perpanjangan; UPS mengambil daya dari kabel perpanjangan.'],
  bench: ['Meja kerja', 'Meja keramik bebas getaran.'], outlet: ['Stop kontak dinding', 'Sumber 220–230 V AC berarde.'],
  strip: ['Kabel perpanjangan', 'Menyalurkan daya ke UPS dan printer. Saat selesai, cabut semua kabel dari perpanjangan kecuali kabel UPS (IK C.9).'],
  ups: ['UPS ICA', 'Uninterruptible Power Supply: menstabilkan tegangan dan memberi daya cadangan bagi UV-1700, CPU, dan monitor. Dinyalakan pertama dan dimatikan terakhir (IK C.8).'],
  upsled: ['Indikator UPS', 'Lampu hijau menunjukkan UPS aktif dan tingkat beban.'],
  cpu: ['CPU (unit komputer)', 'Menjalankan UVProbe dan berkomunikasi dengan UV-1700 melalui kabel RS-232C. Label inventaris 19.01.01.24.2005 / 2.08.01.11.138.'],
  monitor: ['Monitor HP', 'Menampilkan UVProbe. Matikan dengan tombol power monitor setelah komputer shutdown (IK C.5).'], screen: ['Layar monitor', 'Tampilan UVProbe; lihat juga layar besar di bawah.'],
  keyboard: ['Papan ketik', 'Mengetik nama file, sample ID, dan konsentrasi.'], mouse: ['Mouse', 'Klik menu UVProbe.'], mousepad: ['Mouse pad', ''],
  printer: ['Printer', 'Mencetak tabel peak pick, kurva kalibrasi standar, dan data sampel.'], paper: ['Kertas cetak', 'Hasil cetak dari UVProbe.'],
  rack: ['Rak kuvet', 'Tempat kuvet referensi (belakang) dan kuvet sampel (depan) saat diisi dan dilap.'],
  tissue: ['Kotak tisu', 'Tisu untuk mengelap dinding bening kuvet dari tetesan dan sidik jari sebelum dimasukkan. Dua botol kecil putih di atasnya berisi cairan pembersih lensa/kuvet.'],
  cuvcase: ['Kotak kuvet kuarsa', 'Wadah penyimpanan kuvet kuarsa setelah kering.'], bottle: ['Botol semprot akuades', 'Botol semprot (wash bottle) berisi akuades untuk membilas kuvet sebelum dibilas dengan larutan yang akan diukur.'], chair: ['Kursi kerja', 'Setelah selesai, kursi dirapikan ke tempat aman (IK C.11).'], floor: ['Lantai', ''],
  beaker: ['Gelas piala Sampel X', 'Berisi sampel dengan konsentrasi yang tidak diketahui; dituang ke kuvet saat pengukuran sampel.'], etanol: ['Etanol', 'Pembilas pertama saat mengeringkan kuvet (IK D.1).'], aseton: ['Aseton', 'Pembilas terakhir; cepat menguap sehingga kuvet kering sempurna (IK D.1).'],
  ikbook: ['Buku Instruksi Kerja IK/LAB.INST/6', 'Klik untuk membuka. Gunakan tombol Halaman untuk membalik; isinya sama dengan prosedur pada panel langkah.'], logbook: ['Log Book alat', 'Catat semua kejadian dan tanda tangani dengan nama jelas (IK C.1).'], drymat: ['Alas tisu pengering', 'Tempat kuvet dibalik hingga kering.'],
  flask: ['Labu takar 50 mL', 'Berisi blanko, deret standar, atau Sampel X; larutan mencapai garis tera di leher, warna makin pekat seiring konsentrasi. Sumbat dibuka saat larutan dituang ke kuvet.'],
  silica: ['Silica gel', 'Penyerap lembap di kompartemen. Dikeluarkan sebelum pengukuran dan dimasukkan kembali setelah alat dimatikan (IK C.7).'],
  cuvR: ['Kuvet referensi', 'Kuvet kuarsa 1 cm berisi blanko, di sel referensi (belakang).'], cuvS: ['Kuvet sampel', 'Kuvet kuarsa 1 cm untuk blanko, standar, atau sampel, di sel sampel (depan).'],
  lid: ['Tutup kompartemen', 'Buka untuk memasukkan kuvet; tutup rapat sebelum Baseline, Auto Zero, dan pembacaan.'], compartment: ['Kompartemen sampel', 'Ruang gelap berisi sel referensi dan sel sampel.'],
  holderR: ['Sel referensi', 'Dudukan kuvet blanko.'], holderS: ['Sel sampel', 'Dudukan kuvet sampel.'],
  power: ['Tombol power UV-1700', 'Terletak di bagian kiri alat (IK C.7).'], lcd: ['Monitor (LCD) UV-1700', 'Menampilkan λ, absorbansi, lampu, dan status. Ditutup (dilipat) setelah alat dimatikan (IK C.7).'],
  keypad: ['Papan tombol UV-1700', 'Tombol MODE digunakan untuk keluar dari mode PC CONTROL sebelum mematikan alat (IK C.6).'],
  plate: ['Pelat identitas', 'UV-1700 PharmaSpec.'], body: ['Rumah instrumen', ''], cover: ['Penutup atas', ''], base: ['Port belakang', 'RS-232C (merah), Centronics, soket daya.'],
  vent: ['Kisi ventilasi', 'Berpendar oranye saat lampu WI menyala.'], card: ['IC card · CONTRAST', ''], warn: ['Label HOT SURFACE', 'Dinginkan lampu sebelum diganti.'],
  d2: ['Lampu D₂', 'Sumber UV.'], wi: ['Lampu WI', 'Sumber tampak–NIR.'], mirror: ['Cermin sumber', ''], slit1: ['Celah masuk', ''], mono: ['Monokromator', ''], grating: ['Kisi difraksi', ''],
  slit2: ['Celah keluar', ''], split: ['Pembagi berkas', ''], mirror2: ['Cermin pengarah', ''], detR: ['Detektor referensi', ''], detS: ['Detektor sampel', ''], pcb: ['Papan elektronik', ''], lamphouse: ['Rumah lampu', ''], tw: ['Tisu', '']
};

/* ================= kamera & render ================= */
const cv = $('#c'), ctx = cv.getContext('2d'), pcc = $('#pc'), pctx = pcc.getContext('2d');
const VIEWS = { meja: [-14, 26, 270, [-18, 6, 0]], alat: [-30, 24, 100, [0, 9, 0]], kompartemen: [-12, 58, 62, [-8, 8, 8]], kerja: [-26, 34, 100, [-40, 3, 16]],
  komputer: [-6, 10, 105, [-62, 20, -6]], ups: [-34, 18, 115, [44, 8, -12]], belakang: [200, 34, 230, [-25, 4, -25]], printer: [-30, 22, 75, [-116, 4, -4]],
  kering: [-20, 45, 70, [-20, 2, 22]], log: [-15, 50, 70, [-96, 0, 20]], ik: [-1, 78, 46, [-118, -1, 26.5]], isi: [-18, 32, 128, [-52, 5, 18]], kursi: [-24, 24, 210, [-24, -26, 50]] };
const CAM = { yaw: -14, pitch: 26, dist: 270, target: [-18, 6, 0] };
let camAnim = null, viewName = 'meja';
function setView(name) { if (name === 'ik') ST.ikT = Math.max(1, ST.ikT); else ST.ikT = 0; $('#ikbar').classList.toggle('show', name === 'ik'); const v = VIEWS[name]; if (!v || name === viewName && !camAnim) { if (v) viewName = name; return; } viewName = name; camAnim = { from: [CAM.yaw, CAM.pitch, CAM.dist, CAM.target.slice()], to: v, t: 0 };
  $$('.views button').forEach(b => b.classList.toggle('on', b.dataset.v === name)); dirty = true; }
function camXf(p) { const t = CAM.target; const x = p[0] - t[0], y = p[1] - t[1], z = p[2] - t[2];
  const cy = Math.cos(-CAM.yaw * D2R), sy = Math.sin(-CAM.yaw * D2R); const x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
  const cp = Math.cos(CAM.pitch * D2R), sp = Math.sin(CAM.pitch * D2R); return [x1, y * cp - z1 * sp, y * sp + z1 * cp]; }
let W = 800, H = 600, CX = 400, CY = 300, FOC = 800;
function proj(c) { const zc = CAM.dist - c[2]; return [CX + FOC * c[0] / zc, CY - FOC * c[1] / zc, zc]; }
function xf(p, part) {
  let q = p;
  if (part.tf) for (const [ax, piv, deg] of part.tf) q = ax === 'x' ? rotX(q, piv, deg * D2R) : ax === 'z' ? rotZ(q, piv, deg * D2R) : rotY(q, piv, deg * D2R);
  if (part.leaf != null) { const a = Math.max(0, Math.min(1, ST.ikp - part.leaf)); if (a > 0) q = rotZ(q, [IKX, IKB + IKL * IKTH / 2, 0], a * Math.PI); return q; }
  const g = part.grp;
  if (g === 'lid') q = rotX(q, [0, 15, -2.2], -102 * ST.lid * D2R);
  else if (g === 'lcdp') q = rotX(q, [11.5, 16.5, 9], 84 * ST.lcdFold * D2R);
  else if (g === 'cuvR' || g === 'cuvS' || g === 'liqR' || g === 'liqS') {
    const c = ST.cuv[g[3]];
    if (g[0] === 'l') q = [q[0] * 0.98, 0.3 + (q[1] - 0.3) * Math.max(0.001, c.level), q[2] * 0.98];
    if (c.flip) q = rotX(q, [0, 3.8, 0], 180 * c.flip * D2R);
    q = [q[0] + c.pos[0], q[1] + c.pos[1], q[2] + c.pos[2]];
  } else if (g === 'tw') { const t = ST.tissue; if (t) { q = rotY(q, [0, 0, 0], t.rot * D2R); q = [q[0] + t.pos[0], q[1] + t.pos[1], q[2] + t.pos[2]]; } }
  else if (g === 'silica') q = [q[0] + ST.silicaPos[0], q[1] + ST.silicaPos[1], q[2] + ST.silicaPos[2]];
  else if (g === 'paper') q = [q[0], q[1], 6 + (q[2] / 12) * 12 * ST.paper];
  else if (g === 'wash' && ST.wash) { q = rotZ(q, WASH0, ST.wash.ang * D2R); q = [q[0] + ST.wash.d[0], q[1] + ST.wash.d[1], q[2] + ST.wash.d[2]]; }
  else if (g.startsWith('fl_') && ST.pour && ST.pour.k === g.slice(3)) { const f = FL[ST.pour.k]; q = rotZ(q, [f.x, FLY, f.z], ST.pour.ang * D2R); q = [q[0] + ST.pour.d[0], q[1] + ST.pour.d[1], q[2] + ST.pour.d[2]]; }
  return q;
}
function visible(part) {
  const g = part.grp; const closed = ST.xray < 0.04;
  if (g === 'tw') return !!ST.tissue;
  if (g === 'paper') return ST.paper > 0.02;
  if (g.startsWith('fl_')) return !ST.tidy;
  if (g.startsWith('st_')) return !ST.tidy && !(ST.pour && ST.pour.k === g.slice(3));
  if (g.startsWith('liq')) { const c = ST.cuv[g[3]]; if (!c.content || c.level < 0.02) return false; }
  if (g.startsWith('cuv') || g.startsWith('liq')) { const c = ST.cuv[g[3]]; return !(c.where === 'sel' && closed && ST.lid < 0.05); }
  if (g === 'silica') return !(ST.silica === 'comp' && closed && ST.lid < 0.05);
  if (g === 'inner' && closed) return false;
  if (g === 'comp' && closed && ST.lid < 0.05) return false;
  return true;
}
const LIGHT = norm([0.45, 0.85, 0.55]);
let dirty = true, drawn = [], selId = null;
function draw() {
  const r = cv.getBoundingClientRect(); W = Math.max(300, r.width || 800); H = Math.max(240, r.height || 600);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); CX = W * 0.46; CY = H * 0.43; FOC = Math.min(W, H * 1.45);
  const g = ctx.createLinearGradient(0, 0, 0, H); g.addColorStop(0, '#f6f7fb'); g.addColorStop(1, '#e4e7f1'); ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  const lampUV = ST.lam < 340.8, on = ST.inst !== 'off';
  const coverA = 1 - 0.82 * ST.xray, wallA = 1 - 0.75 * ST.xray;
  const bg = [], items = [];
  for (const part of P) {
    if (!visible(part)) continue;
    let alpha = part.alpha;
    if (part.grp === 'cover' || part.grp === 'lid' || part.grp === 'lcdp') alpha *= coverA;
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
    let col = part.col, emi = part.emi; const dy = part.dyn;
    if (dy) {
      if (dy === 'd2') { emi = on && lampUV ? 0.45 : 0; col = on && lampUV ? '#e4dcff' : '#9a96b5'; }
      else if (dy === 'wi') { emi = on && !lampUV ? 0.45 : 0; col = on && !lampUV ? '#ffe7a0' : '#a89c74'; }
      else if (dy === 'glow') { col = on && !lampUV ? '#f0873a' : '#bfa868'; emi = on && !lampUV ? 0.22 : 0; }
      else if (dy === 'lcd') { col = on ? '#a99af2' : '#5c5777'; emi = on ? 0.08 : 0; }
      else if (dy.startsWith('upsled')) { const i = +dy[6]; const lit = ST.ups && i <= (1 + (ST.inst !== 'off') + (ST.pc ? 1 : 0) + 1); col = lit ? (i === 4 ? '#f59e0b' : '#22c55e') : '#2b2f2b'; emi = lit ? 0.5 : 0; if (i === 4) { col = ST.ups ? '#22c55e' : '#2b2f2b'; } }
      else if (dy === 'upsbtn') { col = ST.ups ? '#16351f' : '#0d0e10'; }
      else if (dy === 'pcled') { col = ST.pc ? '#38bdf8' : '#7d838c'; emi = ST.pc ? 0.35 : 0; }
      else if (dy === 'monled') { col = ST.pc ? '#3b82f6' : '#222'; emi = ST.pc ? 0.5 : 0; }
      else if (dy === 'prled') { col = ST.printer ? '#22c55e' : '#444'; emi = ST.printer ? 0.5 : 0; }
      else if (dy === 'pwr') { col = on ? '#3d9a5a' : '#2a2d33'; }
      else if (dy === 'modekey') { col = ST.keyFlash > 0 ? '#facc15' : '#cfd4dc'; emi = ST.keyFlash > 0 ? 0.3 : 0; }
      else if (dy === 'liqR' || dy === 'liqS') { const c = ST.cuv[dy[3]]; col = c.content ? SOL[c.content].col : '#fff'; }
    }
    if (selId && part.id === selId) emi += 0.15;
    const it = { part, pts: cs.map(proj), depth: CAM.dist - cen[2] + part.bias + (part.grp === 'inner' || part.grp === 'comp' ? 0.35 : 0) + (facing ? 0 : 0.02), fill: shade(col, lk, emi), alpha, facing, cs };
    (part.bg ? bg : items).push(it);
  }
  bg.sort((a, b) => b.depth - a.depth); paint(bg);
  ctx.fillStyle = 'rgba(20,18,58,.10)';
  [[-25, 25, -26.5, 26.5], [27, 45, -23, 9], [-107, -87, -23, 15], [-84, -32, -22, -10], [-127, -107, -17, 7], [-88, -55, 16, 36]].forEach(([x0, x1, z0, z1]) => {
    const pts = [[x0 + 1, z0 + 1], [x1 + 1.5, z0 + 1], [x1 + 1.5, z1 + 1.5], [x0 + 1, z1 + 1.5]].map(([x, z]) => proj(camXf([x, -1.36, z])));
    if (pts.some(p => p[2] < 4)) return; ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.closePath(); ctx.fill();
  });
  CABLES.forEach((c, ci) => {
    const prog = ST.cab[ci]; if (prog <= 0) return;
    const nseg = c.seg.length - 1, upto = prog * nseg;
    for (let i = 0; i < nseg && i < upto; i++) {
      const tEnd = Math.min(1, upto - i); const a = c.seg[i], b = lerp3(c.seg[i], c.seg[i + 1], tEnd);
      const ca = camXf(a), cb = camXf(b); if (CAM.dist - ca[2] < 4 || CAM.dist - cb[2] < 4) continue;
      items.push({ cable: true, pa: proj(ca), pb: proj(cb), depth: CAM.dist - (ca[2] + cb[2]) / 2 - 0.2, col: c.col, w: c.w, end: i === nseg - 1 && prog >= 1 });
    }
  });
  if (ST.stream) { const ca = camXf(ST.stream.a), cb = camXf(ST.stream.b); if (CAM.dist - ca[2] > 4 && CAM.dist - cb[2] > 4) items.push({ cable: true, pa: proj(ca), pb: proj(cb), depth: CAM.dist - (ca[2] + cb[2]) / 2 - 0.5, col: ST.stream.col, w: ST.stream.w, end: false }); }
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
    if (it.cable) { ctx.save(); const wpx = Math.max(1.2, it.w * FOC / Math.max(10, it.pa[2]) * 0.9);
      ctx.strokeStyle = it.col; ctx.lineWidth = wpx; ctx.lineCap = 'round'; ctx.beginPath(); ctx.moveTo(it.pa[0], it.pa[1]); ctx.lineTo(it.pb[0], it.pb[1]); ctx.stroke();
      if (it.end) { ctx.fillStyle = '#2b2b2b'; ctx.beginPath(); ctx.arc(it.pb[0], it.pb[1], wpx * 1.3, 0, 7); ctx.fill(); } ctx.restore(); continue; }
    if (it.beam) { ctx.save(); ctx.globalAlpha = 0.3 * it.att * it.a; ctx.strokeStyle = it.col; ctx.lineWidth = 7; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(it.pa[0], it.pa[1]); ctx.lineTo(it.pb[0], it.pb[1]); ctx.stroke();
      ctx.globalAlpha = 0.95 * it.att * it.a; ctx.lineWidth = 2.2; if (ST.lam < 380) ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(it.pa[0], it.pa[1]); ctx.lineTo(it.pb[0], it.pb[1]); ctx.stroke(); ctx.restore(); continue; }
    const p = it.pts; ctx.globalAlpha = it.alpha;
    ctx.beginPath(); ctx.moveTo(p[0][0], p[0][1]); for (let i = 1; i < 4; i++) ctx.lineTo(p[i][0], p[i][1]); ctx.closePath();
    ctx.fillStyle = it.fill; ctx.fill();
    if (it.alpha > 0.97) { ctx.strokeStyle = it.fill; ctx.lineWidth = 1; ctx.stroke(); }
    if (selId && it.part.id === selId) { ctx.strokeStyle = '#7c3aed'; ctx.lineWidth = 1.2; ctx.stroke(); }
    ctx.globalAlpha = 1;
    if (it.part.text && it.facing && (it.part.leaf == null || ikShow(it.part))) faceText(it.part, p, it);
    if (it.facing && it.alpha > 0.3 && !it.part.bg) drawn.push({ id: it.part.id, p });
  }
}
let SCRCS = null;
function screenHit(x, y) {
  if (!SCRCS) return null; const N = 12, M = 8, bl = (u, v) => proj(lerp3(lerp3(SCRCS[0], SCRCS[1], u), lerp3(SCRCS[3], SCRCS[2], u), v));
  for (let i = 0; i < N; i++) for (let j = 0; j < M; j++) { const u0 = i / N, u1 = (i + 1) / N, v0 = j / M, v1 = (j + 1) / M, q = [bl(u0, v0), bl(u1, v0), bl(u1, v1), bl(u0, v1)];
    let c = false; for (let a = 0, b = 3; a < 4; b = a++) { if (((q[a][1] > y) !== (q[b][1] > y)) && (x < (q[b][0] - q[a][0]) * (y - q[a][1]) / (q[b][1] - q[a][1]) + q[a][0])) c = !c; }
    if (c) { const TL = q[3], TR = q[2], BL = q[0], dx = x - TL[0], dy = y - TL[1], ax = TR[0] - TL[0], ay = TR[1] - TL[1], bx = BL[0] - TL[0], by = BL[1] - TL[1], det = ax * by - ay * bx || 1; const fu = (dx * by - dy * bx) / det, fv = (ax * dy - ay * dx) / det;
      return { x: (u0 + fu * (u1 - u0)) * 1040, y: (1 - v1 + fv * (v1 - v0)) * 610 }; } }
  return null;
}
function pcPress(x, y) { const b = pcBtns.find(b => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h); if (!b) return;
  if (b.act.startsWith('tab')) { ACT[b.act](); drawPC(); } else if (ACT[b.act]) run(() => ACT[b.act]()); }
function screenMesh(t, cs) {
  SCRCS = cs;
  const N = 12, M = 8, bl = (u, v) => proj(lerp3(lerp3(cs[0], cs[1], u), lerp3(cs[3], cs[2], u), v)), dpr = cv.width / Math.max(1, W), sw = t.w / N, sh = t.h / M;
  for (let i = 0; i < N; i++) for (let j = 0; j < M; j++) {
    const u0 = i / N, u1 = (i + 1) / N, v0 = j / M, v1 = (j + 1) / M, TL = bl(u0, v1), TR = bl(u1, v1), BL = bl(u0, v0);
    ctx.save(); ctx.setTransform((TR[0] - TL[0]) / sw * dpr, (TR[1] - TL[1]) / sw * dpr, (BL[0] - TL[0]) / sh * dpr, (BL[1] - TL[1]) / sh * dpr, TL[0] * dpr, TL[1] * dpr);
    ctx.drawImage(pcc, u0 * t.w, (1 - v1) * t.h, sw, sh, -1, -1, sw + 2, sh + 2); ctx.restore();
  }
}
function faceText(part, p, it) {
  if (part.text.screen && it && it.cs) { screenMesh(part.text, it.cs); return; }
  const t = part.text, TL = p[3], TR = p[2], BL = p[0]; const dpr = cv.width / Math.max(1, W);
  ctx.save(); ctx.setTransform((TR[0] - TL[0]) / t.w * dpr, (TR[1] - TL[1]) / t.w * dpr, (BL[0] - TL[0]) / t.h * dpr, (BL[1] - TL[1]) / t.h * dpr, TL[0] * dpr, TL[1] * dpr);
  if (t.ik != null) { if (t.rot) { ctx.translate(t.w, t.h); ctx.rotate(Math.PI); } ctx.drawImage(ikTex(t.ik), 0, 0, t.w, t.h); } else if (t.screen) ctx.drawImage(pcc, 0, 0, t.w, t.h); else if (t.lcd) drawLCD(ctx);
  else for (const [s, x, y, al, font, col] of t.lines) { ctx.font = font; ctx.fillStyle = col; ctx.textAlign = al; ctx.fillText(s, x, y); }
  ctx.restore();
}
const mmss = s => String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(Math.floor(s % 60)).padStart(2, '0');
function drawLCD(c) {
  if (ST.inst === 'off') return; c.fillStyle = '#1b1248';
  if (ST.inst === 'init') { c.font = '500 8.5px "JetBrains Mono",monospace'; c.textAlign = 'left'; ST.initLines.slice(-9).forEach((s, i) => c.fillText(s, 6, 11 + i * 9.6)); return; }
  if (ST.pcctrl) { c.font = '600 15px "JetBrains Mono",monospace'; c.textAlign = 'center'; c.fillText('PC CONTROL', 65, 36);
    c.font = '500 10px "JetBrains Mono",monospace'; c.fillText(ST.connected ? 'CONNECTED' : 'WAITING FOR PC', 65, 54); c.fillText(fx(ST.lam, 1).replace(',', '.') + 'nm  ' + (ST.lastA > 3 ? 'OVER' : ST.lastA.toFixed(3)) + ' A', 65, 72);
    c.fillStyle = '#4a3fa0'; c.fillText(ST.inst === 'warm' ? 'WARM-UP ' + mmss(ST.warmLeft) : (ST.lam < 340.8 ? 'D2' : 'WI'), 65, 89); return; }
  c.fillStyle = 'rgba(255,255,255,.85)'; [16, 44, 72].forEach(yy => c.fillRect(8, yy - 6, 24, 5));
  c.fillStyle = '#1b1248'; c.font = '500 17px "JetBrains Mono",monospace'; c.textAlign = 'right';
  c.fillText(ST.lam.toFixed(1) + 'nm', 124, 22); c.fillText((ST.lastA > 3 ? 'OVER' : ST.lastA.toFixed(3)) + ' ABS', 124, 50);
  c.font = '500 13px "JetBrains Mono",monospace'; c.textAlign = 'left'; c.fillText(ST.lam < 340.8 ? 'D2' : 'WI', 40, 77);
  c.textAlign = 'right'; c.fillStyle = '#4a3fa0'; c.font = '500 9.5px "JetBrains Mono",monospace'; c.fillText(ST.inst === 'warm' ? 'WARM-UP ' + mmss(ST.warmLeft) : 'MODE MENU', 124, 90);
}

/* ================= layar UVProbe ================= */
let pcBtns = [];
const F = (w, s) => (w || 400) + ' ' + (s || 12) + 'px Segoe UI, Inter, Arial';
function btn3(c, x, y, w, h, label, act, enabled, hi) {
  c.fillStyle = hi ? '#cce4f7' : '#f0f0f0'; c.fillRect(x, y, w, h); c.strokeStyle = hi ? '#0078d4' : '#adadad'; c.lineWidth = 1; c.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  c.fillStyle = enabled === false ? '#9ca3af' : '#111'; c.font = F(400, 12); c.textAlign = 'center'; c.fillText(label, x + w / 2, y + h / 2 + 4);
  if (act && enabled !== false) pcBtns.push({ x, y, w, h, act });
}
function regress() {
  const rows = ST.stdTable.filter(r => r.A != null); if (rows.length < 1) return null;
  const zero = !ST.pmethod || ST.pmethod.zero;
  const xs = rows.map(r => r.conc), ys = rows.map(r => r.A), n = xs.length;
  let a, b = 0;
  if (zero) { const sxy = xs.reduce((s, x, i) => s + x * ys[i], 0), sxx = xs.reduce((s, x) => s + x * x, 0); a = sxy / sxx; }
  else { if (n < 2) return null; const mx = xs.reduce((s, v) => s + v) / n, my = ys.reduce((s, v) => s + v) / n; let sxx = 0, sxy = 0; xs.forEach((x, i) => { sxx += (x - mx) ** 2; sxy += (x - mx) * (ys[i] - my); }); a = sxy / sxx; b = my - a * mx; }
  const my = ys.reduce((s, v) => s + v) / n; let ssr = 0, sst = 0; xs.forEach((x, i) => { ssr += (ys[i] - (a * x + b)) ** 2; sst += (ys[i] - my) ** 2; });
  return { a, b, r2: sst > 0 ? 1 - ssr / sst : 1, zero, n };
}
function drawPC() {
  const c = pctx, Wp = pcc.width, Hp = pcc.height; pcBtns = []; c.setTransform(1, 0, 0, 1, 0, 0);
  $('#pcst').textContent = !ST.pc ? 'MATI' : ST.pc === 1 ? 'MEMULAI' : ST.pc === 4 ? 'SHUTDOWN' : ST.connected ? 'UVPROBE · TERHUBUNG' : ST.sw ? 'UVPROBE · TIDAK TERHUBUNG' : 'DESKTOP';
  if (!ST.pc) { c.fillStyle = '#020203'; c.fillRect(0, 0, Wp, Hp); c.fillStyle = '#1f2937'; c.font = F(500, 16); c.textAlign = 'center'; c.fillText(ST.ups ? 'Komputer mati' : 'Tidak ada daya', Wp / 2, Hp / 2); return; }
  if (ST.pc === 1 || ST.pc === 4) { c.fillStyle = '#0a1b3d'; c.fillRect(0, 0, Wp, Hp); c.fillStyle = '#e0e7ff'; c.font = F(300, 26); c.textAlign = 'center';
    c.fillText(ST.pc === 1 ? 'Memulai Windows…' : 'Mematikan komputer…', Wp / 2, Hp / 2 - 24);
    for (let i = 0; i < 5; i++) { const a = performance.now() / 300 + i * 0.6; c.beginPath(); c.arc(Wp / 2 + 26 * Math.cos(a), Hp / 2 + 26 + 26 * Math.sin(a), 3.5, 0, 7); c.fill(); } return; }
  // desktop
  const gr = c.createLinearGradient(0, 0, Wp, Hp); gr.addColorStop(0, '#0b3a75'); gr.addColorStop(1, '#0e7490'); c.fillStyle = gr; c.fillRect(0, 0, Wp, Hp);
  const TB = Hp - 32; c.fillStyle = '#101418'; c.fillRect(0, TB, Wp, 32); c.fillStyle = '#fff'; c.fillRect(10, TB + 8, 16, 16);
  c.fillStyle = '#2b2f36'; c.fillRect(36, TB + 5, 260, 22); c.fillStyle = '#9ca3af'; c.font = F(400, 12); c.textAlign = 'left'; c.fillText('Search the web and Windows', 46, TB + 20);
  const d = new Date(); c.fillStyle = '#e5e7eb'; c.textAlign = 'right'; c.fillText(String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0') + '   ' + dstr.replace(/-/g, '/'), Wp - 12, TB + 20);
  if (!ST.sw) {
    c.fillStyle = 'rgba(255,255,255,.12)'; c.fillRect(24, 24, 84, 84); c.fillStyle = '#ffffff'; c.fillRect(44, 36, 44, 34); c.fillStyle = '#2563eb'; c.fillRect(48, 40, 36, 26);
    c.strokeStyle = '#fbbf24'; c.lineWidth = 2; c.beginPath(); c.moveTo(50, 62); c.quadraticCurveTo(60, 38, 70, 56); c.quadraticCurveTo(76, 64, 82, 46); c.stroke();
    c.fillStyle = '#fff'; c.font = F(500, 12); c.textAlign = 'center'; c.fillText('UVProbe', 66, 96); pcBtns.push({ x: 24, y: 24, w: 84, h: 84, act: 'software' });
    if (ST.pc === 2 && ST.done.size) { c.fillStyle = 'rgba(255,255,255,.85)'; c.font = F(400, 14); c.fillText('Klik dua kali ikon UVProbe untuk membuka perangkat lunak', Wp / 2, Hp / 2); }
    drawDlg(c, Wp, Hp); return;
  }
  // jendela UVProbe
  const X = 0, Y = 0, WW = Wp, HH = TB; c.fillStyle = '#ffffff'; c.fillRect(X, Y, WW, HH);
  c.fillStyle = '#ffffff'; c.fillRect(0, 0, WW, 26); c.fillStyle = '#111'; c.font = F(400, 12); c.textAlign = 'left';
  c.fillText('UVProbe - [' + (ST.win === 'photometric' ? 'Photometric' : ST.win === 'spectrum' ? 'Spectrum' : 'Report') + ']', 30, 17);
  c.fillStyle = '#2563eb'; c.fillRect(8, 6, 14, 14); c.fillStyle = '#111'; c.textAlign = 'right'; c.fillText('—    ☐    ✕', WW - 12, 17);
  const MENUS = ['File', 'Edit', 'View', 'Operations', 'Graph', 'Instrument', 'Tools', 'Window', 'Help']; let mx = 8; const mpos = {};
  c.font = F(400, 12); c.textAlign = 'left';
  MENUS.forEach(m => { const w = c.measureText(m).width + 16; mpos[m] = mx; if (ST.menu && ST.menu.name === m) { c.fillStyle = '#cce4f7'; c.fillRect(mx, 26, w, 20); } c.fillStyle = '#111'; c.fillText(m, mx + 8, 40); mx += w; });
  c.fillStyle = '#f5f5f5'; c.fillRect(0, 46, WW, 26); c.strokeStyle = '#e5e7eb'; c.beginPath(); c.moveTo(0, 72.5); c.lineTo(WW, 72.5); c.stroke();
  for (let i = 0; i < 22; i++) { c.fillStyle = i % 5 === 4 ? '#f5f5f5' : ['#94a3b8', '#60a5fa', '#a3a3a3', '#f59e0b'][i % 4]; c.fillRect(8 + i * 22, 51, 16, 16); }
  const mainTop = 74, barTop = HH - 64;
  if (!ST.win) { c.fillStyle = '#6b7280'; c.font = F(400, 15); c.textAlign = 'center'; c.fillText('Pilih Window → Spectrum atau Window → Photometric', WW / 2, (mainTop + barTop) / 2); }
  else if (ST.win === 'spectrum') drawSpectrumWin(c, mainTop, barTop, WW);
  else drawPhotoWin(c, mainTop, barTop, WW);
  // bilah instrumen bawah
  c.fillStyle = '#f3f3f3'; c.fillRect(0, barTop, WW, 64); c.strokeStyle = '#d4d4d4'; c.beginPath(); c.moveTo(0, barTop + 0.5); c.lineTo(WW, barTop + 0.5); c.stroke();
  c.fillStyle = '#111'; c.font = F(600, 14); c.textAlign = 'left'; c.fillText(fx(ST.lam, 1) + ' nm', 14, barTop + 22); c.fillText((ST.connected ? fx(ST.lastA, 3) : '-----') + ' Abs', 130, barTop + 22);
  c.font = F(400, 11); c.fillStyle = '#4b5563'; const z = ST.zero ? (ST.zero.type === 'az' ? 'Auto zero ' + fx(ST.zero.lam, 1) + ' nm' : 'Baseline ' + ST.zero.lo + '–' + ST.zero.hi + ' nm') : 'Belum baseline/auto zero';
  c.fillText(z + (ST.lid > 0.5 ? '   ·   KOMPARTEMEN TERBUKA' : ''), 260, barTop + 22);
  const by = barTop + 32, bh = 26; let bx = 10; const hiA = ST.hiBtn;
  const B = (lab, act, en) => { const w = Math.max(84, c.measureText(lab).width + 26); btn3(c, bx, by, w, bh, lab, act, en, hiA === act); bx += w + 6; };
  B('Auto Zero', 'autozero', ST.connected); B(ST.connected ? 'Disconnect' : 'Connect', ST.connected ? 'disconnect' : 'connect', true);
  B('Go To WL', 'goto', ST.connected); if (ST.win === 'spectrum') { B('Baseline', 'baseline', ST.connected); B('Start', 'scan', ST.connected); }
  if (ST.win === 'photometric') { B('Read Std', 'readstd', ST.connected && ST.ptab === 'std'); B('Read Unk', 'readunk', ST.connected && ST.ptab === 'unk'); }
  if (ST.menu) drawMenu(c, mpos[ST.menu.name]);
  drawDlg(c, Wp, Hp);
}
function drawMenu(c, x) {
  const m = ST.menu, w = 210, ih = 22; c.fillStyle = '#ffffff'; c.fillRect(x, 46, w, m.items.length * ih + 6); c.strokeStyle = '#a3a3a3'; c.strokeRect(x + 0.5, 46.5, w, m.items.length * ih + 6);
  m.items.forEach((it, i) => { if (i === m.hi) { c.fillStyle = '#cce4f7'; c.fillRect(x + 2, 49 + i * ih, w - 4, ih); } c.fillStyle = '#111'; c.font = F(400, 12); c.textAlign = 'left'; c.fillText(it, x + 24, 64 + i * ih); });
}
function drawSpectrumWin(c, top, bot, WW) {
  const lw = 380;
  c.fillStyle = '#fafafa'; c.fillRect(0, top, lw, bot - top); c.strokeStyle = '#d4d4d4'; c.strokeRect(0.5, top + 0.5, lw, bot - top - 1);
  c.fillStyle = '#e8eef7'; c.fillRect(1, top + 1, lw - 1, 22); c.fillStyle = '#111'; c.font = F(600, 12); c.textAlign = 'left'; c.fillText('Peak Pick', 10, top + 16);
  const cols = [['No.', 10], ['P/V', 50], ['Wavelength', 100], ['Abs.', 210], ['Description', 280]];
  c.fillStyle = '#f0f0f0'; c.fillRect(1, top + 23, lw - 1, 20); c.fillStyle = '#374151'; c.font = F(600, 11); cols.forEach(([s, x]) => c.fillText(s, x, top + 37));
  (ST.peaks || []).slice(0, 10).forEach((p, i) => { const yy = top + 43 + i * 20; if (ST.peakSel && i === 0) { c.fillStyle = '#cce4f7'; c.fillRect(1, yy, lw - 1, 20); } c.fillStyle = '#111'; c.font = F(400, 12);
    c.fillText(String(i + 1), 10, yy + 14); c.fillText(p.v ? 'V' : 'P', 50, yy + 14); c.fillText(fx(p.l, 2), 100, yy + 14); c.fillText(fx(p.a, 4), 210, yy + 14); c.fillText(i === 0 ? 'λmaks' : '', 280, yy + 14); });
  if (ST.smethod) { c.fillStyle = '#6b7280'; c.font = F(400, 11); const m = ST.smethod; const y = bot - 70;
    ['Metode spektrum:', 'Rentang ' + m.hi + ' – ' + m.lo + ' nm', 'Kecepatan ' + m.speed + ' · interval ' + fx(m.step, 1) + ' nm', 'Mode: Absorbance · celah 1,0 nm'].forEach((s, i) => c.fillText(s, 10, y + i * 15)); }
  graphFrame(c, lw + 10, top + 6, WW - lw - 20, bot - top - 12, 'nm', 'Abs.');
  const sp = ST.spectrum, gx = lw + 10, gy = top + 6, gw = WW - lw - 20, gh = bot - top - 12;
  const [lo, hi] = ST.smethod ? [ST.smethod.lo, ST.smethod.hi] : scanRange();
  if (sp && sp.l.length > 1) {
    const ymax = Math.max(0.1, ...sp.a) * 1.15, ymin = Math.min(0, ...sp.a);
    const PX = l => gx + 50 + (gw - 64) * (l - lo) / (hi - lo), PY = a => gy + gh - 30 - (gh - 50) * (a - ymin) / (ymax - ymin);
    c.fillStyle = '#374151'; c.font = F(400, 10); c.textAlign = 'right'; for (let i = 0; i <= 4; i++) { const a = ymin + (ymax - ymin) * i / 4; c.fillText(fx(a, 3), gx + 46, PY(a) + 3); }
    c.textAlign = 'center'; for (let l = Math.ceil(lo / 50) * 50; l <= hi; l += 50) c.fillText(fx(l, 2), PX(l), gy + gh - 14);
    c.strokeStyle = '#d00000'; c.lineWidth = 1.4; c.beginPath(); sp.l.forEach((l, i) => { const X1 = PX(l), Y1 = PY(sp.a[i]); i ? c.lineTo(X1, Y1) : c.moveTo(X1, Y1); }); c.stroke();
    if (ST.peaks) ST.peaks.slice(0, 3).forEach((p, i) => { c.fillStyle = '#111'; c.font = F(400, 10); c.textAlign = 'center'; c.fillText(fx(p.l, 2), PX(p.l), PY(p.a) - 8); c.beginPath(); c.moveTo(PX(p.l), PY(p.a) - 4); c.lineTo(PX(p.l) - 3, PY(p.a) - 1); c.lineTo(PX(p.l) + 3, PY(p.a) - 1); c.fill(); });
  } else { c.fillStyle = '#9ca3af'; c.font = F(400, 13); c.textAlign = 'center'; c.fillText(ST.smethod ? 'Lakukan Baseline, lalu Start' : 'Atur metode: Edit → Method', gx + gw / 2, gy + gh / 2); }
}
function graphFrame(c, x, y, w, h, xl, yl) {
  c.fillStyle = '#fff'; c.fillRect(x, y, w, h); c.strokeStyle = '#9ca3af'; c.lineWidth = 1; c.strokeRect(x + 50.5, y + 10.5, w - 64, h - 40);
  c.fillStyle = '#374151'; c.font = F(400, 11); c.textAlign = 'center'; c.fillText(xl, x + w / 2, y + h - 2); c.save(); c.translate(x + 12, y + h / 2); c.rotate(-Math.PI / 2); c.fillText(yl, 0, 0); c.restore();
}
function drawPhotoWin(c, top, bot, WW) {
  const lw = 560;
  // tab tabel
  const tabY = bot - 24; c.fillStyle = '#ffffff'; c.fillRect(0, top, lw, tabY - top); c.strokeStyle = '#d4d4d4'; c.strokeRect(0.5, top + 0.5, lw, tabY - top);
  const pm = ST.pmethod, L = pm ? pm.lam : null;
  const head = ['', 'Sample ID', 'Type', 'Ex', 'Conc', L ? 'WL' + fx(L, 1) : 'WL', 'Comments'], xs = [4, 30, 170, 250, 290, 370, 460];
  c.fillStyle = '#f0f0f0'; c.fillRect(1, top + 1, lw - 1, 22); c.fillStyle = '#111'; c.font = F(600, 11); c.textAlign = 'left'; head.forEach((h, i) => c.fillText(h, xs[i], top + 16));
  const rows = ST.ptab === 'std' ? ST.stdTable : ST.unkTable;
  for (let i = 0; i < Math.max(10, rows.length); i++) {
    const yy = top + 23 + i * 21; if (yy > tabY - 21) break; c.strokeStyle = '#eceff3'; c.beginPath(); c.moveTo(0, yy + 21.5); c.lineTo(lw, yy + 21.5); c.stroke();
    c.fillStyle = '#6b7280'; c.font = F(400, 11); c.fillText(String(i + 1), xs[0], yy + 15);
    const r = rows[i]; if (!r) continue;
    if (ST.rowHi === i && ST.ptab === (ST.rowHiTab || ST.ptab)) { c.fillStyle = '#fef9c3'; c.fillRect(24, yy, lw - 25, 21); }
    c.fillStyle = '#111'; c.font = F(400, 12);
    c.fillText(r.id || '', xs[1], yy + 15); c.fillText(ST.ptab === 'std' ? 'Standard' : 'Unknown', xs[2], yy + 15); c.fillText('', xs[3], yy + 15);
    c.fillText(r.conc != null && isFinite(r.conc) ? fx(r.conc, 4) : '', xs[4], yy + 15); c.fillText(r.A != null ? fx(r.A, 4) : '', xs[5], yy + 15);
  }
  // tab di bawah tabel
  [['Standard Table', 'std'], ['Sample Table', 'unk']].forEach(([s, k], i) => { const x = 4 + i * 128; c.fillStyle = ST.ptab === k ? '#ffffff' : '#e5e7eb'; c.fillRect(x, tabY, 124, 22); c.strokeStyle = '#c7c7c7'; c.strokeRect(x + 0.5, tabY + 0.5, 124, 22);
    c.fillStyle = '#111'; c.font = F(ST.ptab === k ? 600 : 400, 12); c.textAlign = 'center'; c.fillText(s, x + 62, tabY + 15); pcBtns.push({ x, y: tabY, w: 124, h: 22, act: 'tab' + k }); });
  // grafik kurva standar
  const gx = lw + 10, gy = top + 6, gw = WW - lw - 20, gh = bot - top - 76; graphFrame(c, gx, gy, gw, gh, 'Conc (' + (pm ? pm.unit : INIT.unit) + ')', 'Abs.');
  c.fillStyle = '#111'; c.font = F(600, 12); c.textAlign = 'left'; c.fillText('Standard Curve', gx + 56, gy + 24);
  const reg = regress(), pts = ST.stdTable.filter(r => r.A != null), unk = ST.unkTable.filter(r => r.A != null);
  if (pts.length) {
    const xmax = Math.max(...ST.stdTable.map(r => r.conc), 0.01) * 1.1, ymax = Math.max(...pts.map(r => r.A), ...unk.map(r => r.A), 0.05) * 1.15;
    const PX = x => gx + 50 + (gw - 64) * x / xmax, PY = y => gy + gh - 30 - (gh - 40) * y / ymax;
    c.fillStyle = '#374151'; c.font = F(400, 10); c.textAlign = 'right'; for (let i = 0; i <= 4; i++) c.fillText(fx(ymax * i / 4, 3), gx + 46, PY(ymax * i / 4) + 3);
    c.textAlign = 'center'; for (let i = 0; i <= 4; i++) c.fillText(fx(xmax * i / 4, 2), PX(xmax * i / 4), gy + gh - 18);
    if (reg) { c.strokeStyle = '#d00000'; c.lineWidth = 1.4; c.beginPath(); c.moveTo(PX(0), PY(reg.b)); c.lineTo(PX(xmax), PY(reg.a * xmax + reg.b)); c.stroke(); }
    c.fillStyle = '#1d4ed8'; pts.forEach(r => { c.fillRect(PX(r.conc) - 3.5, PY(r.A) - 3.5, 7, 7); });
    c.fillStyle = '#16a34a'; unk.forEach(r => { if (isFinite(r.conc)) { c.beginPath(); c.arc(PX(r.conc), PY(r.A), 4.5, 0, 7); c.fill(); } });
  } else { c.fillStyle = '#9ca3af'; c.font = F(400, 13); c.textAlign = 'center'; c.fillText(pm ? 'Read Std untuk membentuk kurva' : 'Atur metode: Edit → Method', gx + gw / 2, gy + gh / 2); }
  c.fillStyle = '#111'; c.font = F(400, 12); c.textAlign = 'left'; const ly = gy + gh + 18;
  if (reg) { c.fillText('Abs = ' + fx(reg.a, 5) + ' * Conc' + (reg.zero ? '' : (reg.b < 0 ? ' − ' : ' + ') + fx(Math.abs(reg.b), 5)), gx + 6, ly); c.fillText('r² = ' + fx(reg.r2, 5) + '    (' + reg.n + ' standar' + (reg.zero ? ', zero interception' : '') + ')', gx + 6, ly + 18); }
  if (pm) c.fillText('Metode: ' + pm.type + ' · ' + fx(pm.lam, 1) + ' nm · ' + pm.unit + '   File: ' + (pm.file || '-'), gx + 6, ly + 36);
}
function drawDlg(c, Wp, Hp) {
  const d = ST.dlg; if (!d) return;
  const w = d.w || 520, h = d.h || 330, x = (Wp - w) / 2, y = (Hp - 32 - h) / 2;
  c.fillStyle = 'rgba(0,0,0,.18)'; c.fillRect(x + 6, y + 6, w, h); c.fillStyle = '#ffffff'; c.fillRect(x, y, w, h); c.strokeStyle = '#0078d4'; c.lineWidth = 1; c.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  c.fillStyle = '#111'; c.font = F(400, 12); c.textAlign = 'left'; c.fillText(d.title, x + 12, y + 19); c.textAlign = 'right'; c.fillText('✕', x + w - 12, y + 19);
  c.strokeStyle = '#e5e7eb'; c.beginPath(); c.moveTo(x, y + 28.5); c.lineTo(x + w, y + 28.5); c.stroke();
  if (d.sub) { c.fillStyle = '#1e3a8a'; c.font = F(600, 14); c.textAlign = 'left'; c.fillText(d.sub, x + 16, y + 52); }
  let yy = y + (d.sub ? 80 : 56);
  (d.fields || []).forEach(f => {
    c.fillStyle = '#111'; c.font = F(400, 12); c.textAlign = 'left'; c.fillText(f[0], x + 18, yy + 4);
    const fxw = f[2] || 200, fxx = x + w - fxw - 24;
    if (f[3] === 'check') { c.strokeStyle = '#555'; c.strokeRect(fxx + 0.5, yy - 9.5, 13, 13); if (f[1]) { c.strokeStyle = '#0078d4'; c.lineWidth = 2; c.beginPath(); c.moveTo(fxx + 3, yy - 3); c.lineTo(fxx + 6, yy + 1); c.lineTo(fxx + 11, yy - 7); c.stroke(); c.lineWidth = 1; } }
    else { c.fillStyle = d.hi === f[0] ? '#fffbe6' : '#fff'; c.fillRect(fxx, yy - 12, fxw, 22); c.strokeStyle = d.hi === f[0] ? '#0078d4' : '#9ca3af'; c.strokeRect(fxx + 0.5, yy - 11.5, fxw, 22);
      c.fillStyle = '#111'; c.fillText(String(f[1]) + (d.hi === f[0] && (performance.now() / 400 | 0) % 2 ? '|' : ''), fxx + 6, yy + 4); }
    yy += 32;
  });
  (d.buttons || []).forEach((b, i) => { const bw = 86; btn3(c, x + w - (d.buttons.length - i) * (bw + 8) - 8, y + h - 38, bw, 26, b, null, true, d.btnHi === b); });
  if (d.progress != null) { c.fillStyle = '#e5e7eb'; c.fillRect(x + 20, y + h - 70, w - 40, 12); c.fillStyle = '#06b025'; c.fillRect(x + 20, y + h - 70, (w - 40) * d.progress, 12); }
}

/* ================= animasi & utilitas ================= */
const tweens = []; let SPD = 1, PAUSED = false, FAST = false;
function flushTweens() { while (tweens.length) { const tw = tweens.shift(); try { tw.fn(1); } catch (e) { } tw.res(); } dirty = true; }
function tween(dur, fn) { if (FAST) { try { fn(1); } catch (e) { console.error(e); } return Promise.resolve(); } return new Promise(res => tweens.push({ el: 0, dur: dur * 1000, fn, res })); }
const ease = t => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
const wait = s => { if (FAST && ST.inst === 'warm') { ST.warmLeft = 0; ST.inst = 'ready'; } return tween(s, () => { }); };
let audio = null;
function beep(f, d) { if (!ST.sound || FAST) return; try { audio = audio || new (window.AudioContext || window.webkitAudioContext)(); const o = audio.createOscillator(), g = audio.createGain(); o.frequency.value = f || 880; g.gain.value = 0.05; o.connect(g); g.connect(audio.destination); o.start(); o.stop(audio.currentTime + (d || 0.12)); } catch (e) { } }
let toastT = 0;
function toast(msg, kind) { const t = $('#toast'); t.textContent = msg; t.className = 'toast show ' + (kind || 'info'); clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove('show'), 3800); }
function log(msg, kind) { const el = $('#log'); const d = document.createElement('div'); if (kind) d.className = kind; d.textContent = mmss((performance.now() - ST.t0) / 1000) + '  ' + msg; el.insertBefore(d, el.firstChild); }
function mistake(code, msg) { if (!ST.errs.has(code)) { ST.errs.add(code); ST.score = Math.max(0, ST.score - 5); log('Kesalahan prosedur (−5): ' + msg, 'e'); } toast(msg, 'err'); beep(220, 0.25); }
function ok(msg) { toast(msg, 'ok'); log(msg, 'o'); }
const ready = () => ST.inst === 'warm' || ST.inst === 'ready';
const selCuv = () => ($$('input[name=cv]').find(i => i.checked) || {}).value || 'S';
function moveCuv(k, pts, d) { const c = ST.cuv[k]; let ch = Promise.resolve(); pts.forEach(to => { ch = ch.then(() => { const from = c.pos.slice(); return tween(d, t => { c.pos = lerp3(from, to, ease(t)); }); }); }); return ch; }
async function menuPick(name, items, idx) { if (ST.pc < 3) return; ST.menu = { name, items, hi: -1 }; await wait(0.35); for (let i = 0; i <= idx; i++) { ST.menu.hi = i; await wait(0.12); } await wait(0.3); ST.menu = null; }
async function typeInto(obj, key, text, dlgField) { const s = String(text); if (ST.dlg) ST.dlg.hi = dlgField; for (let i = 1; i <= s.length; i++) { obj[key] = s.slice(0, i); if (ST.dlg && dlgField) setField(dlgField, s.slice(0, i)); await wait(0.05); } }
function setField(name, val) { const f = (ST.dlg.fields || []).find(f => f[0] === name); if (f) f[1] = val; }
async function pressBtn(act) { ST.hiBtn = act; await wait(0.35); ST.hiBtn = null; }
async function dlgBtn(b) { if (ST.dlg) { ST.dlg.btnHi = b; await wait(0.35); } }
const MENU = { File: ['New', 'Open...', 'Save', 'Save As...', 'Print...', 'Exit'], Edit: ['Undo', 'Copy', 'Paste', 'Method...'], Window: ['Spectrum', 'Photometric', 'Kinetics', 'Report Generator'],
  Operations: ['Peak Pick', 'Point Pick', 'Arithmetic', 'Smoothing'], Windows: ['Shutdown'] };

/* ================= fisika pengukuran ================= */
function cellA(k, l) { const c = ST.cuv[k]; if (c.where !== 'sel') return 0; let a = 0.045 + c.smudge; if (!c.content) return a + 0.035; return a + 0.003 + SOL[c.content].c * absorptivity(l); }
function rawA(l, noisy) {
  let A = cellA('S', l) - cellA('R', l) + instOff(l); const t = (performance.now() - ST.t0) / 1000;
  if (ST.lid > 0.5) A -= 0.05 + 0.06 * rnd();
  if (ST.silica === 'comp') A += 0.35 + 0.05 * rnd();
  A += ST.warmSkipped ? 0.006 * Math.sin(t / 7) + 0.002 * gauss() : 0.0004 * Math.sin(t / 30);
  if (noisy !== false) { const T = Math.pow(10, -A); A += gauss() * 0.4343 * 1.5e-4 / Math.max(T * lampE(l), 1e-4); }
  const s = l < 230 ? 4e-4 : 1e-4; const T = (Math.pow(10, -A) + s) / (1 + s); return Math.min(4, Math.max(-0.5, -Math.log10(T)));
}
function zeroOff(l) { const z = ST.zero; if (!z) return null; if (z.type === 'az' && Math.abs(z.lam - l) < 0.05) return z.off; if (z.type === 'bl' && l >= z.lo && l <= z.hi) return z.offs[Math.round(l) - z.lo]; return null; }
function readAt(l) { const r = rawA(l); const z = zeroOff(l); return z == null ? r : r - z; }
function checks() {
  if (!ready()) { toast(ST.inst === 'off' ? 'UV-1700 belum dinyalakan.' : 'Tunggu inisialisasi selesai.', 'warn'); return false; }
  if (!ST.connected) { toast('Hubungkan UVProbe ke UV-1700 (Connect) terlebih dahulu.', 'warn'); return false; }
  if (ST.lid > 0.5) mistake('lidopen', 'Kompartemen masih terbuka: cahaya ruangan masuk ke detektor.');
  if (ST.silica === 'comp') mistake('silicain', 'Silica gel masih di dalam kompartemen dan menghalangi berkas.');
  if (ST.inst === 'warm' && !ST.warmSkipped) mistake('nowarm', 'Lampu belum stabil (pemanasan belum selesai).');
  return true;
}
async function beamPulse(s) { ST.xrayT = ST.autoX ? 1 : ST.xrayT; await tween(s, t => { ST.beam = Math.sin(Math.PI * t); }); ST.beam = 0; if (!ST.xrayUser) ST.xrayT = 0; }
function findPeaks(l, a) { const out = []; for (let i = 3; i < a.length - 3; i++) { const v = a[i]; if (v > a[i - 1] && v >= a[i + 1] && v > a[i - 3] && v > a[i + 3]) { let L = Math.min(...a.slice(Math.max(0, i - 25), i)), R = Math.min(...a.slice(i + 1, i + 26)); if (v - Math.max(L, R) > 0.01 && v > 0.03) out.push({ l: l[i], a: v }); } } return out.sort((p, q) => q.a - p.a); }

/* ================= animasi botol semprot & labu takar ================= */
function wpt(p, part) { return xf(p, part); }
async function washBottle(c, fill) {
  const p = c.pos, T = [p[0] - 0.2, 11.2, p[2]], th = -72, base = WASHTIP.slice();
  const tipRel = rotZ(WASHTIP, WASH0, th * D2R).map((v, i) => v - WASH0[i]);
  const F = [T[0] - tipRel[0], T[1] - tipRel[1], T[2] - tipRel[2]], d = [F[0] - WASH0[0], F[1] - WASH0[1], F[2] - WASH0[2]];
  await tween(0.7, t => { const e = ease(t); ST.wash = { ang: th * e, d: [d[0] * e, d[1] * e + 6 * Math.sin(Math.PI * e), d[2] * e] }; });
  ST.wash = { ang: th, d };
  await tween(fill ? 1.7 : 1.0, t => { const tip = xf(WASHTIP, { grp: 'wash' }); if (fill) { c.content = 'blank'; c.level = 0.85 * ease(t); } ST.stream = { a: tip, b: [p[0], 5.2 + 0.4 * Math.sin(t * 40), p[2]], col: '#a8d8ff', w: 0.3 }; if (!fill) c.level = Math.max(c.level, 0.3 * Math.sin(Math.PI * t)); });
  ST.stream = null; beep(500, 0.05);
  await tween(0.6, t => { const e = 1 - ease(t); ST.wash = { ang: th * e, d: [d[0] * e, d[1] * e, d[2] * e] }; });
  ST.wash = null; if (!c.content) c.level = 0;
}
async function pourFlask(sid, c, rinse) {
  const f = FL[sid], th = -104, mouth = [f.x + (f.ox || 0), FLY + (f.h || 16.2 * FS), f.z], pv = [f.x, FLY, f.z], p = c.pos;
  const T = [p[0] - 0.6, 10.6, p[2]], rel = rotZ(mouth, pv, th * D2R).map((v, i) => v - pv[i]);
  const F = [T[0] - rel[0], T[1] - rel[1], T[2] - rel[2]], d = [F[0] - pv[0], F[1] - pv[1], F[2] - pv[2]];
  const dur = rinse ? 0.5 : 1.0;
  await tween(0.8, t => { const e = ease(t); ST.pour = { k: sid, ang: th * Math.min(1, e * 1.2), d: [d[0] * e, d[1] * e + 10 * Math.sin(Math.PI * e), d[2] * e] }; });
  ST.pour = { k: sid, ang: th, d }; c.content = sid;
  await tween(dur, t => { ST.stream = { a: xf(mouth, { grp: 'fl_' + sid }), b: [p[0], 4.6 + (rinse ? 0 : 2.2 * t), p[2]], col: SOL[sid].col, w: 0.35 }; c.level = rinse ? 0.3 * t : 0.85 * ease(t); });
  ST.stream = null;
  await tween(0.7, t => { const e = 1 - ease(t); ST.pour = { k: sid, ang: th * e, d: [d[0] * e, d[1] * e + 8 * Math.sin(Math.PI * e), d[2] * e] }; });
  ST.pour = null; beep(700, 0.04);
}

/* ================= aksi dasar (dipakai mode panduan & manual) ================= */
const ACT = {
  async cables() {
    if (ST.cabled) return toast('Semua kabel sudah tersambung.');
    setView('belakang'); log('Memeriksa dan menyambungkan kabel…');
    for (let i = 0; i < CABLES.length; i++) { if (ST.cab[i] >= 1) continue; await tween(0.55, t => { ST.cab[i] = t; }); beep(600, 0.05); log('Tersambung: ' + CABLES[i].nm); }
    ST.cabled = true; ok('Semua kabel daya dan data tersambung.'); await wait(0.4); setView('meja');
  },
  async ups() {
    if (ST.ups) { if (ST.inst !== 'off' || ST.pc) return mistake('upsoffearly', 'UPS dimatikan terakhir, setelah UV-1700 dan komputer mati (IK C.8).'); setView('ups'); await wait(0.5); ST.ups = false; beep(500, 0.2); log('UPS dimatikan (tombol OFF).'); return; }
    if (ST.cab[0] < 1 || ST.cab[1] < 1) return mistake('upsnocable', 'UPS belum mendapat listrik: sambungkan kabel perpanjangan ke stop kontak dan UPS ke perpanjangan.');
    setView('ups'); await wait(0.6); ST.ups = true; beep(1200, 0.15); await wait(0.25); beep(1200, 0.15); ok('UPS menyala; indikator hijau aktif.'); await wait(0.5);
  },
  async silica() {
    const out = ST.silica === 'comp';
    if (ST.lid < 0.5) await ACT.lid();
    setView('kompartemen'); const from = ST.silicaPos.slice(), to = out ? [-30, -1.4, -6] : [-8.5, 6, 8];
    await tween(0.6, t => { ST.silicaPos = lerp3(from, [from[0], 22, from[2]], ease(t)); }); await tween(0.8, t => { ST.silicaPos = lerp3([from[0], 22, from[2]], [to[0], 22, to[2]], ease(t)); }); await tween(0.6, t => { ST.silicaPos = lerp3([to[0], 22, to[2]], to, ease(t)); });
    ST.silica = out ? 'meja' : 'comp'; log(out ? 'Silica gel dikeluarkan dari kompartemen.' : 'Silica gel dimasukkan kembali ke kompartemen.');
  },
  async inst() {
    if (ST.inst !== 'off') {
      if (ST.pcctrl) return mistake('offpcctrl', 'Tekan MODE pada UV-1700 untuk keluar dari PC CONTROL sebelum mematikan alat (IK C.6).');
      setView('alat'); await wait(0.4); ST.inst = 'off'; ST.lastA = 0; ST.zero = null; beep(400, 0.2); log('Tombol power di bagian kiri UV-1700 ditekan: alat mati.');
      await tween(0.8, t => { ST.lcdFold = t; }); log('Monitor (LCD) UV-1700 ditutup.'); return;
    }
    if (!ST.ups || ST.cab[2] < 1) return mistake('instnoups', 'UV-1700 belum mendapat daya: sambungkan kabel dan nyalakan UPS.');
    if (ST.silica === 'comp') mistake('initsilica', 'Silica gel harus dikeluarkan sebelum inisialisasi; kompartemen harus kosong agar pemeriksaan energi berhasil.');
    if (ST.cuv.R.where === 'sel' || ST.cuv.S.where === 'sel') mistake('initcuv', 'Kompartemen harus kosong saat inisialisasi.');
    if (ST.lid > 0.5) await ACT.lid();
    setView('alat'); if (ST.lcdFold > 0) await tween(0.6, t => { ST.lcdFold = 1 - t; });
    ST.inst = 'init'; ST.initLines = []; beep(900, 0.1); log('Tombol power di bagian kiri UV-1700 ditekan; inisialisasi berjalan.');
    const L = ['UV-1700 INITIALIZE', 'LSI INITIALIZE    OK', 'ROM CHECK         OK', 'RAM CHECK         OK', 'FILTER INIT       OK', 'SLIT INIT         OK', 'D2 LAMP ON        OK', 'WI LAMP ON        OK', 'WL ORIGIN 656.1   OK', 'ENERGY CHECK      ' + (ST.silica === 'comp' ? 'NG' : 'OK')];
    for (const s of L) { ST.initLines.push(s); dirty = true; await wait(0.4); }
    beep(1400, 0.2); ST.inst = 'warm'; ST.warmLeft = 15 * 60; ST.lam = AN.lam_ref; ST.lastA = 0; ok('Inisialisasi selesai; lampu mulai pemanasan.');
  },
  async pc() {
    if (ST.pc) return toast('Komputer sudah menyala.');
    if (!ST.ups || ST.cab[3] < 1 || ST.cab[4] < 1) return mistake('pcnoups', 'Komputer dan monitor mendapat daya dari UPS: sambungkan kabel dan nyalakan UPS dahulu.');
    setView('komputer'); ST.pc = 1; log('CPU dan monitor dinyalakan; Windows memulai.'); await wait(2.6); ST.pc = 2;
    if (ST.cab[5] >= 1) { ST.printer = true; log('Printer dinyalakan.'); } ok('Komputer siap.');
  },
  async pcctrl() {
    if (!ready()) return toast('Tunggu inisialisasi UV-1700 selesai.', 'warn');
    setView('alat'); ST.keyFlash = 1; await wait(0.4); ST.keyFlash = 0;
    if (ST.pcctrl) { if (ST.connected) return mistake('modeconn', 'Klik Disconnect di UVProbe sebelum menekan MODE.'); ST.pcctrl = false; log('Tombol MODE ditekan: UV-1700 keluar dari PC CONTROL.'); return; }
    ST.pcctrl = true; beep(900, 0.08); log('MODE → PC CONTROL: UV-1700 menunggu perintah komputer.');
  },
  async software() {
    if (ST.pc < 2) return toast('Nyalakan komputer terlebih dahulu.', 'warn'); if (ST.sw) return toast('UVProbe sudah terbuka.');
    setView('komputer'); ST.pc = 3; ST.dlg = { title: 'UVProbe', sub: 'Memuat UVProbe…', w: 360, h: 140, progress: 0 }; await tween(1.4, t => { ST.dlg.progress = t; });
    ST.dlg = null; ST.sw = true; ok('UVProbe terbuka.');
  },
  async connect() {
    if (!ST.sw) return toast('Buka UVProbe terlebih dahulu.', 'warn'); if (ST.connected) return toast('Sudah terhubung.');
    if (ST.cab[7] < 1) return mistake('nodata', 'Kabel data RS-232C belum tersambung.');
    if (!ready()) return mistake('connearly', 'Klik Connect setelah inisialisasi UV-1700 selesai.');
    if (!ST.pcctrl) return mistake('nopcctrl', 'Aktifkan PC CONTROL pada UV-1700 (tekan MODE) sebelum Connect.');
    setView('komputer'); await pressBtn('connect');
    const items = ['Communication', 'ROM version', 'Lamp status', 'Wavelength accuracy', 'Baseline flatness'];
    ST.dlg = { title: 'Instrument Initialization', sub: 'Menghubungkan ke UV-1700…', w: 440, h: 270, fields: items.map(s => [s, '…', 120]), progress: 0 };
    for (let i = 0; i < items.length; i++) { await wait(0.35); ST.dlg.fields[i][1] = 'Pass'; ST.dlg.progress = (i + 1) / items.length; }
    await wait(0.3); ST.dlg = null; ST.connected = true; beep(1000, 0.1); ok('UVProbe terhubung dengan UV-1700.');
  },
  async disconnect() { if (!ST.connected) return toast('Belum terhubung.'); setView('komputer'); await pressBtn('disconnect'); ST.connected = false; log('Disconnect: komunikasi UVProbe–UV-1700 diputus.'); },
  skipwarm() { if (ST.inst !== 'warm') return toast(ST.inst === 'ready' ? 'Pemanasan sudah selesai.' : 'Nyalakan UV-1700 terlebih dahulu.', 'warn'); ST.warmSkipped = true; ST.inst = 'ready'; ST.warmLeft = 0; mistake('skipwarm', 'Pemanasan dilewati: drift baseline meningkat.'); },
  async lid() { const to = ST.lid > 0.5 ? 0 : 1; if (viewName !== 'kompartemen') setView('kompartemen'); await tween(0.6, t => { ST.lid = to ? t : 1 - t; }); ST.lid = to; log(to ? 'Tutup kompartemen dibuka.' : 'Tutup kompartemen ditutup.'); },
  async fill(k, sid) {
    k = k || selCuv(); sid = sid || $('#sol').value; const c = ST.cuv[k];
    if (c.where !== 'rak') return toast('Keluarkan kuvet dari alat dahulu.', 'warn');
    setView('isi'); if (c.content) await tween(0.5, t => { c.level = 0.85 * (1 - t); });
    await washBottle(c, false);
    log('Kuvet ' + (k === 'R' ? 'referensi' : 'sampel') + ' dibilas akuades dari botol semprot, dibilas dengan ' + SOL[sid].short + ', lalu diisi ±¾ volume.');
    if (sid === 'blank') { await washBottle(c, true); }
    else { await pourFlask(sid, c, true); await tween(0.4, t => { c.level = 0.85 * (1 - t); }); c.content = null; c.level = 0; await pourFlask(sid, c, false); }
    c.dirty = true; c.smudge = 0; c.wet = false;
  },
  async empty() { const k = selCuv(), c = ST.cuv[k]; if (c.where !== 'rak') return toast('Keluarkan kuvet dahulu.', 'warn'); if (!c.content) return; setView('kerja'); await tween(0.5, t => { c.level = 0.85 * (1 - t); }); c.content = null; c.level = 0; log('Isi kuvet dibuang ke gelas buangan.'); },
  async wipe(k) {
    k = k || selCuv(); const c = ST.cuv[k]; if (c.where !== 'rak') return toast('Kuvet harus di luar alat untuk dilap.', 'warn');
    setView('kerja'); const p = c.pos; const start = [-31.5, 9, 23]; ST.tissue = { pos: start, rot: 0 }; const near = [p[0] + 1.6, p[1] + 5, p[2]];
    await tween(0.6, t => { ST.tissue.pos = lerp3(start, near, ease(t)); ST.tissue.rot = 90 * t; });
    await tween(1.4, t => { ST.tissue.pos = [near[0], p[1] + 4 + 2.8 * Math.sin(t * Math.PI * 6), near[2]]; });
    const cur = ST.tissue.pos.slice(); await tween(0.6, t => { ST.tissue.pos = lerp3(cur, start, ease(t)); });
    ST.tissue = null; c.dirty = false; log('Kuvet ' + (k === 'R' ? 'referensi' : 'sampel') + ' dilap dengan tisu (sisi bening bebas tetesan dan sidik jari).');
  },
  async insert(k) {
    k = k || selCuv(); const c = ST.cuv[k];
    if (c.where === 'sel') return toast('Kuvet sudah di dalam alat.');
    if (ST.lid < 0.5) return mistake('lidclosed', 'Buka tutup kompartemen sebelum memasukkan kuvet.');
    if (c.dirty) { mistake('dirty' + k, 'Kuvet dimasukkan tanpa dilap: tetesan/sidik jari menaikkan absorbansi.'); c.smudge = 0.012 + 0.02 * rnd(); }
    setView('kompartemen'); const r = c.pos.slice(), d = CELL[k];
    await moveCuv(k, [[r[0], 22, r[2]], [d[0], 24.6, d[2]], d], 0.6); c.where = 'sel'; beep(700, 0.05);
    log('Kuvet ' + (k === 'R' ? 'referensi → sel referensi (belakang)' : 'sampel → sel sampel (depan)') + '.');
  },
  async remove(k) {
    k = k || selCuv(); const c = ST.cuv[k]; if (c.where !== 'sel') return toast('Kuvet ini tidak di dalam alat.');
    if (ST.lid < 0.5) return mistake('lidclosed2', 'Buka tutup kompartemen untuk mengeluarkan kuvet.');
    setView('kompartemen'); const d = CELL[k], r = RACK[k];
    await moveCuv(k, [[d[0], 24.6, d[2]], [r[0], 22, r[2]], r], 0.6); c.where = 'rak'; c.smudge = 0; log('Kuvet dikeluarkan ke rak.');
  },
  async goto(lam) {
    const v = lam || parseFloat(String($('#lam').value).replace(',', '.')); if (!(v >= 190 && v <= 1100)) return toast('λ harus 190–1100 nm.', 'warn');
    if (!checks()) return; await pressBtn('goto'); ST.lam = Math.round(v * 10) / 10; $('#lam').value = ST.lam; ST.lastA = readAt(ST.lam); beep(800, 0.05); log('Go To WL ' + fx(ST.lam, 1) + ' nm.');
  },
  async baseline() {
    if (!checks()) return; const R = ST.cuv.R, S = ST.cuv.S;
    if (R.where !== 'sel' || S.where !== 'sel' || R.content !== 'blank' || S.content !== 'blank') mistake('blnotblank', 'Baseline memerlukan blanko di sel referensi dan sel sampel.');
    const [lo, hi] = ST.smethod ? [ST.smethod.lo, ST.smethod.hi] : scanRange(); setView('alat'); await pressBtn('baseline');
    ST.dlg = { title: 'Baseline', sub: 'Koreksi baseline ' + hi + ' → ' + lo + ' nm', w: 400, h: 150, progress: 0 };
    const offs = []; for (let l = lo; l <= hi; l++) offs.push(rawA(l, false));
    ST.xrayT = ST.autoX ? 1 : ST.xrayT; await tween(2.4, t => { ST.lam = hi - (hi - lo) * t; ST.beam = 1; ST.dlg.progress = t; }); ST.beam = 0; if (!ST.xrayUser) ST.xrayT = 0;
    ST.dlg = null; ST.zero = { type: 'bl', lo, hi, offs }; ST.lam = AN.lam_ref; ST.lastA = 0; beep(1100, 0.1); ok('Baseline ' + lo + '–' + hi + ' nm selesai.');
  },
  async autozero() {
    if (!checks()) return; const R = ST.cuv.R, S = ST.cuv.S;
    if (R.where !== 'sel') mistake('azref', 'Sel referensi kosong: masukkan kuvet blanko.');
    if (S.where === 'sel' && S.content && S.content !== 'blank') mistake('aznonblank', 'Auto Zero harus dengan blanko di sel sampel (IK B.11).');
    await pressBtn('autozero'); setView('alat'); await beamPulse(0.9);
    ST.zero = { type: 'az', lam: ST.lam, off: rawA(ST.lam, false) }; ST.lastA = 0; beep(1100, 0.1); ok('Auto Zero pada ' + fx(ST.lam, 1) + ' nm: blanko = 0,000 Abs.');
  },
  async scan() {
    if (!checks()) return; if (!ST.smethod) return toast('Atur metode spektrum dahulu (Edit → Method).', 'warn');
    const S = ST.cuv.S; if (S.where !== 'sel') return mistake('nosample', 'Sel sampel kosong.');
    const { lo, hi } = ST.smethod; const zeroed = ST.zero && ST.zero.type === 'bl' && ST.zero.lo <= lo && ST.zero.hi >= hi;
    if (!zeroed) mistake('noblscan', 'Scan tanpa Baseline: spektrum mengandung serapan blanko dan offset instrumen.');
    await pressBtn('scan'); ST.win = 'spectrum'; ST.peaks = null; ST.spectrum = { l: [], a: [], name: S.content ? SOL[S.content].label : 'kosong' }; setView('alat');
    ST.xrayT = ST.autoX ? 1 : ST.xrayT; let li = hi;
    await tween(4.0, t => { const tg = hi - (hi - lo) * t; while (li >= tg) { ST.spectrum.l.unshift(li); ST.spectrum.a.unshift(readAt(li)); li -= ST.smethod.step; } ST.lam = Math.max(lo, tg); ST.lastA = ST.spectrum.a[0] || 0; ST.beam = 1; });
    ST.beam = 0; if (!ST.xrayUser) ST.xrayT = 0; ST.spectrum.done = true; beep(1300, 0.1); ok('Scan selesai.');
  },
  async peakpick() {
    if (!ST.spectrum || !ST.spectrum.done) return toast('Lakukan scan terlebih dahulu.', 'warn');
    setView('komputer'); await menuPick('Operations', MENU.Operations, 0); ST.peaks = findPeaks(ST.spectrum.l, ST.spectrum.a);
    const pk = ST.peaks[0]; if (pk) { ST.lmax = pk.l; $('#lam').value = pk.l; } ok('Peak pick: λmaks = ' + fx(pk ? pk.l : NaN, 1) + ' nm.');
  },
  async readstd() {
    if (!checks()) return; if (!ST.pmethod) return toast('Atur metode fotometri dahulu.', 'warn');
    const S = ST.cuv.S; if (S.where !== 'sel') return mistake('nosample', 'Sel sampel kosong.');
    if (zeroOff(ST.pmethod.lam) == null) mistake('nozero', 'Belum Auto Zero dengan blanko pada λ metode (IK B.11).');
    const i = ST.stdTable.findIndex(r => r.A == null); if (i < 0) return toast('Semua standar sudah dibaca.');
    const want = 's' + i; if (S.content !== want) mistake('stdwrong' + i, 'Isi kuvet tidak sesuai baris standar ' + (i + 1) + ' (' + ST.stdTable[i].id + ').');
    ST.ptab = 'std'; await pressBtn('readstd'); ST.lam = ST.pmethod.lam; setView('alat'); await beamPulse(1.0);
    let s = 0; for (let j = 0; j < 3; j++) s += readAt(ST.lam); const A = s / 3; ST.lastA = A; ST.stdTable[i].A = A; ST.rowHi = i; beep(1300, 0.08);
    updateOut(); ok('Read Std ' + ST.stdTable[i].id + ': ' + fx(A, 4) + ' Abs.');
  },
  async readunk() {
    if (!checks()) return; if (!ST.pmethod) return toast('Atur metode fotometri dahulu.', 'warn');
    const S = ST.cuv.S; if (S.where !== 'sel') return mistake('nosample', 'Sel sampel kosong.');
    if (!ST.stdTable.some(r => r.A != null)) return mistake('nostd', 'Baca standar terlebih dahulu sebelum sampel.');
    let i = ST.unkTable.findIndex(r => r.A == null); if (i < 0) { ST.unkTable.push({ id: 'Sampel X-' + (ST.unkTable.length + 1), A: null }); i = ST.unkTable.length - 1; }
    ST.ptab = 'unk'; await pressBtn('readunk'); ST.lam = ST.pmethod.lam; setView('alat'); await beamPulse(1.0);
    let s = 0; for (let j = 0; j < 3; j++) s += readAt(ST.lam); const A = s / 3; ST.lastA = A; const reg = regress();
    ST.unkTable[i].A = A; ST.unkTable[i].conc = reg ? (A - reg.b) / reg.a : NaN; ST.rowHi = i; beep(1300, 0.08);
    updateOut(); ok('Read Unk ' + ST.unkTable[i].id + ': ' + fx(A, 4) + ' Abs → ' + fx(ST.unkTable[i].conc, 4) + ' ' + ST.pmethod.unit + '.');
  },
  async dry(k) {
    const ks = k ? [k] : ['R', 'S'];
    for (const kk of ks) { const c = ST.cuv[kk]; if (c.where === 'sel') return toast('Keluarkan kuvet dari alat terlebih dahulu.', 'warn'); }
    if (ST.inst !== 'off') mistake('dryon', 'Kuvet dikeringkan setelah praktik selesai dan alat dalam keadaan mati (IK D.1).');
    setView('kerja');
    for (const kk of ks) { const c = ST.cuv[kk];
      if (c.content) { await tween(0.4, t => { c.level = 0.85 * (1 - t); }); c.content = null; }
      for (const [nm, col] of [['etanol', '#f4efe6'], ['aseton', '#e9f2f3']]) { SOL['_' + nm] = { label: nm, c: 0, col }; c.content = '_' + nm; await tween(0.5, t => { c.level = 0.6 * t; }); await tween(0.4, t => { c.level = 0.6 * (1 - t); }); log('Kuvet ' + (kk === 'R' ? 'referensi' : 'sampel') + ' dibilas ' + nm + '.'); }
      c.content = null; c.level = 0; }
    setView('kering');
    for (const kk of ks) { const c = ST.cuv[kk]; const from = c.pos.slice(), to = DRY[kk];
      await tween(0.6, t => { c.pos = lerp3(from, [from[0], 20, from[2]], ease(t)); }); await tween(0.5, t => { c.flip = t; });
      await tween(0.7, t => { c.pos = lerp3([from[0], 20, from[2]], [to[0], 20, to[2]], ease(t)); }); await tween(0.5, t => { c.pos = lerp3([to[0], 20, to[2]], to, ease(t)); });
      c.where = 'kering'; }
    ok('Kuvet dibilas etanol lalu aseton, dibalik di atas tisu, dan dibiarkan hingga kering sempurna.');
  },
  copy() { const t = $('#out'); t.removeAttribute('readonly'); t.select(); let okc = false; try { okc = document.execCommand('copy'); } catch (e) { } t.setAttribute('readonly', '');
    const done = () => toast(okc ? 'Data disalin.' : 'Pilih teks lalu tekan Ctrl+C.', okc ? 'ok' : 'warn');
    if (navigator.clipboard) navigator.clipboard.writeText(t.value).then(() => toast('Data disalin.', 'ok'), done); else done(); },
  csv() { const rows = ['Tabel;Sample ID;Conc (' + INIT.unit + ');Lambda (nm);Abs'];
    const L = ST.pmethod ? fx(ST.pmethod.lam, 1) : '';
    ST.stdTable.forEach(r => rows.push(['Standard', r.id, fx(r.conc, 4), L, r.A != null ? fx(r.A, 4) : ''].join(';')));
    ST.unkTable.forEach(r => rows.push(['Unknown', r.id, fx(r.conc, 4), L, r.A != null ? fx(r.A, 4) : ''].join(';')));
    if (ST.spectrum && ST.spectrum.done) { rows.push(''); rows.push('Spektrum ' + ST.spectrum.name); rows.push('Lambda (nm);Abs'); ST.spectrum.l.forEach((l, i) => rows.push(fx(l, 1) + ';' + fx(ST.spectrum.a[i], 4))); }
    download(new Blob([rows.join('\n')], { type: 'text/csv' }), fname() + '.csv'); },
  tabstd() { ST.ptab = 'std'; }, tabunk() { ST.ptab = 'unk'; }
};
function download(blob, name) { try { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click(); a.remove(); } catch (e) { toast('Unduhan diblokir browser; gunakan Salin data.', 'warn'); } }
function updateOut() {
  const L = ST.pmethod ? ST.pmethod.lam : ST.lam; let s = '#LAMBDA ' + fx(L, 1) + '\n#STANDAR\n';
  ST.stdTable.filter(r => r.A != null).forEach(r => { s += fx(r.conc, 4) + ';' + fx(r.A, 4) + '\n'; });
  s += '#SAMPEL\n' + ST.unkTable.filter(r => r.A != null).map(r => fx(r.A, 4)).join('; ') + '\n'; $('#out').value = s;
}

/* ================= aksi UVProbe berbasis menu ================= */
async function printDoc(kind) {
  if (!ST.printer) return toast('Printer belum menyala atau kabelnya belum tersambung.', 'warn');
  setView('komputer'); ST.dlg = { title: 'Print', sub: 'Mencetak ke printer…', w: 360, h: 140, progress: 0 }; await tween(0.8, t => { ST.dlg.progress = t; }); ST.dlg = null;
  setView('printer'); await tween(2.0, t => { ST.paper = t; }); beep(700, 0.05);
  const cvp = document.createElement('canvas'); cvp.width = 595; cvp.height = 842; const c = cvp.getContext ? cvp.getContext('2d') : null;
  if (c) {
    c.fillStyle = '#fff'; c.fillRect(0, 0, 595, 842); c.fillStyle = '#111'; c.font = '700 16px Arial'; c.textAlign = 'left';
    const title = kind === 'peak' ? 'Peak Pick — Spektrum' : kind === 'std' ? 'Standard Table & Standard Curve' : 'Sample Table';
    c.fillText('UVProbe · ' + title, 40, 50); c.font = '12px Arial'; c.fillText('File: ' + fname() + (kind === 'peak' ? '.spc' : '.pho') + '    Tanggal: ' + dstr + '    Operator: ' + (ID.nama || '-'), 40, 72);
    c.fillText('Instrumen: Shimadzu UV-1700 PharmaSpec    Analit: ' + AN.nama, 40, 90); c.strokeStyle = '#111'; c.beginPath(); c.moveTo(40, 100); c.lineTo(555, 100); c.stroke();
    let y = 126; c.font = '700 12px Arial';
    if (kind === 'peak') { c.fillText('No.   P/V   Wavelength (nm)   Abs.', 40, y); c.font = '12px Arial'; (ST.peaks || []).slice(0, 12).forEach((p, i) => { y += 20; c.fillText((i + 1) + '      P      ' + fx(p.l, 2) + '                 ' + fx(p.a, 4), 40, y); });
      if (ST.spectrum) { const gx = 40, gy = y + 30, gw = 515, gh = 260; c.strokeRect(gx, gy, gw, gh); const sp = ST.spectrum, lo = sp.l[0], hi = sp.l[sp.l.length - 1], ymax = Math.max(...sp.a) * 1.1;
        c.strokeStyle = '#c00'; c.beginPath(); sp.l.forEach((l, i) => { const X1 = gx + gw * (l - lo) / (hi - lo), Y1 = gy + gh - gh * Math.max(0, sp.a[i]) / ymax; i ? c.lineTo(X1, Y1) : c.moveTo(X1, Y1); }); c.stroke(); } }
    else { const rows = kind === 'std' ? ST.stdTable : ST.unkTable; c.fillText('Sample ID              Type           Conc (' + INIT.unit + ')        Abs', 40, y); c.font = '12px Arial';
      rows.forEach(r => { y += 20; c.fillText((r.id || '').padEnd(22, ' ') + (kind === 'std' ? 'Standard' : 'Unknown').padEnd(16, ' ') + fx(r.conc, 4).padEnd(18, ' ') + (r.A != null ? fx(r.A, 4) : ''), 40, y); });
      const reg = regress(); if (reg) { y += 30; c.font = '700 12px Arial'; c.fillText('Abs = ' + fx(reg.a, 5) + ' * Conc' + (reg.zero ? '' : ' + ' + fx(reg.b, 5)) + '     r² = ' + fx(reg.r2, 5), 40, y); } }
    c.font = '10px Arial'; c.fillStyle = '#555'; c.fillText('Dicetak dari praktikum virtual SHIMADZU UV-1700 (simulasi pendidikan)', 40, 820);
    let url = ''; try { url = cvp.toDataURL('image/png'); } catch (e) { }
    ST.prints.push({ title, url }); renderPrints();
  } else ST.prints.push({ title: kind === 'peak' ? 'Peak Pick' : kind === 'std' ? 'Standard Table' : 'Sample Table', url: '' });
  await wait(0.4); ST.paper = 0; ok('Tercetak: ' + (kind === 'peak' ? 'data peak pick' : kind === 'std' ? 'kurva kalibrasi standar' : 'data sampel') + '.');
}
function renderPrints() { const el = $('#prints'); el.innerHTML = ''; ST.prints.forEach((p, i) => { const a = document.createElement('a'); a.href = p.url || '#'; a.download = fname() + '_cetak' + (i + 1) + '.png'; a.innerHTML = (p.url ? '<img src="' + p.url + '">' : '') + p.title; el.appendChild(a); }); }
async function saveAs(ext) {
  setView('komputer'); await menuPick('File', MENU.File, ext === 'spc' ? 3 : 2);
  ST.dlg = { title: ext === 'spc' ? 'Save As' : 'Save', sub: 'Simpan file', w: 520, h: 210, fields: [['File name:', ''], ['Save as type:', ext === 'spc' ? 'Spectrum Files (*.spc)' : 'Photometric Files (*.pho)']], buttons: ['Save', 'Cancel'] };
  const o = {}; await typeInto(o, 'f', fname() + '.' + ext, 'File name:'); await dlgBtn('Save'); ST.dlg = null; ST.files.push(fname() + '.' + ext); beep(900, 0.05); ok('File tersimpan: ' + fname() + '.' + ext);
}

/* ================= langkah Instruksi Kerja ================= */
const STD_MID = 's' + Math.floor(AN.std.length / 2);
async function ensureLid(open) { if ((ST.lid > 0.5) !== !!open) await ACT.lid(); }
async function putSample(sid) {
  const S = ST.cuv.S; await ensureLid(1);
  if (S.where === 'sel') await ACT.remove('S');
  if (S.where === 'kering') { S.where = 'rak'; S.flip = 0; S.pos = RACK.S.slice(); }
  await ACT.fill('S', sid); await ACT.wipe('S'); await ACT.insert('S'); await ensureLid(0);
}
async function setupRef() {
  const R = ST.cuv.R; if (R.where === 'sel' && R.content === 'blank') return; await ensureLid(1);
  if (R.where === 'sel') await ACT.remove('R'); await ACT.fill('R', 'blank'); await ACT.wipe('R'); await ACT.insert('R');
}
function smethodDlg() { const m = ST.smethod_tmp; return { title: 'Spectrum Method', sub: 'Measurement', w: 560, h: 330, fields: [['Wavelength Range (nm)  Start:', m.hi, 120], ['End:', m.lo, 120], ['Scan Speed:', m.speed, 160], ['Sampling Interval:', fx(m.step, 1), 120], ['Scan Mode:', 'Single', 160], ['Measuring Mode:', 'Absorbance', 160]], buttons: ['OK', 'Cancel'] }; }
function pDlg(page) {
  const m = ST.pm_tmp; const pages = [
    { sub: 'Wavelength', fields: [['Wavelength Type:', 'Point', 160], ['Wavelength (nm):', m.wlText || '', 120], ['Daftar WL:', m.lam ? fx(m.lam, 1) + ' nm' : '-', 160]], buttons: ['Add', '< Back', 'Next >'] },
    { sub: 'Calibration', fields: [['Type:', m.type || '', 180], ['Unit:', m.unit || '', 120], ['WL1:', m.wl1 || '', 160], ['Formula Type:', 'Fixed Wave Length', 180], ['Zero Interception', !!m.zero, 0, 'check']], buttons: ['< Back', 'Next >'] },
    { sub: 'Measurement Parameters (Standard)', fields: [['Repetitions:', '1', 80], ['Averaging:', 'Average', 120]], buttons: ['< Back', 'Next >'] },
    { sub: 'Measurement Parameters (Sample)', fields: [['Repetitions:', '1', 80], ['Averaging:', 'Average', 120]], buttons: ['< Back', 'Next >'] },
    { sub: 'File Properties', fields: [['File name:', m.file || '', 300], ['Folder:', 'D:\\Data Praktikum\\', 300]], buttons: ['...', '< Back', 'Finish'] }];
  const p = pages[page]; return Object.assign({ title: 'Photometric Method Wizard', w: 600, h: 330 }, p);
}
async function wiz(page) { ST.dlg = pDlg(page); ST.wizPage = page; await wait(0.3); }
const STEPS = [
  // A — rekonstruksi
  { sec: 'A. Menyalakan Alat', rk: 1, t: 'Periksa dan sambungkan kabel: stop kontak → kabel perpanjangan → UPS; UV-1700, CPU, dan monitor ke UPS; printer ke perpanjangan; kabel VGA, USB, dan RS-232C.', done: () => ST.cabled, run: () => ACT.cables() },
  { sec: 'A. Menyalakan Alat', rk: 1, t: 'Tekan tombol ON pada UPS dan tunggu indikator hijau menyala.', done: () => ST.ups, run: () => ACT.ups() },
  { sec: 'A. Menyalakan Alat', rk: 1, t: 'Buka tutup kompartemen, keluarkan silica gel, lalu tutup kembali.', done: () => ST.silica === 'meja', run: async () => { await ACT.silica(); await ensureLid(0); } },
  { sec: 'A. Menyalakan Alat', rk: 1, t: 'Tekan tombol power di bagian kiri UV-1700; tunggu inisialisasi sampai semua pemeriksaan OK.', done: () => ST.inst !== 'off' && ST.inst !== 'init', run: () => ACT.inst() },
  { sec: 'A. Menyalakan Alat', rk: 1, t: 'Nyalakan CPU, monitor, dan printer.', done: () => ST.pc >= 2, run: () => ACT.pc() },
  { sec: 'A. Menyalakan Alat', rk: 1, t: 'Pada UV-1700 tekan MODE lalu pilih PC CONTROL.', done: () => ST.pcctrl, run: () => ACT.pcctrl() },
  { sec: 'A. Menyalakan Alat', rk: 1, t: 'Klik dua kali ikon UVProbe, lalu klik Connect; tunggu pemeriksaan instrumen selesai (Pass).', done: () => ST.connected, run: async () => { await ACT.software(); await ACT.connect(); } },
  { sec: 'A. Menyalakan Alat', rk: 1, t: 'Biarkan lampu pemanasan ±15–30 menit agar intensitas stabil (waktu dipercepat).', done: () => ST.inst === 'ready', run: async () => { setView('alat'); toast('Menunggu pemanasan lampu…'); while (ST.inst === 'warm') await wait(0.25); } },
  // B — spektrum (1–5 rekonstruksi; 6 sesuai IK)
  { sec: 'B. Pengukuran Spektrum', rk: 1, t: 'Klik Window → Spectrum.', done: () => ST.done.has('b1'), run: async () => { setView('komputer'); await menuPick('Window', MENU.Window, 0); ST.win = 'spectrum'; ST.done.add('b1'); } },
  { sec: 'B. Pengukuran Spektrum', rk: 1, t: 'Klik Edit → Method; atur rentang panjang gelombang, kecepatan scan, dan interval sampling → OK.', done: () => !!ST.smethod, run: async () => {
      setView('komputer'); await menuPick('Edit', MENU.Edit, 3); const [lo, hi] = scanRange(); ST.smethod_tmp = { lo: '', hi: '', speed: 'Medium', step: 1 }; ST.dlg = smethodDlg(); const o = {};
      await typeInto(o, 'v', String(hi), 'Wavelength Range (nm)  Start:'); await typeInto(o, 'v', String(lo), 'End:'); await dlgBtn('OK'); ST.dlg = null; ST.smethod = { lo, hi, speed: 'Medium', step: 1 }; ok('Metode spektrum: ' + hi + '–' + lo + ' nm, Medium, 1,0 nm.'); } },
  { sec: 'B. Pengukuran Spektrum', rk: 1, t: 'Isi kuvet referensi dan kuvet sampel dengan blanko, lap dengan tisu, masukkan ke sel, tutup kompartemen → klik Baseline.', done: () => ST.zero && ST.zero.type === 'bl' || ST.done.has('b3'),
    run: async () => { await setupRef(); await putSample('blank'); await ACT.baseline(); ST.done.add('b3'); } },
  { sec: 'B. Pengukuran Spektrum', rk: 1, t: 'Ganti kuvet sampel dengan larutan standar (konsentrasi tengah) → klik Start.', done: () => !!(ST.spectrum && ST.spectrum.done), run: async () => { await putSample(STD_MID); await ACT.scan(); } },
  { sec: 'B. Pengukuran Spektrum', rk: 1, t: 'Klik Operations → Peak Pick untuk menampilkan tabel puncak; catat λmaks.', done: () => !!ST.peaks, run: () => ACT.peakpick() },
  { sec: 'B. Pengukuran Spektrum', no: 6, t: 'Klik File → Save As → beri nama spektrum dengan format Tanggal_Kelas_Kelompok → Save.', done: () => ST.files.some(f => f.endsWith('.spc')), run: () => saveAs('spc') },
  { sec: 'B. Print Data', no: 1, t: 'Klik 1× pada tabel peak pick, klik kanan → Print.', done: () => ST.prints.some(p => p.title.startsWith('Peak')), run: async () => { setView('komputer'); ST.peakSel = true; await wait(0.5); await printDoc('peak'); ST.peakSel = false; } },
  // B — analisa sampel (sesuai IK)
  { sec: 'B. Analisa Sampel', no: 1, t: 'Klik Window → Photometric.', done: () => ST.done.has('p1'), run: async () => { setView('komputer'); await menuPick('Window', MENU.Window, 1); ST.win = 'photometric'; ST.done.add('p1'); } },
  { sec: 'B. Analisa Sampel', no: 2, t: 'Klik Edit → Method.', done: () => ST.done.has('p2'), run: async () => { setView('komputer'); await menuPick('Edit', MENU.Edit, 3); ST.pm_tmp = {}; await wiz(0); ST.done.add('p2'); } },
  { sec: 'B. Analisa Sampel', no: 3, t: 'Pada kolom Wavelength (nm) ketik panjang gelombang yang diinginkan (λmaks) → Add.', done: () => ST.done.has('p3'), run: async () => {
      const L = ST.lmax || AN.lam_ref; if (!ST.dlg) await wiz(0); await typeInto(ST.pm_tmp, 'wlText', fx(L, 1), 'Wavelength (nm):'); await dlgBtn('Add'); ST.pm_tmp.lam = L; ST.dlg = pDlg(0); ST.done.add('p3'); } },
  { sec: 'B. Analisa Sampel', no: 4, t: 'Klik Next.', done: () => ST.done.has('p4'), run: async () => { await dlgBtn('Next >'); await wiz(1); ST.done.add('p4'); } },
  { sec: 'B. Analisa Sampel', no: 5, t: 'Pada kolom Type pilih metode: Multi Point (standar lebih dari satu konsentrasi).', done: () => ST.done.has('p5'), run: async () => { if (ST.wizPage !== 1) await wiz(1); await typeInto(ST.pm_tmp, 'type', 'Multi Point', 'Type:'); ST.dlg = pDlg(1); ST.done.add('p5'); } },
  { sec: 'B. Analisa Sampel', no: 6, t: 'Pada kolom Unit ketik satuan yang diinginkan.', done: () => ST.done.has('p6'), run: async () => { if (ST.wizPage !== 1) await wiz(1); await typeInto(ST.pm_tmp, 'unit', INIT.unit, 'Unit:'); ST.dlg = pDlg(1); ST.done.add('p6'); } },
  { sec: 'B. Analisa Sampel', no: 7, t: 'Pada kolom WL1 pilih panjang gelombang yang diinginkan.', done: () => ST.done.has('p7'), run: async () => { if (ST.wizPage !== 1) await wiz(1); await typeInto(ST.pm_tmp, 'wl1', 'WL' + fx(ST.pm_tmp.lam || ST.lmax || AN.lam_ref, 1), 'WL1:'); ST.dlg = pDlg(1); ST.done.add('p7'); } },
  { sec: 'B. Analisa Sampel', no: 8, t: 'Klik Zero Interception → klik Next 3×.', done: () => ST.done.has('p8'), run: async () => { if (ST.wizPage !== 1) await wiz(1); ST.pm_tmp.zero = true; ST.dlg = pDlg(1); await wait(0.5);
      for (let i = 0; i < 3; i++) { await dlgBtn('Next >'); await wiz(2 + i); } ST.done.add('p8'); } },
  { sec: 'B. Analisa Sampel', no: 9, t: 'Pada kolom File name klik (...) → pilih folder → beri nama Tanggal_Kelas_Kelompok → Open → Finish.', done: () => !!ST.pmethod, run: async () => {
      if (ST.wizPage !== 4) await wiz(4); await dlgBtn('...'); await typeInto(ST.pm_tmp, 'file', fname() + '.pho', 'File name:'); ST.dlg = pDlg(4); await dlgBtn('Finish'); ST.dlg = null;
      const t = ST.pm_tmp; ST.pmethod = { lam: t.lam || ST.lmax || AN.lam_ref, type: t.type || 'Multi Point', unit: t.unit || INIT.unit, zero: t.zero !== false, file: t.file || fname() + '.pho' };
      ST.lam = ST.pmethod.lam; $('#lam').value = ST.lam; ok('Metode fotometri tersimpan: ' + fx(ST.pmethod.lam, 1) + ' nm, Multi Point, zero interception.'); } },
  { sec: 'B. Analisa Sampel', no: 10, t: 'Pada Standard Table ketik nama standar pada kolom Sample ID dan konsentrasinya pada kolom Conc.', done: () => ST.stdTable.length === AN.std.length, run: async () => {
      setView('komputer'); ST.ptab = 'std'; ST.stdTable = [];
      for (let i = 0; i < AN.std.length; i++) { const r = { id: '', conc: null, A: null }; ST.stdTable.push(r); ST.rowHi = i; await typeInto(r, 'id', 'Std ' + (i + 1)); r.conc = AN.std[i]; await wait(0.15); } ST.rowHi = null; ok('Standard Table terisi ' + AN.std.length + ' standar.'); } },
  { sec: 'B. Analisa Sampel', no: 11, t: 'Ganti isi kuvet di sample compartment dengan blanko → klik Auto Zero.', done: () => ST.zero && ST.zero.type === 'az' && ST.pmethod && Math.abs(ST.zero.lam - ST.pmethod.lam) < 0.05,
    run: async () => { await setupRef(); await putSample('blank'); ST.lam = ST.pmethod.lam; await ACT.autozero(); } },
  { sec: 'B. Analisa Sampel', no: 12, t: 'Ganti dengan standar → klik Read Std (ulangi untuk semua deret standar).', multi: true,
    done: () => ST.stdTable.length > 0 && ST.stdTable.every(r => r.A != null), prog: () => ST.stdTable.filter(r => r.A != null).length + '/' + AN.std.length,
    run: async () => { const i = ST.stdTable.findIndex(r => r.A == null); await putSample('s' + i); await ACT.readstd(); } },
  { sec: 'B. Analisa Sampel', no: 13, t: 'Klik Sample Table → beri nama sampel pada kolom Sample ID.', done: () => ST.unkTable.length >= 3, run: async () => {
      setView('komputer'); ST.ptab = 'unk'; for (let i = ST.unkTable.length; i < 3; i++) { const r = { id: '', A: null }; ST.unkTable.push(r); ST.rowHi = i; await typeInto(r, 'id', 'Sampel X-' + (i + 1)); } ST.rowHi = null; } },
  { sec: 'B. Analisa Sampel', no: 14, t: 'Ganti dengan sampel → klik Read Unk (ulangi untuk semua sampel; di sini triplo).', multi: true,
    done: () => ST.unkTable.length >= 3 && ST.unkTable.every(r => r.A != null), prog: () => ST.unkTable.filter(r => r.A != null).length + '/' + Math.max(3, ST.unkTable.length),
    run: async () => { if (ST.cuv.S.content !== 'X' || ST.cuv.S.where !== 'sel') await putSample('X'); await ACT.readunk(); } },
  { sec: 'B. Analisa Sampel', no: 15, t: 'Setelah semua pengukuran selesai klik File → Save.', done: () => ST.files.some(f => f.endsWith('.pho')), run: () => saveAs('pho') },
  { sec: 'B. Print Kurva Kalibrasi Standar', no: 1, t: 'Klik kanan pada Standard Table → Print.', done: () => ST.prints.some(p => p.title.startsWith('Standard')), run: async () => { ST.ptab = 'std'; await printDoc('std'); } },
  { sec: 'B. Print Data Sampel', no: 1, t: 'Klik kanan pada Sample Table → Print.', done: () => ST.prints.some(p => p.title.startsWith('Sample')), run: async () => { ST.ptab = 'unk'; await printDoc('unk'); } },
  // C — mematikan alat (sesuai IK)
  { sec: 'C. Mematikan Alat', no: 1, t: 'Catat semua kejadian pada Log Book Alat dan tanda tangani dengan nama jelas.', done: () => !!ST.logbook, run: () => openLog() },
  { sec: 'C. Mematikan Alat', no: 2, t: 'Klik Disconnect.', done: () => ST.done.has('c2'), run: async () => { if (ST.connected) await ACT.disconnect(); ST.done.add('c2'); } },
  { sec: 'C. Mematikan Alat', no: 3, t: 'Klik File → Exit.', done: () => ST.done.has('c3'), run: async () => { if (ST.connected) await ACT.disconnect(); setView('komputer'); await menuPick('File', MENU.File, 5); ST.sw = false; ST.win = null; ST.pc = 2; ST.done.add('c3'); log('UVProbe ditutup (File → Exit).'); } },
  { sec: 'C. Mematikan Alat', no: 4, t: 'Klik menu Windows → tekan tombol Shut down.', done: () => ST.done.has('c4'), run: async () => { setView('komputer'); ST.sw = false; ST.pc = 4; await wait(2.2); ST.pc = 0; ST.done.add('c4'); log('Komputer di-shutdown.'); } },
  { sec: 'C. Mematikan Alat', no: 5, t: 'Tekan tombol power monitor komputer dan printer.', done: () => ST.done.has('c5'), run: async () => { ST.printer = false; await wait(0.3); ST.done.add('c5'); log('Monitor dan printer dimatikan.'); } },
  { sec: 'C. Mematikan Alat', no: 6, t: 'Tekan MODE pada UV-1700 (keluar dari PC CONTROL).', done: () => ST.done.has('c6'), run: async () => { if (ST.connected) await ACT.disconnect(); if (ST.pcctrl) await ACT.pcctrl(); ST.done.add('c6'); } },
  { sec: 'C. Mematikan Alat', no: 7, t: 'Tekan tombol power di bagian kiri UV-1700, tutup monitor UV-1700, dan masukkan kembali silica gel ke dalam compartment.', done: () => ST.inst === 'off' && ST.done.has('c6') && ST.silica === 'comp' && ST.lcdFold > 0.9,
    run: async () => { await ensureLid(1); if (ST.cuv.S.where === 'sel') await ACT.remove('S'); if (ST.cuv.R.where === 'sel') await ACT.remove('R'); await ensureLid(0); if (ST.inst !== 'off') await ACT.inst(); if (ST.silica !== 'comp') await ACT.silica(); await ensureLid(0); } },
  { sec: 'C. Mematikan Alat', no: 8, t: 'Tekan tombol OFF pada UPS.', done: () => ST.done.has('c6') && !ST.ups, run: () => ACT.ups() },
  { sec: 'C. Mematikan Alat', no: 9, t: 'Cabut kabel dari kabel perpanjangan, kecuali kabel UPS.', done: () => ST.done.has('c9'), run: async () => { setView('belakang'); await tween(0.8, t => { ST.cab[5] = 1 - t; }); ST.cabled = false; ST.done.add('c9'); log('Kabel printer dicabut dari kabel perpanjangan; kabel UPS tetap terpasang.'); } },
  { sec: 'C. Mematikan Alat', no: 10, t: 'Rapikan area kerja di sekitar instrumen dan komputer dari sampel dan standar.', done: () => ST.tidy, run: async () => { setView('isi'); await wait(0.6); ST.tidy = true; log('Labu sampel dan standar dikembalikan; area kerja rapi.'); await wait(0.4); } },
  { sec: 'C. Mematikan Alat', no: 11, t: 'Rapikan kursi pada area kerja ke tempat aman.', done: () => ST.done.has('c11'), run: async () => { await wait(0.5); ST.done.add('c11'); log('Kursi dirapikan ke tempat aman.'); } },
  // D — mengeringkan kuvet
  { sec: 'D. Mengeringkan Kuvet', no: 1, t: 'Setelah alat mati, keringkan kuvet dengan membilas etanol lalu aseton; balik kuvet (posisi aman) dan tunggu hingga kering sempurna.', done: () => ST.cuv.R.where === 'kering' && ST.cuv.S.where === 'kering', run: () => ACT.dry() }
];
function isDone(s) { if (!s._d && s.done()) s._d = true; return !!s._d; }
function curStep() { return STEPS.findIndex(s => !isDone(s)); }
function openLog() { return new Promise(res => { const m = $('#logm'); $('#lg_t').value = dstr; $('#lg_n').value = ID.nama; $('#lg_k').value = 'Alat berfungsi normal. Pengukuran ' + AN.nama + ' pada ' + fx(ST.pmethod ? ST.pmethod.lam : AN.lam_ref, 1) + ' nm; ' + ST.stdTable.filter(r => r.A != null).length + ' standar, ' + ST.unkTable.filter(r => r.A != null).length + ' sampel. Tidak ada kendala.';
  $('#lg_s').value = ID.nama; m.classList.add('show'); setView('log'); logResolve = res; if (PL.on || FAST) setTimeout(() => { const b = $('#lg_ok'); if (b.click) b.click(); }, FAST ? 0 : 2200 / SPD); }); }
let logResolve = null;

/* ================= panel ================= */
function pipSync() { $('#pip').classList.toggle('hide', !ST.pc || ST.pipClosed !== false); }
function sync() { pipSync();
  const R = ST.cuv.R, S = ST.cuv.S, ch = [];
  const chip = (t, s) => ch.push('<span class="chip ' + (s === 2 ? 'on' : s === 1 ? 'mid' : '') + '">' + t + '</span>');
  chip('Kabel', ST.cabled ? 2 : (ST.cab.some(x => x > 0) ? 1 : 0)); chip('UPS', ST.ups ? 2 : 0); chip('Silica ' + (ST.silica === 'comp' ? 'di alat' : 'keluar'), ST.silica === 'comp' ? 1 : 2);
  chip('UV-1700 ' + (ST.inst === 'warm' ? 'pemanasan ' + mmss(ST.warmLeft) : ST.inst.toUpperCase()), ST.inst === 'ready' ? 2 : ST.inst === 'off' ? 0 : 1);
  chip('PC ' + (ST.pc >= 2 ? 'ON' : ST.pc ? '…' : 'OFF'), ST.pc >= 2 ? 2 : ST.pc ? 1 : 0); chip(ST.pcctrl ? 'PC CTRL' : 'Mode lokal', ST.pcctrl ? 2 : 0); chip(ST.connected ? 'Terhubung' : 'Tidak terhubung', ST.connected ? 2 : 0);
  chip('Tutup ' + (ST.lid > 0.5 ? 'terbuka' : 'tertutup'), ST.lid > 0.5 ? 1 : 2);
  chip('Ref: ' + (R.content ? SOL[R.content].short || R.content : 'kosong') + ' · ' + R.where + (R.dirty ? ' · belum dilap' : ''), R.where === 'sel' ? 2 : 1);
  chip('Sampel: ' + (S.content ? SOL[S.content].short || S.content : 'kosong') + ' · ' + S.where + (S.dirty ? ' · belum dilap' : ''), S.where === 'sel' ? 2 : 1);
  $('#chips').innerHTML = ch.join('');
  const ci = curStep(); const st = STEPS[ci];
  if (st) { $('#nowsec').textContent = secT(st.sec).toUpperCase() + ' · ' + U('step') + ' ' + (ci + 1) + '/' + STEPS.length + (st.prog ? ' · ' + st.prog() : '');
    $('#nowtx').textContent = stepT(ci); $('#nowwhy').textContent = ST.busy ? U('busy') : U('idle');
    $('#gobtn').textContent = st.multi ? 'Kerjakan (' + st.prog() + ')' : 'Kerjakan langkah ini'; $('#prog2').style.width = (100 * ci / STEPS.length) + '%'; }
  else { $('#nowsec').textContent = U('done'); $('#nowtx').textContent = U('donetx'); $('#nowwhy').textContent = U('donewhy'); $('#gobtn').textContent = 'Selesai'; }
  $('#gobtn').disabled = ST.busy || !st;
  let html = '', last = '';
  STEPS.forEach((s, i) => { if (s.sec !== last) { html += '<div class="grp">' + secT(s.sec) + '</div>'; last = s.sec; }
    const d = isDone(s); html += '<div class="it ' + (d ? 'done' : (i === ci ? 'cur' : '')) + '" data-i="' + i + '"><span class="d">' + (d ? '✓' : (s.no || '•')) + '</span><span>' + stepT(i) + (s.rk ? '<span class="rk">REK</span>' : '') + (s.multi && !d ? ' <b>(' + s.prog() + ')</b>' : '') + '</span></div>'; });
  $('#iklist').innerHTML = html; const nd = STEPS.filter(isDone).length; $('#ikn').textContent = nd + '/' + STEPS.length; $('#prog').style.width = (100 * nd / STEPS.length) + '%';
  $$('button.a').forEach(b => { b.disabled = ST.busy && !['copy', 'csv'].includes(b.dataset.act); });
  $('#score').textContent = ST.score; $('#ident').textContent = (ID.nama || 'Praktikan') + ' · ' + ID.kelas + ' · Kel. ' + ID.kel;
  const [lo, hi] = scanRange(); $('#scaninfo').textContent = 'Rentang scan: ' + lo + '–' + hi + ' nm · λ referensi metode ' + AN.lam_ref + ' nm.';
  if (ci >= 0) { const el = $('#iklist .cur'); if (el && el.scrollIntoView && ST.autoScroll) el.scrollIntoView({ block: 'nearest' }); }
  updateCtl(); dirty = true;
}
async function run(fn, label) {
  if (ST.busy) return; ST.busy = true; ST.busyMsg = label || null; sync();
  try { await fn(); } catch (e) { console.error(e); toast('Galat: ' + e.message, 'err'); }
  ST.busy = false; ST.busyMsg = null; sync(); drawPC();
}

/* ================= loop ================= */
let last = performance.now(), lastPC = 0;
function loop(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  for (let i = tweens.length - 1; i >= 0; i--) { const tw = tweens[i]; tw.el += PAUSED ? 0 : dt * 1000 * SPD; const t = tw.dur <= 0 ? 1 : Math.min(1, tw.el / tw.dur); tw.fn(t); dirty = true; if (t >= 1) { tweens.splice(i, 1); tw.res(); } }
  const e = (cur, tgt, sp) => { const d = tgt - cur; return Math.abs(d) < 0.002 ? tgt : cur + d * Math.min(1, dt * sp); };
  if (ST.ikp !== ST.ikT) { const d = ST.ikT - ST.ikp; ST.ikp = Math.abs(d) < 0.01 ? ST.ikT : ST.ikp + Math.sign(d) * Math.min(Math.abs(d), dt * 2.4); dirty = true; $('#ikpg').textContent = 'Hal. ' + (2 * Math.round(ST.ikT) - 1) + '–' + Math.min(2 * Math.round(ST.ikT), IKPAGES.length) + ' / ' + IKPAGES.length; }
  const nx = e(ST.xray, ST.xrayT, 5); if (nx !== ST.xray) { ST.xray = nx; dirty = true; }
  if (camAnim) { const a = camAnim; a.t = Math.min(1, a.t + (PAUSED ? 0 : dt * 1.6 * Math.max(1, SPD))); const k = ease(a.t); let dy = a.to[0] - a.from[0]; while (dy > 180) dy -= 360; while (dy < -180) dy += 360;
    CAM.yaw = a.from[0] + dy * k; CAM.pitch = a.from[1] + (a.to[1] - a.from[1]) * k; CAM.dist = a.from[2] + (a.to[2] - a.from[2]) * k; CAM.target = lerp3(a.from[3], a.to[3], k); if (a.t >= 1) camAnim = null; dirty = true; }
  if (ST.inst === 'warm' && !ST.warmSkipped) { ST.warmLeft -= PAUSED ? 0 : dt * 90 * SPD; if (ST.warmLeft <= 0) { ST.warmLeft = 0; ST.inst = 'ready'; beep(1400, 0.2); ok('Pemanasan selesai: lampu stabil.'); sync(); } dirty = true; }
  if (now - lastPC > 200) { lastPC = now; ST.pcSeen = ST.pc; pipSync(); $('#clock').textContent = mmss((now - ST.t0) / 1000); if (ST.inst === 'warm') sync(); drawPC(); dirty = true; }
  if (dirty) { dirty = false; draw(); }
  requestAnimationFrame(loop);
}



/* ================= geometri Buku IK (dibuat setelah isi diketahui) ================= */
let IKL = 4, IKMAX = 4; const IKX = -118, IKW = 12, IKZ0 = 18, IKZ1 = 35, IKTH = 0.14, IKB = -0.9;
function ikGeom() {
  ikBuild(); const Pn = IKPAGES.length;
  if (Pn % 2) { IKL = (Pn + 1) / 2; IKMAX = IKL; } else { IKL = Pn / 2 + 1; IKMAX = Pn / 2; }
  box('ikbook', 'room', IKX, IKX + IKW, -1.4, IKB, IKZ0, IKZ1, '#6b1313', { cell: 99 });
  for (let i = 0; i < IKL; i++) {
    const y0 = IKB + (IKL - 1 - i) * IKTH, y1 = y0 + IKTH, cov = i === 0, col = cov ? '#7f1d1d' : '#fbfaf4';
    box('ikbook', 'ikleaf', IKX, IKX + IKW, y0, y1, IKZ0, IKZ1, col, { cell: 99, leaf: i, faceCol: { f: '#efece2', r: '#efece2', b: '#efece2' } });
    box('ikbook', 'ikleaf', IKX, IKX + IKW, y1, y1 + 0.01, IKZ0, IKZ1, col, { cell: 99, leaf: i, lside: 'f', bias: -2, skip: ['f', 'b', 'l', 'r', 'u'], text: { face: 't', w: 480, h: 680, ik: 2 * i } });
    box('ikbook', 'ikleaf', IKX, IKX + IKW, y0 - 0.01, y0, IKZ0, IKZ1, col, { cell: 99, leaf: i, lside: 'b', bias: -2, skip: ['f', 'b', 'l', 'r', 't'], text: { face: 'u', w: 480, h: 680, ik: 2 * i + 1, rot: true } });
  }
}

/* ================= isi Buku IK ================= */
function ikShow(part) { const i = part.leaf, p = ST.ikp; return part.lside === 'f' ? (i >= Math.floor(p + 1e-6) && i <= Math.floor(p + 1e-6) + 1) : (i <= Math.ceil(p - 1e-6) - 1 && i >= Math.ceil(p - 1e-6) - 2); }
const IKC = {}; let IKPAGES = null; const IKK = 1.5, IKNAME = 'Lilis Sulistiawaty, M.Si';
function ikBuild() {
  const m = document.createElement('canvas').getContext('2d'); m.font = '15px Arial'; const W0 = 410, IND = 30;
  const wrap = (t, w) => { const out = []; let cur = ''; for (const wd of t.split(' ')) { const tt = cur ? cur + ' ' + wd : wd; if (m.measureText(tt).width > w && cur) { out.push(cur); cur = wd; } else cur = tt; } if (cur) out.push(cur); return out; };
  const L = []; let last = '';
  STEPS.forEach((st, i) => { if (st.sec !== last) { last = st.sec; L.push({ k: 'sec', t: st.sec.startsWith('D.') ? 'D. Mengeringkan Kuvet (Kaca)' : st.sec }); }
    const tx = (typeof stepT === 'function' ? stepT(i) : st.t); wrap(tx, W0 - IND).forEach((l, j) => L.push({ k: 'step', n: j === 0 ? (st.no || '') : '', t: l })); L.push({ k: 'gap' }); });
  L.push({ k: 'sign' });
  const pages = [[]]; let y = 0; const H = 482;
  for (const ln of L) { const h = ln.k === 'sec' ? 34 : ln.k === 'gap' ? 6 : ln.k === 'sign' ? 150 : 21; if (y + h > H) { pages.push([]); y = 0; if (ln.k === 'gap') continue; } pages[pages.length - 1].push(ln); y += h; }
  IKPAGES = pages;
}
function ikLogo(c, cx, cy, r) {
  c.save(); c.beginPath(); for (let k = 0; k < 5; k++) { const a = -Math.PI / 2 + k * 2 * Math.PI / 5; c.lineTo(cx + r * Math.cos(a), cy + r * Math.sin(a)); } c.closePath(); c.fillStyle = '#fff'; c.fill(); c.lineWidth = 2.2; c.strokeStyle = '#1e3a8a'; c.stroke();
  c.beginPath(); c.arc(cx, cy + 2, r * 0.46, 0, 7); c.lineWidth = 3.2; c.strokeStyle = '#dc2626'; c.stroke(); c.beginPath(); c.arc(cx, cy + 2, r * 0.22, 0, 7); c.fillStyle = '#1e3a8a'; c.fill(); c.restore();
}
function ikTex(s) {
  if (IKC[s]) return IKC[s];
  if (!IKPAGES) ikBuild();
  const cv2 = document.createElement('canvas'); cv2.width = 480 * IKK; cv2.height = 680 * IKK; const c = cv2.getContext('2d'); c.scale(IKK, IKK);
  c.fillStyle = s === 0 ? '#7f1d1d' : '#fbfaf4'; c.fillRect(0, 0, 480, 680); c.textBaseline = 'alphabetic';
  if (s === 0) {
    c.strokeStyle = '#e8c76a'; c.lineWidth = 4; c.strokeRect(22, 22, 436, 636); ikLogo(c, 240, 130, 52); c.fillStyle = '#f3e3b0'; c.textAlign = 'center';
    c.font = '700 22px Arial'; c.fillText('POLITEKNIK AKA BOGOR', 240, 230); c.font = '500 15px Arial'; c.fillText('Laboratorium Instrumentasi', 240, 256);
    c.font = '700 34px Arial'; c.fillText('INSTRUKSI KERJA', 240, 350); c.font = '600 22px Arial'; c.fillText('SPEKTROFOTOMETER UV-VIS', 240, 388); c.fillText('SHIMADZU UV-1700', 240, 420);
    c.font = '700 20px Courier New'; c.fillText('IK/LAB.INST/6', 240, 500); return IKC[s] = cv2;
  }
  c.strokeStyle = '#111'; c.lineWidth = 1.5; c.strokeRect(24, 22, 432, 104); c.beginPath(); c.moveTo(110, 22); c.lineTo(110, 94); c.moveTo(110, 58); c.lineTo(456, 58); c.moveTo(24, 94); c.lineTo(456, 94); c.stroke();
  ikLogo(c, 67, 62, 28); c.fillStyle = '#111'; c.textAlign = 'center'; c.font = '700 13px Arial'; c.fillText('POLITEKNIK AKA BOGOR', 283, 44); c.font = '600 11.5px Arial'; c.fillText('LABORATORIUM INSTRUMENTASI', 283, 74);
  c.font = '600 11px Arial'; c.textAlign = 'left'; c.fillText('Kode/No : IK/LAB.INST/6', 120, 88); c.textAlign = 'right'; c.fillText('Halaman ' + s + ' dari ' + IKPAGES.length, 448, 88);
  c.textAlign = 'center'; c.font = '700 14px Arial'; c.fillText('INSTRUKSI KERJA', 240, 111); c.font = '700 13px Arial'; c.fillText('SPEKTROFOTOMETER UV-VIS SHIMADZU UV-1700', 240, 124);
  const pg = IKPAGES[s - 1]; c.textAlign = 'left'; let y = 160; if (!pg) return IKC[s] = cv2;
  for (const ln of pg) {
    if (ln.k === 'sec') { y += 14; c.fillStyle = '#111'; c.font = '700 16px Arial'; c.fillText(ln.t, 30, y); y += 20; }
    else if (ln.k === 'gap') y += 6;
    else if (ln.k === 'step') { c.fillStyle = '#111'; c.font = '15px Arial'; if (ln.n !== '') { c.textAlign = 'right'; c.fillText(ln.n + '.', 52, y); c.textAlign = 'left'; } c.fillText(ln.t, 60, y); y += 21; }
    else if (ln.k === 'sign') { y += 30; c.font = '15px Arial'; c.fillStyle = '#111'; c.fillText('Mengetahui,', 30, y); y += 20; c.fillText('Kepala Laboratorium Instrumentasi', 30, y); y += 84; c.fillText(IKNAME, 30, y); }
  }
  c.fillStyle = '#6b7280'; c.font = '11px Arial'; c.textAlign = 'center'; c.fillText('— ' + s + ' —', 240, 668);
  return IKC[s] = cv2;
}

/* ================= input ================= */
function pickAt(x, y) {
  const inPoly = (p, px, py) => { let c = false; for (let i = 0, j = 3; i < 4; j = i++) { if (((p[i][1] > py) !== (p[j][1] > py)) && (px < (p[j][0] - p[i][0]) * (py - p[i][1]) / (p[j][1] - p[i][1]) + p[i][0])) c = !c; } return c; };
  for (let i = drawn.length - 1; i >= 0; i--) if (inPoly(drawn[i].p, x, y)) return drawn[i].id; return null;
}
function ikRead() { const o = $('#ikread'); if (!o.classList.contains('show')) return; const n = Math.round(ST.ikT), P = IKPAGES.length;
  [['#ikA', 2 * n - 1], ['#ikB', 2 * n]].forEach(([id, side]) => { const cvx = $(id), cx = cvx.getContext('2d'); cx.clearRect(0, 0, cvx.width, cvx.height); cvx.style.visibility = side <= P ? 'visible' : 'hidden'; if (side <= P) cx.drawImage(ikTex(side), 0, 0, cvx.width, cvx.height); }); }
document.addEventListener('click', e => { const r = e.target.closest && e.target.closest('[data-ikr]'); if (r) { $('#ikread').classList.toggle('show', r.dataset.ikr === '1'); ikRead(); return; }
  const b = e.target.closest && e.target.closest('[data-ik]'); if (!b) return; const d = +b.dataset.ik; if (d === 0) { $('#ikread').classList.remove('show'); setView('meja'); } else ST.ikT = Math.max(1, Math.min(IKMAX, Math.round(ST.ikT) + d)); ikRead(); dirty = true; });
(function bind() {
  let pts = new Map(), start = null, pinch = null;
  cv.addEventListener('pointerdown', e => { try { cv.setPointerCapture(e.pointerId); } catch (_) { } pts.set(e.pointerId, [e.clientX, e.clientY]); start = { x: e.clientX, y: e.clientY, yaw: CAM.yaw, pitch: CAM.pitch, moved: false }; if (pts.size === 2) { const [a, b] = [...pts.values()]; pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), dist: CAM.dist }; } });
  cv.addEventListener('pointermove', e => { if (!pts.has(e.pointerId)) return; pts.set(e.pointerId, [e.clientX, e.clientY]);
    if (pts.size === 2 && pinch) { const [a, b] = [...pts.values()]; CAM.dist = Math.max(40, Math.min(460, pinch.dist * pinch.d / Math.max(10, Math.hypot(a[0] - b[0], a[1] - b[1])))); start.moved = true; dirty = true; return; }
    if (!start) return; const dx = e.clientX - start.x, dy = e.clientY - start.y; if (Math.abs(dx) + Math.abs(dy) > 4) start.moved = true;
    CAM.yaw = start.yaw - dx * 0.4; CAM.pitch = Math.max(-5, Math.min(89, start.pitch + dy * 0.3)); camAnim = null; dirty = true; });
  const up = e => { if (!pts.has(e.pointerId)) return; pts.delete(e.pointerId); if (pts.size < 2) pinch = null;
    if (pts.size === 0) { if (start && !start.moved) { const r = cv.getBoundingClientRect(); const id = pickAt(e.clientX - r.left, e.clientY - r.top); selId = id; if (id === 'screen') { const h = screenHit(e.clientX - r.left, e.clientY - r.top); if (h) pcPress(h.x, h.y); } if (id === 'ikbook' && viewName !== 'ik') setView('ik'); const inf = id && INFO[id];
      $('#ti').textContent = inf ? inf[0] : 'Info objek'; $('#info').textContent = inf ? (inf[1] || 'Komponen bangku optik UV-1700.') : 'Klik objek pada meja untuk penjelasan.'; dirty = true; } start = null; } };
  cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
  cv.addEventListener('wheel', e => { e.preventDefault(); CAM.dist = Math.max(40, Math.min(460, CAM.dist * (e.deltaY > 0 ? 1.08 : 1 / 1.08))); camAnim = null; dirty = true; }, { passive: false });
  pcc.addEventListener('click', e => { const r = pcc.getBoundingClientRect(); const x = (e.clientX - r.left) * pcc.width / r.width, y = (e.clientY - r.top) * pcc.height / r.height;
    const b = pcBtns.find(b => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h); if (!b) return;
    if (b.act.startsWith('tab')) { ACT[b.act](); drawPC(); } else if (ACT[b.act]) run(() => ACT[b.act]()); });
  document.addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return;
    if (b.id === 'b_play') { onPlay(); return; } if (b.id === 'b_next') { onNext(); return; } if (b.id === 'b_prev') { onPrev(); return; } if (b.id === 'b_rst') { onReset(); return; }
    if (b.dataset.spd) { SPD = +b.dataset.spd; updateCtl(); return; } if (b.dataset.lng) { LANG = b.dataset.lng; sync(); return; } if (b.id === 'icx') { INNER.closed = true; showInner(false); return; }
    if (b.id === 'gobtn') { const i = curStep(); if (i >= 0) { ST.autoScroll = true; run(() => STEPS[i].run(), 'Mengerjakan: ' + STEPS[i].t); } return; }
    if (b.id === 'lg_ok') { ST.logbook = { t: dstr, n: $('#lg_n').value, k: $('#lg_k').value, s: $('#lg_s').value }; $('#logm').classList.remove('show'); ok('Log Book tercatat dan ditandatangani oleh ' + (ST.logbook.s || '-') + '.'); if (logResolve) { logResolve(); logResolve = null; } sync(); return; }
    if (b.dataset.act) { const a = b.dataset.act; if (ACT[a]) run(() => ACT[a]()); return; }
    if (b.dataset.v) { setView(b.dataset.v); return; }
    if (b.dataset.t) { const k = b.dataset.t;       if (k === 'pip') { if (!ST.pc) toast('Komputer belum dinyalakan.', 'warn'); else { setView('komputer'); } return; } if (k === 'pipx') { ST.pipClosed = true; pipSync(); return; } if (k === 'pipbig') { const bg = $('#pip').classList.toggle('big'); b.textContent = bg ? 'Kecilkan' : 'Perbesar'; return; } const on = b.getAttribute('aria-pressed') !== 'true'; b.setAttribute('aria-pressed', String(on));
      if (k === 'xray') { ST.xrayUser = on; ST.xrayT = on ? 1 : 0; } if (k === 'sound') ST.sound = on; } });
  const s = $('#sol'); SOL_KEYS.forEach(k => { const o = document.createElement('option'); o.value = k; o.textContent = SOL[k].label; s.appendChild(o); });
  $('#lam').value = AN.lam_ref;
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(() => { dirty = true; }).observe(cv.parentElement);
})();

/* ================= pemutar otomatis, bahasa, tampilan dalam ================= */
let LANG = 'id'; const PL = { on: false, loop: false };
const INNER = { closed: false, shown: false };
const BOOT = (() => { try { return JSON.parse(window.name || '{}') || {}; } catch (e) { return {}; } })();
if (BOOT.lang) LANG = BOOT.lang; if (BOOT.spd) SPD = BOOT.spd;
const STEN = ["Check and connect the cables: wall socket → extension cord → UPS; UV-1700, CPU, and monitor to the UPS; printer to the extension cord; VGA, USB, and RS-232C cables.", "Press the ON button on the UPS and wait for the green indicator to light.", "Open the sample compartment lid, take out the silica gel, then close the lid again.", "Press the power button on the left side of the UV-1700; wait until initialization finishes and all checks are OK.", "Turn on the CPU, monitor, and printer.", "On the UV-1700 press MODE, then select PC CONTROL.", "Double-click the UVProbe icon, then click Connect; wait until the instrument check finishes (Pass).", "Let the lamps warm up for about 15–30 minutes so that the intensity stabilizes (time accelerated).", "Click Window → Spectrum.", "Click Edit → Method; set the wavelength range, scan speed, and sampling interval → OK.", "Fill the reference and sample cuvettes with blank, wipe them with tissue, place them in the cells, close the compartment → click Baseline.", "Replace the sample cuvette with the standard solution (mid concentration) → click Start.", "Click Operations → Peak Pick to show the peak table; record λmax.", "Click File → Save As → name the spectrum as Date_Class_Group → Save.", "Click once on the peak pick table, right-click → Print.", "Click Window → Photometric.", "Click Edit → Method.", "In the Wavelength (nm) column type the desired wavelength (λmax) → Add.", "Click Next.", "In the Type column choose the method: Multi Point (more than one standard concentration).", "In the Unit column type the desired unit.", "In the WL1 column select the desired wavelength.", "Click Zero Interception → click Next 3×.", "In the File name field click (...) → choose the folder → name it Date_Class_Group → Open → Finish.", "In the Standard Table type the standard name in the Sample ID column and its concentration in the Conc column.", "Replace the cuvette content in the sample compartment with blank → click Auto Zero.", "Replace with a standard → click Read Std (repeat for the whole standard series).", "Click Sample Table → enter the sample name in the Sample ID column.", "Replace with the sample → click Read Unk (repeat for all samples; here in triplicate).", "After all measurements are finished click File → Save.", "Right-click the Standard Table → Print.", "Right-click the Sample Table → Print.", "Record all events in the instrument Log Book and sign with a legible name.", "Click Disconnect.", "Click File → Exit.", "Click the Windows menu → press the Shut down button.", "Press the power button of the computer monitor and the printer.", "Press MODE on the UV-1700 (leave PC CONTROL).", "Press the power button on the left side of the UV-1700, close the UV-1700 monitor, and put the silica gel back into the compartment.", "Press the OFF button on the UPS.", "Unplug the cables from the extension cord, except the UPS cable.", "Tidy the work area around the instrument and computer; remove the samples and standards.", "Move the chair in the work area to a safe place.", "After the instrument is off, dry the cuvettes by rinsing with ethanol then acetone; turn the cuvettes upside down (safe position) and wait until completely dry."];
const SECEN = { 'A. Menyalakan Alat': 'A. Powering On the Instrument', 'B. Pengukuran Spektrum': 'B. Spectrum Measurement', 'B. Analisa Sampel': 'B. Sample Analysis', 'B. Print Data': 'B. Printing Data', 'B. Print Data Sampel': 'B. Printing Sample Data', 'B. Print Kurva Kalibrasi Standar': 'B. Printing the Standard Calibration Curve', 'C. Mematikan Alat': 'C. Shutting Down', 'D. Mengeringkan Kuvet': 'D. Drying the Cuvettes' };
const UI = {
  id: { step: 'LANGKAH', busy: 'Sedang berjalan… perhatikan animasi 3D, layar komputer, dan tampilan bagian dalam alat.', idle: 'Tekan Putar untuk menjalankan otomatis, atau Berikutnya untuk satu langkah.', done: 'SELESAI', donetx: 'Seluruh langkah Instruksi Kerja selesai.', donewhy: 'Salin data ke Kurva Kalibrasi untuk mengolah hasil.', play: '▶ Putar', pause: '❚❚ Jeda', prev: '◀ Sebelumnya', next: 'Berikutnya ▶', rst: '⟲ Ulang', spd: 'Kecepatan', lng: 'Bahasa', badge: 'BAGIAN DALAM UV-1700 · ZOOM', close: 'Tutup' },
  en: { step: 'STEP', busy: 'Running… watch the 3D animation, the computer screen, and the inside view of the instrument.', idle: 'Press Play to run automatically, or Next for a single step.', done: 'FINISHED', donetx: 'All steps of the work instruction are complete.', donewhy: 'Copy the data to the Calibration Curve to process the results.', play: '▶ Play', pause: '❚❚ Pause', prev: '◀ Previous', next: 'Next ▶', rst: '⟲ Restart', spd: 'Speed', lng: 'Language', badge: 'INSIDE THE UV-1700 · ZOOM', close: 'Close' } };
const U = k => UI[LANG][k];
const stepT = i => LANG === 'en' && STEN[i] ? STEN[i] : STEPS[i].t;
const secT = s => LANG === 'en' && SECEN[s] ? SECEN[s] : s;
const TT = [
  { t: ['Lampu D₂ dan WI', 'D₂ and WI lamps'], f: ['Menyediakan radiasi kontinu UV (D₂) dan tampak–inframerah dekat (WI).', 'Provide continuous UV (D₂) and visible–near-IR (WI) radiation.'] },
  { t: ['Cermin pemilih sumber dan celah masuk', 'Source-selector mirror and entrance slit'], f: ['Memilih berkas lampu aktif dan membentuk berkas sempit menuju monokromator.', 'Select the active lamp beam and form a narrow beam toward the monochromator.'] },
  { t: ['Kisi difraksi (monokromator)', 'Diffraction grating (monochromator)'], f: ['Memisahkan cahaya polikromatik menjadi panjang gelombang tunggal.', 'Disperse polychromatic light into single wavelengths.'] },
  { t: ['Celah keluar', 'Exit slit'], f: ['Meneruskan pita sempit di sekitar λ terpilih.', 'Pass a narrow band around the selected λ.'] },
  { t: ['Pembagi berkas', 'Beam splitter'], f: ['Membagi berkas menjadi berkas referensi dan berkas sampel.', 'Split the beam into reference and sample beams.'] },
  { t: ['Kompartemen sampel dan kuvet', 'Sample compartment and cuvettes'], f: ['Menempatkan blanko dan sampel pada jalur berkas.', 'Place the blank and sample in the beam path.'] },
  { t: ['Detektor fotodioda', 'Photodiode detectors'], f: ['Mengubah intensitas cahaya menjadi arus listrik.', 'Convert light intensity into electric current.'] },
  { t: ['Pengolahan sinyal', 'Signal processing'], f: ['Menguatkan, mendigitalkan, dan mengoreksi sinyal detektor.', 'Amplify, digitize, and correct the detector signal.'] },
  { t: ['Layar dan keluaran', 'Display and output'], f: ['Menampilkan λ, absorbansi atau %T, dan lampu aktif.', 'Display λ, absorbance or %T, and the active lamp.'] }];
const INNER_SRC = /*__INNER__*/"";
const INNER_ON = typeof INNER_SRC === 'string' && INNER_SRC.length > 50;
if (INNER_ON) $('#innerf').srcdoc = INNER_SRC;
function postInner(m) { try { $('#innerf').contentWindow.postMessage(Object.assign({ src: 'uv1700' }, m), '*'); } catch (e) { } }
function showInner(on) { INNER.shown = on; $('#inner').classList.toggle('show', on); $('#sub').style.visibility = on ? 'hidden' : 'visible'; }
function innerCap(k) { const L = LANG === 'en' ? 1 : 0; $('#icT').textContent = (k + 1) + '. ' + TT[k].t[L]; $('#icF').textContent = TT[k].f[L]; }
async function innerSeq(ks, dwell) {
  if (!INNER_ON || FAST || INNER.closed) return;
  const S = ST.cuv.S, col = (S.content && SOL[S.content] && SOL[S.content].col) || '#d9f3f7'; let A = 0; try { A = Math.max(0, readAt(ST.pmethod ? ST.pmethod.lam : AN.lam_ref)); } catch (e) { }
  postInner({ cmd: 'sol', v: col }); postInner({ cmd: 'lam', v: ST.lam || AN.lam_ref }); postInner({ cmd: 'abs', v: isFinite(A) ? A : 0 });
  showInner(true); await wait(0.5);
  for (const k of ks) { if (INNER.closed || FAST) break; postInner({ cmd: 'tour', k }); innerCap(k); await wait(dwell); }
  showInner(false);
}
const INNER_MAP = { 3: [[0], 3.5], 7: [[0, 1], 3.5], 10: [[0, 1, 2, 3, 4, 5, 6, 7, 8], 3.2], 11: [[1, 2, 5, 6], 3], 25: [[0, 5, 6], 2.6], 26: [[0, 5, 6], 2.6], 28: [[0, 5, 6], 2.6] };
Object.keys(INNER_MAP).forEach(k => { const s = STEPS[k], f = s.run, [ks, d] = INNER_MAP[k]; s.run = async () => { const p = f(); await Promise.all([p, innerSeq(ks, d)]); }; });

const CMPL = [
  { n: ['Lampu deuterium (D₂)', 'Deuterium lamp (D₂)'], f: ['Sumber radiasi kontinu daerah ultraviolet (±190–350 nm).', 'Continuous radiation source for the ultraviolet region (≈190–350 nm).'] },
  { n: ['Lampu wolfram-iodin (WI)', 'Tungsten-iodine lamp (WI)'], f: ['Sumber radiasi kontinu daerah tampak dan inframerah dekat (±340–1100 nm).', 'Continuous source for the visible and near-infrared region (≈340–1100 nm).'] },
  { n: ['Cermin pemilih sumber', 'Source selector mirror'], f: ['Memilih berkas dari lampu D₂ atau WI sesuai panjang gelombang.', 'Select the beam from the D₂ or WI lamp according to wavelength.'] },
  { n: ['Celah masuk', 'Entrance slit'], f: ['Membatasi berkas yang masuk ke monokromator.', 'Limit the beam entering the monochromator.'] },
  { n: ['Kisi difraksi', 'Diffraction grating'], f: ['Menyebarkan cahaya polikromatik menjadi spektrum; sudutnya menentukan λ.', 'Disperse polychromatic light into a spectrum; its angle sets λ.'] },
  { n: ['Celah keluar', 'Exit slit'], f: ['Meloloskan pita sempit di sekitar λ terpilih.', 'Pass a narrow band around the selected λ.'] },
  { n: ['Pembagi berkas', 'Beam splitter'], f: ['Membagi berkas menjadi berkas referensi dan sampel.', 'Split the beam into reference and sample beams.'] },
  { n: ['Kuvet dan kompartemen', 'Cuvettes and compartment'], f: ['Menempatkan blanko dan sampel pada jalur berkas.', 'Hold the blank and the sample in the beam path.'] },
  { n: ['Detektor referensi', 'Reference detector'], f: ['Mengukur intensitas berkas referensi (I₀).', 'Measure the reference beam intensity (I₀).'] },
  { n: ['Detektor sampel', 'Sample detector'], f: ['Mengukur intensitas berkas yang melewati sampel (I).', 'Measure the intensity of the beam through the sample (I).'] },
  { n: ['Papan elektronik', 'Main board'], f: ['Menguatkan, mendigitalkan, dan mengolah sinyal detektor.', 'Amplify, digitize and process the detector signal.'] },
  { n: ['Kipas dan catu daya', 'Cooling fan and power supply'], f: ['Mendinginkan lampu/elektronik dan menyuplai daya.', 'Cool the lamps/electronics and supply power.'] },
  { n: ['Layar LCD dan tombol', 'LCD and keypad'], f: ['Menampilkan λ, absorbansi atau %T, dan lampu aktif; menerima perintah.', 'Show λ, absorbance or %T and the active lamp; accept commands.'] }];
function zoomComp(k) {
  if (!INNER_ON) { toast('Model dalam tidak tersedia.', 'warn'); return; }
  INNER.closed = false; const S = ST.cuv.S, col = (S.content && SOL[S.content] && SOL[S.content].col) || '#d9f3f7';
  postInner({ cmd: 'sol', v: col }); postInner({ cmd: 'lam', v: ST.lam || AN.lam_ref }); showInner(true);
  setTimeout(() => postInner({ cmd: 'cmp', k }), 250); setTimeout(() => postInner({ cmd: 'cmp', k }), 900);
  const L = LANG === 'en' ? 1 : 0; $('#icT').textContent = CMPL[k].n[L]; $('#icF').textContent = CMPL[k].f[L];
}
function buildCmps() { const L = LANG === 'en' ? 1 : 0; $('#cmps').innerHTML = CMPL.map((c, k) => '<button class="chip" data-cmp="' + k + '">' + c.n[L].replace(/ \(.*\)/, '') + '</button>').join(''); $('#cmpT').textContent = L ? 'ZOOM INSTRUMENT COMPONENTS' : 'ZOOM KOMPONEN UV-1700'; }
buildCmps();
document.addEventListener('click', e => { const b = e.target.closest && e.target.closest('[data-cmp]'); if (b) zoomComp(+b.dataset.cmp); });
function updateCtl() {
  const ci = curStep(), playing = PL.on && !PAUSED;
  $('#b_play').textContent = playing ? U('pause') : U('play'); $('#b_prev').textContent = U('prev'); $('#b_next').textContent = U('next'); $('#b_rst').textContent = U('rst'); $('#lb_spd').textContent = U('spd'); $('#lb_lng').textContent = U('lng');
  $('#b_prev').disabled = ci === 0 && !ST.busy; $('#b_next').disabled = ci < 0 && !ST.busy; $('#b_play').disabled = ci < 0;
  $$('[data-spd]').forEach(b => b.classList.toggle('on', +b.dataset.spd === SPD)); $$('[data-lng]').forEach(b => b.classList.toggle('on', b.dataset.lng === LANG));
  $('#ibadge').textContent = U('badge'); $('#icx').textContent = U('close'); buildCmps();
  $('#sub').textContent = ci >= 0 ? (ci + 1) + '. ' + stepT(ci) : '';
}
async function advance() {
  const i = curStep(); if (i < 0 || ST.busy) return false; const s = STEPS[i]; let g = 0; INNER.closed = false;
  do { await run(() => s.run(), 'Mengerjakan: ' + s.t); g++; } while (curStep() === i && s.multi && g < 40);
  return true;
}
async function playLoop() {
  if (PL.loop) return; PL.loop = true;
  try { while (PL.on && curStep() >= 0) { const i = curStep(); await advance(); if (curStep() === i) { PL.on = false; break; } if (!PL.on) break; await wait(0.9); } }
  finally { PL.loop = false; if (curStep() < 0) PL.on = false; updateCtl(); }
}
function onPlay() {
  if (PL.on && !PAUSED) { PAUSED = true; updateCtl(); return; }
  if (PAUSED) { PAUSED = false; PL.on = true; updateCtl(); if (!PL.loop) playLoop(); return; }
  PL.on = true; ST.autoScroll = true; updateCtl(); playLoop();
}
async function finishFast() { FAST = true; flushTweens(); const t0 = performance.now(); while ((ST.busy || PL.loop) && performance.now() - t0 < 30000) { await new Promise(r => setTimeout(r, 15)); flushTweens(); } FAST = false; showInner(false); }
async function onNext() {
  if (PL.nb) return; PL.nb = true;
  try { PL.on = false; PAUSED = false; if (ST.busy || PL.loop) await finishFast(); else { ST.autoScroll = true; await advance(); } }
  finally { PL.nb = false; updateCtl(); }
}
function restartTo(n) { try { window.name = JSON.stringify({ replay: Math.max(0, n), lang: LANG, spd: SPD }); location.reload(); } catch (e) { toast('Muat ulang halaman untuk mengulang.', 'warn'); } }
function onPrev() { const i = curStep() < 0 ? STEPS.length : curStep(); restartTo(i - 1); }
function onReset() { restartTo(0); }
ikGeom();
window.LAB = { ST, ACT, STEPS, run, curStep, readAt, ID, finishLog: () => { ST.logbook = { t: dstr, n: ID.nama, k: 'uji', s: ID.nama }; if (logResolve) { logResolve(); logResolve = null; } } };
log('Praktikum virtual: ' + AN.nama + '. Ikuti kotak "Langkah sekarang".');
updateOut(); sync(); drawPC(); requestAnimationFrame(loop);
if (BOOT.replay > 0) { (async () => { FAST = true; for (let k = 0; k < BOOT.replay && curStep() >= 0; k++) { const b = curStep(); await advance(); if (curStep() === b) break; } FAST = false; ST.t0 = performance.now(); sync(); })(); }
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

try:  # paksa tema terang agar teks tidak putih di atas latar terang (mode gelap peramban/Windows)
    from streamlit import config as _stcfg
    _stcfg.set_option("theme.base", "light")
except Exception:
    pass
st.set_page_config(page_title="SHIMADZU UV-1700", page_icon=":material/biotech:", layout="wide", initial_sidebar_state="expanded")


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
/* teks tetap terbaca di mode gelap: area utama selalu memakai tinta gelap di atas latar terang */
:root, .stApp { color-scheme: light; }
.stApp { color: var(--ink); }
[data-testid="stMain"] :is(p, li, label, h1, h2, h3, h4, h5, h6, small, span, td, th, strong, em, summary, code):not(.pill):not(.pill *):not(.hero *):not(.lcd *):not(.verdict *):not([data-testid="stAlert"] *):not(.kpi *):not(.step *):not(.eq *):not(.banner *):not(.feat *):not(.homeimg *):not(.ph *):not(.stButton *):not(.stDownloadButton *) { color: var(--ink); }
[data-testid="stMain"] [data-testid="stCaptionContainer"], [data-testid="stMain"] [data-testid="stCaptionContainer"] * { color: var(--muted) !important; }
[data-testid="stMain"] input, [data-testid="stMain"] textarea, [data-testid="stMain"] [data-baseweb="select"] > div, [data-testid="stMain"] [data-baseweb="input"] { background-color:#ffffff !important; color: var(--ink) !important; }
[data-testid="stMain"] [data-testid="stExpander"] details { background:#ffffff; border-radius:12px; }

h1, h2, h3, h4 { font-family:'Space Grotesk','Inter',sans-serif !important; letter-spacing:-.01em; color:var(--ink); }
.stApp { background: radial-gradient(1100px 480px at 100% -8%, #ede8ff 0%, rgba(237,232,255,0) 60%), radial-gradient(900px 420px at -8% 0%, #e3f4ff 0%, rgba(227,244,255,0) 55%), #f7f8fc; }
.block-container { padding-top: 1.1rem; max-width: 1400px; }
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
.homeimg { max-width: 1000px; margin: 8px auto 6px; border-radius: 24px; overflow: hidden; background:#fff; border:1px solid var(--line); box-shadow: 0 30px 60px -30px rgba(37,29,107,.45); }
.homeimg img { display:block; width:100%; height:auto; }
.homeimg .cap { text-align:center; padding: 16px 12px 18px; font: 700 1.7rem 'Space Grotesk', sans-serif; color: var(--ink); letter-spacing:-.01em; }
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


def model3d(height=560, lid=False, xray=True, compact=False, auto=False, tour=False):
    init = {"lam": float(I["lam"]), "abs": float(I["last_A"]) if I["power"] == "ready" else 0.0,
            "status": "READY" if I["power"] == "ready" else "OFF", "solColor": "#d9f3f7" if I["solution"] == "blanko" else an()["warna"],
            "lampSwitch": float(I["lamp_switch"]), "xray": xray, "lid": lid, "compact": compact, "auto": auto, "tour": tour}
    components.html(MODEL3D_HTML.replace("/*__INIT__*/null", json.dumps(init)), height=height, scrolling=False)


def csv_bytes(df):
    k = S["settings"]["dec"] == "koma"
    return df.to_csv(index=False, sep=";" if k else ",", decimal="," if k else ".")


# ---------------------------------------------------------------------------
# Halaman: Beranda
# ---------------------------------------------------------------------------
HOME_IMG = "data:image/jpeg;base64,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"


def page_home():
    html(f'''<div class="homeimg"><img src="{HOME_IMG}" alt="Model 3D Shimadzu UV-1700">
<div class="cap">Spektrofotometer UV-Vis Shimadzu UV-1700 PharmaSpec</div></div>''')
    _, mid, _ = st.columns([1.2, 1, 1.2])
    if mid.button("Mulai praktikum", type="primary", icon=":material/play_arrow:", key="home_go"):
        st.switch_page(PG_LAB)


# ---------------------------------------------------------------------------
# Halaman: Model 3D
# ---------------------------------------------------------------------------
def page_model3d():
    header("Instrumen", "Bagian dalam dan prinsip kerja")
    model3d(height=720, xray=True, tour=True)


# ---------------------------------------------------------------------------
# Halaman: Simulasi Instrumen
# ---------------------------------------------------------------------------



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
    keys = list(ANALYTES)
    c1, c2 = st.columns([1.5, 1], vertical_alignment="bottom")
    with c1:
        html('<div class="ph" style="margin:0"><div class="eb">Praktikum · IK/LAB.INST/6</div><h1 style="font-size:1.6rem">3D UV-1700</h1></div>')
    with c2:
        akey = st.selectbox("Percobaan", keys, index=keys.index(S.get("lab_an", "fephen")), format_func=lambda k: f"{ANALYTES[k]['nama']} · {ANALYTES[k]['lam_ref']} nm", key="lab_an_sel")
    if akey != S.get("lab_an"):
        S["lab_an"] = akey
    a = ANALYTES[akey]
    init = {"seed": int(S["seed"]) % 2147483647, "hidden": hidden_conc(akey, S["seed"]), "dec": S["settings"]["dec"], "unit": "mg/L",
            "analyte": {"nama": a["nama"], "rumus": a["rumus"], "pelarut": a["pelarut"], "lam_ref": a["lam_ref"], "k": float(a["k"]), "warna": a["warna"],
                        "bands": [list(b) for b in a["bands"]], "std": [float(x) for x in a["std"]]}}
    child = MODEL3D_HTML.replace("/*__INIT__*/null", json.dumps({"tour": True, "slave": True, "lam": 550.0, "abs": 0.3, "solColor": "#d9f3f7", "lampSwitch": 340.8, "xray": True}))
    inner = json.dumps(child).replace("</", "<\\/").replace("<!--", "<\\!--")
    components.html(LAB_HTML.replace('/*__INNER__*/""', inner).replace("/*__LAB__*/null", json.dumps(init)), height=900, scrolling=True)
    st.caption("Mengganti percobaan atau memuat ulang halaman mengulang praktikum dari awal; salin data dari panel di bawah 3D sebelum berpindah. "
               "Tanda REK = langkah disusun dari prosedur baku UVProbe karena halaman 1 IK belum tersedia.")
    with st.container(border=True):
        st.markdown("**Impor hasil praktikum ke Kurva Kalibrasi**")
        txt = st.text_area("Tempel teks dari tombol Salin data", key="lab_paste", height=150,
                           placeholder="#LAMBDA 510,0\n#STANDAR\n0,500;0,1012\n…\n#SAMPEL\n0,4891; 0,4895; 0,4893")
        if st.button("Impor hasil praktikum", type="primary"):
            res = parse_lab_result(txt)
            if len(res["std"]) < 2:
                st.error("Data standar tidak ditemukan; pastikan teks diawali #STANDAR dan berisi pasangan C;A.")
            else:
                stdv = list(res["std"])
                if not any(abs(c_) < 1e-12 for c_, _ in stdv):
                    stdv = [(0.0, 0.0)] + stdv
                set_calibration(pd.DataFrame({"C": [c_ for c_, _ in stdv], "A": [a_ for _, a_ in stdv], "Pakai": True}), "simulasi",
                                name=a["rumus"], unit="mg/L", lam=res["lam"], analyte=akey)
                if res["smp"]:
                    nan_ = float("nan"); O_ = S.setdefault("olah", _olah_defaults()); n_ = len(res["smp"])
                    O_["s"] = pd.DataFrame({"Sampel": ["Blanko Sampel"] + [f"Sampel {i + 1}" for i in range(n_)], "Bobot (kg)": [nan_] * (n_ + 1), "Volume labu takar (L)": [nan_] * (n_ + 1),
                                            "Volume (L)": [nan_] * (n_ + 1), "Abs": [0.0] + [float(v) for v in res["smp"]], "FP": [nan_] + [1.0] * n_})
                    O_["ver"] = O_.get("ver", 0) + 1
                if res["smp"]:
                    S["samples"] = [s_ for s_ in S["samples"] if s_.get("src") != "praktikum"] + [
                        new_sample(f"Sampel X {a['rumus']} (praktikum)", "; ".join(fmt(v, 4) for v in res["smp"]), "simulasi", akey)]
                log(f"Hasil praktikum virtual diimpor: {len(res['std'])} standar, {len(res['smp'])} pembacaan sampel.")
                st.success(f"{len(res['std'])} titik standar dan {len(res['smp'])} pembacaan sampel dikirim. Buka menu Olah data; lengkapi bobot, volume labu, dan FP sampel.")


# ---------------------------------------------------------------------------
# Halaman: Fotometri
# ---------------------------------------------------------------------------



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



# ---------------------------------------------------------------------------
# Halaman: Analisis Spektrum
# ---------------------------------------------------------------------------



def rpd_pct(c1, c2) -> float:
    return abs(c1 - c2) / ((c1 + c2) / 2) * 100


def page_qc():
    st.subheader("Presisi dan akurasi: %RPD dan %Recovery")
    q1, q2 = st.columns(2, gap="large")
    with q1:
        with st.container(border=True):
            st.markdown("**%RPD (duplo)**")
            d1 = st.text_input("Konsentrasi duplo 1", "", key="rpd_m1")
            d2 = st.text_input("Konsentrasi duplo 2", "", key="rpd_m2")
            lim = st.number_input("Batas %RPD", 0.1, 50.0, 5.0, 0.5, key="rpd_lim")
            v1, v2 = num_in(d1), num_in(d2)
            if not (math.isnan(v1) or math.isnan(v2)) and (v1 + v2) > 0:
                rpd = rpd_pct(v1, v2)
                f4 = lambda x: fmt(x, 4).replace(",", "{,}")
                st.latex(r"\%RPD = \frac{|C_1 - C_2|}{(C_1 + C_2)/2} \times 100")
                st.latex(r"= \frac{|" + f4(v1) + " - " + f4(v2) + r"|}{(" + f4(v1) + " + " + f4(v2) + r")/2} \times 100 = \frac{" + f4(abs(v1 - v2)) + "}{" + f4((v1 + v2) / 2) + r"} \times 100 = " + fmt(rpd, 2) + r"\ \%")
                verdict(rpd <= lim, f"%RPD = {fmt(rpd, 2)} % " + (f"≤ {fmt(lim, 1)} % (memenuhi)" if rpd <= lim else f"> {fmt(lim, 1)} % (tidak memenuhi)"))
            else:
                st.caption("Isi konsentrasi duplo 1 dan duplo 2.")
    with q2:
        with st.container(border=True):
            st.markdown("**%Recovery (spike)**")
            e1 = st.text_input("C sampel (tanpa spike)", "", key="rec_m1")
            e2 = st.text_input("C sampel + spike", "", key="rec_m2")
            ad_mode = st.radio("C spike yang ditambahkan", ["Hitung dari larutan induk spike", "Isi langsung"], horizontal=True, key="rec_admode")
            if ad_mode.startswith("Hitung"):
                g1, g2, g3 = st.columns(3)
                cst = num_in(g1.text_input("C larutan spike", "", key="rec_cst"))
                vsp = num_in(g2.text_input("V spike (mL)", "", key="rec_vsp"))
                vtt = num_in(g3.text_input("V akhir labu (mL)", "", key="rec_vtt"))
                cad_calc = cst * vsp / vtt if not any(math.isnan(x) for x in (cst, vsp, vtt)) and vtt > 0 else float("nan")
                ad = "" if math.isnan(cad_calc) else str(cad_calc)
                if not math.isnan(cad_calc):
                    st.caption("C spike = C larutan spike × V spike / V akhir = " + fmt(cst, 4) + " × " + fmt(vsp, 3) + " / " + fmt(vtt, 3) + " = " + fmt(cad_calc, 4))
            else:
                ad = st.text_input("C spike yang ditambahkan (setelah pengenceran)", "", key="rec_add")
            c0, c1_ = num_in(e1), num_in(e2)
            cad = float(ad) if ad_mode.startswith("Hitung") and ad else num_in(ad)
            if not any(math.isnan(x) for x in (c0, c1_, cad)) and cad > 0:
                Rv = recovery_spike(c0, c1_, cad)
                ok_ = CR["rec_min"] <= Rv <= CR["rec_max"]
                f4 = lambda x: fmt(x, 4).replace(",", "{,}")
                st.latex(r"\%Recovery = \frac{C_{sampel+spike} - C_{sampel}}{C_{spike}} \times 100")
                st.latex(r"= \frac{" + f4(c1_) + " - " + f4(c0) + "}{" + f4(cad) + r"} \times 100 = \frac{" + f4(c1_ - c0) + "}{" + f4(cad) + r"} \times 100 = " + fmt(Rv, 2) + r"\ \%")
                verdict(ok_, f"%Recovery = {fmt(Rv, 2)} % (kriteria {fmt(CR['rec_min'], 0)}–{fmt(CR['rec_max'], 0)} %): " + ("memenuhi" if ok_ else "tidak memenuhi"))
            else:
                st.caption("Isi C sampel, C sampel + spike, dan C spike.")


# ---------------------------------------------------------------------------
# Halaman: Kurva Kalibrasi
# ---------------------------------------------------------------------------
def _fl(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


BCOL, VCOL = "Bobot Sampel (kg)", "Volume Labu Takar (L)"


def _olah_defaults():
    nan = float("nan")
    return {"s": pd.DataFrame({"Sampel": ["Blanko Sampel", "1", "2"], BCOL: [nan] * 3, VCOL: [nan] * 3, "Volume Sampel (L)": [nan] * 3, "Abs": [nan] * 3, "FP": [nan] * 3}),
            "sp": pd.DataFrame({"Sampel": ["3"], BCOL: [nan], "Volume Sampel (L)": [nan], "Abs": [nan], "FP": [nan]}),
            "c2": pd.DataFrame({"Cons. Standar (mg/L)": [nan], "Volume Standar (L)": [nan]}),
            "rpd_lim": 5.0, "ver": 0}


def olah_calc(df, reg, cmin=None, blank=None, vcol=VCOL):
    """Cons terukur = (Abs − b)/a. Koreksi blanko hanya dipakai bila Cons blanko ≥ 0,1 × konsentrasi standar terkecil;
    bila lebih kecil, Cons terkoreksi = Cons terukur. Kadar = Cons terkoreksi × FP × V labu / bobot."""
    nan = float("nan")
    ok = reg is not None and getattr(reg, "ok", False)
    rows, cblank = [], nan
    for _, r in df.iterrows():
        name = str(r.get("Sampel", "") or "")
        A = _fl(r.get("Abs"))
        cons = (A - reg.intercept) / reg.slope if ok and not math.isnan(A) else nan
        isb = name.strip().lower().startswith("blanko")
        if isb and not math.isnan(cons):
            cblank = cons
        rows.append({"name": name, "blank": isb, "A": A, "cons": cons, "fp": _fl(r.get("FP")), "vl": _fl(r.get(vcol)), "m": _fl(r.get(BCOL))})
    lim = 0.1 * cmin if cmin and cmin > 0 else nan
    if blank is not None:
        cb_used, info = blank
    else:
        use = not math.isnan(cblank) and (math.isnan(lim) or cblank >= lim)
        cb_used = cblank if use else 0.0
        info = {"cblank": cblank, "lim": lim, "use": use}
    for q in rows:
        q["corr"] = nan if q["blank"] else q["cons"] - cb_used
        q["kadar"] = q["corr"] * q["fp"] * q["vl"] / q["m"] if not q["blank"] and q["m"] > 0 and not any(math.isnan(v) for v in (q["corr"], q["fp"], q["vl"])) else nan
    return rows, (cb_used, info)


def olah_table(rows):
    f = lambda x, d=4: "" if (x is None or (isinstance(x, float) and math.isnan(x))) else fmt(x, d)
    return pd.DataFrame({"Sampel": [q["name"] for q in rows], "Abs": [f(q["A"], 3) for q in rows], "Cons terukur (mg/L)": [f(q["cons"]) for q in rows],
                         "Cons terkoreksi (mg/L)": ["" if q["blank"] else f(q["corr"]) for q in rows], "Kadar (mg/kg)": ["" if q["blank"] else f(q["kadar"], 2) for q in rows]})


def page_calibration():
    header("Analisis", "Olah data", "Mengikuti lembar kerja Excel: (1) deret standar, (2) kadar sampel, (3) presisi, (4) akurasi.")
    O = S.setdefault("olah", _olah_defaults())
    for k_, v_ in _olah_defaults().items():
        O.setdefault(k_, v_)
    if BCOL not in O["s"].columns or "c2" not in O or "Cons. Standar (mg/L)" not in O["c2"].columns or BCOL not in O["sp"].columns:
        O.update(_olah_defaults())
    c1, c2 = st.columns([2, 1])
    S["cal_name"] = c1.text_input("Analit / metode", S["cal_name"], placeholder="mis. Cr(VI) dengan difenilkarbazida")
    lam_txt = c2.text_input("λ ukur (nm)", fmt(S["cal_lambda"], 1) if S["cal_lambda"] else "")
    S["cal_lambda"] = num_in(lam_txt, None) if lam_txt else None
    S["cal_zero"] = False
    S["cal_unit"] = "mg/L"
    ver = O["ver"]
    if S["settings"]["mode"] == "latihan":
        if st.button("Muat data contoh (lembar kerja Excel Cr(VI))"):
            set_calibration(pd.DataFrame({"C": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "A": [0.0, 0.080, 0.155, 0.234, 0.312, 0.380], "Pakai": True}), "contoh", name="Cr(VI)", unit="mg/L")
            nan = float("nan")
            O["s"] = pd.DataFrame({"Sampel": ["Blanko Sampel", "1", "2"], BCOL: [nan, 0.4006 / 1000, 0.4007 / 1000], VCOL: [nan, 0.25, 0.25], "Volume Sampel (L)": [nan, 0.05, 0.05],
                                   "Abs": [-0.001, 0.191, 0.184], "FP": [nan, 100.0, 100.0]})
            O["sp"] = pd.DataFrame({"Sampel": ["3"], BCOL: [0.4023 / 1000], "Volume Sampel (L)": [0.25], "Abs": [0.193], "FP": [100.0]})
            O["c2"] = pd.DataFrame({"Cons. Standar (mg/L)": [1000.0], "Volume Standar (L)": [0.001]})
            O["ver"] += 1; st.rerun()

    # ---------------- 1. deret standar
    st.subheader("1. Tabel hasil pengukuran deret standar")
    L, Rr = st.columns([1, 1.5], gap="large")
    with L:
        base_df = S["cal_base"] if len(S["cal_base"]) else pd.DataFrame({"C": [float("nan")] * 6, "A": [float("nan")] * 6, "Pakai": [True] * 6})
        edited = st.data_editor(base_df, key=f"cal_{S['cal_ver']}", num_rows="dynamic", hide_index=False, column_order=("C", "A"), **WIDE,
                                column_config={"C": st.column_config.NumberColumn("Konsentrasi (mg/L)", format="%.4f"), "A": st.column_config.NumberColumn("Abs", format="%.4f")})
        S["cal_cur"] = edited
        reg = cal_reg()
        rr = float("nan")
        if reg and reg.ok:
            rr = math.copysign(math.sqrt(max(0.0, reg.r2)), reg.slope)
            st.dataframe(pd.DataFrame({"Parameter": ["r", "r²", "intercept", "slope"], "Nilai": [fmt(rr, 4), fmt(reg.r2, 4), fmt(reg.intercept, 4), fmt(reg.slope, 4)]}), hide_index=True, **WIDE)
            verdict(reg.r2 >= CR["r2_min"], f"r² = {fmt(reg.r2, 4)} " + (f"≥ {fmt(CR['r2_min'], 3)}" if reg.r2 >= CR["r2_min"] else f"< {fmt(CR['r2_min'], 3)} (kriteria linearitas)"))
        else:
            st.dataframe(pd.DataFrame({"Parameter": ["r", "r²", "intercept", "slope"], "Nilai": ["", "", "", ""]}), hide_index=True, **WIDE)
            st.info("Isi minimal 3 titik standar (konsentrasi dan Abs); r, r², intercept, dan slope muncul otomatis.")
    with Rr:
        pts = cal_points()
        fig = fig_base("Konsentrasi (mg/L)", "Abs", 300); fig.update_layout(hovermode="closest", showlegend=False)
        if reg and reg.ok:
            xs = np.linspace(min(0, reg.xmin), reg.xmax * 1.05, 50)
            fig.add_trace(go.Scatter(x=xs, y=reg.slope * xs + reg.intercept, mode="lines", line=dict(color=PAL[0], width=2)))
            fig.update_layout(title=dict(text=f"y = {fmt(reg.slope, 4)}x + {fmt(reg.intercept, 4)}  ·  R = {fmt(rr, 4)}  ·  R² = {fmt(reg.r2, 4)}", font=dict(size=13)))
        fig.add_trace(go.Scatter(x=pts["C"], y=pts["A"], mode="markers", marker=dict(color=PAL[0], size=10, line=dict(color="white", width=1.2))))
        show(fig)
    pts_ = cal_points(); pos_ = pts_[pts_["C"] > 0]["C"]; cmin = float(pos_.min()) if len(pos_) else None
    ok_reg = bool(reg and reg.ok)

    # ---------------- 2. kadar sampel
    st.subheader("2. Tabel hasil perhitungan kadar")
    st.caption("Isi kolom putih (Bobot, Volume, Abs, FP). Cons terukur = (Abs − intercept) / slope  ·  Cons terkoreksi = Cons terukur − Cons blanko (hanya bila Cons blanko ≥ 0,1 × konsentrasi standar terkecil; jika lebih kecil, Cons terkoreksi = Cons terukur)  ·  Kadar = Cons terukur (terkoreksi) × FP × Volume labu takar / Bobot sampel.")
    cfg = {"Bobot Sampel (kg)": st.column_config.NumberColumn(format="%.7f"), "Volume Labu Takar (L)": st.column_config.NumberColumn(format="%.4f"),
           "Volume Sampel (L)": st.column_config.NumberColumn(format="%.4f"), "Abs": st.column_config.NumberColumn(format="%.4f"), "FP": st.column_config.NumberColumn(format="%.2f")}
    O["s"] = st.data_editor(O["s"], key=f"olah_s_{ver}", num_rows="fixed", hide_index=True, column_config=cfg, **WIDE)
    rows, blk = olah_calc(O["s"], reg if ok_reg else None, cmin)
    nanv = float("nan")
    t2 = pd.DataFrame({"Sampel": [q["name"] for q in rows], "Bobot Sampel (kg)": [fmt(q["m"], 7) if not math.isnan(q["m"]) else "" for q in rows],
                       "Volume Labu Takar (L)": [fmt(q["vl"], 4) if not math.isnan(q["vl"]) else "" for q in rows],
                       "Volume Sampel (L)": [fmt(_fl(v), 4) if not math.isnan(_fl(v)) else "" for v in O["s"]["Volume Sampel (L)"]],
                       "Abs": [fmt(q["A"], 3) if not math.isnan(q["A"]) else "" for q in rows], "Cons terukur (mg/L)": [fmt(q["cons"], 4) if not math.isnan(q["cons"]) else "" for q in rows],
                       "FP": [fmt(q["fp"], 2) if not math.isnan(q["fp"]) else "" for q in rows],
                       "Cons Terkoreksi": ["" if (q["blank"] or math.isnan(q["corr"])) else fmt(q["corr"], 4) for q in rows],
                       "Kadar Cr (VI) (mg/kg)": ["" if (q["blank"] or math.isnan(q["kadar"])) else fmt(q["kadar"], 2) for q in rows]})
    st.dataframe(t2, hide_index=True, **WIDE)
    bi = blk[1]
    if not math.isnan(bi["cblank"]):
        if bi["use"]:
            st.caption(f"Cons blanko = {fmt(bi['cblank'], 4)} mg/L ≥ batas {fmt(bi['lim'], 4) if not math.isnan(bi['lim']) else '–'} mg/L (0,1 × standar terkecil) → blanko dipakai sebagai koreksi.")
        else:
            st.caption(f"Cons blanko = {fmt(bi['cblank'], 4)} mg/L < batas {fmt(bi['lim'], 4)} mg/L (0,1 × standar terkecil) → blanko tidak dipakai; Cons terkoreksi = Cons terukur.")
    smp = [q for q in rows if not q["blank"]]

    # ---------------- 3. presisi
    st.subheader("3. Tabel presisi")
    k1 = smp[0]["kadar"] if len(smp) > 0 else nanv
    k2 = smp[1]["kadar"] if len(smp) > 1 else nanv
    have = not (math.isnan(k1) or math.isnan(k2))
    c1_ = (k1 + k2) / 2 if have else nanv
    sel = abs(k1 - k2) if have else nanv
    rpd = rpd_pct(k1, k2) if have else nanv
    g2 = lambda x: "" if math.isnan(x) else fmt(x, 2)
    st.dataframe(pd.DataFrame({"Sampel": [smp[0]["name"] if smp else "1", smp[1]["name"] if len(smp) > 1 else "2", "Rerata (C1)", "Selisih", "RPD%"],
                               "Kadar Cr(VI) (mg/kg)": [g2(k1), g2(k2), g2(c1_), g2(sel), g2(rpd)]}), hide_index=True, **WIDE)
    O["rpd_lim"] = st.number_input("Batas %RPD", 0.1, 50.0, float(O["rpd_lim"]), 0.5)
    if have:
        f2 = lambda x: fmt(x, 2).replace(",", "{,}")
        st.latex(r"\%RPD = \frac{|C_1 - C_2|}{\bar{C}} \times 100 = \frac{" + f2(sel) + "}{" + f2(c1_) + r"} \times 100 = " + f2(rpd) + r"\ \%")
        verdict(rpd <= O["rpd_lim"], f"%RPD = {fmt(rpd, 2)} % " + (f"≤ {fmt(O['rpd_lim'], 1)} %" if rpd <= O["rpd_lim"] else f"> {fmt(O['rpd_lim'], 1)} %"))
    else:
        st.info("Tabel terisi otomatis setelah Bobot, Volume, Abs, dan FP sampel 1 dan 2 pada tabel 2 diisi.")

    # ---------------- 4. akurasi
    st.subheader("4. Tabel akurasi")
    st.markdown("**Tabel perhitungan konsentrasi sampel + spike (C3)**")
    cfg2 = {"Bobot Sampel (kg)": st.column_config.NumberColumn(format="%.7f"), "Volume Sampel (L)": st.column_config.NumberColumn(format="%.4f"),
            "Abs": st.column_config.NumberColumn(format="%.4f"), "FP": st.column_config.NumberColumn(format="%.2f")}
    O["sp"] = st.data_editor(O["sp"], key=f"olah_sp_{ver}", num_rows="fixed", hide_index=True, column_config=cfg2, **WIDE)
    sp_rows, _ = olah_calc(O["sp"], reg if ok_reg else None, cmin, blank=blk, vcol="Volume Sampel (L)")
    q3 = sp_rows[0]
    st.dataframe(pd.DataFrame({"Sampel": [q3["name"]], "Bobot Sampel (kg)": [fmt(q3["m"], 7) if not math.isnan(q3["m"]) else ""], "Volume Sampel (L)": [fmt(q3["vl"], 4) if not math.isnan(q3["vl"]) else ""],
                               "Abs": [fmt(q3["A"], 3) if not math.isnan(q3["A"]) else ""], "Cons. terukur (mg/L)": [fmt(q3["cons"], 4) if not math.isnan(q3["cons"]) else ""],
                               "FP": [fmt(q3["fp"], 2) if not math.isnan(q3["fp"]) else ""], "Cons Terkoreksi": ["" if math.isnan(q3["corr"]) else fmt(q3["corr"], 4)],
                               "Kadar Cr (VI) (mg/kg)": ["" if math.isnan(q3["kadar"]) else fmt(q3["kadar"], 2)]}), hide_index=True, **WIDE)
    c3 = q3["kadar"]
    st.markdown("**Tabel perhitungan konsentrasi spike (C2)**")
    O["c2"] = st.data_editor(O["c2"], key=f"olah_c2_{ver}", num_rows="fixed", hide_index=True, **WIDE,
                             column_config={"Cons. Standar (mg/L)": st.column_config.NumberColumn(format="%.2f"), "Volume Standar (L)": st.column_config.NumberColumn(format="%.5f")})
    cs_, vs_ = _fl(O["c2"].iloc[0, 0]), _fl(O["c2"].iloc[0, 1]); ms_ = q3["m"]
    c2v = cs_ * vs_ / ms_ if not any(math.isnan(x) for x in (cs_, vs_, ms_)) and ms_ > 0 else nanv
    st.dataframe(pd.DataFrame({"Cons. Standar (mg/L)": [fmt(cs_, 2) if not math.isnan(cs_) else ""], "Volume Standar (L)": [fmt(vs_, 5) if not math.isnan(vs_) else ""],
                               "Bobot Sampel (kg)": [fmt(ms_, 7) if not math.isnan(ms_) else ""], "C2 (mg/kg)": ["" if math.isnan(c2v) else fmt(c2v, 2)]}), hide_index=True, **WIDE)
    st.caption("Bobot sampel pada tabel C2 otomatis sama dengan bobot sampel + spike (C3).")
    st.markdown("**Tabel perhitungan %Recovery**")
    ready = not any(math.isnan(x) for x in (c1_, c2v, c3)) and c2v > 0
    Rv = (c3 - c1_) / c2v * 100 if ready else nanv
    ok_ = ready and CR["rec_min"] <= Rv <= CR["rec_max"]
    st.dataframe(pd.DataFrame({"Sampel": [q3["name"]], "Rerata Kadar Cr(VI) C1 (mg/kg)": [g2(c1_)], "Cons. Spike C2 (mg/kg)": [g2(c2v)],
                               "Cons. spl + spike C3 (mg/kg)": [g2(c3)], "%Recovery": [g2(Rv)], "Kesimpulan": [("MS" if ok_ else "TMS") if ready else ""]}), hide_index=True, **WIDE)
    r1, r2 = st.columns(2)
    CR["rec_min"] = r1.number_input("Batas bawah recovery (%)", 0.0, 200.0, float(CR["rec_min"]), 1.0)
    CR["rec_max"] = r2.number_input("Batas atas recovery (%)", 0.0, 200.0, float(CR["rec_max"]), 1.0)
    if ready:
        f2 = lambda x: fmt(x, 2).replace(",", "{,}")
        st.latex(r"\%Recovery = \frac{C_3 - C_1}{C_2} \times 100 = \frac{" + f2(c3) + " - " + f2(c1_) + "}{" + f2(c2v) + r"} \times 100 = " + f2(Rv) + r"\ \%")
        verdict(ok_, f"%Recovery = {fmt(Rv, 2)} % (kriteria {fmt(CR['rec_min'], 0)}–{fmt(CR['rec_max'], 0)} %): " + ("MS (memenuhi syarat)" if ok_ else "TMS (tidak memenuhi syarat)"))
    else:
        st.info("Tabel terisi otomatis setelah tabel presisi (C1), tabel sampel + spike (C3), dan data spike (C2) lengkap.")


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



# ---------------------------------------------------------------------------
# Halaman: Multikomponen
# ---------------------------------------------------------------------------
MIX_X, MIX_Y = "kmno4", "k2cr2o7"



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
    page_qc()
    st.divider()
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
    blocks.append({"t": "note", "text": "Sumber data: " + (", ".join(sorted(srcs)) or "tidak ada") + ". Dibuat dengan SHIMADZU UV-1700; aplikasi tidak terhubung langsung ke instrumen."})
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
PG_HOME = st.Page(page_home, title="Beranda", icon=":material/home:", url_path="beranda", default=True)
PG_LAB = st.Page(page_lab, title="3D UV-1700", icon=":material/science:", url_path="praktikum")
PG_INNER = st.Page(page_model3d, title="Bagian dalam & prinsip", icon=":material/view_in_ar:", url_path="bagian-dalam")
PG_CAL = st.Page(page_calibration, title="Olah data (kurva kalibrasi)", icon=":material/show_chart:", url_path="kalibrasi")
PAGES = [PG_HOME, PG_LAB, PG_INNER, PG_CAL]
nav = st.navigation(PAGES)

with st.sidebar:
    html("<div style='padding:4px 0 2px'><div style='font:700 1.25rem Space Grotesk,sans-serif;letter-spacing:.04em'>SHIMADZU UV-1700</div>"
         "<div style='font:500 .7rem JetBrains Mono,monospace;letter-spacing:.12em;opacity:.7'>PRAKTIKUM UV-VIS · KELOMPOK 2</div>"
         "<div style='height:4px;margin-top:8px;border-radius:4px;background:linear-gradient(90deg,#7b3fe4,#2f5bff,#00b3d6,#22c55e,#facc15,#f97316,#ef4444)'></div></div>")
    st.caption("Politeknik AKA Bogor · Analisis Kimia. Instrumen acuan: Shimadzu UV-1700 PharmaSpec. Simulator edukatif; tidak terhubung ke alat.")

nav.run()
