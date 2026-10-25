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
:root{--fg:#0b1b3a;--muted:#5b6386;--line:#e1e4f2;--bg:#ffffff;--soft:#e3edff;--acc:#1f6fff}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:transparent;color:var(--fg);font:13px/1.45 'IBM Plex Sans',system-ui,sans-serif}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 250px;gap:10px;height:100%;padding:2px}
.wrap.compact{grid-template-columns:1fr}.wrap.compact .side{display:none}
.stage{position:relative;border-radius:16px;overflow:hidden;min-height:260px;border:1px solid var(--line)}
canvas{width:100%;height:100%;display:block;cursor:grab;touch-action:none}
.tools{position:absolute;top:8px;right:8px;display:flex;gap:6px}
.views{position:absolute;left:8px;bottom:8px;display:flex;flex-wrap:wrap;gap:3px;background:rgba(255,255,255,.86);backdrop-filter:blur(6px);padding:4px;border-radius:10px;max-width:calc(100% - 16px)}
.badge{position:absolute;left:10px;top:10px;font:600 10px 'IBM Plex Mono',monospace;letter-spacing:.1em;color:#1f6fff;background:rgba(255,255,255,.86);padding:4px 8px;border-radius:99px}
button{font:inherit;border:1px solid var(--line);background:#fff;color:var(--fg);border-radius:9px;padding:5px 9px;cursor:pointer}
button:hover{border-color:var(--acc)} .views button{border:0;background:none;color:var(--muted);font-size:12px} .views button.on{background:#1f6fff;color:#fff;font-weight:600}
.side{display:flex;flex-direction:column;gap:8px;overflow:auto}
.tg{display:grid;grid-template-columns:1fr 1fr;gap:6px}.tg button{text-align:left;font-size:12px;color:var(--muted)}
.tg button[aria-pressed=true]{border-color:var(--acc);background:var(--soft);color:var(--fg);font-weight:600}
.card{border:1px solid var(--line);border-radius:14px;padding:10px;background:#fff}
h4{margin:0 0 6px;font-size:13px}.hint{color:var(--muted);font-size:12px}
input[type=range]{width:100%;accent-color:var(--acc)}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-1px;margin-right:4px;border:1px solid rgba(0,0,0,.15)}
.wrap.tour{grid-template-columns:minmax(0,1fr) 330px}
.tourcard{display:none}.wrap.tour .tourcard{display:block}.wrap.tour .plain{display:none}
.steps{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px}
.steps button{width:28px;height:28px;padding:0;border-radius:50%;font:600 12px 'IBM Plex Mono',monospace;color:var(--muted)}
.steps button.on{background:var(--acc);border-color:var(--acc);color:#fff}.steps button.seen{background:#e3edff}
.tt{font:700 15px 'IBM Plex Sans Condensed','IBM Plex Sans',sans-serif;margin:0 0 4px;color:#0b1b3a}.tx{font-size:12.5px;color:#2b2f55;margin:0 0 8px}
.kv{display:grid;grid-template-columns:84px 1fr;gap:6px 10px;font-size:12.5px;color:#2b2f55}.kv b{color:#1f6fff;font-size:11px;line-height:1.35}
.eq{font:500 12.5px 'IBM Plex Mono',monospace;background:linear-gradient(135deg,#e3edff,#eaf4ff);border:1px solid #e0dcff;border-radius:10px;padding:7px 10px;color:#2b2177;margin-bottom:8px}
.nav2{display:flex;gap:6px}.nav2 button{flex:1;font-weight:600}.nav2 .pri{background:var(--acc);border-color:var(--acc);color:#fff}
.meter{display:grid;grid-template-columns:auto 1fr auto;gap:4px 8px;align-items:center;font:500 11px 'IBM Plex Mono',monospace;color:var(--muted)}
.meter .b{height:9px;border-radius:5px;background:#eceef9;overflow:hidden}.meter .b i{display:block;height:100%;border-radius:5px}
.meter b{color:var(--fg)}
.rg{margin:6px 0 2px;font-size:12px;display:flex;justify-content:space-between}
@media (max-width:760px){.wrap,.wrap.tour{grid-template-columns:1fr}.stage{height:380px}}
</style></head><body>
<div class="wrap" id="wrap"><div class="stage"><canvas id="c" aria-label="Model 3D spektrofotometer UV-1700"></canvas>
<div class="badge" id="badge">UV-1700 · MODEL 3D</div>
<div class="tools"><button data-z="0.85" title="Perbesar">+</button><button data-z="1.18" title="Perkecil">−</button><button id="rs" title="Tampilan awal">⟲</button></div>
<div class="views"><button data-v="iso" class="on">Isometrik</button><button data-v="depan">Depan</button><button data-v="samping">Kanan</button><button data-v="belakang">Belakang</button><button data-v="atas">Atas</button><button data-v="kompartemen">Kompartemen</button></div></div>
<div class="side"><div class="card tourcard"><div class="steps" id="steps"></div><div class="tt" id="tt"></div><div class="tx" id="tx"></div><div class="nav2"><button id="prev">◀ Sebelumnya</button><button id="next" class="pri">Berikutnya ▶</button></div></div>
<div class="card tourcard"><h4>Eksperimen: λ dan konsentrasi</h4>
<div class="rg"><span>λ = <b id="lv2"></b> nm</span><span id="lamp2" class="hint"></span></div><input type="range" id="lam2" min="190" max="1100" step="1">
<div class="rg"><span>C = <b id="cv2"></b> mg/L</span><span class="hint">ε·b = 0,20 L mg⁻¹ (b = 1 cm)</span></div><input type="range" id="conc" min="0" max="10" step="0.1" value="2.5">
<div class="meter" style="margin-top:6px"><span>I₀</span><span class="b"><i id="b0" style="background:#1f6fff;width:100%"></i></span><b id="v0">100</b><span>I</span><span class="b"><i id="b1" style="background:#0d9488"></i></span><b id="v1"></b><span>%T</span><span class="b"><i id="b2" style="background:#d4a017"></i></span><b id="v2"></b><span>A</span><span class="b"><i id="b3" style="background:#e11d48"></i></span><b id="v3"></b></div></div>
<div class="card"><h4>Tampilan</h4><div class="tg">
<button data-t="lid" aria-pressed="false">Buka tutup</button><button data-t="xray" aria-pressed="false">Tembus pandang</button><button data-t="expl" aria-pressed="false">Terurai</button><button data-t="beam" aria-pressed="false">Jalur cahaya</button><button data-t="labels" aria-pressed="false">Label</button><button data-t="auto" aria-pressed="false">Putar otomatis</button></div></div>
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
      this.beam = true; this.labels = false; this.auto = false; this.sel = null; this.hl = new Set(); this.beamLimit = 99; this.flow = false; this.tour = false;
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
    zoom(f) { this.dist = Math.max(38, Math.min(260, this.dist * f)); this.draw(); }
    focus(v) { this._anim = { from: [this.yaw, this.pitch, this.dist, this.target.slice()], to: v, t: 0, full: true }; this._kick(); }
    reset() { this.dist = 84; this.view('iso'); this.sel = null; if (this.opt.onPick) this.opt.onPick(null); }
    _kick() { if (!this._running) { this._running = true; this._last = performance.now(); requestAnimationFrame(this._loop); } }
    _loop(now) {
      const dt = Math.min(0.05, (now - this._last) / 1000); this._last = now;
      const ease = (cur, tgt, sp) => { const d = tgt - cur; return Math.abs(d) < 0.002 ? tgt : cur + d * Math.min(1, dt * sp); };
      this.lid = ease(this.lid, this.lidT, 5); this.expl = ease(this.expl, this.explT, 4); this.xray = ease(this.xray, this.xrayT, 5);
      if (this._anim) {
        const a = this._anim; a.t = Math.min(1, a.t + dt * 2.2); const k = a.t < 0.5 ? 2 * a.t * a.t : 1 - Math.pow(-2 * a.t + 2, 2) / 2;
        let dy = a.to[0] - a.from[0]; while (dy > 180) dy -= 360; while (dy < -180) dy += 360;
        this.yaw = a.from[0] + dy * k; this.pitch = a.from[1] + (a.to[1] - a.from[1]) * k;
        if (a.full) { this.dist = a.from[2] + (a.to[2] - a.from[2]) * k; this.target = [0, 1, 2].map(i => a.from[3][i] + (a.to[3][i] - a.from[3][i]) * k); }
        if (a.t >= 1) this._anim = null;
      }
      if (this.auto && !this._drag) this.yaw += dt * 18;
      if (this.flow && !this._anim && !this._drag && now - (this._ld || 0) < 30) { requestAnimationFrame(this._loop); return; }
      this._ld = now; this.draw();
      const busy = this.lid !== this.lidT || this.expl !== this.explT || this.xray !== this.xrayT || this._anim || (this.auto && !this._drag) || (this.flow && this.beam);
      if (busy) requestAnimationFrame(this._loop); else this._running = false;
    }
    _xf(p, part) {
      let q = p;
      if (part.tf) for (const [ax, piv, deg] of part.tf) q = ax === 'x' ? rotX(q, piv, deg * D2R) : rotY(q, piv, deg * D2R);
      if (part.grp === 'lid') { q = rotX(q, [0, 15, -2.2], -102 * this.lid * D2R); q = [q[0], q[1] + 16 * this.expl, q[2]]; }
      if (part.grp === 'cover') q = [q[0], q[1] + 16 * this.expl, q[2]];
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
        let k = 0.5 + 0.55 * Math.max(0, dot(wn, light));
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
        const y = 9.5, by = this.lid > 0.5 || this.xray > 0.5 || this.expl > 0.5;
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
        const p = it.pts; ctx.globalAlpha = it.alpha;
        ctx.beginPath(); ctx.moveTo(p[0][0], p[0][1]); for (let i = 1; i < 4; i++) ctx.lineTo(p[i][0], p[i][1]); ctx.closePath();
        ctx.fillStyle = it.fill; ctx.fill();
        if (it.alpha > 0.97) { ctx.strokeStyle = it.fill; ctx.lineWidth = 1.1; ctx.stroke(); }
        if (this.sel && it.part.id === this.sel) { ctx.strokeStyle = '#0aa5c4'; ctx.lineWidth = 1.2; ctx.stroke(); }
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
      const A0 = [['mono', [4, 12.6, -22.5]], ['lamphouse', [16, 4.5, -22]], ['d2', [19, 12, -20]], ['fan', [-18, 9.5, -22.5]], ['wi', [19, 11, -13]], ['mirror', [15.3, 12, -16]], ['slit1', [11.2, 13.5, -15]], ['grating', [3.5, 13, -21]], ['slit2', [3, 13.5, -9.2]], ['split', [3, 12.5, 3]], ['holderR', [-8.5, 13, 3]], ['holderS', [-8.5, 13, 13]], ['detS', [-20.6, 12, 13]], ['detR', [-20.6, 12, 3]], ['pcb', [-17, 5, -14]], ['lcd', [11.5, 29, 6]], ['keypad', [12, 17, 22]], ['lid', [-8.5, 17, 9]], ['card', [23.3, 12, 6]]];
      const A = this.tour ? A0.filter(([id]) => this.hl.has(id)) : A0.filter(([id]) => id !== 'mono' && id !== 'lamphouse');
      ctx.font = '600 11px IBM Plex Sans, Arial';
      const placed = [];
      const L = A.map(([id, p0]) => { const part = this.parts.find(q => q.id === id); const p = part ? this._xf(p0, part) : p0; return { id, s: this._proj(this._cam(p)) }; }).sort((a, b) => a.s[1] - b.s[1]);
      for (const { id, s } of L) {
        const name = INFO[id][0].replace(/ \(.*\)/, '');
        const w = ctx.measureText(name).width + 12; const x = s[0]; let y = s[1] - 16;
        for (let k = 0; k < 8 && placed.some(r => Math.abs(r.x - x) < (r.w + w) / 2 + 2 && Math.abs(r.y - y) < 17); k++) y -= 17;
        placed.push({ x, y, w });
        ctx.strokeStyle = 'rgba(20,18,58,.5)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(s[0], s[1]); ctx.lineTo(x, y + 4); ctx.stroke();
        ctx.fillStyle = id === this.sel ? '#1f6fff' : 'rgba(255,255,255,.95)';
        ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x - w / 2, y - 11, w, 16, 8) : ctx.rect(x - w / 2, y - 11, w, 16); ctx.fill();
        ctx.fillStyle = id === this.sel ? '#fff' : '#0b1b3a'; ctx.textAlign = 'center'; ctx.fillText(name, x, y + 1);
        ctx.fillStyle = '#0aa5c4'; ctx.beginPath(); ctx.arc(s[0], s[1], 2.5, 0, 7); ctx.fill();
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
var INIT = Object.assign({tour: false, lam: 550, abs: 0, status: "READY", solColor: "#8e2c8f", lampSwitch: 340.8, xray: true, lid: false, compact: false, auto: false}, /*__INIT__*/null || {});
(function(){
  var SL = window.SL, $ = function(s){return document.querySelector(s)};
  if (INIT.compact) $('#wrap').classList.add('compact');
  var m = new SL.Model3D($('#c'), { onPick: function(i){ $('#ti').textContent = i ? i.title : 'Komponen'; $('#info').textContent = i ? i.text : 'Klik komponen untuk penjelasan.'; } });
  m.lampSwitch = INIT.lampSwitch; m.abs = INIT.abs; m.status = INIT.status; m.solColor = INIT.solColor;
  function sync(){ var st = m.state(); document.querySelectorAll('[data-t]').forEach(function(b){ b.setAttribute('aria-pressed', String(!!st[b.dataset.t])); }); }
  function setLam(l){ m.setLam(l); $('#lv').textContent = l; $('#lam').value = l; if ($('#lv2')) { $('#lv2').textContent = l; $('#lam2').value = l; $('#lamp2').textContent = (l < m.lampSwitch ? 'Lampu D₂ · ' : 'Lampu WI · ') + (l < 400 ? 'UV' : l <= 780 ? 'tampak' : 'NIR'); } $('#sw').style.background = SL.lamColor(l); $('#lamp').textContent = (l < m.lampSwitch ? 'Lampu D2 aktif' : 'Lampu WI aktif') + ' · ' + (l < 400 ? 'UV' : l <= 780 ? 'sinar tampak' : 'NIR'); }
  var TOUR = [
    { t: 'Lampu D₂ dan WI', ids: ['d2', 'wi', 'lamphouse'], lim: 0, v: [-38, 42, 80, [17, 8, -16]],
      f: 'Menyediakan radiasi kontinu UV (D₂) dan tampak–inframerah dekat (WI).', p: 'Pelepasan listrik pada gas deuterium dan pijaran filamen wolfram menghasilkan spektrum kontinu.', m: 'Lampu deuterium berjendela kuarsa dan lampu wolfram-iodin (halogen).', h: 'Intensitas lampu menentukan derau, dan lampu berganti otomatis di 340,8 nm.' },
    { t: 'Cermin pemilih dan celah masuk', ids: ['mirror', 'slit1'], lim: 1, v: [-30, 50, 74, [13, 9, -16]],
      f: 'Memilih berkas lampu aktif dan membentuk berkas sempit menuju monokromator.', p: 'Pemantulan oleh cermin dan pembatasan berkas oleh celah.', m: 'Cermin berlapis logam reflektif dan celah logam tetap.', h: 'Menentukan energi yang masuk dan lebar pita spektral.' },
    { t: 'Kisi difraksi (monokromator)', ids: ['grating', 'mono'], lim: 2, v: [-20, 56, 74, [5, 9, -17]],
      f: 'Memisahkan cahaya polikromatik menjadi panjang gelombang tunggal.', p: 'Difraksi menurut nλ = d (sin α + sin β); sudut kisi menentukan λ.', m: 'Kisi difraksi holografik dalam rumah monokromator gelap.', h: 'Sudut kisi menetapkan λ pengukuran dan menekan sinar sesat.' },
    { t: 'Celah keluar', ids: ['slit2'], lim: 3, v: [-26, 56, 70, [3, 9, -10]],
      f: 'Meneruskan pita sempit di sekitar λ terpilih.', p: 'Seleksi spasial berkas terdispersi pada bidang fokus.', m: 'Celah logam tetap.', h: 'Menentukan resolusi spektral (SBW) dan jumlah cahaya ke detektor.' },
    { t: 'Pembagi berkas', ids: ['split', 'mirror2'], lim: 5, v: [-46, 46, 68, [3, 9, 8]],
      f: 'Membagi berkas menjadi berkas referensi dan berkas sampel.', p: 'Pemantulan dan transmisi parsial pada permukaan semi-reflektif (double-beam).', m: 'Cermin pembagi berkas semi-reflektif dan cermin pengarah.', h: 'Mengoreksi fluktuasi lampu sehingga baseline stabil.' },
    { t: 'Kompartemen sampel dan kuvet', ids: ['holderR', 'holderS', 'cuvR', 'cuvS'], lim: 9, v: [-16, 52, 66, [-8, 9, 8]],
      f: 'Menempatkan blanko dan sampel pada jalur berkas.', p: 'Analit menyerap radiasi menurut hukum Lambert–Beer, A = ε · b · C.', m: 'Kuvet kuarsa jalur 1 cm pada dudukan hitam.', h: 'Sumber utama sinyal: A sebanding dengan konsentrasi.' },
    { t: 'Detektor fotodioda', ids: ['detR', 'detS'], lim: 9, v: [-32, 40, 72, [-18, 9, 8]],
      f: 'Mengubah intensitas cahaya menjadi arus listrik.', p: 'Efek fotolistrik pada sambungan p–n semikonduktor.', m: 'Fotodioda silikon (referensi dan sampel).', h: 'Mengukur I₀ dan I untuk memperoleh T = I/I₀.' },
    { t: 'Pengolahan sinyal', ids: ['pcb', 'fan'], lim: 9, v: [-30, 42, 90, [-16, 4, -12]],
      f: 'Menguatkan, mendigitalkan, dan mengoreksi sinyal detektor.', p: 'Penguat → ADC → A = −log₁₀ T dikurangi baseline.', m: 'Papan elektronik (PCB) bermikroprosesor.', h: 'Menghasilkan absorbansi terkoreksi yang dibaca atau dikirim ke PC.' },
    { t: 'Layar dan keluaran', ids: ['lcd', 'keypad'], lim: 9, v: [-24, 30, 92, [12, 18, 10]],
      f: 'Menampilkan λ, absorbansi atau %T, dan lampu aktif.', p: 'Pengendali LCD menampilkan data digital; PC menerima data lewat RS-232C.', m: 'Layar LCD dan papan tombol membran.', h: 'Hasil akhir yang dipakai untuk kurva kalibrasi.' }
  ];
  var EPSB = 0.2, cur = 0, seen = {};
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
    m.hl = new Set(s.ids); m.beamLimit = s.lim; m.focus(s.v); m.sel = null;
    $('#tt').textContent = (cur + 1) + '. ' + s.t; $('#tx').innerHTML = '<div class="kv"><b>Fungsi</b><span>' + s.f + '</span><b>Prinsip</b><span>' + s.p + '</span><b>Material / jenis</b><span>' + s.m + '</span><b>Hubungan dengan pengukuran</b><span>' + s.h + '</span></div>';
    document.querySelectorAll('#steps button').forEach(function (b, k) { b.classList.toggle('on', k === cur); b.classList.toggle('seen', k !== cur && !!seen[k]); });
    $('#prev').disabled = cur === 0; $('#next').textContent = cur === TOUR.length - 1 ? 'Ulangi tur ↺' : 'Berikutnya ▶';
    m.labels = true; sync();
  }
  if (INIT.tour) {
    $('#wrap').classList.add('tour'); $('#badge').textContent = 'BAGIAN DALAM UV-1700 · PRINSIP KERJA'; m.tour = true; m.flow = true;
    var sb = $('#steps'); TOUR.forEach(function (_, k) { var b = document.createElement('button'); b.textContent = k + 1; b.title = TOUR[k].t; b.addEventListener('click', function () { goStep(k); }); sb.appendChild(b); });
    $('#prev').addEventListener('click', function () { goStep(cur - 1); }); $('#next').addEventListener('click', function () { goStep(cur === TOUR.length - 1 ? 0 : cur + 1); });
    m.toggle('expl', true); $('#conc').addEventListener('input', meter); $('#lam2').addEventListener('input', function (e) { setLam(+e.target.value); });
    INIT.xray = true;
  }
  m.toggle('beam', true); if (INIT.xray) m.toggle('xray', true); if (INIT.lid) m.toggle('lid', true); if (INIT.auto) m.toggle('auto', true);
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
:root{--fg:#0b1b3a;--muted:#5b6386;--line:#e1e4f2;--acc:#1f6fff;--ok:#0d9488;--bad:#e11d48}
*{box-sizing:border-box}html,body{margin:0;background:transparent;color:var(--fg);font:13px/1.45 'IBM Plex Sans',system-ui,sans-serif}
.top{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:10px}
.stage{position:relative;height:575px;border-radius:16px;overflow:hidden;border:1px solid var(--line);background:#eef0f8}
canvas#c{width:100%;height:100%;display:block;cursor:grab;touch-action:none}
.views{position:absolute;left:8px;top:8px;display:flex;flex-wrap:wrap;gap:3px;background:rgba(255,255,255,.92);padding:4px;border-radius:10px;max-width:calc(100% - 170px)}
.views button{border:0;background:none;color:var(--muted);font:500 12px 'IBM Plex Sans',sans-serif;padding:4px 8px;border-radius:7px;cursor:pointer}
.views button.on{background:var(--acc);color:#fff}
.tools{position:absolute;left:8px;bottom:8px;display:flex;gap:4px;flex-wrap:wrap}
.tools button{border:1px solid var(--line);background:rgba(255,255,255,.94);border-radius:8px;padding:4px 9px;cursor:pointer;font:500 12px 'IBM Plex Sans',sans-serif}
.tools button[aria-pressed=true]{background:#e3edff;border-color:var(--acc);color:#1f4fbf}
.hud{display:none;position:absolute;right:8px;top:8px;background:rgba(20,18,58,.9);color:#fff;border-radius:12px;padding:8px 12px;font:600 11px 'IBM Plex Mono',monospace;text-align:right;min-width:150px}
.hud .sc{font-size:22px;line-height:1.1}.hud .tm{color:#63dcf2}
.bar{height:5px;background:rgba(255,255,255,.2);border-radius:3px;margin-top:5px;overflow:hidden}.bar i{display:block;height:100%;background:linear-gradient(90deg,#22c55e,#a3e635);width:0}
.toast{position:absolute;left:50%;top:50px;transform:translateX(-50%);max-width:86%;padding:9px 14px;border-radius:12px;font-weight:500;box-shadow:0 10px 26px -10px rgba(0,0,0,.45);opacity:0;transition:opacity .25s;pointer-events:none;text-align:center}
.toast.show{opacity:1}.toast.info{background:#0b1b3a;color:#fff}.toast.ok{background:#0d9488;color:#fff}.toast.err{background:#e11d48;color:#fff}.toast.warn{background:#f59e0b;color:#1f1300}
.modal{position:absolute;inset:0;background:rgba(20,18,58,.45);display:none;align-items:center;justify-content:center;z-index:5}
.modal.show{display:flex}
.mcard{background:#fff;border-radius:18px;padding:20px 22px;width:min(520px,92%);box-shadow:0 30px 60px -20px rgba(0,0,0,.5);max-height:94%;overflow:auto}
.mcard h2{margin:0 0 6px;font:700 20px 'IBM Plex Sans Condensed','IBM Plex Sans',sans-serif}.mcard p{margin:4px 0 10px;color:var(--muted)}
.mcard ol{padding-left:18px;margin:6px 0 12px}.mcard li{margin:4px 0}
.mcard label{display:block;font-size:12px;color:var(--muted);margin:8px 0 3px}
.mcard input,.mcard textarea{width:100%;font:inherit;padding:7px 9px;border:1px solid var(--line);border-radius:9px}
.mcard .row3{display:grid;grid-template-columns:2fr 1fr 1fr;gap:8px}
.big{font:600 14px 'IBM Plex Sans',sans-serif;border:0;border-radius:11px;padding:10px 16px;background:linear-gradient(135deg,#1f6fff,#0aa5c4);color:#fff;cursor:pointer;width:100%;margin-top:12px}
.side{display:flex;flex-direction:column;gap:8px;height:575px;overflow:auto;padding-right:2px}
.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:10px}
.card h4{margin:0 0 7px;font:600 11px 'IBM Plex Mono',monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);display:flex;justify-content:space-between;align-items:center}
.chips{display:flex;flex-wrap:wrap;gap:4px}
.chip{font:500 11px 'IBM Plex Mono',monospace;padding:3px 8px;border-radius:99px;background:#f1f2f8;color:#6b7194}
.chip.on{background:#dff7f1;color:#0b7a6e}.chip.mid{background:#fff4d6;color:#9a6a00}
.now{background:linear-gradient(135deg,#e3edff,#eaf4ff);border-color:#d9d4ff}
.now .sec{font:600 11px 'IBM Plex Mono',monospace;color:#0aa5c4;letter-spacing:.06em}
.now .tx{font-weight:600;font-size:14px;margin:4px 0 8px;color:#0b1b3a}
.now .why{color:var(--muted);font-size:12px;margin-bottom:8px}
.go{font:600 13px 'IBM Plex Sans',sans-serif;border:0;border-radius:10px;padding:9px 12px;background:linear-gradient(135deg,#1f6fff,#0aa5c4);color:#fff;cursor:pointer;width:100%}
.go:disabled{opacity:.5;cursor:wait}
.ik{max-height:none}.ik .grp{font:600 11px 'IBM Plex Mono',monospace;color:#1f4fbf;margin:10px 0 4px;letter-spacing:.05em}
.ik .it{display:flex;gap:8px;padding:5px 6px;border-radius:8px;font-size:12px;align-items:flex-start}
.ik .it .d{flex:none;width:20px;height:20px;border-radius:50%;display:grid;place-items:center;font:600 10px 'IBM Plex Mono',monospace;background:#e3edff;color:#1f6fff}
.ik .it.done{color:#6b7280}.ik .it.done .d{background:#0d9488;color:#fff}
.ik .it.cur{background:#f5f3ff;box-shadow:inset 0 0 0 1px #9cc0ff;color:#0b1b3a;font-weight:600}
.ik .rk{font:500 9px 'IBM Plex Mono',monospace;color:#9a6a00;background:#fff4d6;border-radius:5px;padding:1px 5px;margin-left:4px}
details.man summary{cursor:pointer;font:600 11px 'IBM Plex Mono',monospace;letter-spacing:.08em;color:var(--muted);text-transform:uppercase}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:6px}
button.a{font:500 12px 'IBM Plex Sans',sans-serif;border:1px solid var(--line);background:#fff;color:var(--fg);border-radius:9px;padding:7px 8px;cursor:pointer;text-align:left}
button.a:hover{border-color:var(--acc)}button.a:disabled{opacity:.45;cursor:not-allowed}
select,input[type=number],input[type=text]{width:100%;font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:9px;background:#fff;color:var(--fg)}
.seg{display:flex;gap:4px;margin:6px 0}.seg label{flex:1;text-align:center;border:1px solid var(--line);border-radius:9px;padding:5px;cursor:pointer;font-size:12px}
.seg input{display:none}.seg label:has(input:checked){border-color:var(--acc);background:#e3edff;color:#1f4fbf;font-weight:600}
.bottom{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:10px}
.pip{position:absolute;right:8px;bottom:8px;width:35%;min-width:250px;max-width:460px;background:#111214;border-radius:12px;padding:5px 6px 7px;box-shadow:0 14px 34px -10px rgba(0,0,0,.65),inset 0 0 0 2px #2a2c31;z-index:3;transition:width .25s}
.pip.big{width:70%;max-width:none}.pip.hide{display:none}
.pip .ph{display:flex;justify-content:space-between;align-items:center;color:#9aa1b5;font:500 10px 'IBM Plex Mono',monospace;letter-spacing:.1em;padding:0 2px 4px}
.pip .ph button{border:0;background:#2a2c31;color:#cfd3e4;border-radius:6px;padding:2px 7px;font:500 10px 'IBM Plex Mono',monospace;cursor:pointer;margin-left:4px}
.pip canvas{width:100%;display:block;border-radius:4px;background:#000;cursor:pointer}
.prog{height:6px;border-radius:4px;background:#e6e8f5;overflow:hidden;margin:2px 0 8px}.prog i{display:block;height:100%;background:linear-gradient(90deg,#1f6fff,#22c55e);width:0;transition:width .3s}
.pcwrap{background:#111214;border-radius:16px;padding:10px 10px 14px;box-shadow:inset 0 0 0 2px #26282d}
.pchead{color:#9aa1b5;font:500 11px 'IBM Plex Mono',monospace;letter-spacing:.1em;margin:0 0 6px 4px;display:flex;justify-content:space-between}
canvas#pc{width:100%;display:block;border-radius:4px;background:#000;cursor:pointer}
textarea{width:100%;height:110px;font:12px 'IBM Plex Mono',monospace;border:1px solid var(--line);border-radius:9px;padding:6px;resize:vertical}
.log{font:11.5px 'IBM Plex Mono',monospace;color:#4a5072;max-height:170px;overflow:auto;white-space:pre-wrap}
.log .e{color:#be123c}.log .o{color:#0b7a6e}.log .w{color:#9a6a00}
.prints{display:flex;gap:6px;flex-wrap:wrap}.prints a{display:block;border:1px solid var(--line);border-radius:6px;overflow:hidden;width:72px;text-decoration:none;color:var(--muted);font-size:10px;text-align:center}
.prints img{width:100%;display:block}
.row{display:flex;gap:6px;align-items:center}.hint{color:var(--muted);font-size:12px}
@media (max-width:980px){.top,.bottom{grid-template-columns:1fr}.side{height:auto}.stage{height:520px}.pip{width:62%}}
</style></head><body>
<div class="top">
  <div class="stage"><canvas id="c" aria-label="Meja praktikum 3D"></canvas>
    <div class="views">
      <button data-v="meja" class="on">Meja</button><button data-v="alat">UV-1700</button><button data-v="kompartemen">Kompartemen</button>
      <button data-v="isi">Labu & kuvet</button><button data-v="komputer">Komputer</button><button data-v="belakang">Kabel & UPS</button>
    </div>
    <div class="hud"><div>SKOR PROSEDUR</div><div class="sc" id="score">100</div><div class="tm" id="clock">00:00</div><div class="bar"><i id="prog"></i></div></div>
    <div class="tools"><button data-t="xray" aria-pressed="false">Tembus pandang</button><button data-t="sound" aria-pressed="true">Suara</button><button data-t="help">Bantuan</button><button data-t="pip">Layar komputer</button></div>
    <div class="pip" id="pip"><div class="ph"><span>MONITOR · <span id="pcst" style="color:#8b93b8">MATI</span></span><span><button data-t="pipbig" id="pipbig">Perbesar</button><button data-t="pipx">Tutup</button></span></div><canvas id="pc" width="1040" height="610"></canvas></div>
    <div class="toast" id="toast"></div>
    <div class="modal show" id="welcome"><div class="mcard">
      <h2>Praktikum Virtual UV-1700</h2>
      <p>Simulasi penggunaan Spektrofotometer UV-Vis Shimadzu UV-1700 PharmaSpec mengikuti Instruksi Kerja IK/LAB.INST/6 Laboratorium Instrumentasi.</p>
      <ol><li>Isi nama, kelas, dan kelompok (dipakai untuk nama file dan Log Book).</li>
      <li>Baca kotak <b>Langkah sekarang</b> di kanan, lalu tekan <b>Kerjakan langkah ini</b>. Meja 3D dan layar komputer menunjukkan hasilnya.</li>
      <li>Ingin mencoba sendiri? Buka <b>Kontrol manual</b>; urutan yang salah mengurangi skor.</li></ol>
      <div class="row3"><div><label>Nama praktikan</label><input id="nm" placeholder="Nama lengkap"></div><div><label>Kelas</label><input id="kl" placeholder="mis. 2A"></div><div><label>Kelompok</label><input id="kp" placeholder="mis. 2" value="2"></div></div>
      <button class="big" id="startbtn">Mulai praktikum</button></div></div>
    <div class="modal" id="logm"><div class="mcard">
      <h2>Log Book Alat UV-1700</h2><p>Catat semua kejadian selama penggunaan dan tanda tangani dengan nama jelas (IK C.1).</p>
      <label>Tanggal</label><input id="lg_t" readonly><label>Nama pengguna</label><input id="lg_n">
      <label>Kejadian / kondisi alat</label><textarea id="lg_k" style="height:80px"></textarea>
      <label>Tanda tangan (ketik nama jelas)</label><input id="lg_s">
      <button class="big" id="lg_ok">Simpan ke Log Book</button></div></div>
  </div>
  <div class="side">
    <div class="card now"><div class="sec" id="nowsec">LANGKAH SEKARANG</div><div class="tx" id="nowtx"></div><div class="prog"><i id="prog2"></i></div><div class="why" id="nowwhy"></div>
      <button class="go" id="gobtn">Kerjakan langkah ini</button></div>
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
function rotZ(p, piv, ang) { const s = Math.sin(ang), c = Math.cos(ang); const x = p[0] - piv[0], y = p[1] - piv[1]; return [piv[0] + x * c - y * s, piv[1] + x * s + y * c, p[2]]; }
function rotY(p, piv, ang) { const s = Math.sin(ang), c = Math.cos(ang); const x = p[0] - piv[0], z = p[2] - piv[2]; return [piv[0] + x * c + z * s, p[1], piv[2] - x * s + z * c]; }

/* ================= larutan & identitas ================= */
const SOL = { blank: { id: 'blank', label: 'Blanko (' + AN.pelarut + ')', short: 'Blanko', c: 0, col: '#dff4f8' } };
AN.std.forEach((c, i) => { SOL['s' + i] = { id: 's' + i, label: 'Standar ' + (i + 1) + ' · ' + fx(c, c < 1 ? 2 : 1) + ' ' + INIT.unit, short: 'Std ' + (i + 1), c, std: true, idx: i }; });
SOL.X = { id: 'X', label: 'Sampel X (tak diketahui)', short: 'Sampel X', c: INIT.hidden, sample: true };
const cmax = Math.max(...AN.std, INIT.hidden);
Object.values(SOL).forEach(s => { if (s.c > 0) s.col = mix('#eef6f8', AN.warna, Math.min(1, 0.25 + 0.75 * s.c / cmax)); });
const SOL_KEYS = Object.keys(SOL);
const ID = { nama: '', kelas: 'Kelas', kel: 'Kelompok' };
const today = new Date(); const dstr = String(today.getDate()).padStart(2, '0') + '-' + String(today.getMonth() + 1).padStart(2, '0') + '-' + today.getFullYear();
const fname = () => dstr + '_' + (ID.kelas || 'Kelas').replace(/\s+/g, '') + '_' + (ID.kel || 'Kelompok').replace(/\s+/g, '');
const scanRange = () => { const lo = AN.lam_ref < 320 ? 200 : Math.max(190, Math.round((AN.lam_ref - 150) / 10) * 10); return [lo, Math.min(1100, lo + 320)]; };

/* ================= keadaan ================= */
const RACK = { R: [-31, 0.2, 4], S: [-31, 0.2, 10] }, CELL = { R: [-8.5, 6.6, 3], S: [-8.5, 6.6, 13] }, DRY = { R: [-16, -1.25, 30], S: [-11, -1.25, 30] };
const ST = {
  cab: [0, 0, 0, 0, 0, 0, 0, 0, 0], cabled: false, ups: false, pc: 0, pcBoot: 0, printer: false, inst: 'off', initLines: [], warmLeft: 0, warmSkipped: false,
  pcctrl: false, sw: false, connected: false, lid: 0, xray: 0, xrayT: 0, xrayUser: false, autoX: true, sound: true, lam: AN.lam_ref, lastA: 0, lcdFold: 0,
  silica: 'comp', silicaPos: [-8.5, 6, 8], logbook: null, tidy: false, paper: 0, chairZ: 66, wash: null, pour: null, stream: null,
  cuv: { R: { where: 'rak', pos: RACK.R.slice(), content: null, level: 0, dirty: false, smudge: 0, flip: 0, wet: false }, S: { where: 'rak', pos: RACK.S.slice(), content: null, level: 0, dirty: false, smudge: 0, flip: 0, wet: false } },
  zero: null, win: null, menu: null, dlg: null, busyMsg: null, ptab: 'std',
  spectrum: null, peaks: null, smethod: null, pmethod: null, stdTable: [], unkTable: [], files: [], prints: [],
  score: 100, errs: new Set(), done: new Set(), busy: false, tissue: null, beam: 0, t0: performance.now()
};

/* ================= geometri ================= */
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
box('monitor', 'room', -66, -50, -1.4, -0.2, -22, -10, '#141518'); box('monitor', 'room', -60, -56, -0.2, 13, -18, -15.8, '#1b1c20');
box('monitor', 'room', -82, -34, 12, 40, -17, -14.6, '#0e0f12', { cell: 8 });
box('screen', 'room', -81, -35, 13.6, 39.2, -14.6, -14.4, '#000000', { bias: -3, dyn: 'screen', dynFace: 'f', text: { face: 'f', w: 1040, h: 610, screen: true } });
box('monitor', 'room', -37.6, -36.6, 12.6, 13.2, -14.6, -14.3, '#222', { bias: -3.3, dyn: 'monled' });
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
  box('bottle', 'wash', wx - 1.6, wx + 1.6, by + 3.4, by + 9.6, wz + 3.34, wz + 3.5, '#ffffff', { bias: -3, cell: 99, text: { face: 'f', w: 110, h: 130, lines: [['AKUADES', 55, 40, 'center', '700 17px Arial', '#1d4ed8'], ['H\u2082O', 55, 72, 'center', '600 22px Arial', '#1d4ed8'], ['bebas ion', 55, 104, 'center', '500 11px Arial', '#475569']] } });
}
const WASHTIP = [WASH0[0] + 4.9, WASH0[1] + 19.4 + 4.11, WASH0[2]];
// gelas piala buangan (berparuh), etanol dan aseton
cyl('beaker', 'room', -40.5, 33.5, 3.2, -1.4, 6.5, '#dfeff5', { alpha: 0.4, open: true }, 16); cyl('beaker', 'room', -40.5, 33.5, 3.0, -1.3, 1.6, '#c7dce6', { alpha: 0.55 }, 16);
box('beaker', 'room', -37.7, -36.5, 5.6, 6.5, 32.9, 34.1, '#dfeff5', { alpha: 0.5, cell: 99 });
box('logbook', 'room', -104, -88, -1.4, -0.4, 20, 33, '#1e3a8a', { cell: 99, faceCol: { t: '#22409a' }, text: { face: 't', w: 120, h: 100, lines: [['LOG BOOK', 60, 40, 'center', '700 15px Arial', '#e0e7ff'], ['UV-1700', 60, 60, 'center', '600 12px Arial', '#c7d2fe']] } });
box('logbook', 'room', -86, -85.4, -0.6, -0.1, 21, 31, '#111', {});
box('drymat', 'room', -20, -7, -1.4, -1.25, 26, 34, '#ffffff', { cell: 99 });
// labu takar 50 mL: alas datar, badan bulat, leher panjang bergaris tera, sumbat biru
const FLY = -1.4, FS = 0.74;
const FPROF = [[0, 0], [2.5, 0], [3.4, 0.5], [4.0, 1.6], [4.25, 3.1], [4.0, 4.7], [3.2, 6.1], [2.0, 7.3], [1.1, 8.3], [0.78, 9.4], [0.74, 15.2], [0.9, 15.7], [0.98, 16.2]].map(([r, y]) => [r * FS, FLY + y * FS]);
const FMARK = FLY + 12.9 * FS, FTOP = FLY + 16.2 * FS;
const FL = {}; SOL_KEYS.forEach((k, i) => { FL[k] = { x: -83 + (i % 4) * 8, z: 21 + Math.floor(i / 4) * 10 }; });
SOL_KEYS.forEach(k => { const f = FL[k], g = 'fl_' + k;
  lathe('flask', g, f.x, f.z, FPROF, '#e9f4f8', { alpha: 0.34 }, 10);
  lathe('flask', g, f.x, f.z, clipProf(FPROF, FMARK, 0.13), SOL[k].col, { alpha: 0.9, topCol: mix(SOL[k].col, '#ffffff', 0.35) }, 10);
  lathe('flask', g, f.x, f.z, [[0.78 * FS + 0.02, FMARK - 0.05], [0.78 * FS + 0.02, FMARK + 0.1]], '#1f2937', {}, 10);
  box('flask', g, f.x - 1.5, f.x + 1.5, FLY + 1.1, FLY + 3.5, f.z + 3.18, f.z + 3.3, '#ffffff', { bias: -3, cell: 99, text: { face: 'f', w: 90, h: 72, lines: [[k === 'X' ? 'X' : fx(SOL[k].c, 1), 45, 38, 'center', '700 30px Arial', '#111827'], [INIT.unit, 45, 62, 'center', '600 15px Arial', '#374151']] } });
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
  beaker: ['Gelas piala buangan', 'Menampung bilasan dan larutan bekas.'], etanol: ['Etanol', 'Pembilas pertama saat mengeringkan kuvet (IK D.1).'], aseton: ['Aseton', 'Pembilas terakhir; cepat menguap sehingga kuvet kering sempurna (IK D.1).'],
  logbook: ['Log Book alat', 'Catat semua kejadian dan tanda tangani dengan nama jelas (IK C.1).'], drymat: ['Alas tisu pengering', 'Tempat kuvet dibalik hingga kering.'],
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
  kering: [-20, 45, 70, [-20, 2, 22]], log: [-15, 50, 70, [-96, 0, 20]], isi: [-18, 32, 128, [-52, 5, 18]], kursi: [-24, 24, 210, [-24, -26, 50]] };
const CAM = { yaw: -14, pitch: 26, dist: 270, target: [-18, 6, 0] };
let camAnim = null, viewName = 'meja';
function setView(name) { const v = VIEWS[name]; if (!v || name === viewName && !camAnim) { if (v) viewName = name; return; } viewName = name; camAnim = { from: [CAM.yaw, CAM.pitch, CAM.dist, CAM.target.slice()], to: v, t: 0 };
  $$('.views button').forEach(b => b.classList.toggle('on', b.dataset.v === name)); dirty = true; }
function camXf(p) { const t = CAM.target; const x = p[0] - t[0], y = p[1] - t[1], z = p[2] - t[2];
  const cy = Math.cos(-CAM.yaw * D2R), sy = Math.sin(-CAM.yaw * D2R); const x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
  const cp = Math.cos(CAM.pitch * D2R), sp = Math.sin(CAM.pitch * D2R); return [x1, y * cp - z1 * sp, y * sp + z1 * cp]; }
let W = 800, H = 600, CX = 400, CY = 300, FOC = 800;
function proj(c) { const zc = CAM.dist - c[2]; return [CX + FOC * c[0] / zc, CY - FOC * c[1] / zc, zc]; }
function xf(p, part) {
  let q = p;
  if (part.tf) for (const [ax, piv, deg] of part.tf) q = ax === 'x' ? rotX(q, piv, deg * D2R) : ax === 'z' ? rotZ(q, piv, deg * D2R) : rotY(q, piv, deg * D2R);
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
    const it = { part, pts: cs.map(proj), depth: CAM.dist - cen[2] + part.bias + (part.grp === 'inner' || part.grp === 'comp' ? 0.35 : 0) + (facing ? 0 : 0.02), fill: shade(col, lk, emi), alpha, facing };
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
    if (selId && it.part.id === selId) { ctx.strokeStyle = '#0aa5c4'; ctx.lineWidth = 1.2; ctx.stroke(); }
    ctx.globalAlpha = 1;
    if (it.part.text && it.facing) faceText(it.part, p);
    if (it.facing && it.alpha > 0.3 && !it.part.bg) drawn.push({ id: it.part.id, p });
  }
}
function faceText(part, p) {
  const t = part.text, TL = p[3], TR = p[2], BL = p[0]; const dpr = cv.width / Math.max(1, W);
  ctx.save(); ctx.setTransform((TR[0] - TL[0]) / t.w * dpr, (TR[1] - TL[1]) / t.w * dpr, (BL[0] - TL[0]) / t.h * dpr, (BL[1] - TL[1]) / t.h * dpr, TL[0] * dpr, TL[1] * dpr);
  if (t.screen) ctx.drawImage(pcc, 0, 0, t.w, t.h); else if (t.lcd) drawLCD(ctx);
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
const F = (w, s) => (w || 400) + ' ' + (s || 12) + 'px Segoe UI, IBM Plex Sans, Arial';
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
const tweens = [];
function tween(dur, fn) { return new Promise(res => tweens.push({ t0: performance.now(), dur: dur * 1000, fn, res })); }
const ease = t => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
const wait = s => tween(s, () => { });
let audio = null;
function beep(f, d) { if (!ST.sound) return; try { audio = audio || new (window.AudioContext || window.webkitAudioContext)(); const o = audio.createOscillator(), g = audio.createGain(); o.frequency.value = f || 880; g.gain.value = 0.05; o.connect(g); g.connect(audio.destination); o.start(); o.stop(audio.currentTime + (d || 0.12)); } catch (e) { } }
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
async function washBottle(c) {
  const p = c.pos, T = [p[0] - 0.2, 11.2, p[2]], th = -72, base = WASHTIP.slice();
  const tipRel = rotZ(WASHTIP, WASH0, th * D2R).map((v, i) => v - WASH0[i]);
  const F = [T[0] - tipRel[0], T[1] - tipRel[1], T[2] - tipRel[2]], d = [F[0] - WASH0[0], F[1] - WASH0[1], F[2] - WASH0[2]];
  await tween(0.7, t => { const e = ease(t); ST.wash = { ang: th * e, d: [d[0] * e, d[1] * e + 6 * Math.sin(Math.PI * e), d[2] * e] }; });
  ST.wash = { ang: th, d };
  await tween(1.0, t => { const tip = xf(WASHTIP, { grp: 'wash' }); ST.stream = { a: tip, b: [p[0], 5.2 + 0.4 * Math.sin(t * 40), p[2]], col: '#a8d8ff', w: 0.3 }; c.level = Math.max(c.level, 0.3 * Math.sin(Math.PI * t)); });
  ST.stream = null; beep(500, 0.05);
  await tween(0.6, t => { const e = 1 - ease(t); ST.wash = { ang: th * e, d: [d[0] * e, d[1] * e, d[2] * e] }; });
  ST.wash = null; if (!c.content) c.level = 0;
}
async function pourFlask(sid, c, rinse) {
  const f = FL[sid], th = -104, mouth = [f.x, FLY + 16.2 * FS, f.z], pv = [f.x, FLY, f.z], p = c.pos;
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
    await washBottle(c);
    log('Kuvet ' + (k === 'R' ? 'referensi' : 'sampel') + ' dibilas akuades dari botol semprot, dibilas dengan ' + SOL[sid].short + ', lalu diisi ±¾ volume.');
    await pourFlask(sid, c, true); await tween(0.4, t => { c.level = 0.85 * (1 - t); }); c.content = null; c.level = 0;
    await pourFlask(sid, c, false); c.dirty = true; c.smudge = 0; c.wet = false;
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
  $('#lg_s').value = ID.nama; m.classList.add('show'); setView('log'); logResolve = res; }); }
let logResolve = null;

/* ================= panel ================= */
function sync() {
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
  if (st) { $('#nowsec').textContent = st.sec.toUpperCase() + ' · ' + (st.no ? 'LANGKAH ' + st.no : 'LANGKAH') + (st.rk ? ' (REK)' : '') + (st.prog ? ' · ' + st.prog() : '');
    $('#nowtx').textContent = st.t; $('#nowwhy').textContent = ST.busy ? (ST.busyMsg || 'Sedang dikerjakan… perhatikan animasi 3D dan layar komputer.') : 'Tekan tombol di bawah; atau coba sendiri lewat Kontrol manual.';
    $('#gobtn').textContent = st.multi ? 'Kerjakan (' + st.prog() + ')' : 'Kerjakan langkah ini'; $('#prog2').style.width = (100 * ci / STEPS.length) + '%'; }
  else { $('#nowsec').textContent = 'SELESAI'; $('#nowtx').textContent = 'Seluruh langkah Instruksi Kerja selesai. '; $('#nowwhy').textContent = 'Salin data ke Kurva Kalibrasi untuk mengolah hasil.'; $('#gobtn').textContent = 'Selesai'; }
  $('#gobtn').disabled = ST.busy || !st;
  let html = '', last = '';
  STEPS.forEach((s, i) => { if (s.sec !== last) { html += '<div class="grp">' + s.sec + '</div>'; last = s.sec; }
    const d = isDone(s); html += '<div class="it ' + (d ? 'done' : (i === ci ? 'cur' : '')) + '" data-i="' + i + '"><span class="d">' + (d ? '✓' : (s.no || '•')) + '</span><span>' + s.t + (s.rk ? '<span class="rk">REK</span>' : '') + (s.multi && !d ? ' <b>(' + s.prog() + ')</b>' : '') + '</span></div>'; });
  $('#iklist').innerHTML = html; const nd = STEPS.filter(isDone).length; $('#ikn').textContent = nd + '/' + STEPS.length; $('#prog').style.width = (100 * nd / STEPS.length) + '%';
  $$('button.a').forEach(b => { b.disabled = ST.busy && !['copy', 'csv'].includes(b.dataset.act); });
  $('#score').textContent = ST.score; $('#ident').textContent = (ID.nama || 'Praktikan') + ' · ' + ID.kelas + ' · Kel. ' + ID.kel;
  const [lo, hi] = scanRange(); $('#scaninfo').textContent = 'Rentang scan: ' + lo + '–' + hi + ' nm · λ referensi metode ' + AN.lam_ref + ' nm.';
  if (ci >= 0) { const el = $('#iklist .cur'); if (el && el.scrollIntoView && ST.autoScroll) el.scrollIntoView({ block: 'nearest' }); }
  dirty = true;
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
  for (let i = tweens.length - 1; i >= 0; i--) { const tw = tweens[i]; const t = Math.min(1, (now - tw.t0) / tw.dur); tw.fn(t); dirty = true; if (t >= 1) { tweens.splice(i, 1); tw.res(); } }
  const e = (cur, tgt, sp) => { const d = tgt - cur; return Math.abs(d) < 0.002 ? tgt : cur + d * Math.min(1, dt * sp); };
  const nx = e(ST.xray, ST.xrayT, 5); if (nx !== ST.xray) { ST.xray = nx; dirty = true; }
  if (camAnim) { const a = camAnim; a.t = Math.min(1, a.t + dt * 1.6); const k = ease(a.t); let dy = a.to[0] - a.from[0]; while (dy > 180) dy -= 360; while (dy < -180) dy += 360;
    CAM.yaw = a.from[0] + dy * k; CAM.pitch = a.from[1] + (a.to[1] - a.from[1]) * k; CAM.dist = a.from[2] + (a.to[2] - a.from[2]) * k; CAM.target = lerp3(a.from[3], a.to[3], k); if (a.t >= 1) camAnim = null; dirty = true; }
  if (ST.inst === 'warm' && !ST.warmSkipped) { ST.warmLeft -= dt * 90; if (ST.warmLeft <= 0) { ST.warmLeft = 0; ST.inst = 'ready'; beep(1400, 0.2); ok('Pemanasan selesai: lampu stabil.'); sync(); } dirty = true; }
  if (now - lastPC > 200) { lastPC = now; $('#clock').textContent = mmss((now - ST.t0) / 1000); if (ST.inst === 'warm') sync(); drawPC(); dirty = true; }
  if (dirty) { dirty = false; draw(); }
  requestAnimationFrame(loop);
}

/* ================= input ================= */
function pickAt(x, y) {
  const inPoly = (p, px, py) => { let c = false; for (let i = 0, j = 3; i < 4; j = i++) { if (((p[i][1] > py) !== (p[j][1] > py)) && (px < (p[j][0] - p[i][0]) * (py - p[i][1]) / (p[j][1] - p[i][1]) + p[i][0])) c = !c; } return c; };
  for (let i = drawn.length - 1; i >= 0; i--) if (inPoly(drawn[i].p, x, y)) return drawn[i].id; return null;
}
(function bind() {
  let pts = new Map(), start = null, pinch = null;
  cv.addEventListener('pointerdown', e => { try { cv.setPointerCapture(e.pointerId); } catch (_) { } pts.set(e.pointerId, [e.clientX, e.clientY]); start = { x: e.clientX, y: e.clientY, yaw: CAM.yaw, pitch: CAM.pitch, moved: false }; if (pts.size === 2) { const [a, b] = [...pts.values()]; pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), dist: CAM.dist }; } });
  cv.addEventListener('pointermove', e => { if (!pts.has(e.pointerId)) return; pts.set(e.pointerId, [e.clientX, e.clientY]);
    if (pts.size === 2 && pinch) { const [a, b] = [...pts.values()]; CAM.dist = Math.max(40, Math.min(460, pinch.dist * pinch.d / Math.max(10, Math.hypot(a[0] - b[0], a[1] - b[1])))); start.moved = true; dirty = true; return; }
    if (!start) return; const dx = e.clientX - start.x, dy = e.clientY - start.y; if (Math.abs(dx) + Math.abs(dy) > 4) start.moved = true;
    CAM.yaw = start.yaw - dx * 0.4; CAM.pitch = Math.max(-5, Math.min(89, start.pitch + dy * 0.3)); camAnim = null; dirty = true; });
  const up = e => { if (!pts.has(e.pointerId)) return; pts.delete(e.pointerId); if (pts.size < 2) pinch = null;
    if (pts.size === 0) { if (start && !start.moved) { const r = cv.getBoundingClientRect(); const id = pickAt(e.clientX - r.left, e.clientY - r.top); selId = id; const inf = id && INFO[id];
      $('#ti').textContent = inf ? inf[0] : 'Info objek'; $('#info').textContent = inf ? (inf[1] || 'Komponen bangku optik UV-1700.') : 'Klik objek pada meja untuk penjelasan.'; dirty = true; } start = null; } };
  cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
  cv.addEventListener('wheel', e => { e.preventDefault(); CAM.dist = Math.max(40, Math.min(460, CAM.dist * (e.deltaY > 0 ? 1.08 : 1 / 1.08))); camAnim = null; dirty = true; }, { passive: false });
  pcc.addEventListener('click', e => { const r = pcc.getBoundingClientRect(); const x = (e.clientX - r.left) * pcc.width / r.width, y = (e.clientY - r.top) * pcc.height / r.height;
    const b = pcBtns.find(b => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h); if (!b) return;
    if (b.act.startsWith('tab')) { ACT[b.act](); drawPC(); } else if (ACT[b.act]) run(() => ACT[b.act]()); });
  document.addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return;
    if (b.id === 'gobtn') { const i = curStep(); if (i >= 0) { ST.autoScroll = true; run(() => STEPS[i].run(), 'Mengerjakan: ' + STEPS[i].t); } return; }
    if (b.id === 'startbtn') { ID.nama = $('#nm').value.trim(); ID.kelas = $('#kl').value.trim() || 'Kelas'; ID.kel = $('#kp').value.trim() || 'Kelompok'; $('#welcome').classList.remove('show'); log('Praktikum dimulai oleh ' + (ID.nama || 'praktikan') + ' (' + ID.kelas + ', kelompok ' + ID.kel + ').'); sync(); return; }
    if (b.id === 'lg_ok') { ST.logbook = { t: dstr, n: $('#lg_n').value, k: $('#lg_k').value, s: $('#lg_s').value }; $('#logm').classList.remove('show'); ok('Log Book tercatat dan ditandatangani oleh ' + (ST.logbook.s || '-') + '.'); if (logResolve) { logResolve(); logResolve = null; } sync(); return; }
    if (b.dataset.act) { const a = b.dataset.act; if (ACT[a]) run(() => ACT[a]()); return; }
    if (b.dataset.v) { setView(b.dataset.v); return; }
    if (b.dataset.t) { const k = b.dataset.t; if (k === 'help') { $('#welcome').classList.add('show'); return; }
      if (k === 'pip') { $('#pip').classList.remove('hide'); return; } if (k === 'pipx') { $('#pip').classList.add('hide'); return; } if (k === 'pipbig') { const bg = $('#pip').classList.toggle('big'); b.textContent = bg ? 'Kecilkan' : 'Perbesar'; return; } const on = b.getAttribute('aria-pressed') !== 'true'; b.setAttribute('aria-pressed', String(on));
      if (k === 'xray') { ST.xrayUser = on; ST.xrayT = on ? 1 : 0; } if (k === 'sound') ST.sound = on; } });
  const s = $('#sol'); SOL_KEYS.forEach(k => { const o = document.createElement('option'); o.value = k; o.textContent = SOL[k].label; s.appendChild(o); });
  $('#lam').value = AN.lam_ref;
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(() => { dirty = true; }).observe(cv.parentElement);
})();
window.LAB = { ST, ACT, STEPS, run, curStep, readAt, ID, finishLog: () => { ST.logbook = { t: dstr, n: ID.nama, k: 'uji', s: ID.nama }; if (logResolve) { logResolve(); logResolve = null; } } };
log('Praktikum virtual: ' + AN.nama + '. Ikuti kotak "Langkah sekarang".');
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

st.set_page_config(page_title="SHIMADZU UV-1700", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")


def _ver(v: str) -> tuple:
    return tuple(int(p) for p in re.findall(r"\d+", v)[:2])


# lebar penuh: API width="stretch" (Streamlit ≥ 1.50) atau use_container_width (versi lama)
WIDE = {"width": "stretch"} if _ver(st.__version__) >= (1, 50) else {"use_container_width": True}
PAL = ["#1f6fff", "#0aa5c4", "#c2368a", "#d98a00", "#1a9a5f", "#7a4dd8"]
PCFG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 3}}

_tpl = go.layout.Template(pio.templates["plotly_white"])
_tpl.layout.update(font=dict(family="IBM Plex Sans, system-ui, sans-serif", size=12, color="#0b1b3a"), colorway=PAL,
                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
                   xaxis=dict(gridcolor="#eceef8", linecolor="#c9cde6", zeroline=False, ticks="outside", tickcolor="#c9cde6"),
                   yaxis=dict(gridcolor="#eceef8", linecolor="#c9cde6", zeroline=False, ticks="outside", tickcolor="#c9cde6"),
                   hoverlabel=dict(bgcolor="#14123a", font_color="#ffffff", bordercolor="#14123a"))
pio.templates["spectralab"] = _tpl

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{--ink:#0b1b3a;--muted:#5a6b8c;--line:#dbe3f1;--ind:#1f6fff;--vio:#0aa5c4;--teal:#1a9a5f;--amb:#d98a00;--rose:#c3312f}
html, body, [class*="css"], .stMarkdown, .stText { font-family:'IBM Plex Sans', system-ui, sans-serif; }
h1, h2, h3, h4 { font-family:'IBM Plex Sans Condensed','IBM Plex Sans',sans-serif !important; letter-spacing:.01em; color:var(--ink); }
.block-container { padding-top: 1.1rem; max-width: 1400px; }
section[data-testid="stSidebar"] { background: #0b1b3a; }
section[data-testid="stSidebar"] * { color:#dbe6fb; }
section[data-testid="stSidebar"] [data-baseweb="select"] * { color:#0b1b3a; }
.stButton > button, .stDownloadButton > button { width:100%; border-radius:8px; font-weight:600; }
.stButton > button[kind="primary"] { background:#1f6fff; border:0; color:#fff; }
.ph { margin: 0 0 12px; }
.ph .eb { font: 500 .72rem 'IBM Plex Mono', monospace; letter-spacing: .12em; text-transform: uppercase; color: #0aa5c4; }
.ph h1 { margin: 2px 0 4px; font-size: 1.9rem; }
.ph p { color: var(--muted); margin: 0; max-width: 900px; }
.topbar { display:flex; flex-wrap:wrap; gap:6px; margin-bottom: 8px; }
.pill { display:inline-block; font: 500 .72rem 'IBM Plex Mono', monospace; padding: 3px 9px; border-radius: 99px; background:#e3edff; color:#1f4fbf; }
.pill .d { display:none; }
.pill.sim { background:#fdf0d9; color:#a86400; } .pill.ok { background:#dcf3e9; color:#12875a; } .pill.bad { background:#fbe3e2; color:#c3312f; }
.kpi-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 8px 0 14px; }
.kpi { background:#fff; border:1px solid var(--line); border-radius: 10px; padding: 10px 14px; border-left:4px solid var(--ind); }
.kpi.teal{border-left-color:var(--teal)} .kpi.amber{border-left-color:var(--amb)} .kpi.rose{border-left-color:var(--rose)} .kpi.violet{border-left-color:var(--vio)}
.kpi .kl { font: 500 .66rem 'IBM Plex Mono', monospace; letter-spacing:.08em; text-transform:uppercase; color: var(--muted); }
.kpi .kv { font: 600 1.3rem 'IBM Plex Sans', sans-serif; color: var(--ink); margin-top: 3px; line-height: 1.2; word-break: break-word; }
.kpi .ks { font-size: .76rem; color: var(--muted); margin-top: 2px; }
.lcd-bezel { background: #e9e4cf; border:1px solid #d3caa6; border-radius: 10px; padding: 12px; }
.lcd { background:#b9efec; color:#0b3a48; border-radius:6px; padding:10px 16px; font-family:'IBM Plex Mono', monospace;
  box-shadow: inset 0 0 0 3px #d9d2b8, inset 0 0 0 5px #8d8a7c, inset 0 2px 14px rgba(0,0,0,.18); }
.lcd .lr { display:flex; align-items:center; justify-content:space-between; gap: 10px; }
.lcd .bar { width: 34px; height: 7px; background: rgba(255,255,255,.88); border-radius: 1px; flex: none; }
.lcd .big { font-size: 1.5rem; letter-spacing: .08em; } .lcd small { font-size: .7rem; margin-left: 4px; }
.lcd .lf { border-top:1px dashed rgba(11,58,72,.3); margin-top:4px; padding-top:3px; font-size:.72rem; color:#4f8a92; display:flex; justify-content:space-between; }
.lcd-caption { display:flex; justify-content:space-between; font: 500 .62rem 'IBM Plex Mono', monospace; color:#8a8061; margin-top: 6px; letter-spacing: .08em; }
.eq { font:500 1.05rem 'IBM Plex Mono', monospace; background:#edf2f9; border-radius:8px; padding:10px 12px; }
.banner { background:#fdf0d9; border:1px solid #f0c98a; border-radius:8px; padding:9px 12px; font-size:.86rem; margin-bottom: 10px; }
.banner.info { background:#e3edff; border-color:#b9d0ff; color:#1f3f8a; }
.step { display:flex; gap: 12px; align-items:flex-start; padding: 10px 14px; border:1px solid var(--line); border-radius: 10px; background:#fff; margin-bottom: 8px; }
.step .dot { width: 28px; height: 28px; border-radius: 50%; display:grid; place-items:center; font: 500 .8rem 'IBM Plex Mono', monospace; background:#e3edff; color: var(--ind); flex: none; }
.step.done .dot { background: var(--teal); color:#fff; } .step.cur { border-color: var(--ind); }
.step .t { font-weight: 600; color: var(--ink); } .step .s { color: var(--muted); font-size: .86rem; }
.feat { background:#fff; border:1px solid var(--line); border-radius: 10px; padding: 12px 14px; height: 100%; }
.feat .ic { font: 500 .72rem 'IBM Plex Mono', monospace; color: var(--vio); letter-spacing:.1em; }
.feat .h { font: 600 1rem 'IBM Plex Sans Condensed', sans-serif; color: var(--ink); margin: 4px 0; }
.feat .b { color: var(--muted); font-size: .85rem; }
.homeimg { max-width: 1000px; margin: 8px auto 6px; border-radius: 14px; overflow: hidden; background:#fff; border:1px solid var(--line); }
.homeimg img { display:block; width:100%; height:auto; }
.homeimg .cap { text-align:center; padding: 14px 12px 16px; font: 600 1.7rem 'IBM Plex Sans Condensed', sans-serif; color: var(--ink); letter-spacing:.04em; background:linear-gradient(135deg,#0b1b3a,#13295a); color:#fff; }
.verdict { border-radius: 8px; padding: 9px 12px; font-weight: 500; margin: 6px 0; }
.verdict.ok { background:#dcf3e9; color:#0b6b46; border:1px solid #b6e3cf; } .verdict.bad { background:#fbe3e2; color:#a8201e; border:1px solid #f1bcba; }
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
HOME_IMG = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCAGNA+gDASIAAhEBAxEB/8QAHQABAAEFAQEBAAAAAAAAAAAAAAYDBAUHCAECCf/EAFUQAAEDAwEEBQkEBwQHBQcFAQEAAgMEBREGBxIhMRNBUXGBCBQiMmFykaGxM0LB0RUjQ1JigqI0U5KyFiREY3PC4RclNVTwJjZ0o8Pi8UVGg5OVs//EABoBAQADAQEBAAAAAAAAAAAAAAABAgMEBQb/xAA2EQEAAgIBAwIDBgQFBQEAAAAAAQIDEQQSITEFQRNRYRQiMkKRoXGBscEGFVJT8CNi0eHxJP/aAAwDAQACEQMRAD8A6NREVwREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERARF5kdqD1F6GuPqtce4ZXy8iP7QhnvHH1QeosdV36xUZIq73a6cjiRLWxMx8XLDVu0jZ9RZ861tp6LHPNfGfoSglSLXlXtu2UUud/W9tkx/ctkl/wArSsNWeUbsnp87t7ran/gW6U5+ICDbiKNbOtcae19YXXnTlTJLTsmdDIyaPo5I3jjhzcnGQQR2hSVAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBFjNV3uj03pq5X+vDzS2+nfUStZjecGjkM8Mk4A71oqr8qqwNyKXR93kPUZaqJg+QKDohFzBV+VbUcRSaHhHYZriT8gz8ViKvyp9WvyKXTNig7C980hH9QCG3WyLjKq8pfaTKMRNsNP7lAXH+p5WJq/KA2rVGcaihgB6oaCFvzLSURt3Fkdq+g1x5NJ8FwJWbYtqFUCJNcXdoPVG9rB/S0LC1eutbVjcVWr7/KOx1wlx8A5Db9FJP1bd6T0B2u4D5qwq7zZ6NpdV3e3U4HMy1cbfqV+cVRcbhUkuqbhWTE8zJUPd9SrUtY52S1pJ6yOKG36ZU00NTTx1FPNHNDK0PjkjcHNe0jIII4EEdaqKBeT1X/AKR2KaVnzlzKEQO9hjc5mPg0KeokRePO6xzsZwCfkuRK7yrtWF72UulbDDgkZklmkI+YCDrxFxVWeVBtKmyII9P0w/hoS4jxc9Yas8ojaxPnGo4KcHqht8LfmWkojbu3I7V9BrjyaT4L8963bXtSqQRLry7MB6o3sjH9LQsFWbRNa1YxVa1v8o7DcpcfAOTZt+krg5jd57S0drhgKyqbpa6VpdU3OhgA5mSpY36lfmdVXy4VRzU3WvnJ5mSpkd9SrJ8rHnedG557SzJ+abH6TVuvtC0QJq9ZafixzBuMRPwDsrCVe2fZXS/aa6s7j2RPfIf6WlfnoJP3YD/hAXvSzEcIz4uCjY7yrPKI2TU+d3UVRUEdUNvmd8y0BYas8qDZrDkQQagqj/DQtYD4ueuJd6Y/daP5k/XHrYPimzbsCu8rDS7GnzLSV7nd1dLPDGD8CSsNWeVseIotCAdhnuWfk1n4rljdl/vAO5qBjiOMrvABNjo6s8rDVb8ik0rYoOwvmmk/ELC1nlQbS5s9DHYKUf7uhLv8zytF9F2vkP8AMnRM69497imxtqt8ofazUf8A7mipv/h6CFn/AClYSu20bT6rPT69vAz/AHcrY/8AKAoAYo8H0ArLATYl9dr/AFfXZ881lfqjPPfuMp/5lhqq8VNTnzm4VdRn+9ne/wCpWJRRs0ujUQn7mf5UFUwerGR3ABWqJs0uTVu/cPxXyaqT90fFUETaXWnkEy5j1hFnmaN/ylH4rqNcoeQTJ/3nqyLtpaV/9bx+K6vVgREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBEXzI8RxvkJwGtLvgMoPtrHOAIaSDyXpYRzwO8rWejKyo/0ZFRUVcojMkj/AE5Dhjc/ILMvljDGvfUR7rhlpL8gj2LCcswt0pe98TPXnib3vCpurKJvrVkPgcqJxvik+ye14zjI5ZVjSXigqYOmjdKRvOb6nWCQevtCj4snSmxuVvA41BPcwqrTVMNS1zoC4tacEkY4qG01VFUOcI2vG6MkuwpPYG7tvB/eeSrY7TaSY0yCIi2VEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREGn/ACvLx+jNjdTRtdiS6VkNIOPNoJkd8mAeK4q4rpPy47wXV+mdPsdwjimrpBnrcRG35Nd8VzYiJOK8Xq8RD1ERATgvF6gJwREHZ/kcV/nexmOmLsmiuVTDjsBLZB/nK3MucPIXr9/T+qbYXZ6GtgqAOwPY5v1Yuj0Wesxvt3uWRnuX5na8t5t2tb7bjlhprlURYHDlK7C/S88l+fXlHUP6P246thAw2Sv6ce3pGNfn4uKiUS130TesvPe4p0MX7gPevtFA+RHGOTG/BegAcgB4L1FKBERAQIiAiIgIEQICIiArA8yr8kDmQrB/rHHaolMPERFCRERAREQdO+QRJ/7Uanj/AHrZCfhMfzXXS438gyTG0C+x5PpWnOO6Zv5rshXgEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAVhqObzfT1xnzjcpZDn+Uq/Ue2jzdBoe6vH3odz/ABOA/FJ8CDcaXZk/HM0Lvi7h+Ku5Jg+np4ms3GxRhoGewAfgshT0VPPYoKCpjEkRgjDmZxnAB+qumUUJPo0wcfdJXJMbXhbW5witz5TybvOPcB/0Ua040ix0ZPN8e+e9xJ/FTiOhqQwBlJKGdX6sgKpHbKzADaUtA6uATpmTbDWZhDJHEEZIHEKbWpu5boB2tz8SsO201rubWN73rPws6OFkf7rQPktcVZjyiZfaIi2VEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQERfMkjIY3TSnEcbS957GgZPyCDh7yqbx+lttd3ja7MduZFQt72My7+pzlq1ZHVN0fe9TXS8yEl1dWS1Hg95I+RCxyIERecUQ98UREBERAREQb78h2tEWvb/QFxxU2tsjW9pZKPwcV10uHfJGr/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aO44P1CnCzny0jwIiKEi+ZGCSN0bhlr2lp8eC+kQamljMUr4nc2OLT4HCutPz+bahoJs4HS7ju5wx+Kr6qg6C/1bcYDndIO5wysTK5zGiRvrMIeO8HK08wz8S28ip08rZ6eOdpy2RgePEZVRZtBERAREQEREBEXu6ewoPEX0GOXu4e1B8Iqm4O0r3cb2KdI2pIqoAHUF6mjaluu7F7uOVRE0bfHR+1e7g9q+kTQ8DW9i9wOxEUoEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBfE8rKeGSeQgMiaXuJ7AMn6L7Uc2m136P0Fd6gHD3U5iZ3vIb+JQc4SySXG6PkdkyVU5cc9r3Z/FbPa0MaGN5NGB4LXmk4BPfqZpGWsJefAfnhbE6uHNWhEtiaDg6HT7JCOM0jn+HIfRZ5W1qp/NbZTU+MdHE1p78cVcqqYEREBERAREQEREBERAREQEREFFERVWEREBERBrXU1P5tfauMDDS/fb3O4rHKS7QYmtuVPMPWkiwfA8Pqo0tI8M58r3SdR5rqqlJOGzExn+YfnhbOWn5HuhqIJ2HDmPDh3ggrb7TvNDsYyMqtlqvURFVYREAyghW0GDduFPUAfaRFp72n8iow4ZaR28FOdoULTaoJc+kybA8QfyUHWlfDOfLYmjZ/ONNUbicuYwxnvacfksuo9s4H/ccwPECqfgdnAKTqml4lSwewoGO7FVRNG1PcPaF6GDtX2iaHzuNXu6OwL1FIIiIgREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQFrbygq3odK0dCDg1VWCR2tY0n6kLZK0p5RFQ91/tdKfs46R0g73PIP+UIIts+hzWVNQRwZGGA+0n/AKKe2anNVdqSn6nytB7s5PyCiegY2ttEsg9Z8xz4AKf6EjbJqFjnfs4nuHfjH4q3srPlsRERVWEREBERAREQEREBERAREQEREH//2Q=="


def page_home():
    html(f'''<div class="homeimg"><img src="{HOME_IMG}" alt="Model 3D Shimadzu UV-1700">
<div class="cap">Spektrofotometer UV-Vis Shimadzu UV-1700 PharmaSpec</div></div>''')
    _, mid, _ = st.columns([1.2, 1, 1.2])
    mid.button("Mulai praktikum", type="primary", key="home_go", on_click=lambda: st.session_state.update(nav="3D UV-1700"))


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
    components.html(LAB_HTML.replace("/*__LAB__*/null", json.dumps(init)), height=900, scrolling=True)
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



def page_qc():
    st.subheader("Presisi dan akurasi: %RPD dan %Recovery")
    cs = []
    reg = cal_reg()
    for s_ in S["samples"]:
        r_ = sample_res(s_, reg) if reg else None
        if r_:
            cs.append((s_["name"], float(r_["c_final"])))
    opt = ["(isi manual)"] + [f"{n} · {sig(c)}" for n, c in cs]
    q1, q2 = st.columns(2, gap="large")
    with q1:
        with st.container(border=True):
            st.markdown("**%RPD (duplo)**")
            i1 = st.selectbox("Hasil duplo 1", opt, index=1 if len(cs) > 0 else 0, key="rpd_s1")
            i2 = st.selectbox("Hasil duplo 2", opt, index=2 if len(cs) > 1 else 0, key="rpd_s2")
            d1 = st.text_input("Konsentrasi duplo 1 (manual)", "", key="rpd_m1", disabled=i1 != opt[0])
            d2 = st.text_input("Konsentrasi duplo 2 (manual)", "", key="rpd_m2", disabled=i2 != opt[0])
            lim = st.number_input("Batas %RPD", 0.1, 50.0, 5.0, 0.5, key="rpd_lim")
            v1 = cs[opt.index(i1) - 1][1] if i1 != opt[0] else num_in(d1)
            v2 = cs[opt.index(i2) - 1][1] if i2 != opt[0] else num_in(d2)
            if not (math.isnan(v1) or math.isnan(v2)) and (v1 + v2) > 0:
                rpd = abs(v1 - v2) / ((v1 + v2) / 2) * 100
                st.latex(r"\%RPD = \frac{|C_1 - C_2|}{(C_1 + C_2)/2} \times 100 = " + fmt(rpd, 2).replace(",", "{,}") + r"\ \%")
                verdict(rpd <= lim, f"%RPD = {fmt(rpd, 2)} % " + (f"≤ {fmt(lim, 1)} % (memenuhi)" if rpd <= lim else f"> {fmt(lim, 1)} % (tidak memenuhi)"))
            else:
                st.caption("Pilih dua hasil sampel atau isi konsentrasi secara manual.")
    with q2:
        with st.container(border=True):
            st.markdown("**%Recovery (spike)**")
            j1 = st.selectbox("Sampel tanpa spike", opt, index=1 if len(cs) > 0 else 0, key="rec_s1")
            j2 = st.selectbox("Sampel dengan spike", opt, index=2 if len(cs) > 1 else 0, key="rec_s2")
            e1 = st.text_input("C sampel (manual)", "", key="rec_m1", disabled=j1 != opt[0])
            e2 = st.text_input("C sampel + spike (manual)", "", key="rec_m2", disabled=j2 != opt[0])
            ad = st.text_input("C spike yang ditambahkan (setelah pengenceran)", "", key="rec_add")
            c0 = cs[opt.index(j1) - 1][1] if j1 != opt[0] else num_in(e1)
            c1_ = cs[opt.index(j2) - 1][1] if j2 != opt[0] else num_in(e2)
            cad = num_in(ad)
            if not any(math.isnan(x) for x in (c0, c1_, cad)) and cad > 0:
                Rv = recovery_spike(c0, c1_, cad)
                ok_ = CR["rec_min"] <= Rv <= CR["rec_max"]
                st.latex(r"\%Recovery = \frac{C_{spike+sampel} - C_{sampel}}{C_{spike}} \times 100 = " + fmt(Rv, 1).replace(",", "{,}") + r"\ \%")
                verdict(ok_, f"%Recovery = {fmt(Rv, 1)} % (kriteria {fmt(CR['rec_min'], 0)}–{fmt(CR['rec_max'], 0)} %): " + ("memenuhi" if ok_ else "tidak memenuhi"))
            else:
                st.caption("Isi C spike yang ditambahkan dan pilih/isi dua hasil.")


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
    page_qc()


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
ADV = bool(st.session_state.get("adv_menu", False))
MAIN = ["Beranda", "3D UV-1700", "Bagian dalam & prinsip", "Olah data (kurva kalibrasi)", "Laporan"]
EXTRA = ["Validasi data", "Kualifikasi kinerja", "Pembelajaran & kuis", "Panduan praktikum", "Riwayat"]
PAGE_FN = {"Beranda": page_home, "3D UV-1700": page_lab, "Bagian dalam & prinsip": page_model3d, "Olah data (kurva kalibrasi)": page_calibration,
           "Laporan": page_report, "Validasi data": page_validation, "Kualifikasi kinerja": page_pq, "Pembelajaran & kuis": page_learn,
           "Panduan praktikum": page_guide, "Riwayat": page_history}
MENU = MAIN + (EXTRA if ADV else [])
if st.session_state.get("nav") not in MENU:
    st.session_state["nav"] = MENU[0]

with st.sidebar:
    st.markdown("### SHIMADZU UV-1700\n<span style='color:#8ea3c9;font-size:.8rem'>Praktikum UV-Vis · Kelompok 2</span>", unsafe_allow_html=True)
    page = st.radio("Menu", MENU, key="nav", label_visibility="collapsed")
    st.divider()
    st.toggle("Tampilkan semua menu", key="adv_menu", help="Menu lanjutan: validasi, kualifikasi kinerja, kuis, panduan, riwayat.")
    with st.expander("Pengaturan"):
        S["settings"]["mode"] = st.radio("Mode kerja", ["latihan", "analisis"], index=0 if S["settings"]["mode"] == "latihan" else 1,
                                         format_func=lambda m: "Latihan (data contoh)" if m == "latihan" else "Analisis (data saya)")
        S["settings"]["dec"] = st.radio("Pemisah desimal", ["koma", "titik"], index=0 if S["settings"]["dec"] == "koma" else 1,
                                        format_func=lambda d: "Koma (0,512)" if d == "koma" else "Titik (0.512)", horizontal=True)
    st.caption("Politeknik AKA Bogor · Analisis Kimia. Instrumen acuan: Shimadzu UV-1700. Simulator edukatif; tidak terhubung ke alat.")

if ADV:
    _bl = I["baseline"]
    html('<div class="topbar">' + pill("SIMULASI", "sim") +
         pill("Instrumen siap" if I["power"] == "ready" else "Instrumen mati", "ok" if I["power"] == "ready" else "bad") +
         pill(f"Baseline {int(_bl['start'])}–{int(_bl['end'])} nm" if _bl else "Baseline belum", "ok" if _bl else "sim") +
         pill(f"λ {fmt(I['lam'], 1)} nm · {'D2' if I['lam'] < I['lamp_switch'] else 'WI'}") +
         pill(f"{ANALYTES[I['analyte']]['rumus']} · {CUVETTES[I['cuvette']]['nama']}") + "</div>")

PAGE_FN[page]()
