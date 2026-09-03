#!/usr/bin/env python3
"""
compute_lbol.py — Bolometric and spectral luminosity from SuperLite output.

Single-day mode: reads one simulation directory and reports L_bol per timestep,
optionally plotting L_lambda vs wavelength.

Lightcurve mode (--lightcurve): scans a base directory for day_N.0 subfolders,
computes L_bol at each day, and plots/saves the bolometric light curve.

Usage:
    # Single day
    python compute_lbol.py <sim_dir> [--lsun] [--plot] [--save FILE]

    # Light curve over many days
    python compute_lbol.py <base_dir> --lightcurve [--day-min N] [--day-max N]
                           [--lsun] [--plot] [--save FILE]

Examples:
    python compute_lbol.py /path/to/day_2.0/ --lsun --plot
    python compute_lbol.py /path/to/100M_rot_0pt5_0pt1Zbase/ --lightcurve --lsun --plot
"""

import argparse
import re
import numpy as np
from pathlib import Path


def parse_flx_grid(sim_dir):
    """
    Read output.flx_grid.
    Returns ng, nmu, nom, wl_edges (cm), mu_edges, om_edges.
    """
    path = Path(sim_dir) / 'output.flx_grid'
    with open(path) as f:
        header = f.readline()          # '# ng nmu nom'
        ng, nmu, nom = [int(x) for x in header.split()[1:4]]
        wl = np.array(f.readline().split(), dtype=float)   # ng+1 edges in cm
        mu = np.array(f.readline().split(), dtype=float)   # nmu+1 cos(theta) edges
        om = np.array(f.readline().split(), dtype=float)   # nom+1 radian edges
    return ng, nmu, nom, wl, mu, om


def parse_flx_luminos(sim_dir, ng, nmu, nom):
    """
    Read output.flx_luminos.

    Each timestep occupies (nmu * nom) rows in the file.  Each row has ng
    values written as Fortran e12.4 fixed-width fields (12 chars each).
    Values are luminosity in erg/s for one (wavelength, angle) bin.

    Returns
    -------
    lspec : ndarray, shape (niter, ng)
        Spectral luminosity per wavelength bin (erg/s), summed over angles.
    lbol  : ndarray, shape (niter,)
        Bolometric luminosity per timestep (erg/s).
    """
    path = Path(sim_dir) / 'output.flx_luminos'
    rows_per_iter = nmu * nom
    field_width = 12

    lspec_list = []
    cur_spec = np.zeros(ng)
    cur_row = 0

    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if len(line) < ng * field_width:
                continue
            row = np.array(
                [line[i * field_width:(i + 1) * field_width] for i in range(ng)],
                dtype=float,
            )
            cur_spec += row
            cur_row += 1
            if cur_row == rows_per_iter:
                lspec_list.append(cur_spec.copy())
                cur_spec[:] = 0.0
                cur_row = 0

    lspec = np.array(lspec_list)      # (niter, ng)  erg/s per wl bin
    lbol = lspec.sum(axis=1)          # (niter,)      erg/s
    return lspec, lbol


def parse_tot_energy(sim_dir):
    """
    Read output.tot_energy.
    Returns dict with keys eout, evelo, sflux, sthermal (erg), one value per timestep.
    """
    path = Path(sim_dir) / 'output.tot_energy'
    data = np.genfromtxt(path, comments='#')
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return dict(eout=data[:, 0], evelo=data[:, 1],
                sflux=data[:, 2], sthermal=data[:, 3])


def collect_day_dirs(base_dir, day_min, day_max):
    """
    Scan base_dir for subdirectories named day_N.0 where N is an integer.
    Returns a sorted list of (day_number, Path) tuples within [day_min, day_max].
    """
    pattern = re.compile(r'^day_(\d+)\.0$')
    matches = []
    for entry in Path(base_dir).iterdir():
        m = pattern.match(entry.name)
        if m and entry.is_dir():
            day = int(m.group(1))
            if day_min <= day <= day_max:
                matches.append((day, entry))
    return sorted(matches, key=lambda x: x[0])


def run_single(sim_dir, args, L_sun):
    ng, nmu, nom, wl_edges, *_ = parse_flx_grid(sim_dir)
    wl_cen = np.sqrt(wl_edges[:-1] * wl_edges[1:])
    dwl = wl_edges[1:] - wl_edges[:-1]
    wl_ang = wl_cen * 1e8

    print(f"Flux grid  : ng={ng}, nmu={nmu}, nom={nom}")
    print(f"Wavelength : {wl_edges[0]:.3e} – {wl_edges[-1]:.3e} cm"
          f"  ({wl_edges[0]*1e8:.1f} – {wl_edges[-1]*1e8:.1f} Å)")

    lspec, lbol = parse_flx_luminos(sim_dir, ng, nmu, nom)
    niter = len(lbol)
    print(f"Timesteps  : {niter}\n")

    hdr = f"{'iter':>6}  {'L_bol [erg/s]':>16}"
    if args.lsun:
        hdr += f"  {'L_bol [L_sun]':>14}"
    print(hdr)
    print('-' * len(hdr))
    for i, L in enumerate(lbol):
        row = f"{i + 1:>6}  {L:>16.6e}"
        if args.lsun:
            row += f"  {L / L_sun:>14.6e}"
        print(row)

    try:
        tot = parse_tot_energy(sim_dir)
        print(f"\nCross-check (output.tot_energy):")
        print(f"  eout     : " + ', '.join(f"{e:.4e}" for e in tot['eout']) + " erg/s")
        print(f"  sthermal : " + ', '.join(f"{e:.4e}" for e in tot['sthermal']) + " erg/s")
    except Exception:
        pass

    if args.save:
        save_path = Path(args.save)
        hdr_txt = (
            "wl_ang[Angstrom]  dwl_ang[Angstrom]"
            + ''.join(f"  L_lambda_iter{i+1}[erg/s/Ang]" for i in range(niter))
        )
        L_lambda = lspec / (dwl * 1e8)
        table = np.column_stack([wl_ang, dwl * 1e8] + [L_lambda[i] for i in range(niter)])
        np.savetxt(save_path, table, header=hdr_txt, fmt='%.6e')
        print(f"\nSpectral table saved to: {save_path}")

    if args.plot:
        import matplotlib.pyplot as plt
        L_lambda = lspec / (dwl * 1e8)
        fig, ax = plt.subplots(figsize=(9, 5))
        for i in range(niter):
            ax.plot(wl_ang, L_lambda[i], lw=0.8,
                    label=f'iter {i + 1}  L_bol={lbol[i]:.3e} erg/s')
        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel(r'$L_\lambda$ (erg s$^{-1}$ Å$^{-1}$)')
        ax.set_title(f'SuperLite spectral luminosity — {Path(sim_dir).name}')
        ax.set_xlim(wl_ang[0], min(wl_ang[-1], 2e5))
        ax.set_yscale('log')
        if niter > 1:
            ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def run_lightcurve(base_dir, args, L_sun):
    day_dirs = collect_day_dirs(base_dir, args.day_min, args.day_max)
    if not day_dirs:
        print(f"No day_N.0 directories found in {base_dir} "
              f"for day {args.day_min}–{args.day_max}.")
        return

    print(f"Found {len(day_dirs)} day folders "
          f"(day {day_dirs[0][0]} – day {day_dirs[-1][0]})\n")

    days = []
    lbol_vals = []
    skipped = []

    for day, dpath in day_dirs:
        try:
            ng, nmu, nom, *_ = parse_flx_grid(dpath)
            _, lbol = parse_flx_luminos(dpath, ng, nmu, nom)
            # niter=1 for post-processing runs; take first (and only) iter
            days.append(day)
            lbol_vals.append(lbol[0])
        except Exception as e:
            skipped.append((day, str(e)))

    if skipped:
        print(f"Skipped {len(skipped)} folders:")
        for day, reason in skipped:
            print(f"  day_{day}.0 : {reason}")
        print()

    days = np.array(days, dtype=float)
    lbol_vals = np.array(lbol_vals)

    hdr = f"{'day':>6}  {'L_bol [erg/s]':>16}"
    if args.lsun:
        hdr += f"  {'L_bol [L_sun]':>14}"
    print(hdr)
    print('-' * len(hdr))
    for d, L in zip(days, lbol_vals):
        row = f"{d:>6.1f}  {L:>16.6e}"
        if args.lsun:
            row += f"  {L / L_sun:>14.6e}"
        print(row)

    if args.save:
        save_path = Path(args.save)
        hdr_txt = "day  L_bol[erg/s]  L_bol[L_sun]"
        table = np.column_stack([days, lbol_vals, lbol_vals / L_sun])
        np.savetxt(save_path, table, header=hdr_txt, fmt='%.6e')
        print(f"\nLight curve saved to: {save_path}")

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        y = lbol_vals / L_sun if args.lsun else lbol_vals
        ylabel = r'$L_\mathrm{bol}$ ($L_\odot$)' if args.lsun \
            else r'$L_\mathrm{bol}$ (erg s$^{-1}$)'
        ax.plot(days, y, 'o-', ms=3, lw=1.2)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Bolometric light curve — {Path(base_dir).name}')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Compute bolometric luminosity from SuperLite output.')
    parser.add_argument('sim_dir',
                        help='Simulation directory (single day) or base directory '
                             '(with --lightcurve)')
    parser.add_argument('--lightcurve', action='store_true',
                        help='Scan sim_dir for day_N.0 subfolders and compute '
                             'the bolometric light curve')
    parser.add_argument('--day-min', type=int, default=1,
                        help='First day to include in light curve (default: 1)')
    parser.add_argument('--day-max', type=int, default=180,
                        help='Last day to include in light curve (default: 180)')
    parser.add_argument('--lsun', action='store_true',
                        help='Print/plot luminosity in solar units')
    parser.add_argument('--plot', action='store_true',
                        help='Show plot')
    parser.add_argument('--save', metavar='FILE',
                        help='Save results to FILE')
    args = parser.parse_args()

    L_sun = 3.828e33  # erg/s

    if args.lightcurve:
        run_lightcurve(args.sim_dir, args, L_sun)
    else:
        run_single(args.sim_dir, args, L_sun)


if __name__ == '__main__':
    main()
