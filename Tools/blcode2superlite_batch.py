#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for blcode2superlite.py.

Finds all hydrovars_nt_* / composition_nt_* pairs in --path, reads the time
from each hydrovars file header, converts to days (rounded to 2 decimal places),
and runs blcode2superlite.py for each pair, saving output to:
    <out-path>/day_<time_days>/

Example:
>> python blcode2superlite_batch.py \\
    -p /path/to/blcode/Data_folder \\
    -o /path/to/sim-superlite/model_name \\
    --tau-path /path/to/tau_files \\
    --lum-path /path/to/lightcurve.dat \\
    -t --tau-thresh 100
"""

import os
import sys
import glob
import argparse
import subprocess

SECONDS_PER_DAY = 3600 * 24

def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch-run blcode2superlite.py over all hydrovars/composition pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to folder containing hydrovars_nt_* and composition_nt_* files')
    parser.add_argument('-o', '--out-path', type=str, required=True,
                        help='Base output path; each day saved to <out-path>/day_<N>')
    parser.add_argument('--tau-path', type=str, required=True,
                        help='Path to folder containing tau_* files')
    parser.add_argument('--lum-path', type=str, required=True,
                        help='Path to luminosity .dat file')
    parser.add_argument('-t', '--truncate', action='store_true',
                        help='Truncate profile at tau-thresh')
    parser.add_argument('--tau-thresh', type=float, default=100,
                        help='Optical depth threshold for truncation')
    parser.add_argument('-n_par', '--n-particles', type=int, default=20,
                        help='Number of particles for in_src_n2s')
    parser.add_argument('-s', '--sanity-check', action='store_true',
                        help='Run consistency checks in blcode2superlite')
    parser.add_argument('-n', '--renorm', action='store_true',
                        help='Renormalize mass fractions to 1')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output from blcode2superlite')
    parser.add_argument('--out-prefix', type=str, default='Model',
                        help='Prefix for output profile plots')
    return parser.parse_args()


def day_from_hyd_file(hyd_file):
    with open(hyd_file) as f:
        first_line = f.readline().strip()
    time_s = float(first_line.split()[-1])
    return round(time_s / SECONDS_PER_DAY, 2)


def build_cmd(blcode2sl, hyd_basename, comp_basename, args, day_outdir):
    cmd = [
        sys.executable, blcode2sl,
        '-p', args.path,
        '-o', day_outdir,
        '--hyd_file', hyd_basename,
        '--comp_file', comp_basename,
        '--tau_path', args.tau_path,
        '--lum_path', args.lum_path,
        '--tau-thresh', str(args.tau_thresh),
        '-n_par', str(args.n_particles),
        '--out-prefix', args.out_prefix,
    ]
    if args.truncate:     cmd.append('-t')
    if args.sanity_check: cmd.append('-s')
    if args.renorm:       cmd.append('-n')
    if args.verbose:      cmd.append('-v')
    return cmd

def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    blcode2sl = os.path.join(script_dir, 'blcode2superlite.py')

    hyd_files = sorted(glob.glob(os.path.join(args.path, 'hydrovars_nt_*')))
    if not hyd_files:
        print(f"No hydrovars_nt_* files found in {args.path}")
        sys.exit(1)

    print(f"Found {len(hyd_files)} hydrovars file(s) in {args.path}\n")
    failed = []

    for hyd_file in hyd_files:
        hyd_basename = os.path.basename(hyd_file)
        suffix = hyd_basename[len('hydrovars'):]          # '_nt_XXXXXXXXXX_time_T...'
        comp_basename = 'composition' + suffix
        comp_file = os.path.join(args.path, comp_basename)

        if not os.path.exists(comp_file):
            print(f"[SKIP] {hyd_basename}: no matching {comp_basename}")
            continue

        day = day_from_hyd_file(hyd_file)
        day_outdir = os.path.join(args.out_path, f'day_{day}')
        os.makedirs(day_outdir, exist_ok=True)

        cmd = build_cmd(blcode2sl, hyd_basename, comp_basename, args, day_outdir)

        print(f"[day {day}] {hyd_basename}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: blcode2superlite.py exited with code {e.returncode}")
            failed.append((day, hyd_basename))

    print(f"\nDone. {len(hyd_files) - len(failed)} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed days:")
        for day, name in failed:
            print(f"  day {day}  ({name})")
        sys.exit(1)


if __name__ == '__main__':
    main()
