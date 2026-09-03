# SuperLite — Codebase Guide

## What it is
IMC-DDMC (Implicit Monte Carlo + Discrete Diffusion Monte Carlo) radiation transport code for simulating supernovae. Primary language: Fortran 77/90. Python utility scripts in `Tools/`. Licensed GNU GPLv3.

## Build system
- **CMake** (primary): `cmake .. && make`
  - Optional flags: `-DUSE_MPI=ON`, `-DUSE_OPENMP=ON`
  - Links against BLAS/LAPACK
- **Traditional make** (fallback): `make`
- Compiler configs in `System/` (gfortran, Intel, IBM BlueGene, g95)
- Main executable: `superlite`
- Auto-generated dependency files: `Makefile.depend`, `Makefile.dependmod`

## Entry points
- `superlite.f90` — main program (init → MC loop → output)
- `particle_advance.f90` — core particle transport loop
- Runtime input: `input.par` (parameters), `input.str` (domain structure), `input.wlgrid` (wavelength grid)

## Directory structure

| Path | Purpose |
|---|---|
| `superlite.f90` | Main program entry point |
| `particle_advance.f90` | Core MC particle transport loop |
| `TRANSPORT1/` | 1D spherical transport (`transport11.f90`) and diffusion (`diffusion11.f90`) |
| `GAS/` | EOS, opacity, gas physics (LTE and NLTE) |
| `GRID/` | Spatial grid setup, emission probabilities (`emitgroup.f`) |
| `SOURCE/` | Particle generation (boundary, interior, energy assignment) |
| `OUTPUT/` | Flux, grid, and profile output routines |
| `MISC/` | Utilities: binary search, sorting, interpolation |
| `Data/` | Physical data files: 173 atomic files (`Data/Atoms/data.atom.*`), `data.ion`, cross-sections |
| `Tools/` | Python converters: `blcode2superlite.py`, `blcode2superlite_batch.py`, `stella2superlite.py` |
| `Testsuite/` | 4 test suites: A4, s18, sn1999em-like, sn2017hcc-like |
| `System/` | Compiler flag definitions |

## Key Fortran modules
- `physconstmod.f` — physical constants
- `inputparmod.f` — input parameters
- `particlemod.f90` — particle/packet type definitions
- `gridmod.f` — spatial grid and geometry
- `gasmod.f` — gas state (density, temperature, composition)
- `groupmod.f` — energy group structure
- `transportmod.f` — transport method interfaces
- `randommod.f` — random number generation
- `timingmod.f` — performance timing
- `mpimod_mpi.f` / `mpimod_ser.f` — MPI or serial fallback
- `ionsmod.f` — ion species and energy levels
- `nltemod.f` — non-LTE data

## Simulation workflow
1. Read `input.par` + `input.str` → parse and verify configuration
2. Load atomic data, cross-sections from `Data/`
3. Setup grid, gas properties, radiation field
4. **Main MC loop** per timestep:
   - Update gas (EOS, opacity)
   - Calculate source energy and leakage opacity
   - Generate source particles
   - Advance particles through domain (transport/diffusion)
   - Tally radiation per group
   - Write output
5. Print timing stats, finalize MPI, clean up memory

## Python tools (`Tools/`)
- `blcode2superlite.py` — converts B&L explosion model → SuperLite input format; default `--n_particles 23`
- `blcode2superlite_batch.py` — batch wrapper; loops over hydrovars/composition pairs, outputs to `day_N/` subfolders
- `stella2superlite.py` — converts STELLA code output → SuperLite format
- `myfuncts.py` — shared utility functions
- `parse_hdf5_sndata.py` — HDF5 data parsing
- `blcode2superlite.ipynb` — Jupyter notebook with conversion examples

## Git remotes
- `thuyn25` — push target (thuyn25/superlite)
- `origin` — gururajw/superlite (no write access)

## Output files
All files are written to the working directory where `superlite` is executed. All formats are plain ASCII text.

### Written once per run
| File | Contents |
|---|---|
| `output.grd_grid` | Grid geometry: dimensions, coordinate arrays, cell index map |
| `output.flx_grid` | Flux grid: wavelength bins, polar/azimuthal angle bins |
| `output.grp_grid` | Multigroup energy structure: group count, wavelength bin edges |
| `output.name` | Simulation name (`in_name` parameter) |
| `output.logdata` | Verbose atomic data log: ion level counts, ionization energies, `read_atom` success/fail per element and ion |
| `output.log` | Main stdout: banner, namelist echo, setup info, per-iteration MC progress, counters, timing summary. Written only if `in_io_grabstdout=.true.`; otherwise goes to console |

### Written every timestep (appended)
| File | Contents |
|---|---|
| `output.tot_energy` | Global luminosity scalars (erg/s): `eout` (escaped), `evelo` (velocity-frame), `sflux`, `sthermal` (= `in_L_bol`) |
| `output.grd_temp` | Material temperature per cell |
| `output.grd_radtemp` | Radiation temperature per cell |
| `output.grd_nvol` | Volume index per cell |
| `output.grd_eraddens` | Radiation energy density per cell |
| `output.flx_luminos` | Outbound luminosity (erg/s) per wavelength × angle bin |
| `output.flx_lumnum` | Escaped particle count per wavelength × angle bin |
| `output.flx_lumdev` | Luminosity variance per wavelength × angle bin |
| `output.src_number` | Source particle count per cell per energy group |
| `output.src_luminos` | Source luminosity per cell per energy group |
| `output.timing` | CPU time breakdown (gas update, EOS, opacity, MPI, transport, etc.) |
| `output.counters` | Algorithm counters (particles created, transported, method swaps, etc.) |

### Conditional outputs
| File | Condition | Contents |
|---|---|---|
| `output.grp_cap` / `output.grp_capemit` / `output.grp_emiss` | `in_io_opacdump='one'` or `'all'` | Multigroup absorption/emission opacity and emissivity per cell |
| `output.grd_capgrey` / `output.grd_capemitgrey` / `output.grd_capross` / `output.grd_sig` | `in_io_opacdump != 'off'` | Grey and Rosseland opacities, scattering opacity per cell |
| `output.grd_methodswap` | `in_io_dogrdtally=.true.` | IMC↔DDMC transport method swap count per cell |
| `output.pdens` | `in_io_pdensdump='on'` | Ionization state populations and electron density per cell |
| `output.profile` | `in_io_profdump=.true.`, final timestep only | Radial profile: density, temperature, velocity, composition, electron fraction |

### Key output control parameters (in `input.par`)
| Parameter | Effect |
|---|---|
| `in_io_grabstdout` | Redirect stdout to `output.log` |
| `in_io_nogriddump` | Disable all per-cell grid outputs (except `output.tot_energy`) |
| `in_io_opacdump='off\|one\|all'` | Control multigroup opacity output |
| `in_io_dogrdtally` | Enable `output.grd_methodswap` |
| `in_io_pdensdump='off\|on'` | Control partial density output |
| `in_io_profdump` | Enable `output.profile` at final timestep |

## Diagnostic stderr messages

`GAS/physical_opacity_nlte.f` (and the LTE counterpart `physical_opacity.f`) runs a sanity
check after every opacity calculation, over every `(cell, wavelength-group)` pair with
`gas_mass > 0`, on three arrays: `gas_cap` (absorption opacity), `gas_capemit`
(opacity-weighted emission), `gas_emiss` (emissivity). For each array it ORs together flags
across the whole grid and prints via `write(0,*)` (stderr, not `output.log`) — **these are
`write` statements, not `stop`/`error stop`, so the run continues regardless**:

| Message | Meaning |
|---|---|
| `opacity_calc: some cap==0` / `some capemit==0` | *At least one* (cell, group) pair has zero opacity — mild, often expected (see below) |
| `opacity_calc: some cap<0` / `<0` / `==NaN` / `==inf` | At least one bad value — worth investigating |
| `opacity_calc: all cap==0` / `all capemit==0` / etc. | The *entire* grid has zero/negative/NaN/inf opacity — this is the fatal-looking one; only printed as a follow-up when the "some" check already fired |

`some cap==0`/`some capemit==0` printed repeatedly (e.g. ~10 pairs under one `it: N` line)
is routine NLTE solver chatter, not a crash signal — it corresponds to one opacity
recomputation per gas/ionization-temperature sub-iteration while the solver converges
within that MC step. It's especially expected when `in_nlte_nelem` covers few elements
and/or `read_atom failed` was reported for some element/ionization stage at startup (see
`output.logdata`): many `(cell, group)` combinations genuinely have zero line/continuum
opacity there. Only the "all cap==0"-style messages (or the run stopping before
`SuperLite finished`) indicate an actual failure.
