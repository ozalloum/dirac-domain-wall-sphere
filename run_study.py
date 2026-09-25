#!/usr/bin/env python3
"""Reproduce numerical tables, raw CSV data, and figures for the paper.

Usage: python run_study.py --output .
The full study uses NumPy, SciPy, and Matplotlib.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from sphere_dirac_galerkin import PoleRegularBasis, cosine_double, cosine_single
from benchmark_methods import run_cost_benchmark, run_independent_benchmark

EPS = np.array([0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01])
EXCITED_EPS = np.array([0.08, 0.04, 0.02, 0.01, 0.005])


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nearest(vals, target):
    return float(vals[np.argmin(np.abs(vals - target))])


def style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10.5,
        'axes.labelsize': 11, 'axes.titlesize': 12,
        'legend.fontsize': 9, 'xtick.labelsize': 9,
        'ytick.labelsize': 9, 'lines.linewidth': 1.8,
        'axes.linewidth': 0.8, 'figure.dpi': 150,
        'savefig.dpi': 400, 'savefig.bbox': 'tight',
        'mathtext.fontset': 'stix', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    })


def savefig(fig, pathbase: Path):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(pathbase.with_suffix('.' + ext), dpi=400 if ext == 'png' else None)
    plt.close(fig)


def run(out: Path, nmain: int = 112, nq: int = 1000, quick: bool = False):
    data, figs = out / 'data', out / 'figures'
    data.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    style()
    if quick:
        eps_grid = np.array([0.08, 0.04, 0.02, 0.01])
        excited_grid = np.array([0.08, 0.04, 0.02, 0.01])
        nmain = min(nmain, 72)
        nq = min(nq, 600)
    else:
        eps_grid, excited_grid = EPS, EXCITED_EPS

    # Exact free and constant-mass tests in three half-integer sectors.
    validation_rows = []
    for nu in (0.5, 1.5, -0.5):
        basis = PoleRegularBasis.build(nu, 20, max(360, 8*20))
        free = basis.solve(1.0, lambda t: np.zeros_like(t))
        constant_mass = basis.solve(0.7, lambda t: np.full_like(t, 0.8))
        expected_n = np.array([abs(nu) + 0.5 + j for j in range(4)])
        free_pos = free[free > 0][:4]
        const_pos = constant_mass[constant_mass > 0][:4]
        expected_const = np.sqrt(0.8**2 + 0.7**2 * expected_n**2)
        for test, computed, exact in (
            ('massless', free_pos, expected_n),
            ('constant_mass', const_pos, expected_const),
        ):
            for j, (val, ref) in enumerate(zip(computed, exact), start=1):
                validation_rows.append({
                    'test': test, 'nu': nu, 'mode_index': j,
                    'computed_eigenvalue': f'{val:.14g}',
                    'exact_eigenvalue': f'{ref:.14g}',
                    'absolute_error': f'{abs(val-ref):.8e}',
                    'basis_size_per_component': basis.size,
                })
    write_csv(data / 'exact_spectrum_validation.csv', validation_rows)

    # Single equatorial wall: chiral branch and its curvature correction.
    nu, mu, radius = 0.5, 1.0, 1.0
    n_hi = nmain
    n_lo = max(28, nmain - 16)
    basis_hi = PoleRegularBasis.build(nu, n_hi, nq)
    basis_lo = PoleRegularBasis.build(nu, n_lo, nq)
    single_rows = []
    for eps in eps_grid:
        vals = basis_hi.solve(eps, lambda t: cosine_single(t, mu), radius)
        coarse = basis_lo.solve(eps, lambda t: cosine_single(t, mu), radius)
        leading = -nu * eps / radius
        second = leading - nu * eps**2 / (4*mu*radius**2)
        num = nearest(vals, leading)
        num_coarse = nearest(coarse, leading)
        single_rows.append({
            'epsilon': f'{eps:.8g}', 'nu': nu, 'mu': mu, 'radius': radius,
            'lambda_numeric': f'{num:.14g}',
            'leading_order': f'{leading:.14g}',
            'two_term_asymptotic': f'{second:.14g}',
            'error_vs_two_term': f'{abs(num-second):.8e}',
            'error_over_epsilon_2': f'{abs(num-second)/eps**2:.8e}',
            'basis_change_Nminus16_to_N': f'{abs(num-num_coarse):.8e}',
            'basis_size_N': n_hi, 'quadrature_order': nq,
        })
    write_csv(data / 'single_wall_chiral_branch.csv', single_rows)

    # Symmetric pair of walls: algebraic low-energy gap for two sectors.
    gap_rows = []
    gap_basis = {}
    for nu_gap in (0.5, 1.5):
        basis = PoleRegularBasis.build(nu_gap, nmain, nq)
        gap_basis[nu_gap] = basis
        for eps in eps_grid:
            vals = basis.solve(eps, lambda t: cosine_double(t, mu), radius)
            numeric = 2.0 * float(vals[vals > 0][0])
            symmetry_error = float(np.max(np.abs(vals + vals[::-1])))
            leading = 2*np.sqrt(2)*abs(nu_gap)*eps/radius
            second = leading + 3*np.sqrt(2)*abs(nu_gap)*eps**2/(4*mu*radius**2)
            gap_rows.append({
                'epsilon': f'{eps:.8g}', 'nu': nu_gap, 'mu': mu, 'radius': radius,
                'gap_numeric': f'{numeric:.14g}',
                'leading_order': f'{leading:.14g}',
                'two_term_asymptotic': f'{second:.14g}',
                'absolute_error_two_term': f'{abs(numeric-second):.8e}',
                'relative_error_two_term': f'{abs(numeric-second)/numeric:.8e}',
                'discrete_spectral_pairing_error': f'{symmetry_error:.8e}',
                'basis_size_per_component': nmain, 'quadrature_order': nq,
            })
    write_csv(data / 'symmetric_two_wall_gap.csv', gap_rows)

    # First excited positive double-wall cluster, including the two-wall pair.
    excited_basis = PoleRegularBasis.build(0.5, max(nmain, 140), max(nq, 1000))
    excited_rows = []
    for eps in excited_grid:
        vals_hi = excited_basis.solve(eps, lambda t: cosine_double(t, mu), radius)
        positives = vals_hi[vals_hi > 0]
        local_scale = 2*np.sqrt(mu*eps/radius)
        pair = np.sort(positives[np.argsort(np.abs(positives-local_scale))[:2]])
        # Compare a slightly shorter basis to estimate spectral truncation error.
        basis_lower = PoleRegularBasis.build(0.5, excited_basis.size-20,
                                             max(nq, 1000))
        vals_lo = basis_lower.solve(eps, lambda t: cosine_double(t, mu), radius)
        positives_lo = vals_lo[vals_lo > 0]
        pair_lo = np.sort(positives_lo[np.argsort(np.abs(positives_lo-local_scale))[:2]])
        center = 2*np.sqrt(eps) - 3.0/8.0 * eps**1.5
        for j, value in enumerate(pair, 1):
            excited_rows.append({
                'epsilon': f'{eps:.8g}', 'nu': 0.5, 'positive_pair_member': j,
                'eigenvalue_numeric': f'{value:.14g}',
                'local_asymptotic_center': f'{center:.14g}',
                'absolute_error_center': f'{abs(value-center):.8e}',
                'error_over_epsilon_2': f'{abs(value-center)/eps**2:.8e}',
                'basis_change_Nminus20_to_N': f'{abs(pair[j-1]-pair_lo[j-1]):.8e}',
                'basis_size_per_component': excited_basis.size,
                'quadrature_order': max(nq, 1000),
            })
    write_csv(data / 'excited_double_wall_pair.csv', excited_rows)
    separation_rows = []
    for eps in excited_grid:
        pair_values = sorted(float(r['eigenvalue_numeric']) for r in excited_rows
                             if float(r['epsilon']) == float(eps))
        delta = pair_values[1] - pair_values[0]
        separation_rows.append({
            'epsilon': f'{eps:.8g}', 'nu': 0.5,
            'positive_excited_level_separation': f'{delta:.14g}',
            'separation_over_epsilon_2': f'{delta/eps**2:.14g}',
            'basis_size_per_component': excited_basis.size,
        })
    write_csv(data / 'excited_pair_separation.csv', separation_rows)

    # Basis convergence against high-order Galerkin reference, and exact free spectrum.
    conv_rows = []
    conv_sizes = [8, 12, 16, 24, 32, 48, 64, 80]
    eps_conv = 0.02
    nu_conv = 0.5
    reference_basis = PoleRegularBasis.build(nu_conv, max(nmain+12, 124), max(nq, 1000))
    reference_vals = reference_basis.solve(eps_conv, cosine_single)
    reference_chiral = nearest(reference_vals, -nu_conv*eps_conv)
    for size in conv_sizes:
        basis = PoleRegularBasis.build(nu_conv, size, max(420, 8*size))
        free = basis.solve(1.0, lambda t: np.zeros_like(t))
        free_err = float(np.max(np.abs(free[(free > 0)][:3] - np.array([1.,2.,3.]))))
        wall_vals = basis.solve(eps_conv, cosine_single)
        wall_chiral = nearest(wall_vals, -nu_conv*eps_conv)
        conv_rows.append({
            'basis_size_per_component': size, 'free_spectrum_max_abs_error_first3': f'{free_err:.8e}',
            'single_wall_epsilon': eps_conv, 'single_wall_lambda': f'{wall_chiral:.14g}',
            'absolute_difference_high_order_reference': f'{abs(wall_chiral-reference_chiral):.8e}',
            'high_order_reference_N': reference_basis.size,
        })
    write_csv(data / 'basis_convergence.csv', conv_rows)

    # Eigenfunction densities for one and two walls; use the saved quadrature coefficients.
    theta_grid = np.linspace(0.002, np.pi-0.002, 1500)
    density_rows = []
    eplot = 0.02
    for profile_name, prof in (('single_wall', cosine_single), ('two_wall', cosine_double)):
        vals, vecs = basis_hi.solve(eplot, prof, radius, vectors=True)
        if profile_name == 'single_wall':
            targets = [('chiral', -nu*eplot)]
        else:
            targets = [('negative', -np.sqrt(2)*nu*eplot),
                       ('positive', np.sqrt(2)*nu*eplot)]
        mass = prof(theta_grid)
        for branch, target in targets:
            idx = int(np.argmin(np.abs(vals - target)))
            u, v = basis_hi.reconstruct(vecs[:, idx], theta_grid)
            rho = np.abs(u)**2 + np.abs(v)**2
            # Integrate explicitly to keep compatibility with NumPy 1.x and 2.x.
            density_norm = np.sum(0.5 * (rho[:-1] + rho[1:]) * np.diff(theta_grid))
            rho /= density_norm
            for th, r, m in zip(theta_grid, rho, mass):
                density_rows.append({'profile': profile_name, 'branch': branch,
                                     'eigenvalue': f'{vals[idx]:.12g}', 'epsilon': eplot,
                                     'theta': f'{th:.12g}', 'theta_over_pi': f'{th/np.pi:.12g}',
                                     'probability_density': f'{r:.12g}', 'mass': f'{m:.12g}'})
    write_csv(data / 'eigenfunction_density.csv', density_rows)

    # Figure 1: single-wall chiral branch.
    sr = sorted(single_rows, key=lambda r: float(r['epsilon']))
    ex = np.array([float(r['epsilon']) for r in sr])
    ys = -np.array([float(r['lambda_numeric']) for r in sr])
    y1 = -np.array([float(r['leading_order']) for r in sr])
    y2 = -np.array([float(r['two_term_asymptotic']) for r in sr])
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(ex, ys, 'o', ms=4.8, color='#146C94', label='Galerkin eigenvalue')
    ax.plot(ex, y1, '--', color='#E76F51', label=r'leading $O(\epsilon)$')
    ax.plot(ex, y2, '-', color='#2A9D8F', label='two-term asymptotic')
    ax.set(xlabel=r'Semiclassical parameter $\epsilon$', ylabel=r'$-\lambda_{\epsilon,1/2}$',
           title='Single equatorial wall: chiral branch')
    ax.legend(frameon=False); ax.grid(alpha=.2)
    savefig(fig, figs/'single_wall_chiral_branch')

    # Figure 2: two-wall gap, two angular sectors.
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    colors = {0.5:'#146C94', 1.5:'#D97706'}
    for nuse in (0.5,1.5):
        rr = [r for r in gap_rows if float(r['nu']) == nuse]
        rr.sort(key=lambda r: float(r['epsilon']))
        x = np.array([float(r['epsilon']) for r in rr])
        ax.plot(x, [float(r['gap_numeric']) for r in rr], 'o', ms=4.5,
                color=colors[nuse], label=fr'Galerkin, $|\nu|={nuse:g}$')
        ax.plot(x, [float(r['two_term_asymptotic']) for r in rr], '-', lw=1.3,
                color=colors[nuse], alpha=.8, label=fr'asymptotic, $|\nu|={nuse:g}$')
    ax.set(xlabel=r'$\epsilon$', ylabel=r'Spectral gap $\Delta_{\epsilon,\nu}$',
           title=r'Symmetric two-wall profile $m(\theta)=\cos(2\theta)$')
    ax.legend(frameon=False,ncol=2); ax.grid(alpha=.2)
    savefig(fig, figs/'symmetric_two_wall_gap')

    # Figure 3: two localized copies of the first excited positive branch.
    erows = sorted(excited_rows, key=lambda r:(float(r['epsilon']),int(r['positive_pair_member'])))
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for member, color, marker in ((1,'#146C94','o'),(2,'#D97706','s')):
        rr=[r for r in erows if int(r['positive_pair_member'])==member]
        x=np.array([float(r['epsilon']) for r in rr]); y=np.array([float(r['eigenvalue_numeric']) for r in rr])
        ax.plot(x,y,marker+'-',ms=4,color=color,label=f'Galerkin member {member}')
    x=np.array(sorted(set(float(r['epsilon']) for r in erows)))
    ax.plot(x,2*np.sqrt(x),'--',color='#777777',label=r'leading $2\sqrt{\epsilon}$')
    ax.plot(x,2*np.sqrt(x)-3*x**1.5/8,':',color='#111111',label='local two-term center')
    ax.set(xlabel=r'$\epsilon$',ylabel='Positive eigenvalue',
           title=r'First excited double-wall cluster ($\nu=1/2$)')
    ax.legend(frameon=False); ax.grid(alpha=.2)
    savefig(fig, figs/'excited_double_wall_pair')

    # Figure 4: normalized probability densities and sign-changing masses.
    fig, axes=plt.subplots(2,1,figsize=(6.2,5.2),sharex=True,constrained_layout=True)
    colors={'single_wall':'#146C94','two_wall':'#9B5DE5'}
    for ax, name, title in zip(axes,('single_wall','two_wall'),('One wall','Two symmetric walls')):
        rows=[r for r in density_rows if r['profile']==name]
        reference_branch = 'chiral' if name=='single_wall' else 'positive'
        ref_rows=[r for r in rows if r['branch']==reference_branch]
        xx=np.array([float(r['theta_over_pi']) for r in ref_rows])
        mass=np.array([float(r['mass']) for r in ref_rows])
        branch_specs = ([('chiral',colors['single_wall'],r'chiral mode')]
                        if name=='single_wall' else
                        [('negative','#E76F51',r'$\lambda<0$ mode'),
                         ('positive','#9B5DE5',r'$\lambda>0$ mode')])
        for branch,color,label in branch_specs:
            rr=[r for r in rows if r['branch']==branch]
            xb=np.array([float(r['theta_over_pi']) for r in rr])
            rho=np.array([float(r['probability_density']) for r in rr])
            ax.plot(xb,rho,color=color,label=label)
        ax.set_ylabel('Probability density')
        twin=ax.twinx(); twin.plot(xx,mass,color='#555555',alpha=.5,ls='--',lw=1.2,label=r'$m(\theta)$')
        twin.axhline(0,color='#999999',lw=.6); twin.set_ylabel('Mass profile',color='#555555')
        ax.set_title(title,loc='left',fontsize=10.5); ax.grid(alpha=.15)
        if name=='single_wall': walls=[.5]
        else: walls=[.25,.75]
        for wloc in walls: ax.axvline(wloc,color='#777777',lw=.8,ls=':')
        ha,la=ax.get_legend_handles_labels(); ht,lt=twin.get_legend_handles_labels()
        if name == 'two_wall':
            ax.legend(ha+ht,la+lt,frameon=False,ncol=1,loc='upper center')
        else:
            ax.legend(ha+ht,la+lt,frameon=False,ncol=2,loc='upper right')
    axes[-1].set_xlabel(r'Latitude $\theta/\pi$')
    savefig(fig,figs/'localized_eigenfunction_density')

    # Figure 5: numerical convergence in basis size.
    fig, ax1=plt.subplots(figsize=(5.4,3.8))
    cs=np.array([int(r['basis_size_per_component']) for r in conv_rows])
    ew=np.array([float(r['absolute_difference_high_order_reference']) for r in conv_rows])
    ef=np.array([float(r['free_spectrum_max_abs_error_first3']) for r in conv_rows])
    ax1.semilogy(cs,ew,'o-',color='#146C94',label='single-wall eigenvalue error')
    ax1.set(xlabel='Basis functions per component',ylabel='Absolute error',
            title=r'Pole-regular Galerkin convergence ($\epsilon=0.02$)')
    ax2=ax1.twinx(); ax2.semilogy(cs,np.maximum(ef,1e-17),'s--',color='#D97706',label='free-spectrum error')
    ax2.set_ylabel('Free-spectrum error',color='#D97706')
    ax2.tick_params(axis='y',colors='#D97706')
    ax2.spines['right'].set_color('#D97706')
    lines=ax1.lines+ax2.lines
    ax1.legend(lines,[l.get_label() for l in lines],frameon=False,
               ncol=2,loc='upper center')
    ax1.grid(alpha=.2)
    savefig(fig,figs/'basis_convergence')

    # Figure 6: finite-range scaling of the two positive excited levels.
    sep_eps=np.array([float(r['epsilon']) for r in separation_rows])
    sep_val=np.array([float(r['positive_excited_level_separation']) for r in separation_rows])
    sep_ratio=np.array([float(r['separation_over_epsilon_2']) for r in separation_rows])
    order=np.argsort(sep_eps); sep_eps,sep_val,sep_ratio=sep_eps[order],sep_val[order],sep_ratio[order]
    fig,(ax_top,ax_bot)=plt.subplots(2,1,figsize=(5.4,5.3),sharex=True,
                                     gridspec_kw={'height_ratios':[1.35,1]})
    ax_top.loglog(sep_eps,sep_val,'o-',color='#6A4C93',label=r'computed $\delta\lambda$')
    ax_top.loglog(sep_eps,0.5*sep_eps**2,'--',color='#E76F51',label=r'$0.5\epsilon^2$ guide')
    ax_top.set_ylabel(r'Level separation $\delta\lambda$')
    ax_top.legend(frameon=False); ax_top.grid(which='both',alpha=.18)
    ax_bot.semilogx(sep_eps,sep_ratio,'o-',color='#146C94')
    ax_bot.set(xlabel=r'$\epsilon$',ylabel=r'$\delta\lambda/\epsilon^2$',
               title='Finite-range scaling diagnostic')
    ax_bot.grid(which='both',alpha=.18)
    savefig(fig,figs/'excited_pair_separation')

    independent_summary = None
    cost_summary = None
    if not quick:
        independent_summary = run_independent_benchmark(out, nmain, nq)
        cost_summary = run_cost_benchmark(out)

    max_free = max(float(r['absolute_error']) for r in validation_rows if r['test']=='massless')
    max_const = max(float(r['absolute_error']) for r in validation_rows if r['test']=='constant_mass')
    max_pairing = max(float(r['discrete_spectral_pairing_error']) for r in gap_rows)
    summary={
        'study':'Pole-regular Jacobi-Galerkin simulations for axial Dirac domain walls on S^2',
        'dimensionless_parameters':{'R':1.0,'mu':1.0},
        'main_basis_per_component':nmain,'main_quadrature_order':nq,
        'max_exact_massless_spectrum_error':max_free,
        'max_exact_constant_mass_spectrum_error':max_const,
        'max_discrete_spectral_pairing_error':max_pairing,
        'independent_shooting_benchmark':independent_summary,
        'solver_cost_benchmark':cost_summary,
        'simulation_parameter_epsilon':eps_grid.tolist(),
        'excited_branch_epsilon':excited_grid.tolist(),
        'profiles':{'single_wall':'m(theta)=mu*cos(theta)',
                    'two_wall':'m(theta)=mu*cos(2 theta)'},
        'note':'Exponential tunneling-scale eigenvalue splitting is not claimed to be resolved.'
    }
    (out/'simulation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT)
    parser.add_argument('--basis-size',type=int,default=112)
    parser.add_argument('--quadrature-order',type=int,default=1000)
    parser.add_argument('--quick',action='store_true',help='short smoke run; not paper tables')
    args=parser.parse_args()
    print(json.dumps(run(args.output.resolve(),args.basis_size,args.quadrature_order,args.quick),indent=2))

if __name__=='__main__':
    main()
