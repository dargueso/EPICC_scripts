#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simplified version of *plot_probability_hourly_vs_daily_changes.py*
(without functions).  It plots wet‑hour probability heat‑maps for *Present*,
*Future* and the Δ (*Future – Present*).

Changes in this version
-----------------------
* **No functions** – every step is executed top‑level for clarity.
* **Y‑axis ticks duplicated on the right** for the last (Δ) column.
* **Δ colour‑bar reversed** so that **blue = positive**.
All other behaviour, styling and constants are identical.
"""


# ─── Imports ────────────────────────────────────────────────────────────
import numpy as np
import string
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ─── User settings ──────────────────────────────────────────────────────
SCALE       = "linear"        # "linear" | "log"
MASK_ZERO   = True             # mask exact‑zero cells
SHOW_DELTA  = True
XLIMHRLY = False
COLORMAP    = "viridis"       # base cmap for the two experiments
DELTA_CMAP  = "coolwarm_r"    # *reversed* so blue shows positive Δ
DATAFILEP    = "/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5/condprob_01H_given_DAY.nc"
DATAFILEF    = "/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5_CMIP6anom/condprob_01H_given_DAY.nc"



# ─── Load & pre‑process data ────────────────────────────────────────────
ds_p          = xr.open_dataset(DATAFILEP)
ds_f          = xr.open_dataset(DATAFILEF)

med_mask = xr.open_dataset('/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5/my_coastal_med_mask.nc')['combined_mask'].values

locs_names = {'Med_all':1,'Med_Ocean':3,'Med_Coast':2}

ds = xr.concat([ds_p, ds_f], dim="experiment")

# Variable names from condprob_01H_given_DAY.nc
# dims: cond_prob_n_wet     -> (exp, n_wet_timesteps, bin_DAY, y, x)
#       cond_prob_intensity -> (exp, bin_01H, bin_DAY, y, x)
#       n_events            -> (exp, bin_DAY, y, x)
cond_prob_n_wet     = np.nan_to_num(ds.cond_prob_n_wet.data,     nan=0.0)
cond_prob_intensity = np.nan_to_num(ds.cond_prob_intensity.data, nan=0.0)
n_events            = ds.n_events.data

whdp_weighted = cond_prob_n_wet     * n_events[:, np.newaxis, :, :, :]  # (exp, n_wet, bin_DAY, y, x)
hidp_weighted = cond_prob_intensity * n_events[:, np.newaxis, :, :, :]  # (exp, bin_01H, bin_DAY, y, x)

drain_centers = ds.bin_DAY.values          # bin centres: 2.5, 7.5, 12.5 … mm
hrain_centers = ds.bin_01H.values          # hourly bin centres
n_wet_vals    = ds.n_wet_timesteps.values  # 1, 2, ... 24

# Derive bin edges from centres (assumes uniform spacing)
_step = drain_centers[1] - drain_centers[0]
drain_edges = np.concatenate(([drain_centers[0] - _step / 2],
                               drain_centers + _step / 2)).astype(int)

hrain_edges = np.concatenate(([0], (hrain_centers[1:] + hrain_centers[:-1]) / 2, [hrain_centers[-1] + 1]))
hrain_edges[-1] = 100
hrain_edges = hrain_edges.astype(int)

labels  = [f"{lo}-{hi}" for lo, hi in zip(drain_edges[:-1], drain_edges[1:])]
labels.append(f'>{drain_edges[-1]}')
hlabels = [f"{lo}-{hi}" for lo, hi in zip(hrain_edges[:-2], hrain_edges[1:-1])]
hlabels.append(f'>{hrain_edges[-2]}')

for loc_name in locs_names.keys():

    loc = locs_names[loc_name]
    print(loc_name)

    mask = med_mask > 1 if loc_name == "Med_all" else med_mask == loc

    comp_samp = n_events[:, :, mask].sum(axis=2)          # (exp, bin_DAY)
    comp_wWet = whdp_weighted[:, :, :, mask].sum(axis=3)  # (exp, n_wet, bin_DAY)
    comp_wHr  = hidp_weighted[:, :, :, mask].sum(axis=3)  # (exp, bin_01H, bin_DAY)

    comp_wWet /= comp_samp[:, np.newaxis, :]
    comp_wHr  /= comp_samp[:, np.newaxis, :]

    # Mean wet hours per wet day: E[N_wet] = sum_k(k * P(N=k)) per bin and overall
    E_n_wet = (comp_wWet * n_wet_vals[np.newaxis, :, np.newaxis]).sum(axis=1)  # (exp, bin_DAY)
    mean_wet_hrs_pres = float((E_n_wet[0] * comp_samp[0]).sum() / comp_samp[0].sum())
    mean_wet_hrs_fut  = float((E_n_wet[1] * comp_samp[1]).sum() / comp_samp[1].sum())
    abs_change = mean_wet_hrs_fut - mean_wet_hrs_pres
    rel_change = 100.0 * abs_change / mean_wet_hrs_pres
    print(f"  {'Bin (mm)':<12} {'Present':>9} {'Future':>9} {'Abs':>8} {'Rel (%)':>9}")
    print(f"  {'-'*50}")
    for j, (lo, hi) in enumerate(zip(drain_edges[:-1], drain_edges[1:])):
        p = float(E_n_wet[0, j])
        f = float(E_n_wet[1, j])
        ac = f - p
        rc = 100.0 * ac / p if p > 0 else float('nan')
        print(f"  {lo}-{hi} mm{'':<4} {p:>9.2f} {f:>9.2f} {ac:>+8.2f} {rc:>+9.1f}%")
    print(f"  {'-'*50}")
    print(f"  {'Overall':<12} {mean_wet_hrs_pres:>9.2f} {mean_wet_hrs_fut:>9.2f} "
          f"{abs_change:>+8.2f} {rel_change:>+9.1f}%")
    # Overall excluding first bin (0-5 mm)
    p_exc = float((E_n_wet[0, 1:] * comp_samp[0, 1:]).sum() / comp_samp[0, 1:].sum())
    f_exc = float((E_n_wet[1, 1:] * comp_samp[1, 1:]).sum() / comp_samp[1, 1:].sum())
    ac_exc = f_exc - p_exc
    rc_exc = 100.0 * ac_exc / p_exc
    print(f"  {'Overall >5mm':<12} {p_exc:>9.2f} {f_exc:>9.2f} "
          f"{ac_exc:>+8.2f} {rc_exc:>+9.1f}%")
    # Overall for bins >30 mm
    idx30 = np.searchsorted(drain_edges, 30)   # first edge >= 30 → bins from that index onward
    p_30 = float((E_n_wet[0, idx30:] * comp_samp[0, idx30:]).sum() / comp_samp[0, idx30:].sum())
    f_30 = float((E_n_wet[1, idx30:] * comp_samp[1, idx30:]).sum() / comp_samp[1, idx30:].sum())
    ac_30 = f_30 - p_30
    rc_30 = 100.0 * ac_30 / p_30
    print(f"  {'Overall >30mm':<12} {p_30:>9.2f} {f_30:>9.2f} "
          f"{ac_30:>+8.2f} {rc_30:>+9.1f}%")

    wet_prob = xr.DataArray(
        comp_wWet * 100.0,
        coords={"experiment": ["Present", "Future"], "n_wet_timesteps": n_wet_vals, "bin_DAY": drain_centers},
        dims=["experiment", "n_wet_timesteps", "bin_DAY"],
        name="wet_prob"
    )

    int_prob = xr.DataArray(
        comp_wHr * 100.0,
        coords={"experiment": ["Present", "Future"], "bin_01H": hrain_centers, "bin_DAY": drain_centers},
        dims=["experiment", "bin_01H", "bin_DAY"],
        name="int_prob"
    )



    n_exp       = 2                                       # Present, Future
    n_panels    = n_exp + int(SHOW_DELTA)                 # +1 for Δ
    fig_width   = 13 if SHOW_DELTA else 8
    fig, axes   = plt.subplots(
        nrows=1, ncols=n_panels, figsize=(fig_width, 7), sharey=False
    )
    if n_panels == 1:          # when SHOW_DELTA = False, axes is not a list
        axes = [axes]          # make it iterable


    # ─── Draw the two experiment panels ─────────────────────────────────────
    experiment_im = []      # store a mappable for the colour‑bar
    for i in range(n_exp):
        da = wet_prob.isel(experiment=i)
        da = da.assign_coords(drain_index=("bin_DAY", range(len(drain_centers))))

        if MASK_ZERO:
            da = da.where(da != 0)

        # ----- colour scaling ----------------------------------------------
        if SCALE == "log":
            positive = da.where(da > 0)
            vmin = float(positive.min().item()) if float(positive.min()) > 0 else 1e-3
            norm = LogNorm(vmin=vmin, vmax=float(da.max().item()))
        else:
            norm = None

        im = da.plot.imshow(
            ax           = axes[i],
            x            = "n_wet_timesteps",
            y            = "drain_index",
            cmap         = COLORMAP,
            norm         = norm,
            add_colorbar = False,
            robust       = (SCALE == "linear")
        )
        experiment_im.append(im)
        axes[i].set_title(f"{string.ascii_lowercase[i]}", size='x-large', weight='bold',loc="left")
        axes[i].set_title(f"{str(int_prob.experiment.values[i])}", loc="center")
        #axes[i].text(0.5,0.99,f'{str(int_prob.experiment.values[i])}',color='k',fontsize='xx-large',verticalalignment = 'top',horizontalalignment = 'center', transform=axes[i].transAxes,zorder=105)

        # axes[i].set_title(str(int_prob.experiment.values[i]))
        axes[i].set_xlabel("N wet hours")

        # x-axis ticks
        axes[i].set_xticks(n_wet_vals[1::2])
        axes[i].set_xticklabels(n_wet_vals[1::2], fontsize=8)

        # y‑axis ticks & labels only in first column
        if i == 0:
            axes[i].set_ylabel("Daily rainfall (mm)")
            axes[i].set_yticks(range(len(drain_edges)))
            axes[i].set_yticklabels(labels)
        else:
            axes[i].set_ylabel("")
            axes[i].set_yticks([])
            axes[i].set_yticklabels([])
            axes[i].tick_params(axis="y", left=False, labelleft=False)

        axes[i].tick_params(axis="both", which="both", length=0)


    # ─── Δ panel (Future – Present) ─────────────────────────────────────────
    if SHOW_DELTA:
        delta_da = wet_prob.isel(experiment=1) - wet_prob.isel(experiment=0)
        delta_da = delta_da.assign_coords(drain_index=("bin_DAY", range(len(drain_centers))))
        if MASK_ZERO:
            delta_da = delta_da.where(delta_da != 0)

        vmax = float(np.nanmax(np.abs(delta_da)))
        delta_norm = plt.Normalize(vmin=-1.5, vmax=1.5)

        delta_im = delta_da.plot.imshow(
            ax           = axes[-1],
            x            = "n_wet_timesteps",
            y            = "drain_index",
            cmap         = DELTA_CMAP,     # *reversed* cmap
            norm         = delta_norm,
            add_colorbar = False,
            robust       = (SCALE == "linear")
        )

        axes[-1].set_title(f"{string.ascii_lowercase[2]}", size='x-large', weight='bold',loc="left")
        #axes[-1].text(0.5,0.99,'Future – Present',color='k',fontsize='xx-large',verticalalignment = 'top',horizontalalignment = 'center', transform=axes[-1].transAxes,zorder=105)

        axes[-1].set_title("Future – Present", loc="center")
        axes[-1].set_xlabel("N wet hours")

        # duplicate y‑axis ticks on the right
        axes[-1].set_ylabel("")
        axes[-1].set_yticks(range(len(drain_edges)))
        axes[-1].set_yticklabels(labels)
        axes[-1].tick_params(axis="y", labelright=True, right=True, length=0)
        axes[-1].tick_params(
            axis="y",
            labelright=True,
            right=True,
            labelleft=False,
            left=False,
            length=0,
        )

        # match x‑axis ticks
        axes[-1].set_xticks(n_wet_vals[1::2])
        axes[-1].set_xticklabels(n_wet_vals[1::2], fontsize=8)
        #axes[-1].tick_params(axis="x", which="both", length=0, rotation=90)
        axes[-1].tick_params(axis="both", which="both", length=0)


    # ─── Colour‑bars ────────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.10, 0.07, 0.50, 0.025])
    cbar_exp = plt.colorbar(
        experiment_im[0], cax=cbar_ax, orientation="horizontal", shrink=0.9
    )
    cbar_exp.set_label("Probability (%)")

    if SHOW_DELTA:
        cbar_axd = fig.add_axes([0.70, 0.07, 0.20, 0.025])
        cbar_del = plt.colorbar(
            delta_im, cax=cbar_axd, orientation="horizontal", shrink=0.9
        )
        cbar_del.set_label("Δ probability (%)")


    # ─── Layout & display ───────────────────────────────────────────────────
    #fig.suptitle("Wet hours probability by daily rainfall")
    fig.subplots_adjust(
        left=0.05, right=0.95, top=0.90, bottom=0.2, wspace=0.10, hspace=0.1
    )
    plt.savefig(f"/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/wet_hour_probability_heatmaps-{loc_name}_5mm.png", dpi=300, bbox_inches="tight")


    # # ###############################################################################
    # # ##### PLOT 2 hourly vs daily intensity
    # # # ###############################################################################

    # if XLIMHRLY:
    #     int_prob_plot = int_prob.sel(hrain_bin=slice(10,200),drain_bin=slice(10,200))
    #     drain_edges = int_prob_plot.drain_bin.values                     # 0, 5, 10 … mm
    #     hrain_centers = int_prob_plot.hrain_bin.values                  # 0.5, 1.5, 2.5 … mm
    #     hrain_edges = np.concatenate(([0], (hrain_centers[1:] + hrain_centers[:-1]) / 2, [hrain_centers[-1] + 1]))
    #     hrain_edges[-1]=100
    #     hrain_edges = hrain_edges.astype(int)

    # else:
    #     int_prob_plot = int_prob

    # fig, axes   = plt.subplots(
    #     nrows=1, ncols=n_panels, figsize=(fig_width, 10), sharey=False
    # )
    # if n_panels == 1:          # when SHOW_DELTA = False, axes is not a list
    #     axes = [axes]          # make it iterable


    # labels = [f"{lo}–{hi}" for lo, hi in zip(drain_edges[:-1], drain_edges[1:])]
    # labels.append(f'>{drain_edges[-1]}')

    # hlabels = [f"{lo}–{hi}" for lo, hi in zip(hrain_edges[:-2], hrain_edges[1:-1])]
    # hlabels.append(f'>{hrain_edges[-2]}')

    # # ─── Draw the two experiment panels ─────────────────────────────────────
    # experiment_im = []      # store a mappable for the colour‑bar
    # for i in range(n_exp):
    #     da = int_prob_plot.isel(experiment=i)
    #     da = da.assign_coords(drain_index=("drain_bin", range(len(drain_edges))))
    #     da = da.assign_coords(hrain_index=("hrain_bin", range(len(hrain_centers))))
        

    #     if MASK_ZERO:
    #         da = da.where(da != 0)

    #     # ----- colour scaling ----------------------------------------------
    #     if SCALE == "log":
    #         positive = da.where(da > 0)
    #         vmin = float(positive.min().item()) if float(positive.min()) > 0 else 1e-3
    #         norm = LogNorm(vmin=vmin, vmax=float(da.max().item()))
    #     else:
    #         norm = None

    #     im = da.plot.imshow(
    #         ax           = axes[i],
    #         x            = "hrain_index",
    #         y            = "drain_index",
    #         cmap         = COLORMAP,
    #         norm         = norm,
    #         add_colorbar = False,
    #         robust       = (SCALE == "linear")
    #     )
    #     experiment_im.append(im)

    #     axes[i].set_title(str(ds.experiment.values[i]))
    #     axes[i].set_xlabel("Hourly rainfall (mm)")

    #     # x‑axis ticks
    #     axes[i].set_xticks(range(0,len(hrain_centers),2))
    #     axes[i].set_xticklabels(hlabels[::2], fontsize=8)
    #     axes[i].tick_params(axis="x", which="both", length=0, rotation=90)

    #     # y‑axis ticks & labels only in first column
    #     if i == 0:
    #         axes[i].set_ylabel("Daily rainfall (mm)")
    #         axes[i].set_yticks(range(len(drain_edges)))
    #         axes[i].set_yticklabels(labels)
    #     else:
    #         axes[i].set_ylabel("")
    #         axes[i].set_yticks([])
    #         axes[i].set_yticklabels([])
    #         axes[i].tick_params(axis="y", left=False, labelleft=False)

    #     if XLIMHRLY:
    #         axes[i].set_xlim([10,100])


    #     axes[i].tick_params(axis="both", which="both", length=0)


    # # ─── Δ panel (Future – Present) ─────────────────────────────────────────
    # if SHOW_DELTA:
    #     delta_da = int_prob_plot.isel(experiment=1) - int_prob_plot.isel(experiment=0)
    #     delta_da = delta_da.assign_coords(drain_index=("drain_bin", range(len(drain_edges))))
    #     delta_da = delta_da.assign_coords(hrain_index=("hrain_bin", range(len(hrain_centers))))
    #     if MASK_ZERO:
    #         delta_da = delta_da.where(delta_da != 0)

    #     vmax = float(np.nanmax(np.abs(delta_da)))
    #     delta_norm = plt.Normalize(vmin=-vmax, vmax=vmax)

    #     delta_im = delta_da.plot.imshow(
    #         ax           = axes[-1],
    #         x            = "hrain_index",
    #         y            = "drain_index",
    #         cmap         = DELTA_CMAP,     # *reversed* cmap
    #         norm         = delta_norm,
    #         add_colorbar = False,
    #     )

    #     axes[-1].set_title("Future – Present")
    #     axes[-1].set_xlabel("Hourly rainfall (mm)")

    #     # duplicate y‑axis ticks on the right
    #     axes[-1].set_ylabel("")
    #     axes[-1].set_yticks(range(len(drain_edges)))
    #     axes[-1].set_yticklabels(labels)
    #     axes[-1].tick_params(axis="y", labelright=True, right=True, length=0)
    #     axes[-1].tick_params(
    #         axis="y",
    #         labelright=True,
    #         right=True,
    #         labelleft=False,
    #         left=False,
    #         length=0,
    #     )

    #     # match x‑axis ticks
    #     axes[-1].set_xticks(range(0,len(hrain_centers),2))
    #     axes[-1].set_xticklabels(hlabels[::2], fontsize=8)
    #     axes[-1].tick_params(axis="x", which="both", length=0, rotation=90)
    #     axes[-1].tick_params(axis="both", which="both", length=0)
        

    # # ─── Colour‑bars ────────────────────────────────────────────────────────
    # cbar_ax = fig.add_axes([0.10, 0.07, 0.50, 0.025])
    # cbar_exp = plt.colorbar(
    #     experiment_im[0], cax=cbar_ax, orientation="horizontal", shrink=0.9
    # )
    # cbar_exp.set_label("Probability (%)")

    # if SHOW_DELTA:
    #     cbar_axd = fig.add_axes([0.70, 0.07, 0.20, 0.025])
    #     cbar_del = plt.colorbar(
    #         delta_im, cax=cbar_axd, orientation="horizontal", shrink=0.9
    #     )
    #     cbar_del.set_label("Δ probability (%)")


    # # ─── Layout & display ───────────────────────────────────────────────────
    # fig.suptitle("Hourly vs daily rainfall intensity")
    # fig.subplots_adjust(
    #     left=0.05, right=0.95, top=0.90, bottom=0.2, wspace=0.10, hspace=0.1
    # )
    # plt.savefig(f"/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/intensity_probability_heatmaps-{loc_name}_5mm.png", dpi=300, bbox_inches="tight")
