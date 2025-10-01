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
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import uniform_filter

def _window_sum(arr, radius):
    """
    Sliding-window *sum* over the last two spatial axes (ny, nx).
    Works for both 3-D and 4-D arrays by building the size tuple at run-time.
    """
    k = 2 * radius + 1
    size = [1] * (arr.ndim - 2) + [k, k]   # e.g. (1,1,k,k) or (1,k,k)
    return uniform_filter(arr, size=size, mode="nearest") * (k * k)

# ─── User settings ──────────────────────────────────────────────────────
SCALE       = "linear"        # "linear" | "log"
MASK_ZERO   = True             # mask exact‑zero cells
SHOW_DELTA  = True
XLIMHRLY = False
COLORMAP    = "viridis"       # base cmap for the two experiments
DELTA_CMAP  = "coolwarm_r"    # *reversed* so blue shows positive Δ
DATAFILEP    = "/home/dargueso//postprocessed/EPICC/EPICC_2km_ERA5/rainfall_probability_optimized_conditional_5mm_bins.nc" # 
DATAFILEF    = "/home/dargueso//postprocessed/EPICC/EPICC_2km_ERA5_CMIP6anom/rainfall_probability_optimized_conditional_5mm_bins.nc" #



# ─── Load & pre‑process data ────────────────────────────────────────────
ds_p          = xr.open_dataset(DATAFILEP)
ds_f          = xr.open_dataset(DATAFILEF)
buffer= 25


med_mask = xr.open_dataset('/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5/my_coastal_med_mask.nc')['combined_mask'].values

locs_names = {'Med_all':1,'Med_Ocean':3,'Med_Coast':2}

ds = xr.concat([ds_p, ds_f], dim="experiment")

wet_hour_dist_present = ds.wet_hours_distribution.data
hourly_intensity_dist_present = ds.hourly_distribution.data
samples_per_bin_present = ds.samples_per_bin.data

wet_hour_dist_present = np.nan_to_num(wet_hour_dist_present, nan=0.0)
hourly_intensity_dist_present = np.nan_to_num(hourly_intensity_dist_present, nan=0.0)

whdp_weighted = wet_hour_dist_present * samples_per_bin_present[:, :, np.newaxis, :, :]
hidp_weighted = hourly_intensity_dist_present * samples_per_bin_present[:, :, np.newaxis, :, :]

for loc_name in locs_names.keys():
    
    loc = locs_names[loc_name]
    print(loc_name)
    
    if loc_name == "Med_all":
        comp_samp = samples_per_bin_present[:,:,med_mask>1].sum(axis=2)
        comp_wWet = whdp_weighted[:,:,:,med_mask>1].sum(axis=3)
        comp_wHr = hidp_weighted[:,:,:,med_mask>1].sum(axis=3)
    else:
        comp_samp = samples_per_bin_present[:,:,med_mask==loc].sum(axis=2)
        comp_wWet = whdp_weighted[:,:,:,med_mask==loc].sum(axis=3)
        comp_wHr = hidp_weighted[:,:,:,med_mask==loc].sum(axis=3)

    comp_wWet /= comp_samp[:, :, np.newaxis]
    comp_wHr /= comp_samp[:, :, np.newaxis]

    drain_edges = ds.drain_bin.values                     # 0, 5, 10 … mm
    hrain_centers = ds.hrain_bin.values                  # 0.5, 1.5, 2.5 … mm
    hrain_edges = np.concatenate(([0], (hrain_centers[1:] + hrain_centers[:-1]) / 2, [hrain_centers[-1] + 1]))
    hrain_edges[-1]=100
    hrain_edges = hrain_edges.astype(int)
    hour_vals   = ds.hour.values                          # 0 … 23

    labels = [f"{lo}–{hi}" for lo, hi in zip(drain_edges[:-1], drain_edges[1:])]
    labels.append(f'>{drain_edges[-1]}')

    hlabels = [f"{lo}–{hi}" for lo, hi in zip(hrain_edges[:-2], hrain_edges[1:-1])]
    hlabels.append(f'>{hrain_edges[-2]}')


    wet_prob = xr.DataArray(
        comp_wWet[:, :, :] * 100.0,
        coords={"experiment":["Present", "Future"],"drain_bin": drain_edges, "hour": hour_vals},
        dims=["experiment","drain_bin", "hour"],
        name="wet_prob"
    )

    int_prob = xr.DataArray(
        comp_wHr[:, :, :] * 100.0,
        coords={"experiment":["Present", "Future"],"drain_bin": drain_edges, "hrain_bin": hrain_centers},
        dims=["experiment","drain_bin", "hrain_bin"],
        name="int_prob"
    )



    n_exp       = 2                                       # Present, Future
    n_panels    = n_exp + int(SHOW_DELTA)                 # +1 for Δ
    fig_width   = 21 if SHOW_DELTA else 14
    fig, axes   = plt.subplots(
        nrows=1, ncols=n_panels, figsize=(fig_width, 10), sharey=False
    )
    if n_panels == 1:          # when SHOW_DELTA = False, axes is not a list
        axes = [axes]          # make it iterable


    # ─── Draw the two experiment panels ─────────────────────────────────────
    experiment_im = []      # store a mappable for the colour‑bar
    for i in range(n_exp):
        da = wet_prob.isel(experiment=i)
        da = da.assign_coords(drain_index=("drain_bin", range(len(drain_edges))))

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
            x            = "hour",
            y            = "drain_index",
            cmap         = COLORMAP,
            norm         = norm,
            add_colorbar = False,
            robust       = (SCALE == "linear")
        )
        experiment_im.append(im)

        axes[i].set_title(str(ds.experiment.values[i]))
        axes[i].set_xlabel("Hour of day")

        # x‑axis ticks
        axes[i].set_xticks(hour_vals)
        axes[i].set_xticklabels(hour_vals, fontsize=8)

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
        delta_da = delta_da.assign_coords(drain_index=("drain_bin", range(len(drain_edges))))
        if MASK_ZERO:
            delta_da = delta_da.where(delta_da != 0)

        vmax = float(np.nanmax(np.abs(delta_da)))
        delta_norm = plt.Normalize(vmin=-vmax, vmax=vmax)

        delta_im = delta_da.plot.imshow(
            ax           = axes[-1],
            x            = "hour",
            y            = "drain_index",
            cmap         = DELTA_CMAP,     # *reversed* cmap
            norm         = None,
            add_colorbar = False,
            robust       = (SCALE == "linear")
        )

        axes[-1].set_title("Future – Present")
        axes[-1].set_xlabel("Hours of rain")

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
        axes[-1].set_xticks(hour_vals)
        axes[-1].set_xticklabels(hour_vals, fontsize=8)
        #axes[-1].tick_params(axis="x", which="both", length=0, rotation=90)
        axes[-1].tick_params(axis="both", which="both", length=0)


    # ─── Colour‑bars ────────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.10, 0.07, 0.50, 0.025])
    cbar_exp = plt.colorbar(
        experiment_im[0], cax=cbar_ax, orientation="horizontal", shrink=0.9
    )
    cbar_exp.set_label("Wet‑hour probability (%)")

    if SHOW_DELTA:
        cbar_axd = fig.add_axes([0.70, 0.07, 0.20, 0.025])
        cbar_del = plt.colorbar(
            delta_im, cax=cbar_axd, orientation="horizontal", shrink=0.9
        )
        cbar_del.set_label("Δ probability (%)")


    # ─── Layout & display ───────────────────────────────────────────────────
    fig.suptitle("Wet‑hour probability by daily rainfall")
    fig.subplots_adjust(
        left=0.05, right=0.95, top=0.90, bottom=0.2, wspace=0.10, hspace=0.1
    )
    plt.savefig(f"/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/wet_hour_probability_heatmaps-{loc_name}_5mm.png", dpi=300, bbox_inches="tight")


    # ###############################################################################
    # ##### PLOT 2 hourly vs daily intensity
    # # ###############################################################################

    if XLIMHRLY:
        int_prob_plot = int_prob.sel(hrain_bin=slice(10,200),drain_bin=slice(10,200))
        drain_edges = int_prob_plot.drain_bin.values                     # 0, 5, 10 … mm
        hrain_centers = int_prob_plot.hrain_bin.values                  # 0.5, 1.5, 2.5 … mm
        hrain_edges = np.concatenate(([0], (hrain_centers[1:] + hrain_centers[:-1]) / 2, [hrain_centers[-1] + 1]))
        hrain_edges[-1]=100
        hrain_edges = hrain_edges.astype(int)

    else:
        int_prob_plot = int_prob

    fig, axes   = plt.subplots(
        nrows=1, ncols=n_panels, figsize=(fig_width, 10), sharey=False
    )
    if n_panels == 1:          # when SHOW_DELTA = False, axes is not a list
        axes = [axes]          # make it iterable


    labels = [f"{lo}–{hi}" for lo, hi in zip(drain_edges[:-1], drain_edges[1:])]
    labels.append(f'>{drain_edges[-1]}')

    hlabels = [f"{lo}–{hi}" for lo, hi in zip(hrain_edges[:-2], hrain_edges[1:-1])]
    hlabels.append(f'>{hrain_edges[-2]}')

    # ─── Draw the two experiment panels ─────────────────────────────────────
    experiment_im = []      # store a mappable for the colour‑bar
    for i in range(n_exp):
        da = int_prob_plot.isel(experiment=i)
        da = da.assign_coords(drain_index=("drain_bin", range(len(drain_edges))))
        da = da.assign_coords(hrain_index=("hrain_bin", range(len(hrain_centers))))
        

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
            x            = "hrain_index",
            y            = "drain_index",
            cmap         = COLORMAP,
            norm         = norm,
            add_colorbar = False,
            robust       = (SCALE == "linear")
        )
        experiment_im.append(im)

        axes[i].set_title(str(ds.experiment.values[i]))
        axes[i].set_xlabel("Hourly rainfall (mm)")

        # x‑axis ticks
        axes[i].set_xticks(range(0,len(hrain_centers),2))
        axes[i].set_xticklabels(hlabels[::2], fontsize=8)
        axes[i].tick_params(axis="x", which="both", length=0, rotation=90)

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

        if XLIMHRLY:
            axes[i].set_xlim([10,100])


        axes[i].tick_params(axis="both", which="both", length=0)


    # ─── Δ panel (Future – Present) ─────────────────────────────────────────
    if SHOW_DELTA:
        delta_da = int_prob_plot.isel(experiment=1) - int_prob_plot.isel(experiment=0)
        delta_da = delta_da.assign_coords(drain_index=("drain_bin", range(len(drain_edges))))
        delta_da = delta_da.assign_coords(hrain_index=("hrain_bin", range(len(hrain_centers))))
        if MASK_ZERO:
            delta_da = delta_da.where(delta_da != 0)

        vmax = float(np.nanmax(np.abs(delta_da)))
        delta_norm = plt.Normalize(vmin=-vmax, vmax=vmax)

        delta_im = delta_da.plot.imshow(
            ax           = axes[-1],
            x            = "hrain_index",
            y            = "drain_index",
            cmap         = DELTA_CMAP,     # *reversed* cmap
            norm         = delta_norm,
            add_colorbar = False,
        )

        axes[-1].set_title("Future – Present")
        axes[-1].set_xlabel("Hourly rainfall (mm)")

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
        axes[-1].set_xticks(range(0,len(hrain_centers),2))
        axes[-1].set_xticklabels(hlabels[::2], fontsize=8)
        axes[-1].tick_params(axis="x", which="both", length=0, rotation=90)
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
    fig.suptitle("Hourly vs daily rainfall intensity")
    fig.subplots_adjust(
        left=0.05, right=0.95, top=0.90, bottom=0.2, wspace=0.10, hspace=0.1
    )
    plt.savefig(f"/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/intensity_probability_heatmaps-{loc_name}_5mm.png", dpi=300, bbox_inches="tight")
