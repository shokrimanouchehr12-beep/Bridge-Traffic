#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
import pandas as pd
import os
import math
from datetime import datetime

# ============================================================
# PART 1: Execute the first code - "ارتباط تعمیرات و شدت بسته شدن.py"
# ============================================================

#random.seed(0)
#np.random.seed(0)

LIFETIME_YEARS = 100

def sample_beta_scaled(a, b, low, high):
    x = random.betavariate(a, b)
    return low + x * (high - low)

ADT_BASE = sample_beta_scaled(2.5, 2.5, 10000, 120000)
HGV_SHARE = sample_beta_scaled(2.5, 2.5, 0.05, 0.25)
CAR_SHARE = 1.0 - HGV_SHARE

BASE_TRAFFIC = {
    "ADT": ADT_BASE,
    "HGV_share": HGV_SHARE,
    "CAR_share": CAR_SHARE
}

MAINTENANCE_TABLE = [
     ("نئوپرن الاستومری",        20, 35, 0.027777778, 0.5),
    ("درز انبساط",             10, 40, 0.042857143, 0.3),
    ("سطح روسازی",             10, 20, 1.125,        7.5),
    ("گاردریل و نرده‌ها",      20, 40, 1.68,        11.2),
    ("موانع ایمنی",            25, 40, 1.68,        11.2),
    ("پوشش ضدخوردگی فولاد",    25, 35, 0.25,       1.5),
    ("تیرک لبه‌ای بتنی",       25, 40, 1.0,        7.84),
]

REPAIR_CLOSURE_SEVERITY = {
    "نئوپرن الاستومری":        {"partial": "low",    "major": "medium"},
    "درز انبساط":             {"partial": "medium", "major": "high"},
    "سطح روسازی":             {"partial": "medium", "major": "high"},
    "گاردریل و نرده‌ها":      {"partial": "low",    "major": "medium"},
    "موانع ایمنی":            {"partial": "low",    "major": "medium"},
    "پوشش ضدخوردگی فولاد":    {"partial": "low",    "major": "medium"},
    "تیرک لبه‌ای بتنی":       {"partial": "medium", "major": "high"},
}

CLOSURE_SCENARIOS = {
    "low": {
        "scenario_keys": [4, 7, 8],
        "description": "Minimal disruption"
    },
    "medium": {
        "scenario_keys": [2, 3, 7],
        "description": "Partial disruption"
    },
    "high": {
        "scenario_keys": [1, 5, 6],
        "description": "Severe disruption"
    }
}

scenarios = {
    1: ("Full closure",        {"qc_factor": (0.05,0.15), "v_queue": (0.3,1.0),  "path_factor": (1.3,1.7), "detour": True}),
    2: ("Partial lane closure",{"qc_factor": (0.4,0.6),   "v_queue": (1.5,3.0), "path_factor": (1.0,1.0), "detour": False}),
    3: ("Shoulder use/shift",  {"qc_factor": (0.6,0.8),   "v_queue": (4.0,6.0), "path_factor": (1.0,1.0), "detour": False}),
    4: ("Night-time/Weekend",  {"qc_factor": (0.85,0.95), "v_queue": (10.0,14.0),"path_factor": (1.0,1.0), "detour": False}),
    5: ("Contraflow",          {"qc_factor": (0.5,0.7),   "v_queue": (2.0,4.0), "path_factor": (1.0,1.0), "detour": False}),
    6: ("Crossover/bypass",    {"qc_factor": (0.4,0.6),   "v_queue": (5.0,7.0), "path_factor": (1.2,1.4), "detour": True}),
    7: ("Staged construction", {"qc_factor": (0.7,0.8),   "v_queue": (3.0,5.0), "path_factor": (1.0,1.0), "detour": False}),
    8: ("Dynamic lane mgmt",   {"qc_factor": (0.7,0.9),   "v_queue": (5.0,7.0), "path_factor": (1.0,1.0), "detour": False}),
}

closure_events = []

for item, t_partial, t_major, gwp_p, gwp_m in MAINTENANCE_TABLE:

    # Partial repairs
    n_partial = int(LIFETIME_YEARS // t_partial)
    sev_p = REPAIR_CLOSURE_SEVERITY[item]["partial"]

    for _ in range(n_partial):
        closure_events.append({
            "item": item,
            "repair_type": "partial",
            "interval_years": t_partial,
            "closure_severity": sev_p,
            "gwp_repair_kg": gwp_p
        })

    # Major repairs
    n_major = int(LIFETIME_YEARS // t_major)
    sev_m = REPAIR_CLOSURE_SEVERITY[item]["major"]

    for _ in range(n_major):
        closure_events.append({
            "item": item,
            "repair_type": "major",
            "interval_years": t_major,
            "closure_severity": sev_m,
            "gwp_repair_kg": gwp_m
        })

closure_events_detailed = []

for event in closure_events:
    severity = event["closure_severity"]
    valid_scenario_keys = CLOSURE_SCENARIOS[severity]["scenario_keys"]
    scenario_key = random.choice(valid_scenario_keys)
    scenario_name, scenario_params = scenarios[scenario_key]

    event_detailed = event.copy()
    event_detailed.update({
        "scenario_key": scenario_key,
        "scenario_name": scenario_name,
        "scenario_params": scenario_params
    })
    closure_events_detailed.append(event_detailed)

df_closures_detailed = pd.DataFrame(closure_events_detailed)

df_summary = (
    df_closures_detailed
    .groupby(["item", "repair_type", "closure_severity"], as_index=False)
    .agg(
        num_closures=("item", "count"),
        gwp_total_kg=("gwp_repair_kg", "sum")
    )
)

GWP_MAINTENANCE_TOTAL = df_closures_detailed["gwp_repair_kg"].sum()

scenario_summary_detailed = (
    df_closures_detailed
    .groupby(["item", "repair_type"])["scenario_key"]
    .apply(lambda x: sorted(set(x)))  # 
    .reset_index()
    .rename(columns={"scenario_key": "selected_scenarios"})
)

MODEL_CONTEXT = {
    "base_traffic": BASE_TRAFFIC,
    "closure_events": df_closures_detailed,
    "closure_summary": df_summary,
    "scenario_summary_detailed": scenario_summary_detailed,
    "gwp_maintenance_total_kg": GWP_MAINTENANCE_TOTAL,
    "lifetime_years": LIFETIME_YEARS
}

# ============================================================
# PART 2: Integrate with the second code - "bridge_micro model_full_weighted random clean_adt repair.py"
# Use outputs from Part 1: ADT, HGV_SHARE, selected scenarios from df_closures_detailed, number of closures, etc.
# ============================================================

# Extract key parameters from Part 1
#ADT_FROM_PART1 = BASE_TRAFFIC["ADT"]
#HGV_SHARE_FROM_PART1 = BASE_TRAFFIC["HGV_share"] * 100  # Convert to percent for Part 2
#SELECTED_SCENARIOS_FROM_PART1 = df_closures_detailed["scenario_key"].tolist()  # List of all selected scenario keys for all repairs
#NUM_REPAIRS_FROM_PART1 = len(SELECTED_SCENARIOS_FROM_PART1)  # Total number of repairs over lifetime

# Override random seeds if needed, but keep consistent
#random.seed(0)
#np.random.seed(0)

# Scenarios from Part 1 are already defined above, no need to redefine

scenario_keys = np.array(list(scenarios.keys()))

input_ranges = {
    "K": (0.07, 0.12),
    "D": (0.5, 0.65),
    "num_lanes_before": (2,4),
    "num_lanes_after": (1,4),
    "v_f": (70, 120),
    "dist_upstream": (300, 2000),
    "h_d_mean": (1.5,2.5),
    "h_d_sigma": (0.1,0.3),
    "a_max_mean": (1.2,2.0),
    "a_max_sigma": (0.2,0.4),
    "b_comf": (1.5,3.0),
    "tau_brake": (1.0,2.0),
    "tau_acc": (1.0,3.0),
}
logic_ranges = {
    "num_lanes_before": (2, 3),
    "num_lanes_closed": (0, 3),
    "duration_days": (5, 30),
    "interval_years": (10, 25),  # Will be overridden per repair from Part 1
    "extra_path_km": (2, 10)
}

# Scenario weights are not used since we take scenarios from Part 1
emission_factors = {
    "car": 0.2,
    "truck": 1.10
}

BASE_LANE_CAPACITY = 1800.0
L_VEH = 4.5
MIN_HEADWAY = 0.8

def sample_range(rng):
    return random.uniform(*rng)

def sample_int_range(rng):
    a, b = rng
    a_i = int(math.ceil(a))
    b_i = int(math.floor(b))
    if b_i < a_i:
        return a_i
    return random.randint(a_i, b_i)

def rand_param(name):
    if name in logic_ranges:
        rng = logic_ranges[name]
        if "lane" in name or name in ("num_lanes_closed",):
            return sample_int_range(rng)
        else:
            return sample_range(rng)
    else:
        val = sample_range(input_ranges[name])
        if "lane" in name:
            return int(val)
        if name == "v_f":
            return val / 3.6
        return val

def sample_scenario(sc_key):
    sc_name, params = scenarios[sc_key]
    qc_factor = sample_range(params["qc_factor"])
    v_queue = sample_range(params["v_queue"])
    path_factor = sample_range(params["path_factor"])
    return {"qc_factor": qc_factor, "v_queue": v_queue, "path_factor": path_factor}, params["detour"], sc_name

def bridge_modifiers(position):
    if position == "زیر":
        return {"qc_mult":0.9, "v_mult":1.05, "h_mult":1.0, "path_add":0.1}
    else:
        return {"qc_mult":0.7, "v_mult":0.8, "h_mult":1.2, "path_add":0.3}



def simulate_brae_micro(sc_key, N_sample=300, dt=0.5, sim_duration=900.0,
                        add_stochastic=True, detour_share_override=None,
                        save_all_trajectories=False, traj_sample_count=3,
                        verbose=False, adt_override=None, hgv_percent_override=None):
    max_tries = 200
    for attempt in range(max_tries):
        lanes_before = rand_param("num_lanes_before")
        num_lanes_closed = rand_param("num_lanes_closed")
        if num_lanes_closed > lanes_before:
            num_lanes_closed = lanes_before
        lanes_after = max(0, lanes_before - num_lanes_closed)

        v_f = rand_param("v_f")
        ADT = adt_override if adt_override is not None else sample_beta_scaled(2.5, 2.5, 10000, 120000)
    

        K = rand_param("K")
        D = rand_param("D")
        dist_upstream = rand_param("dist_upstream")
        h_d_mean = rand_param("h_d_mean")
        h_d_sigma = rand_param("h_d_sigma")
        a_max_mean = rand_param("a_max_mean")
        a_max_sigma = rand_param("a_max_sigma")
        b_comf = rand_param("b_comf")
        tau_brake = rand_param("tau_brake")
        tau_acc = rand_param("tau_acc")

        repair_duration_days = rand_param("duration_days")
        repair_interval_years = rand_param("interval_years")
        extra_path_km = rand_param("extra_path_km")

        sc_params, detour_flag, sc_name = sample_scenario(sc_key)

        qc_nominal = BASE_LANE_CAPACITY * max(1, lanes_after) * sc_params["qc_factor"]

        PHV = ADT * K * D
        q_in = PHV / 3600.0

        if qc_nominal <= max(1, lanes_after) * BASE_LANE_CAPACITY and sc_params["v_queue"] <= v_f*3.6:
            break
    else:
        raise RuntimeError("Sampling parameters failed")

    bridge_pos = random.choice(["روی","زیر"])
    mod = bridge_modifiers(bridge_pos)

    qc = qc_nominal * mod["qc_mult"]
    v_queue_mps = min(sc_params["v_queue"] * mod["v_mult"], v_f)
    path_factor = sc_params["path_factor"] + (mod["path_add"] if sc_params["path_factor"]!=1.0 else 0.0)

    if detour_flag:
        detour_distance = extra_path_km * 1000.0
    else:
        detour_distance = 0.0

    time = np.arange(0.0, sim_duration + dt, dt)
    steps = len(time)

    mean_interarrival = 1.0 / max(q_in, 1e-9)
    arrival_times = np.cumsum(np.random.exponential(scale=mean_interarrival, size=N_sample))
    arrival_times = arrival_times[arrival_times <= sim_duration]
    N = len(arrival_times)
    positions_init = -v_f * arrival_times

    heavy_share = hgv_percent_override / 100.0 if hgv_percent_override is not None else sample_beta_scaled(2.5, 2.5, 0.05, 0.25)
    vehicle_types = np.random.choice(['car','truck'], size=N, p=[1-heavy_share, heavy_share])

    if add_stochastic and N>0:
        a_max_perveh = np.clip(np.random.normal(loc=a_max_mean, scale=a_max_sigma, size=N), 0.01, None)
        tau_acc_perveh = np.clip(np.random.normal(loc=tau_acc, scale=0.5, size=N), 0.01, None)
        tau_brake_perveh = np.clip(np.random.normal(loc=tau_brake, scale=0.3, size=N), 0.01, None)
        hd_mean_truck = h_d_mean * 1.2
        hd_means = np.where(vehicle_types == 'truck', hd_mean_truck, h_d_mean)
        hd_sigma_arr = np.where(vehicle_types == 'truck', h_d_sigma*1.2, h_d_sigma)
    else:
        a_max_perveh = np.full(N, a_max_mean) if N>0 else np.array([])
        tau_acc_perveh = np.full(N, tau_acc) if N>0 else np.array([])
        tau_brake_perveh = np.full(N, tau_brake) if N>0 else np.array([])
        hd_means = np.full(N, h_d_mean) if N>0 else np.array([])
        hd_sigma_arr = np.full(N, h_d_sigma) if N>0 else np.array([])

    Lveh = L_VEH
    s_queue = max(Lveh + 2.0, 1.0)
    k_queue = 1.0 / s_queue

    L = np.zeros(steps)
    for i, t in enumerate(time):
        veh_in_queue = max(0.0, (PHV - qc) * (t / 3600.0))
        L[i] = veh_in_queue / max(k_queue, 1e-9)

    t_arr = np.full(N, np.nan)
    for idx in range(N):
        x0 = positions_init[idx]
        arrived = False
        for i, t in enumerate(time):
            x_t = x0 + v_f * t
            if x_t >= -L[i]:
                t_arr[idx] = t
                arrived = True
                break
        if not arrived:
            t_arr[idx] = max(0.0, -x0 / max(v_f, 1e-6))

    t_dep = np.full(N, np.nan)
    in_queue_idx = np.where(t_arr <= sim_duration)[0]
    in_queue_sorted = in_queue_idx[np.argsort(t_arr[in_queue_idx])] if len(in_queue_idx) > 0 else np.array([], dtype=int)

    if len(in_queue_sorted) > 0:
        hd_samples = np.maximum(np.random.normal(loc=hd_means[in_queue_sorted], scale=hd_sigma_arr[in_queue_sorted]), MIN_HEADWAY)
        t_service_start = t_arr[in_queue_sorted[0]]
        cumsum_hd = np.cumsum(hd_samples)
        for j, vid in enumerate(in_queue_sorted):
            service_time_for_vid = t_service_start + cumsum_hd[j]
            t_dep[vid] = max(t_arr[vid], service_time_for_vid)

    if detour_flag:
        if detour_share_override is None:
            detour_share = 1.0
        else:
            detour_share = float(detour_share_override)
        detour_mask = np.random.rand(N) < detour_share
        detour_dist_per_vehicle = np.where(detour_mask, detour_distance, 0.0)
    else:
        detour_mask = np.zeros(N, dtype=bool)
        detour_dist_per_vehicle = np.zeros(N)

    if save_all_trajectories:
        v = np.zeros((steps, N))
        a = np.zeros((steps, N))
        x = np.zeros((steps, N))
    else:
        sample_idxs = [0, N//2, N-1][:traj_sample_count] if N >= 3 else list(range(N))
        v_samples = {vid: np.zeros(steps) for vid in sample_idxs}
        a_samples = {vid: np.zeros(steps) for vid in sample_idxs}
        x_samples = {vid: np.zeros(steps) for vid in sample_idxs}

    for i in range(N):
        for k in range(steps):
            t = time[k]
            if t <= t_arr[i]:
                if save_all_trajectories:
                    v[k, i] = v_f
                    x[k, i] = positions_init[i] + v_f * t
                    a[k, i] = 0.0
                else:
                    if i in v_samples:
                        v_samples[i][k] = v_f
                        x_samples[i][k] = positions_init[i] + v_f * t
                        a_samples[i][k] = 0.0
            else:
                break

    for i in range(N):
        ta = t_arr[i]
        td = t_dep[i]
        if np.isnan(ta) or np.isnan(td):
            continue
        idx_ta = np.searchsorted(time, ta)
        idx_td = np.searchsorted(time, td)

        tau_b = max(1e-6, tau_brake_perveh[i]) if add_stochastic else max(1e-6, tau_brake)
        for k in range(idx_ta, min(steps, idx_td)):
            tt = time[k] - ta
            denom = max(1e-6, tau_b)
            v_now = v_queue_mps + (v_f - v_queue_mps) * math.exp(-tt / denom)
            v_now = min(max(v_now, v_queue_mps), v_f)
            a_now = -(v_f - v_queue_mps) / denom * math.exp(-tt / denom)
            if save_all_trajectories:
                v[k, i] = v_now
                a[k, i] = a_now
                if k == 0:
                    x[k, i] = positions_init[i] + v_now * time[k]
                else:
                    x[k, i] = x[k-1, i] + v_now * (time[k] - time[k-1])
            else:
                if i in v_samples:
                    v_samples[i][k] = v_now
                    a_samples[i][k] = a_now
                    if k == 0:
                        x_samples[i][k] = positions_init[i] + v_now * time[k]
                    else:
                        x_samples[i][k] = x_samples[i][k-1] + v_now * (time[k] - time[k-1])

        if idx_td < steps:
            a_max = a_max_perveh[i] if add_stochastic else a_max_mean
            tau_a = tau_acc_perveh[i] if add_stochastic else tau_acc
            for k in range(idx_td, steps):
                tt = time[k] - td
                denom = max(1e-6, tau_a)
                v_now_exp = v_f - (v_f - v_queue_mps) * math.exp(-tt / denom)
                a_now = (v_f - v_queue_mps) / denom * math.exp(-tt / denom)
                if a_now > a_max:
                    t_needed = (v_f - v_queue_mps) / a_max if a_max > 1e-6 else float('inf')
                    if tt <= t_needed:
                        v_now = v_queue_mps + a_max * tt
                        a_now = a_max
                    else:
                        v_now = v_f
                        a_now = 0.0
                else:
                    v_now = v_now_exp
                if save_all_trajectories:
                    v[k, i] = min(v_now, v_f)
                    a[k, i] = a_now
                    x[k, i] = x[k-1, i] + v[k, i] * (time[k] - time[k-1])
                else:
                    if i in v_samples:
                        v_samples[i][k] = min(v_now, v_f)
                        a_samples[i][k] = a_now
                        x_samples[i][k] = x_samples[i][k-1] + v_samples[i][k] * (time[k] - time[k-1])

    delay_per_vehicle = np.maximum(0.0, t_dep - t_arr)
    delay_per_vehicle = np.nan_to_num(delay_per_vehicle, 0.0)

    extra_dist_from_delay_m = v_queue_mps * delay_per_vehicle
    extra_dist_total_m = detour_dist_per_vehicle + extra_dist_from_delay_m

    gwp_per_vehicle = np.zeros(N)
    for i in range(N):
        extra_km = extra_dist_total_m[i] / 1000.0
        typ = vehicle_types[i]
        ef = emission_factors["truck"] if typ == 'truck' else emission_factors["car"]
        gwp_per_vehicle[i] = max(0.0, extra_km * ef)

    per_vehicle = pd.DataFrame({
        "vehicle_id": np.arange(N),
        "vehicle_type": vehicle_types,
        "arrival_s": t_arr,
        "departure_s": t_dep,
        "delay_s": delay_per_vehicle,
        "detour_m": detour_dist_per_vehicle,
        "extra_dist_m": extra_dist_total_m,
        "gwp_kgCO2eq": gwp_per_vehicle
    })

    mask_car = (vehicle_types == 'car')
    mask_truck = (vehicle_types == 'truck')
    gwp_cars_total = float(np.sum(gwp_per_vehicle[mask_car])) if N>0 else 0.0
    gwp_trucks_total = float(np.sum(gwp_per_vehicle[mask_truck])) if N>0 else 0.0
    gwp_sample_total = gwp_cars_total + gwp_trucks_total

    vehicles_impacted_total = ADT * repair_duration_days * D
    effective_N = max(1, N)
    scale_factor = vehicles_impacted_total / effective_N
    gwp_total_scaled = gwp_sample_total * scale_factor

    summary = {
        "scenario_key": sc_key,
        "scenario_name": sc_name,
        "bridge_pos": bridge_pos,
        "ADT": ADT,
        "K": K,
        "D": D,
        "lanes_before": lanes_before,
        "lanes_after": lanes_after,
        "num_lanes_closed": num_lanes_closed,
        "repair_duration_days": repair_duration_days,
        "v_f_mps": v_f,
        "v_queue_mps": v_queue_mps,
        "qc_veh_h": qc,
        "PHV_veh_h": PHV,
        "num_vehicles_simulated": int(N),
        "n_cars": int(np.sum(mask_car)),
        "n_trucks": int(np.sum(mask_truck)),
        "mean_delay_s": float(np.mean(delay_per_vehicle)) if N>0 else 0.0,
        "max_delay_s": float(np.max(delay_per_vehicle)) if N>0 else 0.0,
        "total_extra_dist_m_mean_per_vehicle": float(np.mean(extra_dist_total_m)) if N>0 else 0.0,
        "detour_distance_m_mean": float(np.mean(detour_dist_per_vehicle)) if N>0 else 0.0,
        "gwp_cars_kg_sample": gwp_cars_total,
        "gwp_trucks_kg_sample": gwp_trucks_total,
        "gwp_sample_total_kg": gwp_sample_total,
        "gwp_scaled_total_kg": gwp_total_scaled,
        "repair_interval_years_sampled": repair_interval_years,
        "repair_extra_path_km_sampled": extra_path_km,
        "vehicles_impacted_total_est": vehicles_impacted_total,
        "scale_factor_used": scale_factor
    }

    if save_all_trajectories:
        trajectories = {"time": time, "x": x, "v": v, "a": a}
    else:
        trajectories = {"time": time, "x_samples": x_samples, "v_samples": v_samples, "a_samples": a_samples}

    return {"summary": summary, "per_vehicle": per_vehicle, "trajectories": trajectories}

# Modified simulate_bridge_lifetime to use data from Part 1

def simulate_bridge_lifetime(N_sample_per_repair=300, lifetime_years=100,
                             save_all_trajectories=False, verbose=False,
                             monte_carlo_runs=1):
    results_list = []
    for mc in range(monte_carlo_runs):
        # new seed each run
        current_seed = int(datetime.now().timestamp()) + mc * 1000
        random.seed(current_seed)
        np.random.seed(current_seed)

        # ADT و HGV جدید برای این ران
        ADT_BASE = sample_beta_scaled(2.5, 2.5, 10000, 120000)
        HGV_SHARE = sample_beta_scaled(2.5, 2.5, 0.05, 0.25)

        # 
        closure_events = []
        for item, t_partial, t_major, gwp_p, gwp_m in MAINTENANCE_TABLE:
            n_partial = int(lifetime_years // t_partial)
            sev_p = REPAIR_CLOSURE_SEVERITY[item]["partial"]
            for _ in range(n_partial):
                closure_events.append({
                    "item": item,
                    "repair_type": "partial",
                    "interval_years": t_partial,
                    "closure_severity": sev_p,
                    "gwp_repair_kg": gwp_p
                })
            n_major = int(lifetime_years // t_major)
            sev_m = REPAIR_CLOSURE_SEVERITY[item]["major"]
            for _ in range(n_major):
                closure_events.append({
                    "item": item,
                    "repair_type": "major",
                    "interval_years": t_major,
                    "closure_severity": sev_m,
                    "gwp_repair_kg": gwp_m
                })

        closure_events_detailed = []
        for event in closure_events:
            severity = event["closure_severity"]
            valid_scenario_keys = CLOSURE_SCENARIOS[severity]["scenario_keys"]
            scenario_key = random.choice(valid_scenario_keys)
            scenario_name, scenario_params = scenarios[scenario_key]
            event_detailed = event.copy()
            event_detailed.update({
                "scenario_key": scenario_key,
                "scenario_name": scenario_name,
                "scenario_params": scenario_params
            })
            closure_events_detailed.append(event_detailed)

        selected_scenarios = [e["scenario_key"] for e in closure_events_detailed]
        num_repairs = len(selected_scenarios)

        lifetime_gwp_total = 0.0
        repair_breakdown = []
        for idx_rep, sc_key in enumerate(selected_scenarios):
            repair_interval_years = closure_events_detailed[idx_rep]["interval_years"]
            sim_res = simulate_brae_micro(sc_key, N_sample=N_sample_per_repair,
                                          save_all_trajectories=save_all_trajectories,
                                          adt_override=ADT_BASE,
                                          hgv_percent_override=HGV_SHARE * 100)
            gwp_single = sim_res["summary"]["gwp_scaled_total_kg"]
            lifetime_gwp_total += gwp_single
            repair_summary = sim_res["summary"].copy()
            repair_summary.update({
                "repair_index": idx_rep + 1,
                "gwp_single_kg": gwp_single,
                "interval_years": repair_interval_years,
                "item": closure_events_detailed[idx_rep]["item"],
                "repair_type": closure_events_detailed[idx_rep]["repair_type"]
            })
            repair_breakdown.append(repair_summary)

        results_list.append({
            "mc_run": mc+1,
            "num_repairs": num_repairs,
            "selected_scenarios": selected_scenarios,
            "gwp_lifetime_total_kg": lifetime_gwp_total,
            "repair_breakdown": repair_breakdown,
            "ADT": ADT_BASE  # برای استفاده در main
        })

    if monte_carlo_runs == 1:
        return results_list[0]
    else:
        df_rows = []
        for r in results_list:
            df_rows.append({
                "mc_run": r["mc_run"],
                "num_repairs": r["num_repairs"],
                "gwp_lifetime_total_kg": r["gwp_lifetime_total_kg"],
                "selected_scenarios": ",".join(map(str, r["selected_scenarios"]))
            })
        df = pd.DataFrame(df_rows)
        return {"runs_summary_df": df, "detailed_runs": results_list}

def run_lifetime_monte_carlo(n_runs=10, N_sample_per_repair=300, lifetime_years=100,
                             save_all_trajectories=False, out_dir="BRAE_lifetime_outputs_integrated"):
    os.makedirs(out_dir, exist_ok=True)
    all_run_summaries = []
    for i in range(n_runs):
        if (i+1) % 10 == 0:
            print(f"[{datetime.now().isoformat()}] Running lifetime MC {i+1}/{n_runs} ...")
        res = simulate_bridge_lifetime(N_sample_per_repair=N_sample_per_repair,
                                      lifetime_years=lifetime_years,
                                      save_all_trajectories=save_all_trajectories,
                                      verbose=False, monte_carlo_runs=1)
        run_name = f"lifetime_run_{i+1}"
        df_break = pd.DataFrame(res["repair_breakdown"])
        df_break.to_csv(os.path.join(out_dir, f"{run_name}_repairs.csv"), index=False)
        summary_row = {
            "run_index": i+1,
            "num_repairs": res["num_repairs"],
            "gwp_lifetime_total_kg": res["gwp_lifetime_total_kg"],
            "selected_scenarios": ";".join(map(str, res["selected_scenarios"]))
        }
        all_run_summaries.append(summary_row)
    df_all = pd.DataFrame(all_run_summaries)
    df_all.to_csv(os.path.join(out_dir, "lifetime_MC_summary.csv"), index=False)
    print(f"Lifetime Monte Carlo complete. Results saved to {out_dir}")
    return df_all

if __name__ == "__main__":
    # Print outputs from Part 1 for debugging
    print("=== BASE TRAFFIC FROM PART 1 ===")
    print(BASE_TRAFFIC)
    print("\n=== CLOSURE SUMMARY FROM PART 1 ===")
    print(df_summary)
    print(f"\nTOTAL NON-TRAFFIC GWP (kg): {GWP_MAINTENANCE_TOTAL:.2f}")
    print("\n=== SAMPLE DETAILED CLOSURE EVENTS (SORTED BY ITEM) FROM PART 1 ===")
    print(df_closures_detailed.sort_values(["item", "repair_type"]).reset_index(drop=True))
    print("\n=== DETAILED SCENARIO SUMMARY PER ITEM & REPAIR TYPE FROM PART 1 ===")
    for item in scenario_summary_detailed["item"].unique():
        print(f"\n{item}:")
        sub = scenario_summary_detailed[scenario_summary_detailed["item"] == item]
        for _, row in sub.iterrows():
            print(f"    {row['repair_type']}: {row['selected_scenarios']}")

    # Run the integrated Part 2
    N_runs = 1000/
    N_sample_per_repair = 300
    lifetime_years = 100
    out_dir = "BRAE_lifetime_outputs_integrated"
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []
    for run_idx in range(1, N_runs+1):
        seed = int(datetime.now().timestamp()) + run_idx
        random.seed(seed)
        np.random.seed(seed)
        result = simulate_bridge_lifetime(N_sample_per_repair=N_sample_per_repair,
                                 lifetime_years=lifetime_years,
                                 save_all_trajectories=False,
                                 verbose=False,
                                 monte_carlo_runs=1)
        scenario_sequence = result["selected_scenarios"]
        for repair_idx, repair in enumerate(result["repair_breakdown"]):
            row = {
                
                "item": repair["item"],
                "repair_type": repair["repair_type"],

                "scenario_key": repair["scenario_key"],
                "scenario_name": repair["scenario_name"],
                "bridge_pos": repair.get("bridge_pos", ""),
                "ADT": result["ADT"],
                "K": repair.get("K", np.nan),
                "D": repair.get("D", np.nan),
                "lanes_before": repair.get("lanes_before", np.nan),
                "lanes_after": repair.get("lanes_after", np.nan),
                "v_f_mps": repair.get("v_f_mps", np.nan),
                "v_queue_mps": repair.get("v_queue_mps", np.nan),
                "qc_veh_h": repair.get("qc_veh_h", np.nan),
                "PHV_veh_h": repair.get("PHV_veh_h", np.nan),
                "num_vehicles_simulated": repair["num_vehicles_simulated"],
                "n_cars": repair.get("n_cars", np.nan),
                "n_trucks": repair.get("n_trucks", np.nan),
                "mean_delay_s": repair.get("mean_delay_s", np.nan),
                "max_delay_s": repair.get("max_delay_s", np.nan),
                "total_extra_dist_m_mean_per_vehicle": repair.get("total_extra_dist_m_mean_per_vehicle", np.nan),
                "detour_distance_m_mean": repair.get("detour_distance_m_mean", np.nan),
                "gwp_cars_kg": repair.get("gwp_single_kg", np.nan) * 0.5,
                "gwp_trucks_kg": repair.get("gwp_single_kg", np.nan) * 0.5,
                "gwp_total_kg": repair.get("gwp_single_kg", np.nan),
                "interval_years": repair.get("interval_years", np.nan),
                "num_repairs_100yr": result["num_repairs"],
                "gwp_lifetime_100yr_kg": result["gwp_lifetime_total_kg"],
                "gwp_total_kg_scaled_to_ADT": repair.get("gwp_total_kg_scaled_to_ADT", np.nan),
                "gwp_lifetime_scaled_ADT_100yr_kg": result["gwp_lifetime_total_kg"],
                "run_name": f"MC_run_{run_idx}",
                "scenario_sequence": ";".join(map(str, scenario_sequence))
            }

            # ============================================================
            # POST-PROCESSING BLOCK: Traffic-dependent GWP (Partial + Major)
            # ============================================================

            # -------- Traffic correction factors --------
            def traffic_factor_from_adt(adt):
                if adt < 10000:
                    return 1.25
                elif adt < 30000:
                    return 1.00
                elif adt < 60000:
                    return 0.80
                else:
                    return 0.65

            def truck_factor(hgv_percent):
                if hgv_percent < 10:
                    return 1.00
                elif hgv_percent < 20:
                    return 0.90
                else:
                    return 0.80

            # -------- Maintenance Table (تمامی تعمیرات) --------
            MAINTENANCE_TABLE = [
                # item, partial_interval, major_interval, gwp_partial, gwp_major
                ("نئوپرن الاستومری", 20, 35, 0.027777778, 0.5),
                ("درز انبساط", 10, 40, 0.042857143, 0.3),
                ("سطح روسازی", 10, 20, 3.0, 20),
                ("گاردریل و نرده‌ها", 20, 40, 3.0, 20),
                ("موانع ایمنی", 25, 40, 3.0, 20),
                ("پوشش ضدخوردگی فولاد", 25, 35, 0.25, 1.5),
                ("تیرک لبه‌ای بتنی", 25, 40, 1.8, 14),
            ]

            # -------- GWP calculation function --------
            
            def compute_gwp_traffic_all_simple(adt, hgv_percent, lifetime_years=100):

                alpha_t = traffic_factor_from_adt(adt)
                alpha_h = truck_factor(hgv_percent)
                alpha_eff = alpha_t * alpha_h

                gwp_total = 0.0

                for (_, t_partial, t_major, gwp_p, gwp_m) in MAINTENANCE_TABLE:

                    # ✅ PARTIAL → traffic dependent
                    t_partial_adj = max(1e-6, t_partial * alpha_eff)
                    n_partial = int(lifetime_years // t_partial_adj)

                    # ✅ MAJOR → traffic independent
                    n_major = int(lifetime_years // t_major)

                    gwp_total += n_partial * gwp_p + n_major * gwp_m

                return gwp_total, alpha_t, alpha_h


            # ============================================================
            # APPLY TO EACH RUN / REPAIR
            # ============================================================
            try:
                ADT_run = result["ADT"]
                n_trucks = repair["n_trucks"]
                n_total  = repair["num_vehicles_simulated"]
                hgv_percent = 100.0 * n_trucks / max(1, n_total)

                # 
                gwp_repair, alpha_t, alpha_h = compute_gwp_traffic_all_simple(
                    adt=ADT_run,
                    hgv_percent=hgv_percent,
                    lifetime_years=100
                )

                # print and save
                row["gwp_repair"] = gwp_repair
                print(f"GWP_repair (kg CO2-eq): {gwp_repair:.2f}")
                print(f"Alpha factors: alpha_t={alpha_t:.3f}, alpha_h={alpha_h:.3f}")

            except Exception as e:
                row["gwp_repair"] = float("nan")
                print(f"GWP_repair calculation skipped: {e}")

            all_rows.append(row)
   
    df_new = pd.DataFrame(all_rows)
    out_file = os.path.join(out_dir, "lifetime_MC_full_details_with_sequence_integrated.csv")

    if os.path.exists(out_file):
        df_old = pd.read_csv(out_file)
        df_full = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_full = df_new

    df_full.to_csv(out_file, index=False, encoding='utf-8-sig')


    print(f"\nجزئیات کامل هر تعمیر با توالی سناریوها در فایل '{out_file}' ذخیره شد.")

    summary_rows = []
    for run_idx in range(1, N_runs+1):
        run_data = [row for row in all_rows if row["run_name"] == f"MC_run_{run_idx}"]
        if not run_data:
            continue
        scenario_sequence = ";".join(map(str, run_data[0]["scenario_sequence"].split(";")))
        total_gwp = sum(row["gwp_total_kg"] for row in run_data)
        total_ADT = sum(row["ADT"] for row in run_data)
        total_extra_dist = sum(row["total_extra_dist_m_mean_per_vehicle"] * row["num_vehicles_simulated"] for row in run_data)
        total_delay = sum(row["mean_delay_s"] * row["num_vehicles_simulated"] for row in run_data)
        summary_rows.append({
            "run_name": f"MC_run_{run_idx}",
            "scenario_sequence": scenario_sequence,
            "num_repairs": len(run_data),
            "total_gwp_kg": total_gwp,
            "total_ADT": total_ADT,
            "total_extra_dist_m": total_extra_dist,
            "total_delay_s": total_delay
        })
    

    df_new_summary = pd.DataFrame(summary_rows)
    summary_file = os.path.join(out_dir, "lifetime_MC_summary_per_run_integrated.csv")

    if os.path.exists(summary_file):
        df_old_summary = pd.read_csv(summary_file)
        df_summary = pd.concat([df_old_summary, df_new_summary], ignore_index=True)
    else:
        df_summary = df_new_summary

    df_summary.to_csv(summary_file, index=False)

    print(f"فایل summary هر Run در '{summary_file}' ذخیره شد.")

