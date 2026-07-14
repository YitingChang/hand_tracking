try:
    import json
    from pathlib import Path
    import pandas as pd
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    import matplotlib.pyplot as plt
    import sys
    from itertools import groupby
    import gzip
except ModuleNotFoundError as e:
    raise Exception("You probably need to run: conda activate robot_venv")


def find_missing_packet_times(ccsr_nan, timestamps, THRESHOLD):
    """Returns the time of missing packets. Many missing packets in a row (>THRESHOLD) (e.g. due to a pause in the robot motion) are not counted since these are not "lost" but are just the ones not being recorded."""
    timestamps_bool = np.zeros(timestamps[-1] + 1, dtype="bool")
    timestamps_bool[timestamps] = 1
    neighbor_counts = np.zeros(timestamps[-1] + 1, dtype="int")

    idx = 0
    for _, group in groupby(timestamps_bool):
        count = len(list(group))
        neighbor_counts[idx : idx + count] = count
        idx += count

    missing_packets = np.zeros(timestamps[-1] + 1, dtype="bool")
    missing_packets[neighbor_counts < THRESHOLD] = 1
    missing_packets[timestamps_bool] = 0

    missing_packet_times = np.where(missing_packets)[0]
    assert set(timestamps) - set(missing_packet_times) == set(timestamps)

    # TODO: Currently not implemented detection many missing packets (>THRESHOLD) in a row, which are not counted as "lost" but are just the ones not being recorded between two different motions.
    last_100 = [True] * 100
    ccsr_alt = []
    first_non_nan_found = False
    ts_prev = 0
    packet_loss_times_alt = []
    for k in range(len(timestamps)):

        # Read current ccsr value
        ccsr_curr = ccsr_nan[k]

        # Determine timestamp
        ts_curr = timestamps[k]
        if k == 0:
            ts_prev = ts_curr - 1

        # Skip initial NaN values
        if first_non_nan_found is False and np.isnan(ccsr_curr):
            ccsr_alt.append(np.nan)
            ts_prev = ts_curr  # Reset timestamp to current
            continue
        else:
            first_non_nan_found = True

        # Determine if jump in timestamp and add False values
        ts_jump = ts_curr - ts_prev
        if ts_jump > 1:
            for j in range(ts_jump - 1):
                last_100.pop(0)
                last_100.append(False)
                packet_loss_times_alt.append(ts_prev + j + 1)  # Add missing timestamp
        ts_prev = ts_curr  # Reset timestamp to current

        # Calculate expected values
        value_if_true = np.nansum(last_100[1:]) + 1
        value_if_false = np.nansum(last_100[1:])

        # Use known CCSR to determine if True or False
        if ccsr_curr == value_if_true:
            last_100.pop(0)
            last_100.append(True)
        elif ccsr_curr == value_if_false:
            last_100.pop(0)
            last_100.append(False)
            packet_loss_times_alt.append(ts_curr)  # Add current timestamp

        else:
            print(f"ccsr_nan[{k}] = {ccsr_curr} does not match expected values {value_if_true} or {value_if_false}")

        # Calculate CCSR from last 100 values
        ccsr_calc = np.nansum(last_100)
        ccsr_alt.append(ccsr_calc)

        # Check if calculated matches known
        if ccsr_calc != ccsr_curr:
            print(f"Mismatch at index {k}: ccsr_nan={ccsr_nan[k]} ccsr_alt={ccsr_calc}")

        if ccsr_alt[k] != ccsr_nan[k]:
            print(f"Mismatch at index {k}: ccsr_nan={ccsr_nan[k]} ccsr_alt={ccsr_alt[k]}")

    # Assert all previously found times are in new method
    # assert set(missing_packet_times) - set(packet_loss_times_alt) == set()

    return packet_loss_times_alt


def parse_state(state_string):
    # JSON 10x faster than eval to convert to dict
    state_dict = json.loads(state_string)
    return state_dict


def read_data_from_txt_gz_file(txt_gz_file):

    # Read the compressed file
    with gzip.open(txt_gz_file, "rb") as f:
        txt = f.read()

    return txt


def read_data_from_txt_file(txt_path):

    if txt_path.suffix == ".gz":
        with gzip.open(txt_path, "rb") as f:
            txt = f.read()
            txt = txt.decode("utf-8")
    elif txt_path.suffix == ".txt":
        with open(txt_path) as f:
            txt = f.read()
    else:
        raise ValueError("File extension must be .txt or .txt.gz")

    text_decompressed = txt

    # with open(txt_path) as f:
    #     text = f.read()
    #     text_decompressed = text

    # # Make directory if needed
    # if zlb_path.parent.exists() is False:
    #     zlb_path.parent.mkdir(parents=True)

    # # Compress with zlib
    # text_compressed = zlib.compress(text.encode())
    # with open(zlb_path, "wb") as f:
    #     f.write(text_compressed)

    # # Decompress with zlib
    # with open(zlb_path, "rb") as f:
    #     text_decompressed = zlib.decompress(f.read()).decode()

    # Split based on "\n" character
    lines = text_decompressed.split("\n")
    if lines[-1] == "":
        lines = lines[:-1]  # Remove last line, which is empty

    state_dict_list = []
    for l in lines:
        state_dict_list.append(parse_state(l))

    df = pd.DataFrame.from_dict(state_dict_list)

    # Represent dropped timesteps as empty rows

    # Adjust current_errors column so that DataFrame converted correctly
    for i, v in enumerate(df["current_errors"]):
        if len(v) == 0:
            val = ""
        else:
            val = "/".join(v)
        df.loc[i, "current_errors"] = val

    frames = []
    for k in df.columns:
        col_type = type(df[k].iloc[0])
        if col_type == list and k != "current_errors":
            num_sub_cols = len(df[k].iloc[0])
        else:
            num_sub_cols = 1
        multiindex = pd.MultiIndex.from_product([[k], [str(i) for i in range(num_sub_cols)]])

        frames.append(
            pd.DataFrame(
                df[k].to_list(),
                columns=multiindex,
            )
        )
    df2 = pd.concat(frames, axis=1)

    timestamps = df2["time"] - df2["time"].iloc[0]  # Packet losses are the missing times plus changes in ccsr
    timestamps = timestamps.to_numpy().ravel()
    ccsr = df2["control_command_success_rate"].to_numpy().ravel()
    ccsr_nan = ccsr * 100  # Convert to %
    ccsr_nan[ccsr_nan == 0] = np.nan

    packet_loss_times = find_missing_packet_times(ccsr_nan, timestamps, 20)

    return df2, timestamps, packet_loss_times, state_dict_list


def find_true_ranges(bool_arr):
    """Finds the ranges of True values in a boolean array."""

    # Ensure the array is a numpy array
    bool_arr = np.asarray(bool_arr)

    # Append False at the start and end to ensure all ranges are closed
    padded_arr = np.pad(bool_arr, (1, 1), mode="constant", constant_values=False)

    # Find the differences; true starts have 1, ends have -1
    diff = np.diff(padded_arr.astype(int))

    # Start indices are where diff is 1, end indices are where diff is -1 shifted by 1
    start_indices = np.where(diff == 1)[0]
    end_indices = np.where(diff == -1)[0] - 1

    # Form the ranges
    true_ranges = list(zip(start_indices, end_indices))

    return true_ranges


def calc_halt_motion_ranges(df2):
    # Calculate ranges of halt_motion
    halt_motion_ranges = []
    start_index = None
    for i, row in df2.iterrows():
        if row["halt_motion"][0] == True:
            if start_index is None:
                start_index = i
        else:
            if start_index is not None:
                halt_motion_ranges.append((start_index, i - 1))
                start_index = None
    if start_index is not None:
        halt_motion_ranges.append((start_index, i))
    return halt_motion_ranges


def plot_state_data(df2, timestamps, packet_loss_times):
    # Calculate minimum index where stage == "slowdown"
    x_range_dict = {}
    for stage in np.unique(df2["stage"]):
        minx = np.where(df2["stage"] == stage)[0][0]
        maxx = np.where(df2["stage"] == stage)[0][-1]
        x_range_dict[stage] = (minx, maxx)

    # Calculate ranges of halt_motion
    halt_motion_ranges = calc_halt_motion_ranges(df2)

    # Create plots
    fig, axs = plt.subplots(4, 5)
    plt.subplots_adjust(left=0.05, right=0.995, top=0.975, bottom=0.05)
    axs = axs.ravel()

    ax = axs[0]
    ax.set_title("q_d scaled")
    q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    q_d = df2["q_d"]
    q_scaled = (q_d - q_min) / (q_max - q_min) * 2 - 1
    ax.plot(timestamps, q_scaled, label=["j" + str(i) for i in range(7)])
    ax.set_ylabel("Scaled joint position")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(y=1.0, color="r", linestyle="--")
    ax.axhline(y=-1.0, color="r", linestyle="--")
    ax.legend()

    ax = axs[1]
    ax.set_title("dq_d scaled")
    dq_max = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
    dq_min = -dq_max
    dq_d = df2["dq_d"]
    dq_scaled = (dq_d - dq_min) / (dq_max - dq_min) * 2 - 1
    ax.plot(timestamps, dq_scaled, label=["joint_" + str(i) for i in range(7)])
    ax.set_ylabel("Scaled joint velocity")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(y=1.0, color="r", linestyle="--")
    ax.axhline(y=-1.0, color="r", linestyle="--")

    # Estimate ddq_d if missing (happens during cartesian pose control)
    ax = axs[2]
    ddq_max = np.array([15, 7.5, 10, 12.5, 15, 20, 20])
    ddq_min = -ddq_max
    ddq_d = df2["ddq_d"]
    if ddq_d.max().max() == 0:
        ddq_d = np.diff(dq_d, axis=0) / np.diff(timestamps).reshape(-1, 1) / 0.001
        ddq_d = np.vstack((ddq_d[0], ddq_d))
        ddq_scaled = (ddq_d - ddq_min) / (ddq_max - ddq_min) * 2 - 1
        ax.plot(timestamps, ddq_scaled, label=["joint_" + str(i) for i in range(7)])
        ax.set_title("ddq_d estimated")

    else:
        ddq_d = df2["ddq_d"]
        ddq_scaled = (ddq_d - ddq_min) / (ddq_max - ddq_min) * 2 - 1
        ax.plot(timestamps, ddq_scaled, label=["joint_" + str(i) for i in range(7)])
        ax.set_title("ddq_d scaled")
    ax.plot(packet_loss_times, np.zeros(len(packet_loss_times)), ".r")
    ax.set_ylabel("Scaled joint acceleration")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(y=1.0, color="r", linestyle="--")
    ax.axhline(y=-1.0, color="r", linestyle="--")

    ax = axs[3]
    ax.set_title("dddq scaled")
    dddq_max = np.array([7500, 3750, 5000, 6250, 7500, 10000, 10000])
    dddq_min = -dddq_max
    dddq = np.diff(np.vstack((np.zeros(7), ddq_d)), 1, axis=0) / 0.001
    dddq_scaled = (dddq - dddq_min) / (dddq_max - dddq_min) * 2 - 1
    ax.plot(timestamps, dddq_scaled, label=["joint_" + str(i) for i in range(7)])
    ax.set_ylabel("Scaled joint jerk")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(y=1.0, color="r", linestyle="--")
    ax.axhline(y=-1.0, color="r", linestyle="--")

    ax = axs[4]
    ax.set_title("tau_J scaled")
    tau_max = np.array([87, 87, 87, 87, 12, 12, 12])
    tau_min = -tau_max
    tau_J = df2["tau_J"]
    tau_J_scaled = (tau_J - tau_min) / (tau_max - tau_min) * 2 - 1
    ax.plot(timestamps, tau_J_scaled, label=["joint_" + str(i) for i in range(7)])
    ax.set_ylabel("Scaled joint torque")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(y=1.0, color="r", linestyle="--")
    ax.axhline(y=-1.0, color="r", linestyle="--")

    for i, (minx, maxx) in enumerate(halt_motion_ranges):
        ax.axvspan(
            timestamps[minx],
            timestamps[maxx],
            alpha=0.2,
            color="red",
        )

    ax = axs[5]
    ax.set_title("dtau_J scaled")
    dtau_max = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000])
    dtau_min = -dtau_max
    dtau_J = df2["dtau_J"]
    dtau_J_scaled = (dtau_J - dtau_min) / (dtau_max - dtau_min) * 2 - 1
    ax.plot(timestamps, dtau_J_scaled, label=["joint_" + str(i) for i in range(7)])
    ax.set_ylabel("Scaled joint torque derivative")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(y=1.0, color="r", linestyle="--")
    ax.axhline(y=-1.0, color="r", linestyle="--")

    ax = axs[6]
    ax.set_title("Δ O_T_EE XYZ")
    pose_dict = {"x": 12, "y": 13, "z": 14}
    miny = 0.0
    for letter, col in pose_dict.items():
        data = df2["O_T_EE", str(col)]
        data_changed = data - data[0]
        miny = min(miny, data_changed.min())
        ax.plot(timestamps, data_changed, label=letter)

    # Plot stages with a new color for each stage
    colors = [
        "red",
        "green",
        "blue",
        "orange",
        "purple",
        "yellow",
        "cyan",
        "magenta",
        "brown",
        "gray",
    ]
    for i, (stage, (minx, maxx)) in enumerate(x_range_dict.items()):
        ax.axvspan(
            timestamps[minx],
            timestamps[maxx],
            alpha=0.2,
            color=colors[i],
        )
        # place text in the middle of the span
        ax.text(
            round((timestamps[minx] + timestamps[maxx]) / 2),
            miny,
            stage,
            ha="center",
            va="center",
            fontsize=4,
        )
    ax.set_ylabel("EE Position")
    ax.legend()

    ax = axs[7]
    ax.set_title("delbow_c")
    delbow_c = df2["delbow_c", "0"]
    ax.plot(timestamps, delbow_c, label="delbow_c")
    ax.set_ylabel("d Elbow angle")
    ax.legend()

    ax = axs[8]
    ax.set_title("ddelbow_c")
    ddelbow_c = df2["ddelbow_c", "0"]
    ax.plot(timestamps, ddelbow_c, label="ddelbow_c")
    ax.set_ylabel("dd Elbow angle")
    ax.legend()

    # ax = axs[7]
    # ax.set_title("Δ O_T_EE RPY")
    # O_T_EE_rot = df2["O_T_EE"].to_numpy().reshape(-1, 4, 4, order="F")
    # R_initial = R.from_matrix(O_T_EE_rot[0, :3, :3])
    # euler_angles = np.zeros((len(O_T_EE_rot), 3))
    # # for i in range(len(O_T_EE_rot)):
    # #     R_now = R.from_matrix(O_T_EE_rot[i, :3, :3])
    # #     R_diff = R_now * R_initial.inv()
    # #     rot_diff = R_diff.as_matrix()
    # #     angY = np.arcsin(rot_diff[0, 2])
    # #     angX = np.arctan2(-rot_diff[1, 2], rot_diff[2, 2])
    # #     angZ = np.arctan2(-rot_diff[0, 1], rot_diff[0, 0])
    # #     euler_angles[i] = np.array([angX, angY, angZ])
    # # ax.plot(timestamps, euler_angles[:, 0], label="R")
    # # ax.plot(timestamps, euler_angles[:, 1], label="P")
    # # ax.plot(timestamps, euler_angles[:, 2], label="Y")
    # ax.legend()
    # ax.set_ylabel("rad")

    # # Plot smoothed external force
    # ax = axs[8]
    # ax.set_title("O_F_ext_hat_K")
    # F = df2["O_F_ext_hat_K"].to_numpy()[:, 0]
    # window = 200
    # F_smoothed = np.zeros(len(F))
    # for i in range(len(F)):
    #     if i < window:
    #         F_smoothed[i] = np.mean(F[:i])
    #     else:
    #         F_smoothed[i] = np.mean(F[i - window : i])
    # dF = np.diff(F_smoothed) / 0.001
    # dF = np.hstack((dF[0], dF))
    # ax.plot(timestamps, dF, label="dF")
    # ax.plot(timestamps, F, label="raw")
    # ax.plot(timestamps, F_smoothed, label="smoothed")
    # ax.legend()
    # for i, (minx, maxx) in enumerate(halt_motion_ranges):
    #     ax.axvspan(
    #         timestamps[minx],
    #         timestamps[maxx],
    #         alpha=0.2,
    #         color="red",
    #     )

    # # Plot tau_ext_hat_filtered z-scores
    # ax = axs[8]
    # ax.set_title("tau_ext_hat_filtered z-scores")
    # window_start = 500
    # window_end = 1000
    # tau_ext_hat_filtered = df2["tau_ext_hat_filtered"].to_numpy()
    # tau_ext_hat_filtered_means = np.mean(tau_ext_hat_filtered[window_start:window_end], axis=0)

    # tau_ext_hat_stds = np.std(tau_ext_hat_filtered[window_start:window_end], axis=0)
    # print("Means: ", tau_ext_hat_filtered_means)
    # print("Stds : ", tau_ext_hat_stds)
    # z_scores = (tau_ext_hat_filtered - tau_ext_hat_filtered_means) / tau_ext_hat_stds
    # z_score_mean = np.mean(z_scores, axis=1)
    # z_score_mean_abs = np.mean(np.abs(z_scores), axis=1)
    # # ax.plot(timestamps, z_score_mean, label="mean")
    # # ax.plot(timestamps, z_score_mean_abs, label="mean_abs")
    # # z_score_mean_abs_j05 = np.mean(np.abs(z_scores[:, :6]), axis=1)
    # # ax.plot(timestamps, z_score_mean_abs_j05, label="j05")
    # # z_score_mean_abs_j04 = np.mean(np.abs(z_scores[:, :5]), axis=1)
    # # ax.plot(timestamps, z_score_mean_abs_j04, label="j04")
    # z_score_mean_abs_j03 = np.mean(np.abs(z_scores[:, :4]), axis=1)
    # z_score_mean_abs_j03_smooth = np.zeros(len(z_score_mean_abs_j03))
    # window = 200
    # # for i in range(len(z_score_mean_abs_j03)):
    # #     if i < window:
    # #         z_score_mean_abs_j03_smooth[i] = np.mean(z_score_mean_abs_j03[:i])
    # #     else:
    # #         z_score_mean_abs_j03_smooth[i] = np.mean(z_score_mean_abs_j03[i - window : i])
    # # d_zscore_mean_abs_j03_smooth = np.diff(z_score_mean_abs_j03_smooth, axis=0) / 0.001
    # # d_zscore_mean_abs_j03_smooth = np.hstack((d_zscore_mean_abs_j03_smooth[0], d_zscore_mean_abs_j03_smooth))
    # # ax.plot(timestamps, d_zscore_mean_abs_j03_smooth, label="dj03s")
    # # ax.plot(timestamps, z_score_mean_abs_j03, label="j03")
    # # ax.plot(timestamps, z_score_mean_abs_j03_smooth, label="j03s")
    # # ax.plot(timestamps, np.abs(z_scores[:, 5]), label="j5")
    # # for i in range(7):
    # #     ax.plot(timestamps, z_scores[:, i], label="j" + str(i))
    # ax.legend()
    # ax.set_ylabel("z-score")
    # for i, (minx, maxx) in enumerate(halt_motion_ranges):
    #     ax.axvspan(
    #         timestamps[minx],
    #         timestamps[maxx],
    #         alpha=0.2,
    #         color="red",
    #     )

    ax = axs[9]
    ax.set_title("tau_ext_hat")
    tau_ext_hat = df2["tau_ext_hat_filtered"]
    ax.plot(
        timestamps,
        tau_ext_hat,
        label=["joint_" + str(i) for i in range(7)],
    )
    ax.set_ylabel("External joint torque")
    ax.axhline(y=-50.0, color="r", linestyle="--")
    ax.axhline(y=50.0, color="r", linestyle="--")
    ax.axhline(y=-12.0, color="r", linestyle="--")
    ax.axhline(y=12.0, color="r", linestyle="--")

    for i, (minx, maxx) in enumerate(halt_motion_ranges):
        ax.axvspan(
            timestamps[minx],
            timestamps[maxx],
            alpha=0.2,
            color="red",
        )
    ax.set_ylim(-20.0, 20.0)

    ax = axs[10]
    ax.set_title("O_F_ext_hat_K")
    col_dict = {"x": "0", "y": "1", "z": "2"}
    for label, col in col_dict.items():
        data = df2["O_F_ext_hat_K_est", col]
        ax.plot(timestamps, data, "*-", label=label + "e")
        data = df2["O_F_ext_hat_K", col]
        ax.plot(timestamps, data, label=label)
    ax.set_ylabel("EE force")
    ax.legend()

    # Touch detection algorithm using O_F_ext_hat_K
    K_THRES_BIG = 5.0
    K_THRES_SMALL = -4.0
    DX_THRES = 0
    DX_PERSIST_LENGTH = 200
    WINDOW = 100

    x_list = np.zeros(WINDOW)
    dx_list = []
    touching = np.zeros(len(timestamps), dtype="bool")
    calls_since_dx_true = 0
    K_x_arr = df2["O_F_ext_hat_K"]["0"].to_numpy()

    for i in range(len(timestamps)):

        # Store the current position, smooth it with a window of 100
        x = df2["O_T_EE"]["12"].to_numpy()[i]
        if i == 0:
            x_list[:] = x
        else:
            x_list[:-1] = x_list[1:]
            x_list[-1] = x

        # # Smooth and calculate derivative
        # x_smooth = np.mean(x_list)
        # dx = (x - x_smooth) / 0.001
        # dx_list.append(dx)

        # Calculate derivative
        dx = (x_list[-1] - x_list[-2]) / (timestamps[i] - timestamps[i - 1]) / 0.001
        dx_list.append(dx)

        # Get current force
        K_x = K_x_arr[i]

        # Test if the force is above the threshold
        if abs(K_x) > K_THRES_BIG:
            touching[i] = True
            calls_since_dx_true = DX_PERSIST_LENGTH
        elif K_x < K_THRES_SMALL and abs(K_x) < K_THRES_BIG and dx > DX_THRES:
            touching[i] = True
            calls_since_dx_true = DX_PERSIST_LENGTH
        elif calls_since_dx_true > 0:
            touching[i] = True
        else:
            touching[i] = False

        calls_since_dx_true -= 1

    touching[:500] = False
    touching[-200:] = False

    # Plot touching in green
    touching_ranges = find_true_ranges(touching)
    for i, (minx, maxx) in enumerate(touching_ranges):
        ax.axvspan(
            timestamps[minx],
            timestamps[maxx],
            alpha=0.2,
            color="green",
        )

    # ax = axs[11]
    # ax.set_title("O_F_ext_hat_K")
    # col_dict = {"R": "3", "P": "4", "Y": "5"}
    # for label, col in col_dict.items():
    #     data = df2["O_F_ext_hat_K"][col]
    #     ax.plot(timestamps, data, label=label)
    # ax.set_ylabel("EE wrench")
    # ax.legend()

    ax = axs[11]
    ax.set_title("ddx")
    dx_list_smoothed = np.zeros(len(dx_list))
    WINDOW = 50

    preceding_x = np.zeros(WINDOW)
    preceding_dx = np.zeros(WINDOW)
    preceding_dx_smooth = np.zeros(WINDOW)
    preceding_ddx = np.zeros(WINDOW)

    x = df2["O_T_EE"]["12"].to_numpy()
    dx_from_smoothing = np.zeros(len(x))
    ddx_from_smoothing = np.zeros(len(x))
    for i in range(len(dx_list)):

        if i == 0:
            preceding_x[:] = x[i]
            preceding_dx[:] = 0
            preceding_dx_smooth[:] = 0
            preceding_ddx[:] = 0
        else:
            preceding_x[:-1] = preceding_x[1:]
            preceding_x[-1] = x[i]

            preceding_dx[:-1] = preceding_dx[1:]
            dx_now = (
                (preceding_x[-1] - preceding_x[-2])
                / (timestamps[i] - timestamps[i - 1])
                / (timestamps[i] - timestamps[i - 1])
                / 0.001
            )
            preceding_dx[-1] = dx_now
            dx_smooth = np.mean(preceding_dx)

            preceding_dx_smooth[:-1] = preceding_dx_smooth[1:]
            preceding_dx_smooth[-1] = dx_smooth

            preceding_ddx[:-1] = preceding_ddx[1:]
            ddx_now = (preceding_dx_smooth[-1] - preceding_dx_smooth[-2]) / (timestamps[i] - timestamps[i - 1]) / 0.001
            preceding_ddx[-1] = ddx_now
            ddx_smooth = np.mean(preceding_ddx)

            dx_from_smoothing[i] = dx_smooth
            ddx_from_smoothing[i] = ddx_smooth
    ax.plot(timestamps, ddx_from_smoothing, label="ddx")
    # ax.plot(timestamps, dx, label="dx")

    # for i in range(len(dx_list)):
    #     if i < WINDOW:
    #         dx_list_smoothed[i] = np.mean(dx_list[:i])
    #     else:
    #         dx_list_smoothed[i] = np.mean(dx_list[i - WINDOW : i])
    # ddx = np.diff(dx_list_smoothed) / np.diff(timestamps) / 0.001
    # ddx = np.hstack((ddx[0], ddx))
    # ddx_smoothed = np.zeros(len(ddx))
    # for i in range(len(ddx)):
    #     if i < WINDOW:
    #         ddx_smoothed[i] = np.mean(ddx[:i])
    #     else:
    #         ddx_smoothed[i] = np.mean(ddx[i - WINDOW : i])
    # ax.plot(timestamps, ddx_smoothed, label="ddx_s")
    ax.set_ylabel("EE Acceleration")

    ax = axs[12]
    ax.set_title("Control Command Success Rate")
    ccsr = df2["control_command_success_rate"]
    ax.axhline(y=95, color="r", linestyle="--")
    ccsr_nan = ccsr.to_numpy(dtype="float") * 100  # Convert to %
    ccsr_nan[ccsr_nan == 0] = np.nan
    ax.plot(timestamps, ccsr_nan)
    ax.text(
        round(timestamps[-1] / 2),
        70,
        "{} dropped packets".format(len(packet_loss_times)),
        ha="center",
        va="center",
    )
    ax.set_ylabel("% of success last 100 commands")
    ax.set_ylim(50, 105)

    # # Iterate over all timestamps (including skipped)
    # ccsr_nan_full = np.zeros(timestamps[-1] + 1, dtype="float")
    # ccsr_nan_full[:] = np.nan  # Initialize with NaN
    # ccsr_nan_full[timestamps[:]] = ccsr_nan.ravel()
    # timestamps_full = np.zeros(timestamps[-1] + 1, dtype="bool")
    # ccsr_ts_full = np.zeros(timestamps[-1] + 1, dtype="float")

    # # Count leading NaN as True
    # num_leading_nan = 0
    # for j in range(i + 1):
    #     if np.isnan(ccsr_nan_full[j]):
    #         num_leading_nan += 1
    #     else:
    #         break

    # for i in range(timestamps[-1] + 1):

    #     # Determine possible values of ccsr_nan_full
    #     if i < num_leading_nan:
    #         value_if_true = np.nan
    #         value_if_false = np.nan
    #     elif i < 100 + num_leading_nan:
    #         value_if_true = np.nansum(timestamps_full[:i]) + (100 - i)
    #         value_if_false = np.nansum(timestamps_full[:i]) + (100 - i) - 1
    #     else:
    #         value_if_true = np.nansum(timestamps_full[i - 99 : i]) + 1
    #         value_if_false = np.nansum(timestamps_full[i - 99 : i])

    #     if i < num_leading_nan:
    #         timestamps_full[i] = True
    #     elif np.isnan(ccsr_nan_full[i]):
    #         timestamps_full[i] = False
    #     elif ccsr_nan_full[i] == value_if_true:
    #         timestamps_full[i] = True
    #     elif ccsr_nan_full[i] == value_if_false:
    #         timestamps_full[i] = False
    #     else:
    #         raise ValueError(
    #             f"ccsr_nan_full[{i}] = {ccsr_nan_full[i]} does not match expected values {value_if_true} or {value_if_false}"
    #         )

    #     # Calculate ccsr
    #     if np.all(np.isnan(ccsr_nan_full[: i + 1])):
    #         ccsr_ts_full[i] = np.nan
    #     elif np.isnan(ccsr_nan_full[i]):
    #         ccsr_ts_full[i] = np.nan
    #     elif i < 100 + num_leading_nan:

    #         ccsr_ts_full[i] = np.nansum(timestamps_full[:i]) + (100 - i)
    #     else:
    #         ccsr_ts_full[i] = np.nansum(timestamps_full[i - 99 : i + 1])

    #     # Check
    #     if (ccsr_ts_full[i] != ccsr_nan_full[i]) and not (np.isnan(ccsr_ts_full[i]) and np.isnan(ccsr_nan_full[i])):
    #         print(f"Mismatch at index {i}: ccsr_nan_full={ccsr_nan_full[i]} ccsr_ts_full={ccsr_ts_full[i]}")

    # i = None  # Reset variable to avoid confusion in the next loop

    # # Same but from robot's perspective (missing packets)
    # last_100 = [True] * 100
    # ccsr_alt = []
    # first_non_nan_found = False
    # ts_prev = 0
    # packet_loss_times_alt = []
    # for k in range(len(timestamps)):

    #     # Read current ccsr value
    #     ccsr_curr = ccsr_nan[k]

    #     # Determine timestamp
    #     ts_curr = timestamps[k]
    #     if k == 0:
    #         ts_prev = ts_curr - 1

    #     # Skip initial NaN values
    #     if first_non_nan_found is False and np.isnan(ccsr_curr):
    #         ccsr_alt.append(np.nan)
    #         ts_prev = ts_curr  # Reset timestamp to current
    #         continue
    #     else:
    #         first_non_nan_found = True

    #     # Determine if jump in timestamp and add False values
    #     ts_jump = ts_curr - ts_prev
    #     if ts_jump > 1:
    #         for j in range(ts_jump - 1):
    #             last_100.pop(0)
    #             last_100.append(False)
    #             packet_loss_times_alt.append(ts_prev + j + 1)  # Add missing timestamp
    #     ts_prev = ts_curr  # Reset timestamp to current

    #     # Calculate expected values
    #     value_if_true = np.nansum(last_100[1:]) + 1
    #     value_if_false = np.nansum(last_100[1:])

    #     # Use known CCSR to determine if True or False
    #     if ccsr_curr == value_if_true:
    #         last_100.pop(0)
    #         last_100.append(True)
    #     elif ccsr_curr == value_if_false:
    #         last_100.pop(0)
    #         last_100.append(False)
    #         packet_loss_times_alt.append(ts_curr)  # Add current timestamp
    #     else:
    #         raise ValueError(
    #             f"ccsr_nan[{k}] = {ccsr_curr} does not match expected values {value_if_true} or {value_if_false}"
    #         )

    #     # Calculate CCSR from last 100 values
    #     ccsr_calc = np.nansum(last_100)
    #     ccsr_alt.append(ccsr_calc)

    #     # Check if calculated matches known
    #     if ccsr_calc != ccsr_curr:
    #         print(f"Mismatch at index {k}: ccsr_nan={ccsr_nan[k]} ccsr_alt={ccsr_calc}")

    #     if ccsr_alt[k] != ccsr_nan[k]:
    #         print(f"Mismatch at index {k}: ccsr_nan={ccsr_nan[k]} ccsr_alt={ccsr_alt[k]}")

    ax.plot(packet_loss_times, 100 * np.ones(len(packet_loss_times)), "xr")

    # # Calculate alternative command success rate
    # ccsr_alt = ccsr_nan.copy()
    # ccsr_alt[1:] = np.nan
    # success_list = [True] * 100
    # packet_loss_times_alt = []
    # timestamps_full = []
    # timestamps_full_ccsr = []
    # for i in range(len(ccsr_nan)):

    #     if i == 1308:
    #         print("Debugging at index 1308")

    #     # Skip NaN values
    #     if np.isnan(ccsr_nan[i]):
    #         timestamps_full.append(True)

    #     # Previous value is NaN and current is not, assume success
    #     elif np.isnan(ccsr_nan[i - 1]) and ccsr_nan[i] == 100.0:
    #         success_list.pop(0)
    #         success_list.append(True)  # If previous value is NaN, assume success
    #         timestamps_full.append(True)

    #     # Increasing success rate is always success
    #     elif ccsr_nan[i] > ccsr_nan[i - 1]:
    #         success_list.pop(0)
    #         success_list.append(True)
    #         timestamps_full.append(True)

    #     # Constant success rate is success if previous 100th command was True
    #     elif ccsr_nan[i] == ccsr_nan[i - 1] and success_list[0] == True:
    #         success_list.pop(0)
    #         success_list.append(True)
    #         timestamps_full.append(True)

    #     # Constant success rate is NOT success if previous 100th command was False
    #     elif ccsr_nan[i] == ccsr_nan[i - 1] and success_list[0] == False:
    #         success_list.pop(0)
    #         success_list.append(False)
    #         packet_loss_times_alt.append(timestamps[i])  # Does this skip confirmed packet loss times?
    #         timestamps_full.append(False)

    #     # Static success rate is False if previous 100 command was True
    #     else:

    #         # Handle multiple skipped timesteps
    #         timestep_gap = timestamps[i] - timestamps[i - 1]
    #         for j in range(timestep_gap - 1, -1, -1):
    #             success_list.pop(0)
    #             success_list.append(False)
    #             packet_loss_times_alt.append(timestamps[i] - j)
    #             timestamps_full.append(False)

    #     ts_ccsr = 0.0
    #     if len(timestamps_full) > 100:
    #         ts_ccsr = np.mean(timestamps_full[-100:]) * 100

    #         if ts_ccsr != ccsr_nan[i]:
    #             print("hi")
    #             pass

    #     ccsr_alt[i] = np.mean(success_list) * 100  # Convert to %
    #     timestamps_full_ccsr.append(ts_ccsr)
    #     if ccsr_alt[i] != ccsr_nan[i]:
    #         pass

    # ax.plot(packet_loss_times_alt, 100 * np.ones(len(packet_loss_times_alt)), "xb")

    # for i in range(1500):
    #     print(
    #         i,
    #         timestamps[i],
    #         ccsr_nan[i],
    #         ccsr_alt[i],
    #         timestamps_full_ccsr[timestamps[i]],
    #         df2.loc[i, "time"][0],
    #         sep="\t",
    #     )

    # assert np.all(
    #     np.isclose(ccsr_nan, ccsr_alt, equal_nan=True)
    # ), "CCSR and CCSR_ALT should be equal except for NaN values"

    print(f"Num dropped packets      : {len(packet_loss_times)}")
    # Plot difference between O_T_EE_c and O_T_EE
    ax = axs[13]
    ax.set_title("dx ")
    ax.plot(timestamps, dx_from_smoothing, label="dx")
    # ax.plot(timestamps, dx_list_smoothed, label="dx_s")
    ax.hlines(0, 0, timestamps[-1], color="r", linestyle="--")
    for i, (minx, maxx) in enumerate(halt_motion_ranges):
        ax.axvspan(
            timestamps[minx],
            timestamps[maxx],
            alpha=0.2,
            color="red",
        )
    ax.set_ylabel("EE Position")
    ax.legend()

    # Plot difference between O_T_EE and O_T_EE_c x
    ax = axs[14]
    ax.set_title("O_T_EE_c - O_T_EE")
    pose_dict = {"x": "12", "y": "13", "z": "14"}
    for k, v in pose_dict.items():

        O_T_EE = df2["O_T_EE", v].to_numpy()
        O_T_EE_c = df2["O_T_EE_c", v].to_numpy()
        diff = O_T_EE_c - O_T_EE
        ax.plot(timestamps, diff, label=k)
    ax.legend()

    # # Plot tau_ext_hat with mean subtracted
    # ax = axs[14]
    # teh_sub = np.abs(tau_ext_hat_filtered - tau_ext_hat_filtered_means)
    # teh_total = np.sum(teh_sub, axis=1)
    # teh_total_smoothed = np.zeros(len(teh_total))
    # window = 200
    # for i in range(len(teh_total)):
    #     if i < window:
    #         teh_total_smoothed[i] = np.mean(teh_total[:i])
    #     else:
    #         teh_total_smoothed[i] = np.mean(teh_total[i - window : i])

    # d_teh_total = np.diff(teh_total_smoothed, axis=0) / 0.001
    # d_teh_total = np.hstack((d_teh_total[0], d_teh_total))
    # # for i in range(7):
    # #     ax.plot(timestamps, teh_sub[:, i], label="j" + str(i))
    # ax.plot(timestamps, teh_total, label="1-7")
    # ax.plot(timestamps, d_teh_total, label="d(1-7)")

    # # Estimate ranges of contact
    # teh_thres = 7.5
    # teh_above = teh_total > teh_thres

    # dx_thres = 0
    # dx_teh_thres = 2.0
    # dx_above = (d_xyz < dx_thres) & (teh_total > dx_teh_thres)
    # teh_above = teh_above | dx_above

    # teh_above[:500] = False
    # teh_above[-200:] = False
    # teh_delta = np.diff(teh_above.astype("int"))
    # above_indices = np.where(teh_delta == 1)[0]
    # below_indices = np.where(teh_delta == -1)[0]
    # teh_ranges = list(zip(above_indices, below_indices))
    # for i, (minx, maxx) in enumerate(teh_ranges):
    #     ax.axvspan(
    #         timestamps[minx],
    #         timestamps[maxx],
    #         alpha=0.2,
    #         color="green",
    #     )

    # ax.legend()
    # ax.set_title("tau_ext_hat_filtered_sub")

    ax = axs[15]
    ax.set_title("O_T_EE_c (rel.)")
    pose_dict = {"x": 12, "y": 13, "z": 14}
    maxdiff = 0
    mindiff = 1
    for letter, col in pose_dict.items():
        # Subtract from first value to get relative position
        diff = df2["O_T_EE_c", str(col)] - df2["O_T_EE_c", str(col)][0]
        maxdiff = max(maxdiff, diff.max())
        mindiff = min(mindiff, diff.min())
        ax.plot(timestamps, diff, label=letter)
    ax.set_ylabel("EE Position")
    for i, (minx, maxx) in enumerate(halt_motion_ranges):
        ax.axvspan(
            timestamps[minx],
            timestamps[maxx],
            alpha=0.2,
            color="red",
        )
    for i, (stage, (minx, maxx)) in enumerate(x_range_dict.items()):
        ax.vlines(maxx, mindiff, maxdiff, colors="k", linestyles="dashed")

    ax.legend()

    # Change in Euler Angles
    ax = axs[16]
    ax.set_title("Δ O_T_EE_C Euler Angles")
    d = df2["O_dP_EE_c"].to_numpy()
    e = d[:, 3:] * 0.001
    euler_angles = np.cumsum(e, axis=0)
    # data = df2["O_T_EE_c"]
    # euler_angles = np.zeros((len(data), 3))
    # A = data.iloc[0].to_numpy().reshape(4, 4, order="F")[:3, :3]
    # for i in range(len(data)):
    #     arr = data.iloc[i].to_numpy().reshape(4, 4, order="F")
    #     B = arr[:3, :3]
    #     C = B @ A.T
    #     # r = R.from_matrix(C)

    #     angY = np.arcsin(C[0, 2])
    #     angX = np.arctan2(-C[1, 2], C[2, 2])
    #     angZ = np.arctan2(-C[0, 1], C[0, 0])
    #     euler_angles[i] = np.array([angX, angY, angZ])
    #     # euler_angles[i] = r.as_euler("zyx", degrees=False)  # Change 'zyx' to your desired sequence
    ax.plot(timestamps, euler_angles[:, 0], label="R")
    ax.plot(timestamps, euler_angles[:, 1], label="P")
    ax.plot(timestamps, euler_angles[:, 2], label="Y")
    ax.plot(packet_loss_times, np.zeros(len(packet_loss_times)), ".r")
    ax.legend()
    ax.set_ylabel("rad")

    # # Calculate velocity of O_T_EE and plot
    # O_T_EE_x = df2["O_T_EE", "12"]
    # O_T_EE_y = df2["O_T_EE", "13"]
    # O_T_EE_z = df2["O_T_EE", "14"]
    # xyz = np.vstack((O_T_EE_x, O_T_EE_y, O_T_EE_z)).T
    # dxyz = np.vstack((np.zeros((1,3)), np.diff(xyz, axis=0) / 0.001))

    # ax = axs[13]
    # ax.set_title("Measured Cartesian Velocity")
    # ax.plot(timestamps, dxyz)
    # ax.set_ylabel("EE Velocity (m/s)")
    #

    # The plotting for rotational velocity is inadequate and should be fixed if rotation becomes important
    ax = axs[17]
    ax.set_title("O_dP_EE_c")
    col_dict = {"x": "0", "y": "1", "z": "2", "R": "3", "P": "4", "Y": "5"}
    O_dP_EE_c_max = 1.7
    O_dP_EE_c_min = -O_dP_EE_c_max
    for label, col in col_dict.items():
        data = df2["O_dP_EE_c"][col]
        ax.plot(timestamps, data, label=label)
    ax.set_ylabel("EE Velocity (m/s)")
    ax.set_ylim(O_dP_EE_c_min + O_dP_EE_c_min * 0.05, O_dP_EE_c_max + O_dP_EE_c_max * 0.05)
    ax.axhline(y=O_dP_EE_c_max, color="r", linestyle="--")
    ax.axhline(y=O_dP_EE_c_min, color="r", linestyle="--")
    ax.legend()

    ax.plot(packet_loss_times, np.zeros(len(packet_loss_times)), ".r")

    ax = axs[18]
    ax.set_title("O_ddP_EE_c")
    O_ddP_EE_c_max = 13.0
    O_ddP_EE_c_min = -O_ddP_EE_c_max
    col_dict = {"x": "0", "y": "1", "z": "2", "R": "3", "P": "4", "Y": "5"}
    for label, col in col_dict.items():
        data = df2["O_ddP_EE_c"][col]
        ax.plot(timestamps, data, label=label)
    ax.plot(packet_loss_times, np.zeros(len(packet_loss_times)), ".r")
    ax.set_ylabel("EE Acceleration (m/s^2)")
    ax.set_ylim(O_ddP_EE_c_min + O_ddP_EE_c_min * 0.05, O_ddP_EE_c_max + O_ddP_EE_c_max * 0.05)
    ax.axhline(y=O_ddP_EE_c_max, color="r", linestyle="--")
    ax.axhline(y=O_ddP_EE_c_min, color="r", linestyle="--")

    ax = axs[19]
    ax.set_title("O_dddP_EE")
    O_dddP_EE_max = 6500
    O_dddP_EE_min = -O_dddP_EE_max
    col_dict = {"x": "0", "y": "1", "z": "2", "R": "3", "P": "4", "Y": "5"}
    for label, col in col_dict.items():
        data = df2["O_ddP_EE_c"][col]
        diff = np.diff(np.hstack((np.zeros(1), data.to_numpy())), 1, axis=0) / 0.001
        ax.plot(timestamps, diff, label=label)
    ax.set_ylabel("EE Jerk (m/s^3)")
    ax.set_ylim(O_dddP_EE_min + O_dddP_EE_min * 0.05, O_dddP_EE_max + O_dddP_EE_max * 0.05)
    ax.axhline(y=O_dddP_EE_max, color="r", linestyle="--")
    ax.axhline(y=O_dddP_EE_min, color="r", linestyle="--")

    # plt.show()

    # def test_v_a_j(dataframe, timestamp):

    #     v = dataframe["O_dP_EE_c"].iloc[1980]
    #     a = dataframe["O_ddP_EE_c"].iloc[1980]

    # Add in dropped packets
    df2["time_from_start"] = df2.loc[:, [("time", "0")]] - df2.loc[0, [("time", "0")]]
    df2.set_index("time_from_start", inplace=True)
    df5 = pd.DataFrame(columns=df2.columns, index=range(0, timestamps[-1] + 1))
    df5.loc[df2.index] = df2

    for i in range(20):
        if i < 15:
            ax = axs[i]
            ax.set_xticks([], [])

    # # Calculated velocity of euler angles
    # ax = axs[17]
    # ax.set_title("calc. O_dP_EE_c Euler Angles")
    # cdP = np.diff(euler_angles, axis=0) / np.diff(timestamps).reshape(-1, 1) / 0.001
    # cdP = np.vstack([np.zeros((1, 3)), cdP])
    # ax.plot(timestamps, cdP[:, 0], label="Y")
    # ax.plot(timestamps, cdP[:, 1], label="P")
    # ax.plot(timestamps, cdP[:, 2], label="R")
    # ax.legend()
    # ax.set_ylabel("rad/s")

    # # Calculated acceleration of euler angles
    # ax = axs[18]
    # ax.set_title("calc. O_ddP_EE_c Euler Angles")
    # cddP = np.diff(euler_angles, n=2, axis=0) / np.diff(timestamps[:-1]).reshape(-1, 1) / 0.001
    # cddP = np.vstack([np.zeros((2, 3)), cddP])
    # ax.plot(timestamps, cddP[:, 0], label="Y")
    # ax.plot(timestamps, cddP[:, 1], label="P")
    # ax.plot(timestamps, cddP[:, 2], label="R")
    # ax.legend()
    # ax.set_ylabel("rad/s^2")

    # print(np.diff(euler_angles[48:60], n=1,axis=0))

    # if len(packet_loss_times) == 0:
    #     # Command in timestep after packet loss
    #     # a = df3.loc[:, "O_ddP_EE_c"].loc[["0", "1", "2"]].to_numpy(dtype="float")
    #     # v = df3.loc[:, "O_dP_EE_c"].loc[["0", "1", "2"]].to_numpy(dtype="float")
    #     # p = df3.loc[:, "O_T_EE_c"].loc[["12", "13", "14"]].to_numpy(dtype="float")

    #     print(
    #         df5.loc[
    #             :,
    #             [
    #                 ("O_T_EE_c", "12"),
    #                 ("O_T_EE_c", "13"),
    #                 ("O_T_EE_c", "14"),
    #                 ("O_dP_EE_c", "0"),
    #                 ("O_dP_EE_c", "1"),
    #                 ("O_dP_EE_c", "2"),
    #                 ("O_ddP_EE_c", "0"),
    #                 ("O_ddP_EE_c", "1"),
    #                 ("O_ddP_EE_c", "2"),
    #             ],
    #         ]
    #     )
    # else:
    #     t = packet_loss_times[-1]
    #     print(
    #         df5.loc[t - 5 :, [("O_T_EE_c", "12"), ("O_dP_EE_c", "0"), ("O_ddP_EE_c", "0")]]
    #     )

    # Initialize vertical lines (invisible at start)
    vlines = [ax.axvline(x=0, color="red", lw=1, visible=False) for ax in axs]

    def on_move(event):
        if event.inaxes:
            for line in vlines:
                line.set_xdata(event.xdata)
                line.set_visible(True)
            fig.canvas.draw_idle()

    # Connect the event handler
    fig.canvas.mpl_connect("motion_notify_event", on_move)

    # plt.show()

    plt.show()


# # Calculate estimated EE force using joint torques and jacobian
# def calculate_ee_force(df2):

#     # Load data
#     tau_J = df2["tau_J"].to_numpy()
#     tau_ext_hat = df2["tau_ext_hat_filtered"].to_numpy()
#     O_Jac_EE = df2["O_Jac_EE"].to_numpy()
#     O_Jac_EE_inv = np.linalg.pinv(O_Jac_EE)

#     # Calculate estimated EE force
#     tau_J_ext = tau_J + tau_ext_hat
#     F_ext_hat = O_Jac_EE_inv @ tau_J_ext.T

#     # Plot estimated EE force
#     fig, axs = plt.subplots(3, 1, figsize=(10, 10))
#     axs[0].set_title("Estimated EE force")
#     axs[0].plot(F_ext_hat.T)
#     axs[0].set_ylabel("Force (N)")
#     axs[0].legend(["x", "y", "z"])

#     axs[1].set_title("Joint torques")
#     axs[1].plot(tau_J)
#     axs[1].set_ylabel("Torque (Nm)")
#     axs[1].legend(["j" + str(i) for i in range(7)])

#     axs[2].set_title("External torques")
#     axs[2].plot(tau_ext_hat)
#     axs[2].set_ylabel("Torque (Nm)")
#     axs[2].legend(["j" + str(i) for i in range(7)])

#     plt.show()


if __name__ == "__main__":
    # Use command line argument if given, otherwise use sample file
    if len(sys.argv) == 2:
        file = sys.argv[1]
    else:
        dirname = Path("/home/oconnorlabmatlab/Data")
        # dirname = Path("/mnt/data12/William/Data/2024-03-29")
        file_list = sorted(dirname.rglob("*state.txt"))
        file = file_list[-1]

    # Load data
    txt_path = Path(file)
    print("Showing state data for: ", txt_path)

    assert txt_path.exists()

    assert txt_path.suffix == ".txt"
    df2, timestamps, packet_loss_times, _ = read_data_from_txt_file(txt_path)

    # # Print last 10 values of O_T_EE_c[14] at full resolution
    # pd.options.display.precision = 16
    # print(df2["O_T_EE_c", "14"].iloc[-10:])
    # print(df2["O_dP_EE_c", "2"].iloc[-10:])
    # print(df2["O_ddP_EE_c", "2"].iloc[-10:])
    # print(df2["elbow_c", "0"].iloc[-10:])
    # print(df2["delbow_c", "0"].iloc[-10:])
    # print(df2["ddelbow_c", "0"].iloc[-10:])

    # Plot data
    plot_state_data(df2, timestamps, packet_loss_times)
