# %%
import matplotlib.pyplot as plt
import numpy as np


# convert dot dictionary to position in global coordinate system
def dot2coord(dot):
    # For steps defined on an 8 to 5 grid
    step_size = 5 / 8

    field_width = 160 / 3
    front_hash = 60 / 3
    # back_hash = field_width - 60 / 3
    back_hash = front_hash + 12.5

    if dot["side"] == "side1":
        if dot["io"] == "inside":
            x_coord = dot["yd_io"] + dot["steps_io"] * step_size
        elif dot["io"] == "outside":
            x_coord = dot["yd_io"] - dot["steps_io"] * step_size

    elif dot["side"] == "side2":
        if dot["io"] == "inside":
            x_coord = (100 - dot["yd_io"]) - dot["steps_io"] * step_size
        elif dot["io"] == "outside":
            x_coord = (100 - dot["yd_io"]) + dot["steps_io"] * step_size

    if dot["marker_fb"] == "front sideline":
        if dot["fb"] == "in front":
            ycoord = -dot["steps_fb"] * step_size
        elif dot["fb"] == "behind":
            ycoord = dot["steps_fb"] * step_size

    elif dot["marker_fb"] == "back sideline":
        if dot["fb"] == "in front":
            ycoord = field_width - dot["steps_fb"] * step_size
        elif dot["fb"] == "behind":
            ycoord = field_width + dot["steps_fb"] * step_size

    elif dot["marker_fb"] == "front hash":
        if dot["fb"] == "in front":
            ycoord = front_hash - dot["steps_fb"] * step_size
        elif dot["fb"] == "behind":
            ycoord = front_hash + dot["steps_fb"] * step_size

    elif dot["marker_fb"] == "back hash":
        if dot["fb"] == "in front":
            ycoord = back_hash - dot["steps_fb"] * step_size
        elif dot["fb"] == "behind":
            ycoord = back_hash + dot["steps_fb"] * step_size

    return x_coord, ycoord


# calculate the velocity along the line between the source and the observer as well as the frequency shift at the
# observer
def doppler(m, n, l, tempo, counts, t_start, f_s):
    r = (
        n - m
    )  # n and m must be numpy arrays of shape (3, 1). l must be the same shape too
    set_dist = np.linalg.norm(r)

    step_size = 8 / ((set_dist / counts) / (5 / 8))

    Vs = set_dist / ((60 / tempo) * counts)  # in yds per sec
    t_end = set_dist / Vs + t_start
    num_pts = 100
    t = np.linspace(t_start, t_end, num_pts)
    t = t.reshape(1, num_pts)

    counts_list = t * (tempo / 60)

    Ps = m + r * ((t - t_start) / (t_end - t_start))

    d_so = np.linalg.norm(Ps, axis=0)

    m1 = m[0, 0]
    m2 = m[1, 0]
    m3 = m[2, 0]
    n1 = n[0, 0]
    n2 = n[1, 0]
    n3 = n[2, 0]
    l1 = l[0, 0]
    l2 = l[1, 0]
    l3 = l[2, 0]

    V_so_der = (
        1.0
        / np.sqrt(
            (
                l1
                - m1
                + Vs
                * (m1 - n1)
                * (t - t_start)
                * 1.0
                / np.sqrt((m1 - n1) ** 2 + (m2 - n2) ** 2 + (m3 - n3) ** 2)
            )
            ** 2
            + (
                l2
                - m2
                + Vs
                * (m2 - n2)
                * (t - t_start)
                * 1.0
                / np.sqrt((m1 - n1) ** 2 + (m2 - n2) ** 2 + (m3 - n3) ** 2)
            )
            ** 2
            + (
                l3
                - m3
                + Vs
                * (m3 - n3)
                * (t - t_start)
                * 1.0
                / np.sqrt((m1 - n1) ** 2 + (m2 - n2) ** 2 + (m3 - n3) ** 2)
            )
            ** 2
        )
        * (
            Vs
            * (m1 - n1)
            * (
                l1
                - m1
                + Vs
                * (m1 - n1)
                * (t - t_start)
                * 1.0
                / np.sqrt((m1 - n1) ** 2 + (m2 - n2) ** 2 + (m3 - n3) ** 2)
            )
            * 1.0
            / np.sqrt((m1 - n1) ** 2 + (m2 - n2) ** 2 + (m3 - n3) ** 2)
            * 2.0
            + Vs
            * (m2 - n2)
            * (
                l2
                - m2
                + Vs
                * (m2 - n2)
                * (t - t_start)
                * 1.0
                / np.sqrt((m1 - n1) ** 2 + (m2 - n2) ** 2 + (m3 - n3) ** 2)
            )
            * 1.0
            / np.sqrt((m1 - n1) ** 2 + (m2 - n2) ** 2 + (m3 - n3) ** 2)
            * 2.0
            + Vs
            * (m3 - n3)
            * (
                l3
                - m3
                + Vs
                * (m3 - n3)
                * (t - t_start)
                * 1.0
                / np.sqrt((m1 - n1) ** 2 + (m2 - n2) ** 2 + (m3 - n3) ** 2)
            )
            * 1.0
            / np.sqrt((m1 - n1) ** 2 + (m2 - n2) ** 2 + (m3 - n3) ** 2)
            * 2.0
        )
    ) / 2.0

    c = 375.44  # sound speed in air in units of yd/s
    f_o = f_s * (c / (c + V_so_der))

    cents = (1200 / np.log(2)) * np.log(f_o / f_s)

    return V_so_der, t, counts_list, f_o, cents, step_size


# USER INPUTS ------------------------------------------------------------
dot_obs = {
    "side": "side1",
    "steps_io": 0,
    "io": "inside",
    "yd_io": 50,
    "steps_fb": 0,
    "fb": "in front",
    "marker_fb": "front hash",
}

dot1 = {
    "side": "side1",
    "steps_io": 0,
    "io": "inside",
    "yd_io": 40,
    "steps_fb": 0,
    "fb": "in front",
    "marker_fb": "front hash",
}

dot2 = {
    "side": "side1",
    "steps_io": 0,
    "io": "inside",
    "yd_io": 45,
    "steps_fb": 0,
    "fb": "in front",
    "marker_fb": "front hash",
}

z_coord_obs = 2
z_coord1 = 2
z_coord2 = 2

tempo = 160
counts = 8
f_s = 440
t_start = 0
# ------------------------------------------------------------------------

# Important constants, NCAA field with 8 to 5 standard step size
field_length = 100
field_width = 160 / 3
front_hash = 60 / 3
back_hash = field_width - 60 / 3
yd_line_dist = 5
step_size = yd_line_dist / 8

# convert dots to global coordinates
x_coord_obs, y_coord_obs = dot2coord(dot_obs)
x_coord1, y_coord1 = dot2coord(dot1)
x_coord2, y_coord2 = dot2coord(dot2)

l = np.transpose(np.array([[x_coord_obs, y_coord_obs, z_coord_obs]]))
m = np.transpose(np.array([[x_coord1, y_coord1, z_coord1]]))
n = np.transpose(np.array([[x_coord2, y_coord2, z_coord2]]))

V_so_der, time, counts_list, f_o, cents, step_size_march = doppler(
    m, n, l, tempo, counts, t_start, f_s
)

# Yard line locations and markers
yd_markers_locations = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
yd_markers = ["G", 10, 20, 30, 40, 50, 40, 30, 20, 10, "G"]

# Make arrays for the yard lines, four step lines, and two step lines
yd_lines = np.arange(0, field_length + yd_line_dist, yd_line_dist)
four_step_lines_vert = np.arange(0, field_length + yd_line_dist / 2, yd_line_dist / 2)
two_step_lines_vert = np.arange(0, field_length + yd_line_dist / 4, yd_line_dist / 4)
four_step_lines_horz = np.arange(0, field_width, yd_line_dist / 2)
two_step_lines_horz = np.arange(0, field_width, yd_line_dist / 4)

# plot NCAA football field grid, velocity, and cents shift
fig = plt.figure(num=1, figsize=(7, 6), dpi=300)
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax2 = plt.subplot2grid((2, 2), (1, 0))
ax3 = plt.subplot2grid((2, 2), (1, 1))

ax1.vlines(two_step_lines_vert, 0, field_width, color="grey", linewidth=0.25, alpha=0.5)
ax1.hlines(
    two_step_lines_horz, 0, field_length, color="grey", linewidth=0.25, alpha=0.5
)
ax1.vlines(four_step_lines_vert, 0, field_width, color="grey", linewidth=0.5)
ax1.hlines(four_step_lines_horz, 0, field_length, color="grey", linewidth=0.5)
ax1.hlines(front_hash, 0, field_length, color="k", linewidth=1)
ax1.hlines(back_hash, 0, field_length, color="k", linewidth=1)
ax1.hlines(0, 0, field_length, color="k", linewidth=1)
ax1.hlines(field_width, 0, field_length, color="k", linewidth=1)
ax1.vlines(yd_lines, 0, field_width, color="k", linewidth=1)
ax1.plot(x_coord1, y_coord1, "gD", ms=5, alpha=1, mec="k", mew=0.5, label="Start")
ax1.plot(x_coord2, y_coord2, "ro", ms=5, mec="k", mew=0.5, label="End")
ax1.plot(x_coord_obs, y_coord_obs, "bs", ms=5, mec="k", mew=0.5, label="Observer")
ax1.set_xticks(yd_markers_locations, labels=yd_markers)
ax1.tick_params(direction="inout", length=10)
ax1.set_yticks([])
ax1.set_xlim([0, field_length])
ax1.set_ylim([0, field_width])
ax1.axis("equal")
ax1.spines[["left", "right", "top", "bottom"]].set_visible(False)
ax1.legend(fontsize=7, loc="upper left")


ax2.plot(counts_list[0, :], V_so_der[0, :])
ax2.set_xlabel("Count")
ax2.set_ylabel("$V_{s,o}$  (yd/s)")
ax2.set_xlim([np.min(counts_list[0, :]), np.max(counts_list[0, :])])


ax3.plot(counts_list[0, :], cents[0, :])
ax3.set_xlabel("Count")
ax3.set_ylabel("Doppler Shift (cents)")
ax3.set_xlim([np.min(counts_list[0, :]), np.max(counts_list[0, :])])


plt.tight_layout()
plt.show()

print(f"Tempo:          {tempo} bpm")
print(f"Counts:         {counts}")
print(f"Frequency:      {f_s} Hz")
print(f"Step Size:      {np.round(step_size_march,1)}")
print(f"Mean Shift:     {np.round(np.mean(cents[0, :]),1)} cents")
print(f"SD Shift:       {np.round(np.std(cents[0, :]),1)} cents")
print(f"Mean Frequency: {np.round(np.mean(f_o[0, :]),1)} Hz")
print(f"SD Frequency:   {np.round(np.std(f_o[0, :]),1)} Hz")
