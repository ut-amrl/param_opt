import pandas as pd
import numpy as np

normal_speed = 1.0
turbo_speed = 1.0
accel_limit = 6.0
maxTurnRate = 0.25
commandInterval = 1.0 / 20
speed_to_erpm_gain = 5356
speed_to_erpm_offset = 180.0
erpm_speed_limit = 22000
steering_to_servo_gain = -1.2015
steering_to_servo_offset = 0.55
servo_min = 0.01
servo_max = 0.99
wheelbase = 0.324
steer_joystick_idx = 0
drive_joystick_idx = 4


def get_value_at_time(time, times, values):
    if time < times[0]:
        return values[0]
    elif time > times[-1]:
        return values[-1]
    left, right = 0, len(times) - 1
    while right - left > 1:
        mid = (left + right) // 2
        if time > times[mid]:
            left = mid
        elif time < times[mid]:
            right = mid
        else:
            return values[mid]
    return values[left]
    # return values[left] + (values[right] - values[left]) * (time - times[left]) / (
    #     times[right] - times[left]
    # )


def extract_joystick_data():
    data_frame = pd.read_csv("_slash_joystick.csv")

    secs = data_frame["secs"].to_numpy()
    nsecs = data_frame["nsecs"].to_numpy()
    joystick_times = secs + nsecs / 1e9 - secs[0]

    axes_strings = data_frame["axes"].to_numpy()
    axes = []
    for ax in axes_strings:
        ax = ax[1:-1]
        ax = ax.split(", ")
        ax = [float(a) for a in ax]
        axes.append(ax)
    axes = np.array(axes)

    # print(len([axe[0] for axe in axes if axe[0] != 0]))
    # print(len([axe[0] for axe in axes if axe[0] == 0]))

    steer_joystick = axes[:, steer_joystick_idx]
    drive_joystick = axes[:, drive_joystick_idx]
    max_speed = normal_speed
    speed = drive_joystick * max_speed  # array of all the speeds
    steering_angle = steer_joystick * maxTurnRate  # array of all the steering angles?

    last_speed = 0.0
    clipped_speeds = []
    for s in speed:
        smooth_speed = max(s, last_speed - commandInterval * accel_limit)
        smooth_speed = min(smooth_speed, last_speed + commandInterval * accel_limit)
        last_speed = smooth_speed
        erpm = speed_to_erpm_gain * smooth_speed + speed_to_erpm_offset
        erpm_clipped = min(max(erpm, -erpm_speed_limit), erpm_speed_limit)
        clipped_speed = (erpm_clipped - speed_to_erpm_offset) / speed_to_erpm_gain
        clipped_speeds.append(clipped_speed)
    clipped_speeds = np.array(clipped_speeds)

    servo = steering_to_servo_gain * steering_angle + steering_to_servo_offset
    clipped_servo = np.fmin(np.fmax(servo, servo_min), servo_max)
    steering_angle = (clipped_servo - steering_to_servo_offset) / steering_to_servo_gain

    # atan(wheelbase*curvature) == steering_angle
    curvature = np.tan(steering_angle) / wheelbase
    rot_vel = clipped_speeds / wheelbase * np.tan(steering_angle)
    # print(steering_angle)
    # print([i for i in curvature if i != 0])

    return (joystick_times, curvature, clipped_speeds)


def range_data():
    scan = pd.read_csv("_slash_scan.csv")

    range_min = 0.02
    range_max = 30
    angle_incr = scan["angle_increment"][0]
    ranges_list = [eval(i) for i in scan["ranges"]]

    secs = scan["secs"].to_numpy()
    nsecs = scan["nsecs"].to_numpy()
    times = secs + nsecs / 1e9 - secs[0]

    clouds = []

    for ranges in ranges_list:
        theta = scan["angle_min"][0]
        pcloud = []
        for i in ranges:
            if i < range_min or i > range_max:
                pcloud.append([0, 0])
            else:
                pcloud.append([i * np.cos(theta), i * np.sin(theta)])
            theta += angle_incr

        clouds.append(pcloud)

    return (times, clouds)


def write_train_data(n, filename):
    scan_data = range_data()
    joy_data = extract_joystick_data()

    start = scan_data[0][0]
    end = scan_data[0][-1]
    training_data = pd.DataFrame()

    time_points = np.linspace(start + 2, end, n)
    lidar_delay = 0
    joy_curvs = []
    joy_vels = []
    pclouds = []
    xpoints = [[] for i in range(len(scan_data[1][0]))]
    ypoints = [[] for i in range(len(scan_data[1][0]))]
    for t in time_points:
        joy_curvs.append(get_value_at_time(t, joy_data[0], joy_data[1]))
        joy_vels.append(get_value_at_time(t, joy_data[0], joy_data[2]))
        pcloud = get_value_at_time(t + lidar_delay, scan_data[0], scan_data[1])

        for i in range(len(pcloud)):
            xpoints[i].append(pcloud[i][0])
            ypoints[i].append(pcloud[i][1])

    training_data["joy_curvature"] = joy_curvs
    training_data["joy_velocity"] = joy_vels
    for i in range(len(xpoints)):
        training_data["p" + str(i) + "_x"] = xpoints[i]
        training_data["p" + str(i) + "_y"] = ypoints[i]
    print("writing data to " + filename)
    training_data.to_csv(filename)


write_train_data(4000, "dataset.csv")
# rosbagTimestamp,header,seq,stamp,secs,nsecs,frame_id,axes,buttons

# rosbagTimestamp,header,seq,stamp,secs,nsecs,frame_id,angle_min,angle_max,angle_increment,time_increment,scan_time,range_min,range_max,ranges,intensities
