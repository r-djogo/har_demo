import sys, os
import math
import collections
import csv

from statistics import mode
from scipy.io import savemat, loadmat

from wait_timer import WaitTimer
from read_stdin import readline, print_until_first_csi_line

from model.utils.layers import FCCaps, Length, PrimaryCaps, Mask
from model.utils.tools import *


def process(res, perm_amp, csi_dict, save_data, timer):
    # Parser
    all_data = res.split(',')

    # MAC address check
    mac_addr = all_data[2]
    if mac_addr != "24:62:AB:FE:35:F4":
        return 0

    csi_data = all_data[25].split(" ")
    csi_data[0] = csi_data[0].replace("[", "")
    csi_data[-1] = csi_data[-1].replace("]", "")

    csi_data.pop()
    csi_data = [int(c) for c in csi_data if c]
    imaginary = []
    real = []
    for i, val in enumerate(csi_data):
        if i % 2 == 0:
            imaginary.append(val)
        else:
            real.append(val)

    csi_size = len(csi_data)
    amplitudes = []
    if len(imaginary) > 0 and len(real) > 0:
        for j in range(int(csi_size / 2)):
            amplitude_calc = math.sqrt(imaginary[j] ** 2 + real[j] ** 2)
            amplitudes.append(amplitude_calc)

        perm_amp.append(amplitudes)

        if save_data:
            # save CSI to dicts
            csi_dict["time"].append(float(timer))
            csi_dict["csi_amp"].append(amplitudes)

    return 1

def gesture_classification(model, csi):
    # expand input dims
    data_temp = np.expand_dims(np.array(csi), axis=0)
    data_temp = np.expand_dims(data_temp, axis=3)
    # remove inactive subcarriers
    active_sc = list(range(6,32)) + list(range(33,59))
    data_temp = data_temp[:,:,active_sc,:]
    # normalize data
    data_temp = (data_temp - np.mean(data_temp)) / np.std(data_temp)
    # run inference
    activity_caps, pred = model.predict(data_temp, verbose=0)
    # print class
    pred_class = np.argmax(pred)
    gestures = ["left-right", "push-pull", "no gesture detected"]
    # print(gestures[pred_class])

    return gestures[pred_class]

def real_time(csi_data_output, gui_output, stop_event, use_model, use_plot=False, save_data=False, total_len=9999, early_stop=None):
    # Wait Timers
    total_exp_timer = WaitTimer(total_len)
    gesture_wait_timer = WaitTimer(0.25)

    # Deque definition
    perm_amp = collections.deque(maxlen=400)
    gesture_buff = collections.deque(maxlen=5)

    # load the model from file
    capshar = tf.keras.models.load_model("/home/raso/esp/esp32-csi-tool/python_utils/demo_gui/model/capshar_home_demo.hdf5",
                                        custom_objects={"PrimaryCaps": PrimaryCaps, "marginLoss": marginLoss,
                                                        "FCCaps": FCCaps, "Length": Length, "Mask": Mask})
    
    csi_dict = {}
    csi_dict["time"] = []
    csi_dict["csi_amp"] = []
            
    total_exp_timer.update()
    start_time = time.time()
    print("Starting Live CSI")
    
    last_gesture = ""
    lag_count = 0

    while not total_exp_timer.check():
        if stop_event.is_set():
            break
        line = readline()
        if "CSI_DATA" in line:
            if save_data:
                timer = time.time() - start_time
                gui_output.timer.emit(int(timer))
            else: timer = 0

            if process(line, perm_amp, csi_dict, save_data, timer):
                if use_plot:
                    if use_model.is_set():
                        if lag_count == 5:
                            csi_data_output.cb_append_data_array(np.array(perm_amp)[-6:-1,23])
                            lag_count = 0
                        else:
                            lag_count = lag_count + 1

                        if gesture_wait_timer.check() and len(perm_amp) == 400:
                            gesture_wait_timer.update()
                            gesture = gesture_classification(capshar, perm_amp)
                            gesture_buff.append(gesture)


                            current_gesture = mode(gesture_buff)
                            if len(gesture_buff) == 5 and current_gesture != last_gesture:
                                last_gesture = current_gesture
                                # print(current_gesture)
                                gui_output.prediction.emit(current_gesture)
                    else:
                        csi_data_output.cb_append_data_point(np.array(perm_amp)[-1,23])
                elif save_data:
                    if gesture_wait_timer.check() and len(perm_amp) == 400:
                        gesture_wait_timer.update()
                        gesture = gesture_classification(capshar, perm_amp)
                        gesture_buff.append(gesture)


                        current_gesture = mode(gesture_buff)
                        if len(gesture_buff) == 5 and current_gesture != last_gesture:
                            last_gesture = current_gesture
                            print(current_gesture, timer)

    if save_data:
        file_idx = 0
        while True:
            if os.path.isfile(f"/home/raso/esp/esp32-csi-tool/python_utils/demo_gui/csi/csi{file_idx}_gui.mat"):
                file_idx += 1
                continue
            else:
                savemat(f"/home/raso/esp/esp32-csi-tool/python_utils/demo_gui/csi/csi{file_idx}_gui.mat", csi_dict)
                break
        
        if total_len == 330:
            print("Extracting Windows")
            os.makedirs(f"/home/raso/esp/esp32-csi-tool/python_utils/demo_gui/csi/extracted_windows_{file_idx}_gui")
            csi_dict = loadmat(f"/home/raso/esp/esp32-csi-tool/python_utils/demo_gui/csi/csi{file_idx}_gui.mat")
            
            lr_timings = list(range(15,301,15))
            pp_timings = list(range(20,306,15))
            
            count_lr = 0
            count_pp = 0
            count_ng = 0

            for temp in range(54, 1261):
                t = temp/4 # need to increment by 0.25 seconds

                status = 0
                for lr in lr_timings:
                    if t >= (lr-1.5) and t <= lr:
                        status = 1
                if status == 0:
                    for pp in pp_timings:
                        if t >= (pp-1.5) and t <= pp:
                            status = 2
                
                start_idx = np.argmin(np.abs(csi_dict["time"].squeeze() - t))
                end_idx = start_idx + 400
                window = csi_dict["csi_amp"][start_idx:end_idx,:]
                if status == 1:
                    savemat(f"/home/raso/esp/esp32-csi-tool/python_utils/demo_gui/csi/extracted_windows_{file_idx}/lr_run{file_idx}_rep{count_lr}.mat", {"window": window})
                    count_lr = count_lr + 1
                elif status == 2:
                    savemat(f"/home/raso/esp/esp32-csi-tool/python_utils/demo_gui/csi/extracted_windows_{file_idx}/pp_run{file_idx}_rep{count_pp}.mat", {"window": window})
                    count_pp = count_pp + 1
                else:
                    savemat(f"/home/raso/esp/esp32-csi-tool/python_utils/demo_gui/csi/extracted_windows_{file_idx}/ng_run{file_idx}_rep{count_ng}.mat", {"window": window})
                    count_ng = count_ng + 1
                
            print(count_lr, count_pp, count_ng)
            print("Done")

        if not stop_event.is_set():
            early_stop.timer.emit(1)
    
    print("End")

def clear_stdin(stop_clear):
    while True:
        if stop_clear.is_set():
            break
        else:
            line = readline()

def csi_playback(csi_file, csi_output, replay_csv, stop_event, replay_outputs):
    csi_dict = loadmat(csi_file)
    # time = csi_dict["time"]
    # csi = csi_dict["csi_amp"]

    # load csv
    
    pred_gests = []
    true_gests = []
    pred_timings = []
    with open(replay_csv) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",")
        for row in spamreader:
            pred_gests.append(row[0].strip())
            true_gests.append(row[1].strip())
            pred_timings.append(float(row[2]))
    replay_outputs.outputs.emit("no gesture", "no gesture")

    start_time = time.time()
    
    idx = 0
    while idx < len(csi_dict["csi_amp"][:,23]):
        if stop_event.is_set():
            break

        timer = time.time() - start_time

        if len(pred_timings) > 0 and pred_timings[0] < timer:
            replay_outputs.outputs.emit(pred_gests[0], true_gests[0])
            pred_gests.pop(0)
            true_gests.pop(0)
            pred_timings.pop(0)

        if csi_dict["time"][0][idx] < timer:
            # print(idx, csi_dict["time"][0][idx], timer)
            csi_output.cb_append_data_point(csi_dict["csi_amp"][idx,23])
            idx += 1
            time.sleep(0.007)
        