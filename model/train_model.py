# this file is meant to be run on the server, not locally
# this is why the directories are different

import os
import gc
from sklearn.metrics import matthews_corrcoef, f1_score, balanced_accuracy_score, accuracy_score
from utils.layers import FCCaps, Length, PrimaryCaps, Mask
from utils.tools import *

os.environ["CUDA_VISIBLE_DEVICES"]="2"

runs = 10 # number of iterations to perform
dir = "/home/radomir/har/esp32/data/home_data_demo/extracted_windows/" # top level directory of data
gestures = ["lr", "pp", "ng"] # gestures to be included
train_runs = [0, 1, 2] # runs to be included in training set
test_runs = [3] # runs to be included in test set
lr, lr_decay, batch_size, epochs = 5e-4, 0.95, 16, 200  # model training hyperparameters

# get data outside of loop
# x_train, y_train = load_csi(dir, gestures, train_runs) # for regular
# x_test, y_test = load_csi(dir, gestures, test_runs)
# x_test, y_test = load_csi(dir, gestures, test_runs, num=[5, 5, 5]) # for 5 of each gest in test

x_train, y_train = load_csi(dir, gestures, train_runs, num=[140, 140, 927]) # for imperfect windows
x_test, y_test = load_csi(dir, gestures, test_runs, num=[140, 140, 927])

# x_train, y_train = load_csi_2(dir, gestures, train_runs, num=[[14, 16, 36],
#                                                               [17, 20, 28],
#                                                               [10, 11, 47],
#                                                               [13, 19, 41]]) # for imperfect windows 2
# x_test, y_test = load_csi_2(dir, gestures, test_runs, num=[13, 10, 48]])

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

# list of alpha values
alphas = [0.25001]
# alphas = [0.15, 0.20, 0.25]#, 0.30, 0.35, 0.40]

for a_idx, alpha in enumerate(alphas):
    matrices = [] # variable for saving the confusion matrices
    training_times = [] # variable for saving training times
    inference_times = [] # variables for saving inference times
    len_train = 0
    len_test = 0
    hists = [] # variable for saving training histories
    MCC = []
    F1 = []
    B_ACC = []
    ACC = []

    # make directory
    if not os.path.exists(f"./results/{str(alpha)}"):
        os.makedirs(f"./results/{str(alpha)}")

    print(f"Starting Model Training {a_idx+1}/{len(alphas)}")
    pbar = ProgressBar(widgets=[Percentage(), Bar(), Timer()], maxval=runs).start()
    for run in range(runs):
        len_train = x_train.shape[0] * 0.75
        len_test = x_test.shape[0]
        
        # 2 conv eff-capsnet
        input = tf.keras.Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])) # n*400*52*1

        x = tf.keras.layers.Conv2D(filters=32, kernel_size=16, strides=(2,2), activation="relu",
                                padding="valid", kernel_regularizer="l2", bias_regularizer="l2")(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=8, strides=(2,1), activation="relu",
                                padding="valid", kernel_regularizer="l2", bias_regularizer="l2")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = PrimaryCaps(F=16, K=8, N=258, D=8, s=2)(x)

        x = tf.keras.layers.Dropout(rate=0.5)(x)
        activity_caps = FCCaps(len(gestures),16)(x)
        output = Length(name="length_capsnet_output")(activity_caps)

        capsnet = tf.keras.models.Model(inputs=[input], outputs=[activity_caps, output], name="capshar")
        
        # generator graph using Pixel Shuffling
        input_gen = tf.keras.Input(16*len(gestures))
        x_gen = tf.keras.layers.Dense(units=40*6*1, activation="relu")(input_gen)
        x_gen = tf.keras.layers.Reshape(target_shape=(40, 6, 1))(x_gen)
        R = 10
        x_gen = tf.keras.layers.Conv2D(filters=R**2, kernel_size=5, padding="same")(x_gen)
        x_gen = tf.nn.depth_to_space(x_gen, R)
        x_gen = tf.keras.layers.Cropping2D(cropping=((0, 0), (4, 4)))(x_gen)

        generator = tf.keras.models.Model(inputs=input_gen, outputs=x_gen, name="decoder")

        # define combined models
        inputs = tf.keras.Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))
        y_true = tf.keras.Input(shape=(y_train.shape[1]))
        out_caps, out_caps_len = capsnet(inputs)
        masked_by_y = Mask()([out_caps, y_true])  
        masked = Mask()(out_caps)

        x_gen_train = generator(masked_by_y)
        x_gen_eval = generator(masked)

        model = tf.keras.models.Model([inputs, y_true], [out_caps_len, x_gen_train], name="CapsHAR")
        model_test = tf.keras.models.Model(inputs, [out_caps_len, x_gen_eval], name="CapsHAR_test")

        if run == 0 and a_idx == 0:
            capsnet.summary()
            generator.summary()
            model_test.summary()

        # train the model
        adam = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=adam, loss=[marginLoss, "mse"], loss_weights=[alpha, (1-alpha)], metrics=["accuracy"])
        #callbacks
        log = tf.keras.callbacks.CSVLogger(f"./results/{str(alpha)}/log.csv", append=True)
        lr_decay_cb = tf.keras.callbacks.LearningRateScheduler(schedule=lambda epoch: max(lr * (lr_decay ** float(epoch)), 5e-5))
        callbacks = [log, lr_decay_cb]

        #Train
        t_start = time.time()
        history = model.fit([x_train, y_train], [y_train, x_train], batch_size=batch_size, epochs=epochs,
                validation_split=0.25, callbacks=callbacks, shuffle=True, verbose=0)
        training_times.append((time.time() - t_start))
        hists.append(history)
        if run == 0 and a_idx == 0:
            plot_model(history, f"./results/{str(alpha)}/")

        model.save(f"./results/{str(alpha)}/final_model.hdf5")
        capsnet.save(f"./results/{str(alpha)}/capshar.hdf5")

        pred, inference_time = load_model_predictions_generator(x_test, batch_size, model_test)
        inference_times.append(inference_time)

        # get performance metrics
        matrices.append(create_conf_matrix(y_test, pred))
        MCC.append(matthews_corrcoef(np.argmax(y_test, axis=1), np.argmax(pred, axis=1)))
        F1.append(f1_score(np.argmax(y_test, axis=1), np.argmax(pred, axis=1), average='micro'))
        B_ACC.append(balanced_accuracy_score(np.argmax(y_test, axis=1), np.argmax(pred, axis=1)))
        ACC.append(accuracy_score(np.argmax(y_test, axis=1), np.argmax(pred, axis=1)))
        
        print([round(b,3) for b in B_ACC])

        del input, x, activity_caps, output, input_gen, x_gen, inputs, y_true, out_caps, out_caps_len, \
            masked_by_y, masked, capsnet, generator, model, x_gen_train, x_gen_eval, model_test, \
            adam, log, lr_decay_cb, callbacks
        gc.collect()

        pbar.update(run+1)

    # Write results to file
    f_res = open(f"./results/{str(alpha)}/matrices.txt", "w")
    f_res.write(str(matrices))
    all_matrices = np.array(matrices)
    avg_matrix = np.mean(all_matrices, axis=0)
    std_matrix = np.std(all_matrices, axis=0)
    var_matrix = np.var(all_matrices, axis=0)
    f_res.write("\nAvg model matrix:\n")
    f_res.write(str(avg_matrix))
    f_res.write("\nStd model matrix:\n")
    f_res.write(str(std_matrix))
    f_res.write("\nVar model matrix:\n")
    f_res.write(str(var_matrix))
    f_res.write("\nMCC:")
    f_res.write(str(MCC))
    f_res.write("\nAvg MCC:")
    f_res.write(str(np.mean(MCC)))
    f_res.write("\nStd MCC:")
    f_res.write(str(np.std(MCC)))
    f_res.write("\nF1:")
    f_res.write(str(F1))
    f_res.write("\nAvg F1:")
    f_res.write(str(np.mean(F1)))
    f_res.write("\nStd F1:")
    f_res.write(str(np.std(F1)))
    f_res.write("\nBalanced accuracy:")
    f_res.write(str(B_ACC))
    f_res.write("\nAvg balanced accuracy:")
    f_res.write(str(np.mean(B_ACC)))
    f_res.write("\nStd balanced accuracy:")
    f_res.write(str(np.std(B_ACC)))
    f_res.write("\nAccuracy score:")
    f_res.write(str(ACC))
    f_res.write("\nAvg accuracy:")
    f_res.write(str(np.mean(ACC)))
    f_res.write("\nStd accuracy:")
    f_res.write(str(np.std(ACC)))
    f_res.write("\nAvg accuracy (from matrices):")
    f_res.write(str(np.round(np.mean(avg_matrix.diagonal()), 2)))
    f_res.write("\nTraining times:")
    f_res.write(str(training_times))
    f_res.write("\nAvg training time:")
    f_res.write(str(np.mean(training_times)))
    f_res.write("\nAvg training time per sample:")
    f_res.write(str(np.mean(training_times) / epochs / len_train))
    f_res.write("\nInference times:")
    f_res.write(str(inference_times))
    f_res.write("\nAvg inference time:")
    f_res.write(str(np.mean(inference_times)))
    f_res.write("\nAvg inference time per sample:")
    f_res.write(str(np.mean(inference_times) / len_test))
    f_res.write("\n")
    for hist in hists:
        f_res.write("Single run results:\n")
        for value in hist.history:
            f_res.write(str(value))
            f_res.write(str(hist.history[value]))
            f_res.write("\n")
    f_res.close()

    pbar.finish()

    plot_conf_matrix(avg_matrix, f'./results/{str(alpha)}/conf_matrix.png')
