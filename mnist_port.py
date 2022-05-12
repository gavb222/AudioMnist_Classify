import os
import torch
import torchaudio
import random
directory = "D:AudioMNIST/AudioMNIST_ByNumber_1s_8k/"
new_directory_train = "D:AudioMNIST/train/AudioMNIST_ByNumber_1s_8k/"
new_directory_test = "D:AudioMNIST/test/AudioMNIST_ByNumber_1s_8k/"

for subdir, dirs, files in os.walk(directory):
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith(".wav"):
            filename = file[:-4]
            number = file[0]

            wavData, fs = torchaudio.load(filepath)

            #print(wavData.size())
            if fs != 8000:
                wavData = torchaudio.transforms.Resample(fs, 8000)(wavData)
                print("finished resampling " + file)
            else:
                print(file + " is already at appropriate fs")

            length = wavData.squeeze().size()[0]
            n_pad = 8000 - length

            if n_pad > 0:
                pad = torch.nn.ConstantPad1d((0,n_pad),0)
                wavData = pad(wavData)

            if random.random() > .1:
                output_filepath = new_directory_train + number
            else:
                output_filepath = new_directory_test + number
            torchaudio.save(output_filepath + os.sep + file, wavData.squeeze(),8000)
