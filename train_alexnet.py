import torch
import torchaudio
from networks import AudioNet
import audiofolder
from torch.utils.data import DataLoader
import time
import math

torchaudio.set_audio_backend("soundfile")

def train(epochs, lr):

    #setup the network, optimizers, stuff like that
    net = AudioNet(8000,10).cuda()
    print("network created")
    loss_fn = torch.nn.CrossEntropyLoss()
    #criterion = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    print("optimizers and loss created")
    #load data
    dataset = audiofolder.AudioFolder("D:AudioMNIST/train/AudioMNIST_ByNumber_1s_8k/")
    dataset_test = audiofolder.AudioFolder("D:AudioMNIST/test/AudioMNIST_ByNumber_1s_8k/")
    #dataset = audiofolder.AudioFolder("D:AudioMNIST/AudioMNIST_Tiny/")
    print("dataset created, ready to train!")
    #this dataloader gives a tuple of (wav, fs)
    train_dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size = 10, shuffle=False)
    for ep in range(epochs):
        start_time = time.time()
        running_loss_train = 0
        running_loss_test = 0
        for (idx, data) in enumerate(train_dataloader):
            wav, fs = data[0]
            labels = data[1].long().cuda()
            pred = net(wav.cuda())
            loss = loss_fn(pred,labels)
            criterion.zero_grad()
            loss.backward()
            criterion.step()

            running_loss_train += loss.item()

        for (idx,data) in enumerate(test_dataloader):
            wav, fs = data[0]
            labels = data[1].long().cuda()
            pred = net(wav.cuda())
            loss = loss_fn(pred,labels)
            running_loss_test += loss.item()

        print("Epoch {} finished: {} train loss, {} test loss, {} seconds".format(ep+1,(running_loss_train/len(train_dataloader)),(running_loss_test/len(test_dataloader)),math.floor(time.time()-start_time)))

    print("training finished! saving model")
    torch.save(net.state_dict(),"audionet_1kep.pth")
    print("model saved")

    #run through the model
    #print some useful knowledge

train(1000,.0005)
