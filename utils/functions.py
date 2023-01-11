import torch
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.autograd import Variable as var
torch.autograd.set_detect_anomaly(True)   

def create_stim_timecourse(dt, idx, t_steps, dur, start, noise_lbls):

    # define input shape
    input_shape = dt[0][0].shape
    w = input_shape[2]
    h = input_shape[3]
    c = input_shape[1]

    # extract stimuli
    s1 = dt[idx][0][0, 0, :, :, :]
    isi = dt[idx][0][0, 1, :, :, :]
    s2 = dt[idx][0][0, 2, :, :, :]

    # encode timtesteps
    t_steps_label = torch.zeros(t_steps)
    for i in range(len(start)):
        t_steps_label[start[i]:start[i]+dur] = i+1

    # initiate dataframes
    noise_imgs = torch.Tensor(t_steps, 1, c, w, h)
    noise_lbls = dt[idx][1]

    # timecourse
    for t in range(t_steps):

        # assign stimuli to current timestep
        if t_steps_label[t] == 0:           
            noise_imgs[t, 0, :, :, :] = isi.unsqueeze(0)*0
        elif t_steps_label[t] == 1:
            noise_imgs[t, 0, :, :, :] = s1.unsqueeze(0)
        elif t_steps_label[t] == 2: 
            noise_imgs[t, :, :, :] = s2.unsqueeze(0)

    # return noise_imgs, noise_lbls
    return noise_imgs, noise_lbls

def download_data(mnist=False):

    # download data
    if mnist:

        testdt = datasets.MNIST(
                root = 'data', 
                train = True, 
                transform = ToTensor()
            )

        traindt = datasets.MNIST(
                root = 'data', 
                train = False, 
                transform = ToTensor()
            )

    else:

        traindt = datasets.FashionMNIST(root='datasets/', 
                                        train=True, 
                                        transform=ToTensor())
        
        testdt = datasets.FashionMNIST(root='datasets/', 
                                        train=False, 
                                        transform=ToTensor())

    # determine number of samples
    train_n = len(traindt)
    test_n = len(testdt)

    return traindt, testdt, train_n, test_n


def load_data(traindt, testdt, batch_size, shuffle, num_workers):

    # data loader
    ldrs = {
        'train' : torch.utils.data.DataLoader(traindt, 
                                            batch_size=batch_size, 
                                            shuffle=shuffle, 
                                            num_workers=num_workers),
        
        'test'  : torch.utils.data.DataLoader(testdt, 
                                            batch_size=batch_size, 
                                            shuffle=shuffle, 
                                            num_workers=num_workers),
    }

    return ldrs


def train(numepchs, model, ldrs, lossfunct, optim, batch_size, t_steps, run=None, rand_init=None, run_num=None):

    print(30*'-')
    print('Training in progress...')
    print(30*'-')

    # initiate frame to store losses per batch
    run_losses = torch.zeros(numepchs*len(ldrs['train']))
    
    # train model
    model.train()
        
    # Train the model (total amount of steps)
    ttlstp = len(ldrs['train'])
    
    batch_count = 0
    for epoch in range(numepchs): # images and labels
        for a, (imgs, lbls) in enumerate(ldrs['train']):
            # if a == 0:

            # clip last incomplete batch
            if len(lbls) != batch_size:
                continue

            # add inputs and labels
            ax = []
            for t in range(t_steps):
                ax.append(var(imgs[:, t, :, :, :]))
            ay = var(lbls)   # labels

            # zero the parameter gradients
            for param in model.parameters():
                param.grad = None
            # optim.zero_grad()

            outp = model.forward(ax, True)
            losses = lossfunct(outp[len(outp)-1], ay)
            run_losses[batch_count] = losses.item()
    
            losses.backward() 
            optim.step()                
            
            # if (a+1) % 50 == 0:
            #     print ('Random init {} (run {}), Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
            #         .format(rand_init+1, run_num+1, epoch+1, numepchs, a+1, ttlstp, losses.item()))

            if (a+1) % 50 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, numepchs, a+1, ttlstp, losses.item()))

            # print('weight conv1: ', model.conv1.weight.grad[0, 0, 0, 0])
            # print('tau1: ', model.sconv1.tau1.grad)

            # add metric to neptune
            if run != None:
                run["metrics/train/loss"].log(losses.item())
                run["metrics/train/conv1-weight"].log(model.conv1.weight.grad[0, 0, 0, 0])
                run["metrics/train/conv2-weight"].log(model.conv2.weight.grad[0, 0, 0, 0])
                run["metrics/train/conv3-weight"].log(model.conv3.weight.grad[0, 0, 0, 0])
                run["metrics/train/fc1-weight"].log(model.fc1.weight.grad[0, 0])
                # run["metrics/train/tau1"].log(model.sconv1.tau1.grad)
                # run["metrics/train/tau2"].log(model.sconv1.tau2.grad)
                # run["metrics/train/sigma"].log(model.sconv1.sigma.grad)
            
            # increment batch count
            batch_count+=1

    if run != None:
        return run_losses, run
    else:
        return run_losses

def test(model, ldrs, t_steps, batch_size, batch=True, run=None, rand_init=None, run_num=None):

    print(30*'-')
    print('Validation in progress...')
    print(30*'-')

    # Train the model (total amount of steps)
    ttlstp = len(ldrs['test'])

    # initiate frame to store accuracy per batch
    run_accu = torch.zeros(len(ldrs['test']))

    # Test the model
    model.eval()
    with torch.no_grad():
        for a, (imgs, lbls) in enumerate(ldrs['test']):
        # for a, (imgs, lbls) in range(1):
            # if a == 0:

            # clip last incomplete batch
            if len(lbls) != batch_size:
                continue

            imgs_seq = []
            for t in range(t_steps):
                imgs_seq.append(imgs[:, t, : , :, :])
            
            testoutp = model(imgs_seq, batch=True)
            predicy = torch.argmax(testoutp[len(testoutp)-1], dim=1)
            accu = (predicy == lbls).sum().item() / float(lbls.size(0))
            run_accu[a] = accu

            # if (a+1) % 50 == 0:
            #     print ('Random init {} (run {}), Step [{}/{}], accuraciy: {:.4f}' 
            #         .format(rand_init+1, run_num+1, a+1, ttlstp, accu))

            if run != None:
                run["metrics/test/acc"].log(accu)

    if run != None:
        return run_accu, run
    else:
        return run_accu

# def train_feedforward(numepchs, model, ldrs, lossfunct, optim, batch_size, lr, optimizer):

#     run = neptune.init(
#         project="abra1993/adapt-dnn",
#         api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ODkxNGY3NS05NGJlLTQzZDEtOGU5Yy0xMjJlYzI0YzE2YWUifQ==",
#     )  # your credentials

#     params = {"learning_rate": lr, "optimizer": optimizer}
#     run["parameters"] = params

#     print(30*'-')
#     print('Training in progress...')
#     print(30*'-')

#     # initiate frame to store losses per batch
#     run_losses = torch.zeros(numepchs*len(ldrs['train']))
    
#     # train model
#     model.train()
        
#     # Train the model (total amount of steps)
#     ttlstp = len(ldrs['train'])
    
#     for epoch in tqdm (range(numepchs), desc='Training'): # images and labels
#         for a, (img, lbls) in ldrs['train']: # iterates over batches

#             # clipe last batch
#             if len(lbls) != batch_size:
#                 continue

#             ax = var(img) 
#             ay = var(lbls)   # labels

#             outp = model.forward(ax)
#             losses = lossfunct(outp, ay)
#             run_losses[batch_count] = losses.item()
            
#             optim.zero_grad()           
#             losses.backward()  
                      
#             optim.step()                
            
#             if (a+1) % 100 == 0:
#                 print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
#                        .format(epoch + 1, numepchs, a + 1, ttlstp, losses.item()))

#             # add metric neptune
#             run["metrics/train/loss"].log(losses.item())
            
#             # increment batch count
#             batch_count+=1

#     return run_losses, run

# def test_feedforward(model, ldrs, run):

#     # Test the model
#     model.eval()
#     with torch.no_grad():
#         for a, (img, lbls) in enumerate(ldrs['test']):
#             testoutp = model(img)
#             predicy = torch.argmax(testoutp, dim=1)
#             accu = (predicy == lbls).sum().item() / float(lbls.size(0))
#             run["metrics/test/acc"].log(accu)
#             pass
#             print(' Accuracy of the model  %.2f' % accu)

#     return run