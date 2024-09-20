import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from time import time
from pix2pix_torch import Generator,Discriminator
from densetorch import Gen
from torchvision.utils import save_image
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim
# from skimage.metrics import structural_similarity as SSIM
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

# Load pre-trained VGG19 network and select specific layers
dev = torch.device('cuda')
vgg19 = models.vgg19(pretrained=True).features.to(dev).eval()

# Set VGG layers to not require gradients
for param in vgg19.parameters():
    param.requires_grad = False

# Function to extract features from specific layers
def get_features(x, model, layers):
    features = []
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features.append(x)
    return features

# Define the layers to be used for feature extraction
vgg_layers = ['0', '5', '10', '19', '28']  # These layers correspond to conv1_1, conv2_1, conv3_1, conv4_1, conv5_1

def perceptual_loss(gen_output, ref_batch, model, layers):
    gen_features = get_features(gen_output, model, layers)
    ref_features = get_features(ref_batch, model, layers)
    loss = 0
    for gf, rf in zip(gen_features, ref_features):
        loss += F.l1_loss(gf, rf)
    return loss


# def calculate_metrics(gt_image, pred_image):
#     # psnr_value = peak_signal_noise_ratio(gt_image, pred_image)
#     ssim_value, _ = SSIM(gt_image, pred_image, full=True)
#     return ssim_value

# Set constants
CHECKPOINT_PATH = './training_checkpoints/'
LOG_PATH = './logs/'

def convert_to_rgb(image):
    return image.repeat(1, 3, 1, 1)

# Function to extract features from specific layers
def get_features(x, model, layers):
    x = convert_to_rgb(x)  # Convert grayscale to 3-channel
    features = []
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features.append(x)
    return features

# Define batch generator (used in training loop)
def batch_generator(x, n):
    """
    X: data
    n: batch size
    """
    start, stop = 0, n
    while True:
        if start < stop:
            yield torch.tensor(x[start:stop]).float()
        else:
            break
        start = stop
        stop = (stop + n) % len(x)

def save_image(image, path):
    image = image.detach().cpu().numpy()
    plt.imshow(image, cmap='gray')
    # plt.axis('off')  # Turn off axis labels
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory
    
def grad_penalty(D, xr, xf, image_batch,device):
    t = torch.rand(xr.size(0), 1, 1, 1, device=device)
    xm = t * xr + (1 - t) * xf
    xm.requires_grad_(True)
    WDmid = D(xm,image_batch)
    Gradmid = torch.autograd.grad(outputs=WDmid, inputs=xm,
                                  grad_outputs=torch.ones_like(WDmid),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
    Gradmid = Gradmid.view(Gradmid.size(0), -1)
    GP = torch.mean((Gradmid.norm(2, dim=1) - 1) ** 2)
    return GP

def generator_loss(disc_generated_output, gen_output, target, lambda_value, model, layers):
    # loss_object = nn.BCEWithLogitsLoss()
    # gan_loss = loss_object(disc_generated_output, torch.ones_like(disc_generated_output))
    gan_loss=-1e-3*disc_generated_output.mean()
    l1_loss = F.l1_loss(gen_output, target)
    perc_loss = perceptual_loss(gen_output, target, model, layers)
    # print('perc loss * 0.001: ' , 0.001 * perc_loss.item())
    total_gen_loss = gan_loss + (lambda_value * l1_loss) + (0.001 * perc_loss)  
    # print('lambda_value * l1_loss : ' , lambda_value * l1_loss.item())
    return total_gen_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = nn.BCEWithLogitsLoss()
    real_loss = loss_object(disc_real_output, torch.ones_like(disc_real_output))
    generated_loss = loss_object(disc_generated_output, torch.zeros_like(disc_generated_output))
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

# def train_step(image_batch, ref_batch,generator,discriminator, gen_optim, disc_optim, device):
#     image_batch, ref_batch = image_batch.to(device), ref_batch.to(device)
#     gen_optim.zero_grad()
#     disc_optim.zero_grad()
#     gen_output = generator(image_batch)
#     # print(gen_output.shape)
#     disc_real_output = discriminator(ref_batch, image_batch)
#     disc_generated_output = discriminator(gen_output.detach(), image_batch)
#     print('disc fakeout : ', torch.mean(disc_generated_output).item())
#     print('disc realout : ',  torch.mean(disc_real_output).item())
#     GP = 10 * grad_penalty(discriminator, ref_batch, gen_output, image_batch,device)
#     # lossD = disc_generated_output.mean() - disc_real_output.mean() + GP
#     lossD = discriminator_loss(disc_real_output, disc_generated_output)
#     print('GP : ' ,GP.item())
#     print('discloss : ', lossD.item())
#     lossD=lossD+GP
#     # Lloss = torch.mean((ref_batch[0,0,:,:] - gen_output[0,0,:,:]) ** 2)
#     Lloss=generator_loss(disc_generated_output, ref_batch[0,0,:,:] , gen_output[0,0,:,:], lambda_value=100)
#     # print('generator loss shape : ', Lloss.shape)
#     # Adloss = -torch.mean(torch.log(torch.tensor(SSIM(ref_batch[0,0,:,:].cpu().numpy(), gen_output[0,0,:,:].detach().cpu().numpy(), data_range=255.0)).to(device)))
#     ssimtrain=ssim(gen_output,ref_batch)
#     Adloss = 1-ssimtrain
#     print('ssim train : ', ssimtrain.item())

#     # lossG = Lloss + 2e-3* Adloss + -1e-3 * disc_generated_output.mean()
#     # print('discfakeoutshape : ' ,disc_generated_output.shape)
    
#     lossG = Lloss + 2e-2*Adloss 
#     # lossG = Lloss + Adloss + disc_generated_output.mean()
#     # lossG = Gloss
#     lossD.backward(retain_graph=True)
#     lossG.backward()
#     disc_optim.step()
#     gen_optim.step()
    
#     return lossG.item(), lossD.item()

# def train_step(image_batch, ref_batch, generator, discriminator, gen_optim, disc_optim, device, model, layers):
#     image_batch, ref_batch = image_batch.to(device), ref_batch.to(device)
#     gen_optim.zero_grad()
#     disc_optim.zero_grad()
#     gen_output = generator(image_batch)
#     disc_real_output = discriminator(ref_batch, image_batch)
#     disc_generated_output = discriminator(gen_output.detach(), image_batch)
#     GP = 10 * grad_penalty(discriminator, ref_batch, gen_output, image_batch, device)
#     print('disc genrerated output mean * 1e-3 : ',-1e-3*disc_generated_output.mean().item())
#     print('disc real output mean * 1e-3 : ',-1e-3*disc_real_output.mean().item())
#     print('gradient penalty', GP.item())
#     lossD=disc_generated_output.mean()-disc_real_output.mean()+GP
#     gen_output = generator(image_batch)
#     disc_real_output = discriminator(ref_batch, image_batch)
#     disc_generated_output = discriminator(gen_output.detach(), image_batch)
#     GP = 10 * grad_penalty(discriminator, ref_batch, gen_output, image_batch, device)
#     print('disc genrerated output mean * 1e-3 : ',-1e-3*disc_generated_output.mean().item())
#     print('disc real output mean * 1e-3 : ',-1e-3*disc_real_output.mean().item())
#     print('gradient penalty', GP.item())
#     lossD=disc_generated_output.mean()-disc_real_output.mean()+GP
#     lossD.backward()
#     disc_optim.step()

    
#     gen_output = generator(image_batch)
#     disc_real_output = discriminator(ref_batch, image_batch)
#     disc_generated_output = discriminator(gen_output.detach(), image_batch)
#     GP = 10 * grad_penalty(discriminator, ref_batch, gen_output, image_batch, device)
#     print('disc genrerated output mean * 1e-3 : ',-1e-3*disc_generated_output.mean().item())
#     print('disc real output mean * 1e-3 : ',-1e-3*disc_real_output.mean().item())
#     print('gradient penalty', GP.item())
#     lossD=disc_generated_output.mean()-disc_real_output.mean()+GP
#     gen_output = generator(image_batch)
#     disc_real_output = discriminator(ref_batch, image_batch)
#     disc_generated_output = discriminator(gen_output.detach(), image_batch)
#     GP = 10 * grad_penalty(discriminator, ref_batch, gen_output, image_batch, device)
#     print('disc genrerated output mean * 1e-3 : ',-1e-3*disc_generated_output.mean().item())
#     print('disc real output mean * 1e-3 : ',-1e-3*disc_real_output.mean().item())
#     print('gradient penalty', GP.item())
#     lossD=disc_generated_output.mean()-disc_real_output.mean()+GP
#     lossD.backward()
#     disc_optim.step()

#     gen_output = generator(image_batch)
#     disc_real_output = discriminator(ref_batch, image_batch)
#     disc_generated_output = discriminator(gen_output.detach(), image_batch)
#     GP = 10 * grad_penalty(discriminator, ref_batch, gen_output, image_batch, device)
#     print('disc genrerated output mean * 1e-3 : ',-1e-3*disc_generated_output.mean().item())
#     print('disc real output mean * 1e-3 : ',-1e-3*disc_real_output.mean().item())
#     print('gradient penalty', GP.item())
#     lossD=disc_generated_output.mean()-disc_real_output.mean()+GP
#     gen_output = generator(image_batch)
#     disc_real_output = discriminator(ref_batch, image_batch)
#     disc_generated_output = discriminator(gen_output.detach(), image_batch)
#     GP = 10 * grad_penalty(discriminator, ref_batch, gen_output, image_batch, device)
#     print('disc genrerated output mean * 1e-3 : ',-1e-3*disc_generated_output.mean().item())
#     print('disc real output mean * 1e-3 : ',-1e-3*disc_real_output.mean().item())
#     print('gradient penalty', GP.item())
#     lossD=disc_generated_output.mean()-disc_real_output.mean()+GP
#     lossD.backward()
#     disc_optim.step()



#     lossG = generator_loss(disc_generated_output, gen_output, ref_batch, lambda_value=100, model=model, layers=layers)
#     ssimtrain = ssim(gen_output, ref_batch)
#     print('ssim train : ', ssimtrain.item())
#     Adloss = 1 - ssimtrain
#     lossG += 2e-2 * Adloss 

#     lossG.backward()
#     gen_optim.step()

#     return lossG.item(),lossD.item()

def train_step(image_batch, ref_batch, generator, discriminator, gen_optim, disc_optim, device, model, layers,iter,epoch):
    image_batch, ref_batch = image_batch.to(device), ref_batch.to(device)
    # Train discriminator three times
    save_image(image_batch[0,0,:,:], f'project_priyank/pred/artit/arti_{iter}.png')
    save_image(ref_batch[0,0,:,:], f'project_priyank/pred/gtt/GT_{iter}.png')
    for _ in range(5):
        gen_optim.zero_grad()
        disc_optim.zero_grad()
        
        # Generate fake images
        gen_output = generator(image_batch)
        
        # Get discriminator outputs
        disc_real_output = discriminator(ref_batch, image_batch)
        disc_generated_output = discriminator(gen_output.detach(), image_batch)
        
        # Calculate gradient penalty
        GP = 10 * grad_penalty(discriminator, ref_batch, gen_output, image_batch, device)
        
        # Print debug information
        # print('disc generated output mean * 1e-3 - disc real output mean * 1e-3  : ', 1e-3 * disc_generated_output.mean().item()- 1e-3 * disc_real_output.mean().item())
        # print('disc generated output mean * 1e-3 , disc real output mean * 1e-3  : ', 1e-3 * disc_generated_output.mean().item(), 1e-3 * disc_real_output.mean().item())
        # print('gradient penalty', GP.item())
        
        # Calculate discriminator loss
        lossD = disc_generated_output.mean() - disc_real_output.mean() + GP
        # print("Discriminator Loss:", lossD.item())

        
        # Backpropagate discriminator loss
        lossD.backward(retain_graph= True)
        disc_optim.step()
    
    # Train generator once
    gen_optim.zero_grad()
    
    # Generate fake images
    gen_output = generator(image_batch)
    save_image(gen_output[0,0,:,:], f'project_priyank/pred/outt/outt_{iter}.png')
    
    # Get discriminator output for the generated images
    disc_generated_output = discriminator(gen_output, image_batch)
    
    # Calculate generator loss
    lossG = generator_loss(disc_generated_output, gen_output, ref_batch, lambda_value=99, model=model, layers=layers)
    
    # Calculate SSIM
    ssimtrain = ssim(gen_output, ref_batch)    
    # Adjust generator loss with SSIM
    Adloss = 1 - ssimtrain
    lossG += 8e-2 * Adloss 
    
    # Backpropagate generator loss
    lossG.backward()
    gen_optim.step()
    
    return lossG.item(), lossD.item()


# def model_train(X_tr, y_tr, X_val, y_val, epochs=200, img_shape=(1,1,512, 512), lr=2e-4, batch_size=1, device='cuda'):
#     device = torch.device(device)
#     generator,discriminator=Gen(img_shape).to(device),Discriminator().to(device)
#     gen_optim = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
#     disc_optim = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
#     for epoch in range(epochs):
#         print(f"Epoch {epoch}:")
#         batches_X_train = batch_generator(X_tr, batch_size)
#         batches_y_train = batch_generator(y_tr, batch_size)
#         batches_X_test = batch_generator(X_val, batch_size)
#         batches_y_test = batch_generator(y_val, batch_size)
#         for image_batch, ref_batch in tqdm(zip(batches_X_train, batches_y_train)):
#             image_batch = image_batch.unsqueeze(0)
#             ref_batch = ref_batch.unsqueeze(0)
#             lossG, lossD = train_step(image_batch, ref_batch,generator,discriminator, gen_optim, disc_optim, device)
#             print("Generator Loss:", lossG)


def model_train(X_tr, y_tr, X_val, y_val, epochs=100, img_shape=(1,1,512, 512), lr=9e-5, batch_size=1, device='cuda'):
    device = torch.device(device)
    generator, discriminator = Gen(img_shape).to(device), Discriminator().to(device)
    gen_optim = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    disc_optim = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    vgg_layers = ['0', '5', '10', '19', '28']  
    lossG_train=[]
    lossD_train=[]
    lossG_test=[]
    lossD_test=[]
    for epoch in range(epochs):
        print(f"Epoch {epoch}:")
        batches_X_train = batch_generator(X_tr, batch_size)
        batches_y_train = batch_generator(y_tr, batch_size)
        batches_X_test = batch_generator(X_val, batch_size)
        batches_y_test = batch_generator(y_val, batch_size)
        iter=1
        l_G_train=0
        l_D_train=0
        l_G_test=0
        l_D_test=0

        train_batch_size=0
        for image_batch, ref_batch in tqdm(zip(batches_X_train, batches_y_train)):
            image_batch = image_batch.unsqueeze(0)
            ref_batch = ref_batch.unsqueeze(0)
            print(image_batch.shape)
            # if i%4!=0:
            train_batch_size+=1
            lossG,lossD = train_step(image_batch, ref_batch, generator, discriminator, gen_optim, disc_optim, device, vgg19, vgg_layers,iter,epoch)
            print("Generator Loss:", lossG, epoch)
            l_G_train+=lossG 
            l_D_train+=lossD 
            iter+=1
            # lossD = train_step_disc(image_batch, ref_batch, generator, discriminator, gen_optim, disc_optim, device, vgg19, vgg_layers)
            # print("Discriminator Loss:", lossD)
            # lossD = train_step_disc(image_batch, ref_batch, generator, discriminator, gen_optim, disc_optim, device, vgg19, vgg_layers)
            # print("Discriminator Loss:", lossD)
            # lossG = train_step_gen(image_batch, ref_batch, generator, discriminator, gen_optim, disc_optim, device, vgg19, vgg_layers)
            # print("Generator Loss:", lossG)
        
        l_D_train/=train_batch_size
        l_G_train/=train_batch_size
        print("loss G, D: ",epoch,  l_G_train,l_D_train)
        lossG_train.append(l_G_train)
        lossD_train.append(l_D_train)

        test_batch_size=0

        if epoch>=0:
        # and (epoch%5==0 or epoch==epochs-1):            
            iter_test = 0
            l_G_test=0
            lossG=1
            for image_batch_test, ref_batch_test in tqdm(zip(batches_X_test, batches_y_test)):
                image_batch_test = image_batch_test.unsqueeze(0).to(device)  
                ref_batch_test = ref_batch_test.unsqueeze(0).to(device)
                print(image_batch_test.shape)
                pred = generator(image_batch_test)
                
                gen_output=pred
                # disc_generated_output = discriminator(pred, image_batch_test)

    
                # # Calculate generator loss
                # lossG = generator_loss(disc_generated_output, pred, ref_batch_test, lambda_value=99, model=vgg19, layers=vgg_layers)
                
                # # Calculate SSIM
                # ssimtrain = ssim(pred, ref_batch_test)    
                # # Adjust generator loss with SSIM
                # Adloss = 1 - ssimtrain
                # lossG += 8e-2 * Adloss
                 
                l_G_test+=lossG
                if epoch==0:
                    save_image(image_batch_test[0,0,:,:], f'project_priyank/pred/artitest/arti_{iter_test}.png')
                    save_image(ref_batch_test[0,0,:,:], f'project_priyank/pred/gttest/GT_{iter_test}.png')
                save_image(pred[0,0,:,:], f'project_priyank/pred/outtest/pred_{iter_test}.png')           
                iter_test += 1
            l_G_test/=iter_test
            lossG_test.append(l_G_test)
    print("loss G: ", lossG_train, lossG_test)
    print("loss D: ", lossD_train)

    arr = np.arange(1, epochs+1)
    plt.plot(arr, lossG_train, label='Training Loss')
    # plt.plot(arr, lossG_test, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Gen Loss Over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig("project_priyank/graph/gen_loss.jpg")
    plt.close()
    # plt.plot(arr, train_loss_dis, label='Training Loss')

    return generator, discriminator

