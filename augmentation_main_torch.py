from training_torch import model_train
import torch
# torch.autograd.set_detect_anomaly(True)

import numpy as np
import random
import h5py
from  matplotlib import pyplot as plt

j=32
# i=178
def normalize_data(X, axis=0):
    mu = np.mean(X, axis=axis, keepdims=True)
    sigma = np.std(X, axis=axis, keepdims=True)
    X_norm = 2 * (X - mu) / sigma
    return np.nan_to_num(X_norm)

# Load and normalize datasets
w_artifact_file = h5py.File(f'/mnt/useful/mice_sparse{j}_recon.mat', 'r')
variables_w = w_artifact_file.items()
for var in variables_w:
    name_w_artifact = var[0]
    data_w_artifact = var[1]
    if isinstance(data_w_artifact, h5py.Dataset):
        w_artifact = np.array(data_w_artifact)
        w_artifact = normalize_data(w_artifact)

wo_artifact_file = h5py.File(f'/mnt/useful/mice_full_recon.mat', 'r')
variables_wo = wo_artifact_file.items()
for var in variables_wo:
    name_wo_artifact = var[0]
    data_wo_artifact = var[1]
    if isinstance(data_wo_artifact, h5py.Dataset):
        wo_artifact = np.array(data_wo_artifact)
        wo_artifact = normalize_data(wo_artifact)


def augment_images(images):
    augmented_images = []
    for img in images:
        # Original image (0 degrees rotation)
        augmented_images.append(img)
        
        # Flip vertically
        augmented_images.append(np.flipud(img))
        
        # Flip horizontally
        augmented_images.append(np.fliplr(img))
        
        # Rotate 90 degrees clockwise
        augmented_images.append(np.rot90(img))
        
        # Rotate 180 degrees
        augmented_images.append(np.rot90(img, k=2))
        
        # Rotate 270 degrees clockwise (90 degrees counterclockwise)
        augmented_images.append(np.rot90(img, k=3))

    return np.array(augmented_images)

# import numpy as np
# import random
# from skimage import transform, util, exposure

# def augment_images(images, target_count=5000):
#     augmented_images = []
#     total_images = len(images)
#     augmentations_per_image = target_count // total_images

#     for img in images:
#         augmented_images.append(img)  # Original image
#         img=img/255

#         for _ in range(augmentations_per_image):
#             aug_img = img.copy()

#             # Randomly choose augmentations
#             if random.random() > 0.5:
#                 aug_img = np.flipud(aug_img)
#             if random.random() > 0.5:
#                 aug_img = np.fliplr(aug_img)
#             if random.random() > 0.5:
#                 aug_img = np.rot90(aug_img, k=random.choice([1, 2, 3]))

#             # # Random noise
#             # if random.random() > 0.5:
#             #     aug_img = util.random_noise(aug_img)

#             # Ensure non-negative values
#             aug_img = exposure.rescale_intensity(aug_img, out_range=(0, 1))

#             # Random gamma adjustment
#             if random.random() > 0.5:
#                 gamma = random.uniform(0.8, 1.2)
#                 aug_img = exposure.adjust_gamma(aug_img, gamma)

#             # Random crop
#             if random.random() > 0.5:
#                 start_x = random.randint(0, img.shape[1] - int(img.shape[1] * 0.9))
#                 start_y = random.randint(0, img.shape[0] - int(img.shape[0] * 0.9))
#                 aug_img = aug_img[start_y:start_y + int(img.shape[0] * 0.9), start_x:start_x + int(img.shape[1] * 0.9)]
#                 aug_img = transform.resize(aug_img, img.shape, anti_aliasing=True)

#             augmented_images.append(aug_img*255)

#     # Ensure the target count is met
#     if len(augmented_images) < target_count:
#         augmented_images.extend(augmented_images[:target_count - len(augmented_images)])

#     return np.array(augmented_images[:target_count])


def split_train_test(w_artifact, wo_artifact, train_size, test_size):
    # Randomly shuffle the indices
    indices = np.arange(len(w_artifact))
    random.seed(0)
    random.shuffle(indices)
    
    # Split indices
    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size + test_size]
    
    w_artifact_train = w_artifact[train_indices]
    w_artifact_test = w_artifact[test_indices]
    wo_artifact_train = wo_artifact[train_indices]
    wo_artifact_test = wo_artifact[test_indices]
    
    return w_artifact_train, wo_artifact_train, w_artifact_test, wo_artifact_test

print("Original dataset shapes:", w_artifact.shape, wo_artifact.shape)

# Augment the images
w_artifact = augment_images(w_artifact)
wo_artifact = augment_images(wo_artifact)

print("Augmented dataset shapes:", w_artifact.shape, wo_artifact.shape)

# Split into training and testing sets
train_size = 1315
test_size = w_artifact.shape[0]-train_size

w_artifact_train, wo_artifact_train, w_artifact_test, wo_artifact_test = split_train_test(w_artifact, wo_artifact, train_size, test_size)

print("w_artifact_train shape:", w_artifact_train.shape)
print("wo_artifact_train shape:", wo_artifact_train.shape)
print("w_artifact_test shape:", w_artifact_test.shape)
print("wo_artifact_test shape:", wo_artifact_test.shape)

generator_save_path="project_priyank/model/generator.pth"
# generator_save_path1="ki-gan/checkpoint/model_p/gen_best_1.pth"
discriminator_save_path="project_priyank/model/discriminator.pth"


Gen, Disc = model_train(w_artifact_train, wo_artifact_train, w_artifact_test, wo_artifact_test)
# Gen.save('project_priyank/model/gen.h5')
# Disc.save('project_priyank/model/disc.h5')

torch.save(Gen.state_dict(), generator_save_path)
torch.save(Disc.state_dict(), discriminator_save_path)




# # from skimage import io
# # from skimage.metrics import structural_similarity as ssim
# # # import matplotlib.pyplot as plt

# # # Load the images
# # image1 = io.imread('/mnt/project/gttest/GT_1.png',as_gray=True)
# # image2 = io.imread('/mnt/project/artitest/arti_1.png',as_gray=True)
# # # Print the shape of the images to ensure they are loaded correctly
# # print(image1.shape)

# # # Ensure the images are in the correct format (8-bit unsigned integers)
# # image1 = (image1 * 255).astype('uint8')
# # image2 = (image2 * 255).astype('uint8')

# # # Calculate SSIM using OpenCV
# # score = ssim(image1, image2)

# # # Print the SSIM score
# # print(f'SSIM: {score}')
