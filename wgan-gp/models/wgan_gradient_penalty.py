import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import numpy as np
import time as t
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
from utils.tensorboard_logger import Logger
from itertools import chain
from torchvision import utils

# import packages from parent directory
import sys
sys.path.append('..')
from optimizer.tiada import TiAda, TiAda_Adam

from utils.tool import image_grid, get_gradient_norm
from utils.inception_score import get_inception_score

SAVE_PER_TIMES = 200

def get_gradient_norm(model, norm_type=2.0):
    with torch.no_grad():
        total_norm = torch.norm(torch.stack(
            [torch.norm(
                p.grad.detach(), norm_type) \
                        for p in model.parameters()]), norm_type)
    return total_norm

class Generator(torch.nn.Module):
    def __init__(self, channels, in_dim):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=in_dim, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1))
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))


    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


class WGAN_GP(object):
    def __init__(self, args):
        print("WGAN_GradientPenalty init model.")
        self.G = Generator(args.channels, args.z_dim)
        self.D = Discriminator(args.channels)
        self.C = args.channels

        # Check if cuda is available
        self.check_cuda(args.cuda)

        # WGAN values from paper
        # self.learning_rate = 1e-4
        self.learning_rate = args.lr
        self.b1 = 0.5
        self.b2 = 0.9
        self.batch_size = args.batch_size
        self.z_dim = args.z_dim

        self.args = args

        # WGAN_gradient penalty uses ADAM
        if args.optim == 'adam':
            self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
            self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        elif args.optim == 'tiada-adam':
            self.d_optimizer = TiAda_Adam(self.D.parameters(), lr=self.learning_rate,
                    alpha=args.beta, betas=(self.b1, self.b2))
            self.g_optimizer = TiAda_Adam(self.G.parameters(), lr=self.learning_rate,
                    alpha=args.alpha, opponent_optim=self.d_optimizer, betas=(self.b1, self.b2))
        else:
            raise NotImplementedError

        # Set the logger
        dirname = f'./logs/{args.dataset}/{args.optim}_lr_{args.lr}_nc_{args.critic_iter}_bs_{args.batch_size}'
        if 'tiada' in args.optim:
            dirname += f'_a_{args.alpha}_b_{args.beta}'
        self.logger = Logger(dirname)
        self.logger.writer.flush()
        self.number_of_images = 100

        self.generator_iters = args.generator_iters
        self.critic_iter = args.critic_iter
        self.lambda_term = 1e-4

    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)

    def check_cuda(self, cuda_flag=False):
        print(cuda_flag)
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False


    def train(self, train_loader):
        self.t_begin = t.time()

        self.data = self.get_infinite_batches(train_loader)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)

        total_iter = 0
        for g_iter in range(self.generator_iters):
            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True

            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_D = 0

            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                images = self.data.__next__()
                images = self.get_torch_variable(images)

                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                # Train with real images
                d_loss_real = self.D(images)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake images
                z = self.get_torch_variable(torch.randn(images.size(0), self.z_dim, 1, 1))

                fake_images = self.G(z)
                d_loss_fake = self.D(fake_images)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # Train with gradient penalty
                gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data)
                gradient_penalty.backward()


                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = (d_loss_real - d_loss_fake).item()
                self.d_optimizer.step()
                print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')
                total_iter += 1

            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()
            # train generator
            # compute loss with fake images
            z = self.get_torch_variable(torch.randn(self.batch_size, self.z_dim, 1, 1))
            fake_images = self.G(z)
            g_loss = self.D(fake_images)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()
            print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}')
            total_iter += 1
            # Saving model and sampling images every 1000th generator iterations
            if (total_iter) % SAVE_PER_TIMES == 0:
                grad_g = get_gradient_norm(self.G).item()
                grad_d = get_gradient_norm(self.D).item()
                # self.save_model()
                # Workaround because graphic card memory can't store more than 830 examples in memory for generating image
                # Therefore doing loop and generating 800 examples and stacking into list of samples to get 8000 generated images
                # This way Inception score is more correct since there are different generated examples from every class of Inception model
                sample_list = []
                for _ in range(10):
                    # samples  = self.data.__next__()
                    z = Variable(torch.randn(800, self.z_dim, 1, 1)).cuda(self.cuda_index)
                    samples = self.G(z)
                    # samples = samples.mul(0.5).add(0.5)
                    sample_list.append(samples.data.cpu().numpy())

                # # Flattening list of list into one list
                new_sample_list = list(chain.from_iterable(sample_list))
                print("Calculating Inception Score over 8k generated images")
                # # Feeding list of numpy arrays
                inception_score = get_inception_score(new_sample_list, cuda=True, batch_size=64,
                                                      resize=True, splits=10)


                z = self.get_torch_variable(torch.randn(self.number_of_images, self.z_dim, 1, 1))

                # Testing
                time = t.time() - self.t_begin
                print("Real Inception score: {}".format(inception_score))
                print("Generator iter: {}".format(g_iter))
                print("Time {}".format(time))



                # ============ TensorBoard logging ============#
                # (1) Log the scalar values
                info = {
                    'Wasserstein distance': Wasserstein_D,
                    'Loss D': d_loss.data.cpu(),
                    'Loss G': g_cost.data.cpu(),
                    'Loss D Real': d_loss_real.data.cpu(),
                    'Loss D Fake': d_loss_fake.data.cpu(),
                    'Inception Score Mean': inception_score[0],
                    'Inception Score Std': inception_score[1],
                    'Grad Norm G': grad_g,
                    'Grad Norm D': grad_d,

                }

                step = total_iter
                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, step)

                # (3) Log the images
                info = {
                    'generated_images': self.generate_img(z)
                }

                for tag, images in info.items():
                    self.logger.image_summary(tag, images, step)



        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))


    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        z = self.get_torch_variable(torch.randn(self.batch_size, self.z_dim, 1, 1))
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')


    def calculate_gradient_penalty(self, real_images, fake_images):
        # eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1)
        eta = torch.FloatTensor(real_images.size(0),1,1,1).uniform_(0,1)
        eta = eta.expand(real_images.size(0), real_images.size(1), real_images.size(2), real_images.size(3))
        if self.cuda:
            eta = eta.cuda(self.cuda_index)
        else:
            eta = eta

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        if self.cuda:
            interpolated = interpolated.cuda(self.cuda_index)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, 32, 32)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, 32, 32)[:self.number_of_images])

    def generate_img(self, z):
        samples = self.G(z).data.cpu()
        samples = samples.mul(0.5).add(0.5)
        # generated_images = []
        # for sample in samples:
        #     if self.C == 3:
        #         generated_images.append(sample.reshape(self.C, 32, 32))
        #     else:
        #         generated_images.append(sample.reshape(32, 32))
        # return generated_images
        return image_grid(samples, 10, 10)

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        torch.save(self.G.state_dict(), './generator.pkl')
        torch.save(self.D.state_dict(), './discriminator.pkl')
        print('Models save to ./generator.pkl & ./discriminator.pkl ')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        number_int = 10
        # interpolate between twe noise(z1, z2).
        z_intp = torch.FloatTensor(1, self.z_dim, 1, 1)
        z1 = torch.randn(1, self.z_dim, 1, 1)
        z2 = torch.randn(1, self.z_dim, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            images.append(fake_im.view(self.C,32,32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int )
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images.")
