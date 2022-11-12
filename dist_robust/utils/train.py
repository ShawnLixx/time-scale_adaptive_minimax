import torch
from torch import nn

from config import args
from utils.tool import get_gradient_norm
from utils.optim import get_optim

real_label = 1
fake_label = 0

criterion = torch.nn.BCELoss()

def train_step(data_iter, net, optim_x, total_grad_norm_y):
  results = {}


  data, label = next(data_iter)

  # Transfer data tensor to GPU/CPU (sevice)
  data = data.to(args.device)
  label = label.to(args.device)

  # initialize z_hat with data
  z_hat = data.data.clone().to(args.device)
  # z_hat = Variable(z_hat,requires_grad=True)
  z_hat.requires_grad_()

  # running the maximizer for z_hat
  optim_y = get_optim([z_hat], x=False)
  if 'tiada' in args.optim:
      optim_y.total_sum = total_grad_norm_y
      optim_x.opponent_optim = optim_y

  loss_phi = 0 # phi(theta,z0)
  rho = 0 #E[c(Z,Z0)]
  inner_steps = 0
  # stopping criterion
  if 'neada' in args.optim:
      required_err = 1 / (args.outer_step + 1)
  while args.step < args.total_steps:
      optim_x.zero_grad()
      optim_y.zero_grad()
      delta = z_hat - data
      rho = torch.mean((torch.norm(delta.view(len(data),-1),2,1)**2)) 
      loss_zt = nn.functional.cross_entropy(net(z_hat), label)
      loss_phi = - ( loss_zt - args.gamma * rho)
      loss_phi.backward()
      optim_y.step()
      args.step += 1
      y_grad_norm = get_gradient_norm([z_hat]).item()
      if 'neada' in args.optim:
          # using both criterion
          if y_grad_norm ** 2 <= required_err:
              break
          inner_steps += 1
          if inner_steps >= args.outer_step:
              break
      else:
          inner_steps += 1
          if inner_steps >= args.n_inner:
              break
      
  # running the loss minimizer, using z_hat   
  optim_x.zero_grad()
  loss_adversarial = nn.functional.cross_entropy(net(z_hat),label)
  loss_adversarial.backward()         
  optim_x.step()

  if 'tiada' in args.optim:
      total_grad_norm_y = optim_y.total_sum
  
  args.step += 1

  with torch.no_grad():
      delta = z_hat - data
      rho = torch.mean((torch.norm(delta.view(len(data),-1),2,1)**2)) 
      total_loss = loss_adversarial -args.gamma * rho

  # record
  results['x_grad_norm'] = get_gradient_norm(net.parameters()).item()
  results['y_grad_norm'] = get_gradient_norm([z_hat]).item()
  results['classification_loss'] = loss_adversarial.item()
  results['total_loss'] = total_loss.item()
  results['y_total_grad_sum'] = total_grad_norm_y.item()

  return results
