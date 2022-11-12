from torch import optim

from config import args

# import packages from parent directory
import sys
sys.path.append('..')
from optimizer.tiada import TiAda, TiAda_Adam

ADAM_EPSILON = 1e-8

def get_optim(params, x):
  if args.optim == 'adam' or args.optim == 'neada-adam':
    if x:
      optimizer = optim.Adam(params, args.lr_x, eps=ADAM_EPSILON)
    else:
      optimizer = optim.Adam(params, args.lr_y, eps=ADAM_EPSILON)
  elif args.optim == 'adagrad' or args.optim == 'neada-adagrad':
    if x:
      optimizer = optim.Adagrad(params, args.lr_x)
    else:
      optimizer = optim.Adagrad(params, args.lr_y)
  elif args.optim == 'tiada':
    if x:
      optimizer = TiAda(params, args.lr_x, alpha=args.alpha)
    else:
      optimizer = TiAda(params, args.lr_y, alpha=args.beta)
  elif args.optim == 'tiada-adam':
    if x:
      optimizer = TiAda_Adam(params, args.lr_x, alpha=args.alpha, eps=ADAM_EPSILON)
    else:
      optimizer = TiAda_Adam(params, args.lr_y, alpha=args.beta, eps=ADAM_EPSILON)
  else:
    raise NotImplementedError

  return optimizer
