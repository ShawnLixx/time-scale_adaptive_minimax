import numpy as np
import torch

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

#     imgs = np.array(imgs)
#     imgs = [Image.fromarray(img) for img in imgs]

    # switch channels
    # pytorch imgs : [b, c, w, h]
    imgs = imgs.permute(0, 2, 3, 1)

    w, h, c = imgs[0].shape
    grid = np.zeros(shape=(cols * w, rows * h, c))
    
    for i, img in enumerate(imgs):
        grid[i % cols * w : i % cols * w + w, 
          i // cols * h : i // cols * h + h] = img
        # grid.paste(img, box=(i%cols*w, i//cols*h))

    return grid[np.newaxis, :, :, :]

def get_gradient_norm(model, norm_type=2.0):
    with torch.no_grad():
        total_norm = torch.norm(torch.stack(
            [torch.norm(
                p.grad.detach(), norm_type) \
                        for p in model.parameters()]), norm_type)
    return total_norm
