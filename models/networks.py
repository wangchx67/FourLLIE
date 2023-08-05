import torch
import models.archs.FourLLIE as FourLLIE

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'FourLLIE':
        netG = FourLLIE.FourLLIE(nf=opt_net['nf'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG

