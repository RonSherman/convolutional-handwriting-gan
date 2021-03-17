# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT
import contextlib

import torch
from torch.cuda.amp import autocast, GradScaler

from .base_model import BaseModel
from .BigGAN_networks import *
from util.util import toggle_grad, loss_hinge_dis, loss_hinge_gen, ortho, default_ortho, toggle_grad, prepare_z_y, \
    make_one_hot, to_device, multiple_replace, random_word
import pandas as pd
from .OCR_network import *
from torch.nn import CTCLoss, MSELoss, L1Loss
from torch.nn.utils import clip_grad_norm_
import random
import unicodedata
import sys
from functools import partial
activation_dict = {'inplace_relu': nn.ReLU(inplace=True),
                   'relu': nn.ReLU(inplace=False),
                   'ir': nn.ReLU(inplace=True),}

class ScrabbleGANBaseModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(G_shared=False)
        parser.set_defaults(first_layer=True)
        parser.set_defaults(one_hot=True)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        opt.G_activation = activation_dict[opt.G_nl]
        opt.D_activation = activation_dict[opt.D_nl]
        # load saved model to finetune:
        if self.isTrain and opt.saved_model!='':
            opt.G_init = os.path.join(opt.checkpoints_dir, opt.saved_model)
            opt.D_init = os.path.join(opt.checkpoints_dir, opt.saved_model)
            opt.OCR_init = os.path.join(opt.checkpoints_dir, opt.saved_model)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['G', 'D', 'Dreal', 'Dfake', 'OCR_real', 'OCR_fake', 'grad_fake_OCR', 'grad_fake_adv']
        self.loss_G = torch.zeros(1)
        self.loss_D =torch.zeros(1)
        self.loss_Dreal =torch.zeros(1)
        self.loss_Dfake =torch.zeros(1)
        self.loss_OCR_real =torch.zeros(1)
        self.loss_OCR_fake =torch.zeros(1)
        self.loss_grad_fake_OCR =torch.zeros(1)
        self.loss_grad_fake_adv =torch.zeros(1)
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        self.model_names = ['G', 'D', 'OCR']
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        # Next, build the model
        opt.n_classes = len(opt.alphabet)
        self.netG = Generator(**vars(opt))
        self.Gradloss = torch.nn.L1Loss()
        # OUR mixed-precision
        self.autocast_bit = opt.autocast_bit
        self.netconverter = strLabelConverter(opt.alphabet)
        self.netOCR = CRNN(opt).to(self.device)
        if len(opt.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            self.netOCR.to(opt.gpu_ids[0])
            self.netG.to(opt.gpu_ids[0])
            # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
            if len(opt.gpu_ids) > 1:
                self.netOCR = torch.nn.DataParallel(self.netOCR, device_ids=opt.gpu_ids, dim=1, output_device=opt.gpu_ids[0]).cuda()
                self.netG = torch.nn.DataParallel(self.netG, device_ids=opt.gpu_ids, output_device=opt.gpu_ids[0]).cuda()

        self.OCR_criterion = CTCLoss(zero_infinity=True, reduction='none')
        print(self.netG)
        #OUR
        if opt.autocast_bit:
            print("TURNED ON MIXED PRECISION TRANING HELP ME")
            self.scaler=opt.scaler

        if self.isTrain:  # only defined during training time
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.G_lr, betas=(opt.G_B1, opt.G_B2), weight_decay=0, eps=opt.adam_eps)
            self.optimizer_OCR = torch.optim.Adam(self.netOCR.parameters(),
                                                lr=opt.OCR_lr, betas=(opt.OCR_B1, opt.OCR_B2), weight_decay=0,
                                                eps=opt.adam_eps)
            self.optimizers = [self.optimizer_G, self.optimizer_OCR]

            self.optimizer_G.zero_grad()
            self.optimizer_OCR.zero_grad()

        exception_chars = ['ï', 'ü', '.', '_', 'ö', ',', 'ã', 'ñ']
        if opt.lex.endswith('.tsv'):
            self.lex = pd.read_csv(opt.lex, sep='\t')['lemme']
            self.lex = [word.split()[-1] for word in self.lex if
                        (pd.notnull(word) and all(char not in word for char in exception_chars))]
        elif opt.lex.endswith('.txt'):
            with open(opt.lex, 'rb') as f:
                self.lex = f.read().splitlines()
            lex=[]
            for word in self.lex:
                try:
                    word=word.decode("utf-8")
                except:
                    continue
                if len(word)<20:
                    lex.append(word)
            self.lex = lex
        else:
            raise ValueError('could not load lexicon ')
        self.fixed_noise_size = 2
        self.fixed_noise, self.fixed_fake_labels = prepare_z_y(self.fixed_noise_size, opt.dim_z,
                                       len(self.lex), device=self.device,
                                       fp16=opt.G_fp16, seed=opt.seed)
        self.fixed_noise.sample_()
        self.fixed_fake_labels.sample_()
        self.rep_dict = {"'":"", '"':'', ' ':'_', ';':'', '.':''}
        fixed_words_fake = [self.lex[int(i)].encode('utf-8') for i in self.fixed_fake_labels]
        self.fixed_text_encode_fake, self.fixed_text_len = self.netconverter.encode(fixed_words_fake)
        if self.opt.one_hot:
            self.one_hot_fixed = make_one_hot(self.fixed_text_encode_fake, self.fixed_text_len, self.opt.n_classes)
        # Todo change to display names of classes instead of numbers
        self.label_fix = [multiple_replace(word.decode("utf-8"), self.rep_dict) for word in fixed_words_fake]
        visual_names_fixed_noise = ['fake_fixed_' + 'label_' + label for label in self.label_fix]
        visual_names_grad_OCR = ['grad_OCR_fixed_' + 'label_' + label for label in self.label_fix]
        visual_names_grad_G = ['grad_G_fixed_' + 'label_' + label for label in self.label_fix]
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['real', 'fake']
        self.visual_names.extend(visual_names_fixed_noise)
        self.visual_names.extend(visual_names_grad_G)
        self.visual_names.extend(visual_names_grad_OCR)
        self.z, self.label_fake = prepare_z_y(opt.batch_size, opt.dim_z, len(self.lex),
                                   device=self.device, fp16=opt.G_fp16, z_dist=opt.z_dist, seed=opt.seed)
        if opt.single_writer:
            self.fixed_noise = self.z[0].repeat((self.fixed_noise_size, 1))
            self.z = self.z[0].repeat((opt.batch_size, 1)).to(self.device)
            self.z.requires_grad=True
            self.optimizer_z = torch.optim.SGD([self.z], lr=opt.G_lr)
            self.optimizer_z.zero_grad()
        self.l1_loss = L1Loss()
        self.mse_loss = MSELoss()
        self.OCRconverter = OCRLabelConverter(opt.alphabet)
        self.epsilon = 1e-7
        self.real_z = None
        self.real_z_mean = None

    def visualize_fixed_noise(self):
        if self.opt.single_writer:
            self.fixed_noise = self.z[0].repeat((self.fixed_noise_size, 1))
        if self.autocast_bit:
            scale = self.scaler.scale
            factor_scale=lambda x: [item * 1./self.scaler.get_scale() for item in x]
        else:
            scale = lambda x: x
            factor_scale =scale
        with autocast() if self.autocast_bit else contextlib.nullcontext():
            if self.opt.one_hot:
                images = self.netG(self.fixed_noise, self.one_hot_fixed.to(self.device))
            else:
                images = self.netG(self.fixed_noise, self.fixed_text_encode_fake.to(self.device))

            loss_G = loss_hinge_gen(self.netD(**{'x': images, 'z': self.fixed_noise}), self.fixed_text_len.detach(), self.opt.mask_loss)
            # self.loss_G = loss_hinge_gen(self.netD(self.fake, self.rep_label_fake))
            # OCR loss on real data
            pred_fake_OCR = self.netOCR(images)
            preds_size = torch.IntTensor([pred_fake_OCR.size(0)] * len(self.fixed_text_len)).detach()
            # loss_OCR_fake = self.OCR_criterion(pred_fake_OCR.log_softmax(2), self.fixed_text_encode_fake.detach().to(self.device),
            #                                    preds_size, self.fixed_text_len.detach())
            loss_OCR_fake = self.OCR_criterion(pred_fake_OCR.float(), self.fixed_text_encode_fake.detach().to(self.device),
                                               preds_size, self.fixed_text_len.detach())
            loss_OCR_fake = torch.mean(loss_OCR_fake[~torch.isnan(loss_OCR_fake)])


        grad_fixed_OCR = factor_scale(torch.autograd.grad(scale(loss_OCR_fake), images))
        grad_fixed_adv = factor_scale(torch.autograd.grad(scale(loss_G), images))
        _, preds = pred_fake_OCR.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        sim_preds = self.OCRconverter.decode(preds.data, preds_size.data, raw=False)
        raw_preds = self.OCRconverter.decode(preds.data, preds_size.data, raw=True)

        print('######## fake images OCR prediction ########')
        for i in range(self.fixed_noise_size):
            print('%-20s => %-20s, gt: %-20s' % (raw_preds[i], sim_preds[i], self.lex[int(self.fixed_fake_labels[i])]))
            image = images[i].unsqueeze(0).detach()
            grad_OCR = torch.abs(grad_fixed_OCR[0][i]).unsqueeze(0).detach()
            grad_OCR = (grad_OCR / torch.max(grad_OCR)) * 2 - 1
            grad_adv = torch.abs(grad_fixed_adv[0][i]).unsqueeze(0).detach()
            grad_adv = (grad_adv / torch.max(grad_adv)) * 2 - 1
            label = self.label_fix[i]
            setattr(self, 'grad_OCR_fixed_' + 'label_' + label, grad_OCR)
            setattr(self, 'grad_G_fixed_' + 'label_' + label, grad_adv)
            setattr(self, 'fake_fixed_' + 'label_' + label, image)
        #TODO- add prediction on fake of CURRENT real
        #images = self.netG(self.fixed_noise, self.fixed_text_encode_fake.to(self.device))

    def get_current_visuals(self):

        self.visualize_fixed_noise()
        with torch.no_grad():
            preds_size = torch.IntTensor([self.pred_real_OCR.size(0)] * len(self.label)).detach()
            _, preds = self.pred_real_OCR.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = self.OCRconverter.decode(preds.data, preds_size.data, raw=False)
            raw_preds = self.OCRconverter.decode(preds.data, preds_size.data, raw=True)
            print('######## real images OCR prediction ########')
            for i in range(min(3, len(self.label))):
                print('%-20s => %-20s, gt: %-20s' % (
                    raw_preds[i], sim_preds[i], self.label[i].decode('utf-8', 'strict')))
                #TODO- check wtf
                self.netOCR.train()
            ones_img = torch.ones(eval('self.fake_fixed_' + 'label_' +
                                         self.label_fix[0]).shape, dtype=torch.float32)
            #one of 3 (or batch_size) images
            #first param of real is the image of the prediction
            ones_img[:, :, :, 0:min(self.real.shape[3], ones_img.shape[3])] = self.real[0, :, :, 0:min(self.real.shape[3], ones_img.shape[3])]
            #TODO- CAHNGE TO CORRECT READ ABD FAKE
            self.real = ones_img
            #ones_img = torch.ones(eval('self.fake_fixed_' + 'label_' +
                                       #self.label_fix[0]).shape, dtype=torch.float32)
            #ones_img[:, :, :, 0:min(self.real.shape[3], ones_img.shape[3])] = self.real[1, :, :, 0:min(self.real.shape[3], ones_img.shape[3])]
            #TODO- add preditctions of OCR on fake data
            #self.real1=ones_img
            #indexes of 'words'
            print(self.label_fake)
            #true image
            print(f"labels are: {self.label}")
            print(f"fixed_labels are: {self.label_fix}")
            #fake imgages generated
            print(f"current label fake: {self.words}")
            #try to get multiple of these
            ones_img = torch.ones(eval('self.fake_fixed_' + 'label_' +
                                         self.label_fix[0]).shape, dtype=torch.float32)
            ones_img[:, :, :, 0:min(self.fake.shape[3], ones_img.shape[3])] = self.fake[0, :, :, 0:min(self.fake.shape[3], ones_img.shape[3])]
            self.fake = ones_img
            print(f"fake is :{self.fake.shape}")
            print(f"real is :{self.real.shape}")
            if self.fake.shape[0]>1:
                ones_img = torch.ones(eval('self.fake_fixed_' + 'label_' +
                                           self.label_fix[0]).shape, dtype=torch.float32)
                ones_img[:, :, :, 0:min(self.fake.shape[3], ones_img.shape[3])] = self.fake[1, :, :,
                                                                                  0:min(self.fake.shape[3],
                                                                                        ones_img.shape[3])]
            #self.fake1 = ones_img
            self.netG.train()
            return super(ScrabbleGANBaseModel, self).get_current_visuals()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        # if hasattr(self, 'real'): del self.real, self.one_hot_real, self.text_encode, self.len_text
        self.real = input['img'].to(self.device)
        if 'label' in input.keys():
            #used for ocr backward progapate
            self.label = input['label']
            self.text_encode, self.len_text = self.netconverter.encode(self.label)
            if self.opt.one_hot:
                self.one_hot_real = make_one_hot(self.text_encode, self.len_text, self.opt.n_classes).to(self.device).detach()
            self.text_encode = self.text_encode.to(self.device).detach()
            self.len_text = self.len_text.detach()
        self.img_path = input['img_path']  # get image paths
        self.idx_real = input['idx']  # get image paths

    def load_networks(self, epoch):
        BaseModel.load_networks(self, epoch)
        if self.opt.single_writer:
            load_filename = '%s_z.pkl' % (epoch)
            load_path = os.path.join(self.save_dir, load_filename)
            self.z = torch.load(load_path)

    def forward(self, words=None, z=None):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        if hasattr(self, 'fake'): del self.fake, self.text_encode_fake, self.len_text_fake, self.one_hot_fake

        self.label_fake.sample_()
        if words is None:
            words = [self.lex[int(i)] for i in self.label_fake]
            if self.opt.capitalize:
                for i, word in enumerate(words):
                    if random.random()<0.5:
                        word = list(word)
                        word[0] = unicodedata.normalize('NFKD',word[0].upper()).encode('ascii', 'ignore').decode("utf-8")
                        word = ''.join(word)
                    words[i] = word
            words = [word.encode('utf-8') for word in words]
        if z is None:
            if not self.opt.single_writer:
                self.z.sample_()
        else:
            if z.shape[0]==1:
                self.z = z.repeat((len(words), 1))
                self.z = z.repeat((len(words), 1))
            else:
                self.z = z
        self.words = words
        self.text_encode_fake, self.len_text_fake = self.netconverter.encode(self.words)
        self.text_encode_fake = self.text_encode_fake.to(self.device)
        if self.opt.one_hot:
            self.one_hot_fake = make_one_hot(self.text_encode_fake, self.len_text_fake, self.opt.n_classes).to(self.device)
            try:
                self.fake = self.netG(self.z, self.one_hot_fake)
            except:
                print(words)
        else:
            self.fake = self.netG(self.z, self.text_encode_fake)  # generate output image given the input data_A

    def backward_D_OCR(self):
        if self.autocast_bit:
            scale = self.scaler.scale
            factor_scale= lambda x: [item * 1./self.scaler.get_scale() for item in x]
        else:
            scale = lambda x: x
            factor_scale = scale
        # Real
        with autocast() if self.autocast_bit else contextlib.nullcontext():
            if self.real_z_mean is None:
                pred_real = self.netD(self.real.detach())
            else:
                pred_real = self.netD(**{'x': self.real.detach(), 'z': self.real_z_mean.detach()})
            # Fake
            try:
                pred_fake = self.netD(**{'x': self.fake.detach(), 'z': self.z.detach()})
            except:
                print('a')
            # Combined loss
            self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(), self.len_text.detach(), self.opt.mask_loss)
            self.loss_D = self.loss_Dreal + self.loss_Dfake
            # OCR loss on real data
            self.pred_real_OCR = self.netOCR(self.real.detach())
            preds_size = torch.IntTensor([self.pred_real_OCR.size(0)] * self.opt.batch_size).detach()
            loss_OCR_real = self.OCR_criterion(self.pred_real_OCR.float(), self.text_encode.detach(), preds_size, self.len_text.detach())
            self.loss_OCR_real = torch.mean(loss_OCR_real[~torch.isnan(loss_OCR_real)])
            # total loss
            loss_total = self.loss_D + self.loss_OCR_real

            # backward
        scale(loss_total).backward()
        for param in self.netOCR.parameters():
            param.grad[param.grad!=param.grad]=0
            param.grad[torch.isnan(param.grad)]=0
            param.grad[torch.isinf(param.grad)]=0
        if self.opt.clip_grad > 0:
             clip_grad_norm_(self.netD.parameters(), self.opt.clip_grad)


        return loss_total


    def backward_OCR(self):
        # OCR loss on real data
        self.pred_real_OCR = self.netOCR(self.real.detach())
        preds_size = torch.IntTensor([self.pred_real_OCR.size(0)] * self.opt.batch_size).detach()
        loss_OCR_real = self.OCR_criterion(self.pred_real_OCR, self.text_encode.detach(), preds_size, self.len_text.detach())
        self.loss_OCR_real = torch.mean(loss_OCR_real[~torch.isnan(loss_OCR_real)])

        # backward
        self.loss_OCR_real.backward()
        for param in self.netOCR.parameters():
            param.grad[param.grad!=param.grad]=0
            param.grad[torch.isnan(param.grad)]=0
            param.grad[torch.isinf(param.grad)]=0
        if self.opt.clip_grad > 0:
             clip_grad_norm_(self.netD.parameters(), self.opt.clip_grad)
        return self.loss_OCR_real


    def backward_D(self):
        # Real
        if self.real_z_mean is None:
            pred_real = self.netD(self.real.detach())
        else:
            pred_real = self.netD(**{'x': self.real.detach(), 'z': self.real_z_mean.detach()})
        pred_fake = self.netD(**{'x': self.fake.detach(), 'z': self.z.detach()})
        # Combined loss
        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(), self.len_text.detach(), self.opt.mask_loss)
        self.loss_D = self.loss_Dreal + self.loss_Dfake
        # backward
        self.loss_D.backward()

        if self.opt.clip_grad > 0:
             clip_grad_norm_(self.netD.parameters(), self.opt.clip_grad)
        return self.loss_D


    def backward_G(self):
        if self.autocast_bit:
            scale = self.scaler.scale
            factor_scale=1./self.scaler.get_scale() #lambda x: [item * 1./self.scaler.get_scale() for item in x]
        else:
            scale = lambda x: x
            factor_scale = 1#scale
        with autocast() if self.autocast_bit else contextlib.nullcontext():
            self.loss_G = loss_hinge_gen(self.netD(**{'x': self.fake, 'z': self.z}), self.len_text_fake.detach(),
                                         self.opt.mask_loss)
            pred_fake_OCR = self.netOCR(self.fake)
            preds_size = torch.IntTensor([pred_fake_OCR.size(0)] * self.opt.batch_size).detach()
            loss_OCR_fake = self.OCR_criterion(pred_fake_OCR.float(), self.text_encode_fake.detach(), preds_size,
                                               self.len_text_fake.detach())
            self.loss_OCR_fake = torch.mean(loss_OCR_fake[~torch.isnan(loss_OCR_fake)])
            # total loss
            self.loss_T = self.loss_G + self.opt.gb_alpha * self.loss_OCR_fake
        # grad creates scaled tensors, we need to unscale them back, and not owned by any optimizer,
        # so ordinary division is used instead of scaler.unscale_:
        grad_fake_OCR = factor_scale*torch.autograd.grad(scale(self.loss_OCR_fake), self.fake, retain_graph=True)[0]
        grad_fake_adv = factor_scale *torch.autograd.grad(scale(self.loss_G), self.fake, retain_graph=True)[0]
        # calculates more grad, need to do autocast
        with autocast() if self.autocast_bit else contextlib.nullcontext():
            self.loss_grad_fake_OCR = 10**6*torch.mean(grad_fake_OCR**2)

            self.loss_grad_fake_adv = 10**6*torch.mean(grad_fake_adv**2)
        #default not false==true
        if not self.opt.no_grad_balance:
            #until here autocast?

            scale(self.loss_T).backward(retain_graph=True)
            #do unscale
            grad_fake_OCR =  factor_scale*torch.autograd.grad(scale(self.loss_OCR_fake), self.fake, create_graph=True, retain_graph=True)[0]
            grad_fake_adv =  factor_scale*torch.autograd.grad(scale(self.loss_G), self.fake, create_graph=True, retain_graph=True)[0]
            #more calculateions
            with autocast() if self.autocast_bit else contextlib.nullcontext():
                a = self.opt.gb_alpha * torch.div(torch.std(grad_fake_adv), self.epsilon+torch.std(grad_fake_OCR))
                if a is None:
                    print(self.loss_OCR_fake, self.loss_G, torch.std(grad_fake_adv), torch.std(grad_fake_OCR))
                if a>1000 or a<0.0001:
                    print(a)
                b = self.opt.gb_alpha * (torch.mean(grad_fake_adv) -
                                                torch.div(torch.std(grad_fake_adv), self.epsilon+torch.std(grad_fake_OCR))*
                                                torch.mean(grad_fake_OCR))
            # self.loss_OCR_fake = a.detach() * self.loss_OCR_fake + b.detach() * torch.sum(self.fake)
                self.loss_OCR_fake = a.detach() * self.loss_OCR_fake
                self.loss_T = (1-1*self.opt.onlyOCR)*self.loss_G + self.loss_OCR_fake
            scale(self.loss_T).backward(retain_graph=True)
            grad_fake_OCR =  factor_scale*torch.autograd.grad(scale(self.loss_OCR_fake), self.fake, create_graph=False, retain_graph=True)[0]
            grad_fake_adv =  factor_scale*torch.autograd.grad(scale(self.loss_G), self.fake, create_graph=False, retain_graph=True)[0]
            with autocast() if self.autocast_bit else contextlib.nullcontext():
                self.loss_grad_fake_OCR = 10 ** 6 * torch.mean(grad_fake_OCR ** 2)
                self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)
            #TODO- removed this unnsecary
            #with torch.no_grad():
                #scale(self.loss_T).backward()
        else:
            scale(self.loss_T.backward())
        # default false
        if self.opt.clip_grad > 0:
             clip_grad_norm_(self.netG.parameters(), self.opt.clip_grad)
        if any(torch.isnan(loss_OCR_fake)) or torch.isnan(self.loss_G):
            print('loss OCR fake: ', loss_OCR_fake, ' loss_G: ', self.loss_G, ' words: ', self.words)
            sys.exit()

    def optimize_D_OCR(self):
        with autocast() if self.autocast_bit else contextlib.nullcontext():
            self.forward()
            self.set_requires_grad([self.netD], True)
            self.set_requires_grad([self.netOCR], True)
        self.optimizer_D.zero_grad()
        if self.opt.OCR_init in ['glorot', 'xavier', 'ortho', 'N02']:
            self.optimizer_OCR.zero_grad()
        self.backward_D_OCR()

    def optimize_OCR(self):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.set_requires_grad([self.netOCR], True)
        if self.opt.OCR_init in ['glorot', 'xavier', 'ortho', 'N02']:
            self.optimizer_OCR.zero_grad()
        self.backward_OCR()

    def optimize_D(self):
        self.forward()
        self.set_requires_grad([self.netD], True)
        self.backward_D()

    def optimize_D_OCR_step(self):
        if self.opt.autocast_bit:
            self.scaler.step(self.optimizer_D)
        else:
            self.optimizer_D.step()
        if self.opt.OCR_init in ['glorot', 'xavier', 'ortho', 'N02']:
            if self.opt.autocast_bit:
                self.scaler.step(self.optimizer_OCR)
            else:
                self.optimizer_OCR.step()
        self.optimizer_D.zero_grad()
        self.optimizer_OCR.zero_grad()

    def optimize_D_step(self):
        self.optimizer_D.step()
        if any(torch.isnan(self.netD.infer_img.blocks[0][0].conv1.bias)):
            print('D is nan')
            sys.exit()
        self.optimizer_D.zero_grad()

    def optimize_G(self):
        if self.autocast_bit:
            with autocast():
                self.forward()
                self.set_requires_grad([self.netD], False)
                self.set_requires_grad([self.netOCR], False)
            self.backward_G()
        else:
            self.forward()
            self.set_requires_grad([self.netD], False)
            self.set_requires_grad([self.netOCR], False)
            self.backward_G()

    def optimize_G_step(self):
        #check for mixed precision
        if self.opt.autocast_bit:
            f_opt=partial(self.scaler.step,self.optimizer_G)
        else:
            f_opt=self.optimizer_G.step
        if self.opt.single_writer and self.opt.optimize_z:
            self.optimizer_z.step()
            self.optimizer_z.zero_grad()
        if not self.opt.not_optimize_G:
            f_opt()
            self.optimizer_G.zero_grad()

    def optimize_ocr(self):
        self.set_requires_grad([self.netOCR], True)
        # OCR loss on real data
        pred_real_OCR = self.netOCR(self.real)
        preds_size =torch.IntTensor([pred_real_OCR.size(0)] * self.opt.batch_size).detach()
        self.loss_OCR_real = self.OCR_criterion(pred_real_OCR, self.text_encode.detach(), preds_size, self.len_text.detach())
        self.loss_OCR_real.backward()
        self.optimizer_OCR.step()

    def optimize_z(self):
        self.set_requires_grad([self.z], True)


    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def test(self):
        self.visual_names = ['fake']
        self.netG.eval()
        with torch.no_grad():
            self.forward()
    def get_current_fake_labels(self):
        return self.words,self.label
    """
    def train_GD(self):
        self.netG.train()
        self.netD.train()
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(self.real, self.opt.batch_size)
        y = torch.split(self.label, self.opt.batch_size)
        counter = 0

        # Optionally toggle D and G's "require_grad"
        if self.opt.toggle_grads:
            toggle_grad(self.netD, True)
            toggle_grad(self.netG, False)

        for step_index in range(self.opt.num_critic_train):
            self.optimizer_D.zero_grad()
            with torch.set_grad_enabled(False):
                self.forward()
            D_input = torch.cat([self.fake, x[counter]], 0) if x is not None else self.fake
            D_class = torch.cat([self.label_fake, y[counter]], 0) if y[counter] is not None else y[counter]
            # Get Discriminator output
            D_out = self.netD(D_input, D_class)
            if x is not None:
                pred_fake, pred_real = torch.split(D_out, [self.fake.shape[0], x[counter].shape[0]])  # D_fake, D_real
            else:
                pred_fake = D_out
            # Combined loss
            self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(), self.len_text.detach(), self.opt.mask_loss)
            self.loss_D = self.loss_Dreal + self.loss_Dfake
            self.loss_D.backward()
            counter += 1
            self.optimizer_D.step()

        # Optionally toggle D and G's "require_grad"
        if self.opt.toggle_grads:
            toggle_grad(self.netD, False)
            toggle_grad(self.netG, True)
        # Zero G's gradients by default before training G, for safety
        self.optimizer_G.zero_grad()
        self.forward()
        self.loss_G = loss_hinge_gen(self.netD(self.fake, self.label_fake), self.len_text_fake.detach(), self.opt.mask_loss)
        self.loss_G.backward()
        self.optimizer_G.step()
    """
