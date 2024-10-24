import os
import torch
import torch.nn as nn

from bisect import bisect_right
from network import Model
from network.lr import CosineLRScheduler
from tools import os_walk, CrossEntropyLabelSmooth, SupConLoss, TripletLoss_WRT, MSEL, MSEL_Feat, MSEL_Cos

def create_scheduler(optimizer, num_epochs, lr_min, warmup_lr_init, warmup_t, noise_range = None):

    lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=lr_min,
            t_mul= 1.,
            decay_rate=0.1,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_t,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct= 0.67,
            noise_std= 1.,
            noise_seed=42,
        )

    return lr_scheduler

class Base:
    def __init__(self, config):
        self.config = config

        self.pid_num = config.pid_num

        self.max_save_model_num = config.max_save_model_num
        self.output_path = config.output_path
        self.save_model_path = os.path.join(self.output_path, 'models/')
        self.save_logs_path = os.path.join(self.output_path, 'logs/')

        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.milestones = config.milestones

        self.img_h = config.img_h
        self.img_w = config.img_w

        self.stage1_learning_rate = config.stage1_learning_rate
        self.stage2_learning_rate = config.stage2_learning_rate
        self.stage1_weight_decay = config.stage1_weight_decay
        self.stage1_train_epochs = config.stage1_train_epochs
        self.stage1_lr_min = config.stage1_lr_min
        self.stage1_warmup_lr_init = config.stage1_warmup_lr_init
        self.stage1_warmup_epochs = config.stage1_warmup_epochs

        self._init_device()
        self._init_model()
        self._init_creiteron()

    def _init_device(self):
        self.device = torch.device('cuda')

    def _init_model(self):

        self.model = Model(self.pid_num, self.img_h, self.img_w)
        self.model = nn.DataParallel(self.model).to(self.device)

    def _init_creiteron(self):
        self.con_creiteron = SupConLoss(self.device)
        self.pid_creiteron = nn.CrossEntropyLoss()
        self.soft_pid_creiteron = CrossEntropyLabelSmooth()
        self.tri_creiteron = TripletLoss_WRT()
        self.msel_creiteron = MSEL(4)
        self.mselcos_creiteron = MSEL_Cos(4)
        self.mselfeat_creiteron = MSEL_Feat(4)

    def _init_optimizer_stage1(self):
        params = []
        keys = []
        for key, value in self.model.named_parameters():
            if 'prompt_learner1' in key:
                lr = self.stage1_learning_rate
                weight_decay = self.stage1_weight_decay
                params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]
                keys += [[key]]
            if 'prompt_learner2' in key:
                lr = self.stage1_learning_rate
                weight_decay = self.stage1_weight_decay
                params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]
                keys += [[key]]

        self.model_optimizer_stage1 = getattr(torch.optim, 'Adam')(params)
        self.model_lr_scheduler_stage1 = create_scheduler(self.model_optimizer_stage1,
                                                 num_epochs=self.stage1_train_epochs, lr_min=self.stage1_lr_min,
                                                 warmup_lr_init=self.stage1_warmup_lr_init,
                                                 warmup_t=self.stage1_warmup_epochs, noise_range=None)

    def _init_optimizer_stage2(self):
        params = []
        keys = []
        for key, value in self.model.named_parameters():
            if 'attention_fusion' in key:
                lr = self.stage2_learning_rate
                weight_decay = self.stage1_weight_decay
                params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]
                keys += [[key]]

        self.model_optimizer_stage2 = getattr(torch.optim, 'Adam')(params)
        self.model_lr_scheduler_stage2 = create_scheduler(self.model_optimizer_stage2,
                                                 num_epochs=self.stage1_train_epochs, lr_min=self.stage1_lr_min,
                                                 warmup_lr_init=self.stage1_warmup_lr_init,
                                                 warmup_t=self.stage1_warmup_epochs, noise_range=None)

    def _init_optimizer_stage3(self):
        params = []
        keys = []
        for key, value in self.model.named_parameters():
            if 'prompt_learner1' in key:
                value.requires_grad_(False)
                continue
            if 'prompt_learner2' in key:
                value.requires_grad_(False)
                continue
            if 'attention_fusion' in key:
                value.requires_grad_(False)
                continue
            if 'text_encoder' in key:
                value.requires_grad_(False)
                continue
            lr = self.learning_rate
            if 'classifier' in key:
                lr = self.learning_rate * 2
            params += [{'params': [value], 'lr': lr, 'weight_decay': self.weight_decay}]
            keys += [[key]]

        self.model_optimizer_stage3 = getattr(torch.optim, 'Adam')(params)
        self.model_lr_scheduler_stage3 = WarmupMultiStepLR(self.model_optimizer_stage3, self.milestones,
                                             gamma=0.1, warmup_factor=0.01, warmup_iters=10)

    def save_model(self, save_epoch, is_best):
        if is_best:
            model_file_path = os.path.join(self.save_model_path, 'model_{}.pth'.format(save_epoch))
            torch.save(self.model.state_dict(), model_file_path)

        if self.max_save_model_num > 0:
            root, _, files = os_walk(self.save_model_path)
            for file in files:
                if '.pth' not in file:
                    files.remove(file)
            if len(files) > 1 * self.max_save_model_num:
                file_iters = sorted([int(file.replace('.pth', '').split('_')[1]) for file in files], reverse=False)

                model_file_path = os.path.join(root, 'model_{}.pth'.format(file_iters[0]))
                os.remove(model_file_path)

    def resume_last_model(self):
        root, _, files = os_walk(self.save_model_path)
        for file in files:
            if '.pth' not in file:
                files.remove(file)
        if len(files) > 0:
            indexes = []
            for file in files:
                indexes.append(int(file.replace('.pth', '').split('_')[-1]))
            indexes = sorted(list(set(indexes)), reverse=False)
            self.resume_model(indexes[-1])
            start_train_epoch = indexes[-1]
            return start_train_epoch
        else:
            return 0

    def resume_model(self, resume_epoch):
        model_path = os.path.join(self.save_model_path, 'model_{}.pth'.format(resume_epoch))
        self.model.load_state_dict(torch.load(model_path), strict=False)
        print('Successfully resume model from {}'.format(model_path))

    def set_train(self):
        self.model = self.model.train()

        self.training = True

    def set_eval(self):
        self.model = self.model.eval()

        self.training = False

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500,
                 warmup_method='linear', last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of " " increasing integers. Got {}", milestones)

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup method accepted got {}".format(warmup_method))
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha

        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
