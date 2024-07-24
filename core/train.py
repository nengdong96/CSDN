import torch
from tools import MultiItemAverageMeter

def train_stage1(base, num_image, i_ter, batch, visible_labels_list, visible_image_features_list,
                  infrared_labels_list, infrared_image_features_list):
    base.set_train()
    meter = MultiItemAverageMeter()
    iter_list = torch.randperm(num_image).to(base.device)
    for i in range(i_ter):
        b_list = iter_list[i*batch: (i+1)*batch]
        rgb_target = visible_labels_list[b_list].long()
        ir_target = infrared_labels_list[b_list].long()
        rgb_image_features = visible_image_features_list[b_list]
        ir_image_features = infrared_image_features_list[b_list]
        rgb_text_features = base.model(label1=rgb_target, get_text=True)
        ir_text_features = base.model(label2=ir_target, get_text=True)
        image_features = torch.cat([rgb_image_features, ir_image_features], dim=0)
        text_features = torch.cat([rgb_text_features, ir_text_features], dim=0)
        target = torch.cat([rgb_target, ir_target], dim=0)
        loss_i2t = base.con_creiteron(image_features, text_features, target, target)
        loss_t2i = base.con_creiteron(text_features, image_features, target, target)

        loss = loss_i2t + loss_t2i
        base.model_optimizer_stage1.zero_grad()
        loss.backward()
        base.model_optimizer_stage1.step()

        meter.update({'loss_i2t': loss_i2t.data,
                      'loss_t2i': loss_t2i.data,})

    return meter.get_val(), meter.get_str()

def train_stage2(base, num_image, i_ter, batch, labels_list, image_features_list):
    base.set_train()
    meter = MultiItemAverageMeter()
    iter_list = torch.randperm(num_image).to(base.device)
    for i in range(i_ter):
        b_list = iter_list[i*batch: (i+1)*batch]
        target = labels_list[b_list].long()
        image_features = image_features_list[b_list]
        text_features = base.model(label=target, get_fusion_text=True)
        loss_i2t = base.con_creiteron(image_features, text_features, target, target)
        loss_t2i = base.con_creiteron(text_features, image_features, target, target)

        loss = loss_i2t + loss_t2i
        base.model_optimizer_stage2.zero_grad()
        loss.backward()
        base.model_optimizer_stage2.step()

        meter.update({'loss_i2t': loss_i2t.data,
                      'loss_t2i': loss_t2i.data,})

    return meter.get_val(), meter.get_str()

def train(base, loaders, text_features, config):

    base.set_train()
    meter = MultiItemAverageMeter()
    loader = loaders.get_train_loader()
    for i, (input1_0, input1_1, input2, label1, label2) in enumerate(loader):
        rgb_imgs1, rgb_imgs2, rgb_pids = input1_0, input1_1, label1
        ir_imgs, ir_pids = input2, label2
        rgb_imgs1, rgb_imgs2, rgb_pids = rgb_imgs1.to(base.device),  rgb_imgs2.to(base.device), \
                                        rgb_pids.to(base.device).long()
        ir_imgs, ir_pids = ir_imgs.to(base.device), ir_pids.to(base.device).long()

        rgb_imgs = torch.cat([rgb_imgs1, rgb_imgs2], dim=0)
        pids = torch.cat([rgb_pids, rgb_pids, ir_pids], dim=0)

        features, cls_score = base.model(x1=rgb_imgs, x2=ir_imgs)

        n = features[1].shape[0] // 3
        rgb_attn_features = features[1].narrow(0, 0, n)
        ir_attn_features = features[1].narrow(0, 2 * n, n)
        rgb_logits = rgb_attn_features @ text_features.t()
        ir_logits = ir_attn_features @ text_features.t()

        ide_loss = base.pid_creiteron(cls_score[0], pids)
        ide_loss_proj = base.pid_creiteron(cls_score[1], pids)
        triplet_loss = base.tri_creiteron(features[0].squeeze(), pids)
        triplet_loss_proj = base.tri_creiteron(features[1].squeeze(), pids)

        rgb_i2t_ide_loss = base.pid_creiteron(rgb_logits, rgb_pids)
        ir_i2t_ide_loss = base.pid_creiteron(ir_logits, ir_pids)

        loss = ide_loss + ide_loss_proj + config.lambda1 * (triplet_loss + triplet_loss_proj) + \
               config.lambda2 * rgb_i2t_ide_loss + config.lambda3 * ir_i2t_ide_loss
        base.model_optimizer_stage3.zero_grad()
        loss.backward()
        base.model_optimizer_stage3.step()
        meter.update({'pid_loss': ide_loss.data,
                      'pid_loss_proj': ide_loss_proj.data,
                      'triplet_loss': triplet_loss.data,
                      'triplet_loss_proj': triplet_loss_proj.data,
                      'rgb_i2t_pid_loss': rgb_i2t_ide_loss.data,
                      'ir_i2t_pid_loss': ir_i2t_ide_loss.data,
                      })
    return meter.get_val(), meter.get_str()







