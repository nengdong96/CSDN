import copy
import torch
import torchvision
import torch.nn as nn
from .gem_pool import GeneralizedMeanPoolingP

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('InstanceNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class Classifier(nn.Module):
    def __init__(self, pid_num):
        super(Classifier, self, ).__init__()
        self.pid_num = pid_num
        self.GEM = GeneralizedMeanPoolingP()
        self.BN = nn.BatchNorm1d(2048)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(2048, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.l2_norm = Normalize(2)

    def forward(self, features_map):
        features = self.GEM(features_map)
        #bn_features = self.BN(features.squeeze())
        bn_features = self.BN(features.squeeze().unsqueeze(0))
        cls_score = self.classifier(bn_features)
        return features, cls_score, self.l2_norm(bn_features)

class Classifier2(nn.Module):
    def __init__(self, pid_num):
        super(Classifier2, self, ).__init__()
        self.pid_num = pid_num
        self.BN = nn.BatchNorm1d(1024)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(1024, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.l2_norm = Normalize(2)

    def forward(self, features):
        #bn_features = self.BN(features.squeeze())
        bn_features = self.BN(features.squeeze().unsqueeze(0))
        cls_score = self.classifier(bn_features)
        return cls_score, self.l2_norm(features)

class PromptLearner1(nn.Module):
    def __init__(self, num_class, dtype, token_embedding):
        super().__init__()
        ctx_init = "A photo of a X X X X person."
        ctx_dim = 512
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts

class PromptLearner2(nn.Module):
    def __init__(self, num_class, dtype, token_embedding):
        super().__init__()
        ctx_init = "A photo of a X X X X person."
        ctx_dim = 512
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class AttentionFusion(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionFusion, self).__init__()
        self.dropout_rate = 0.1
        self.embed_dim = embed_dim
        self.embed_dim_qkv = embed_dim

        self.embedding_q = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim_qkv),
                                         nn.Tanh(), nn.Dropout(self.dropout_rate))
        self.embedding_k = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim_qkv),
                                         nn.Tanh(), nn.Dropout(self.dropout_rate))
        self.embedding_v = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim_qkv),
                                         nn.Tanh(), nn.Dropout(self.dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(self.embed_dim_qkv, self.embed_dim))
        self.softmax = nn.Softmax(dim=1)

    def q_k_v_product_attention(self, q_emb, k_emb, v_emb):
        weights = torch.bmm(q_emb, k_emb.permute(0, 2, 1))
        weights = torch.div(weights, (self.embed_dim_qkv ** 0.5))
        weights = self.softmax(weights)
        new_v_emb = weights.bmm(v_emb)
        return new_v_emb

    def forward(self, text_features1, text_features2):
        batch_size = text_features1.size(0)
        q_emb = self.embedding_q(text_features1.unsqueeze(1))
        k_emb = self.embedding_k(text_features2.unsqueeze(1))
        v_emb = self.embedding_v(text_features2.unsqueeze(1))
        new_v_emb = self.q_k_v_product_attention(q_emb, k_emb, v_emb)
        new_text_features = self.embedding_common(new_v_emb)
        new_text_features = new_text_features.view(batch_size, self.embed_dim) + text_features1
        return new_text_features

class Model(nn.Module):
    def __init__(self, num_classes, img_h, img_w):
        super(Model, self).__init__()
        self.in_planes = 2048
        self.num_classes = num_classes

        self.h_resolution = int((img_h - 16) // 16 + 1)
        self.w_resolution = int((img_w - 16) // 16 + 1)
        self.vision_stride_size = 16
        clip_model = load_clip_to_cpu('RN50', self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder1 = nn.Sequential(clip_model.visual.conv1, clip_model.visual.bn1, clip_model.visual.conv2,
                                            clip_model.visual.bn2, clip_model.visual.conv3, clip_model.visual.bn3,
                                            clip_model.visual.relu, clip_model.visual.avgpool)
        self.image_encoder2 = copy.deepcopy(self.image_encoder1)

        self.image_encoder = nn.Sequential(clip_model.visual.layer1, clip_model.visual.layer2, clip_model.visual.layer3,
                                           clip_model.visual.layer4)
        self.attnpool = clip_model.visual.attnpool
        self.classifier = Classifier(self.num_classes)
        self.classifier2 = Classifier2(self.num_classes)

        self.prompt_learner1 = PromptLearner1(num_classes, clip_model.dtype, clip_model.token_embedding)
        self.prompt_learner2 = PromptLearner2(num_classes, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)
        self.attention_fusion = AttentionFusion(1024)

    def forward(self, x1=None, x2=None, label1=None, label2=None, label=None, get_image=False, get_text=False,
                get_fusion_text=False):
        if get_image == True:
            if x1 is not None and x2 is None:
                image_features_map1 = self.image_encoder1(x1)
                image_features_map1 = self.image_encoder(image_features_map1)
                image_features1_proj = self.attnpool(image_features_map1)[0]
                return image_features1_proj
            elif x1 is None and x2 is not None:
                image_features_map2 = self.image_encoder2(x2)
                image_features_map2 = self.image_encoder(image_features_map2)
                image_features2_proj = self.attnpool(image_features_map2)[0]
                return image_features2_proj

        if get_text == True:
            if label1 is not None and label2 is None:
                prompts1 = self.prompt_learner1(label1)
                text_features1 = self.text_encoder(prompts1, self.prompt_learner1.tokenized_prompts)
                return text_features1
            if label2 is not None and label1 is None:
                prompts2 = self.prompt_learner2(label2)
                text_features2 = self.text_encoder(prompts2, self.prompt_learner2.tokenized_prompts)
                return text_features2

        if get_fusion_text == True:
            prompts1 = self.prompt_learner1(label)
            text_features1 = self.text_encoder(prompts1, self.prompt_learner1.tokenized_prompts)
            prompts2 = self.prompt_learner2(label)
            text_features2 = self.text_encoder(prompts2, self.prompt_learner2.tokenized_prompts)
            text_features = self.attention_fusion(text_features1, text_features2)
            return text_features

        if x1 is not None and x2 is not None:

            image_features_map1 = self.image_encoder1(x1)
            image_features_map2 = self.image_encoder2(x2)
            image_features_maps = torch.cat([image_features_map1, image_features_map2], dim=0)
            image_features_maps = self.image_encoder(image_features_maps)
            image_features_proj = self.attnpool(image_features_maps)[0]
            features, cls_scores, _ = self.classifier(image_features_maps)
            cls_scores_proj, _ = self.classifier2(image_features_proj)

            return [features, image_features_proj], [cls_scores, cls_scores_proj]


        elif x1 is not None and x2 is None:

            image_features_map1 = self.image_encoder1(x1)

            image_features_map1 = self.image_encoder(image_features_map1)

            # image_features1_proj = self.attnpool(image_features_map1)[0]

            image_features1_proj = self.attnpool(image_features_map1)

            # _, _, test_features1 = self.classifier(image_features_map1)

            _, pred, _ = self.classifier(image_features_map1)

            # _, test_features1_proj = self.classifier2(image_features1_proj)

            pred_proj, _ = self.classifier2(image_features1_proj[0])

            # return torch.cat([test_features1, test_features1_proj], dim=1)

            return image_features_map1, image_features1_proj, pred, pred_proj



        elif x1 is None and x2 is not None:

            image_features_map2 = self.image_encoder2(x2)

            image_features_map2 = self.image_encoder(image_features_map2)

            # image_features2_proj = self.attnpool(image_features_map2)[0]

            image_features2_proj = self.attnpool(image_features_map2)

            # _, _, test_features2 = self.classifier(image_features_map2)

            _, pred, _ = self.classifier(image_features_map2)

            # _, test_features2_proj = self.classifier2(image_features2_proj)

            pred_proj, _ = self.classifier2(image_features2_proj[0])

            return image_features_map2, image_features2_proj, pred, pred_proj

            # return torch.cat([test_features2, test_features2_proj], dim=1)

from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model


