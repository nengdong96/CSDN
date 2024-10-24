
import torchvision.transforms as transforms
from data_loader.dataset import SYSUData, RegDBData, TestData, process_query_sysu, process_gallery_sysu, \
    process_test_regdb, SYSUDataNormalSamples, Dataset, SYSUDataRGBNormalSamples, SYSUDataIRNormalSamples, \
    RegDBDataNormalSamples, RegDBDataRGBSamples, RegDBDataIRSamples
from data_loader.processing import ChannelRandomErasing, ChannelAdapGray, ChannelExchange
from data_loader.sampler import GenIdx, IdentitySampler

import torch.utils.data as data

class Loader:

    def __init__(self, config):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.transform_color1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5)])

        self.transform_color2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelExchange(gray=2)])

        self.transform_thermal = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelAdapGray(probability=0.5)])

        self.transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.img_h, config.img_w)),
            transforms.ToTensor(),
            normalize])

        self.dataset = config.dataset
        self.sysu_data_path = config.sysu_data_path
        self.regdb_data_path = config.regdb_data_path

        self.trial = config.trial

        self.img_w = config.img_w
        self.img_h = config.img_h

        self.num_pos = config.num_pos
        self.stage1_batch_size = config.stage1_batch_size
        self.batch_size = config.batch_size


        self.test_mode = config.test_mode

        self.gall_mode = config.gall_mode

        self.num_workers = config.num_workers

        self._loader()

    def _loader(self):
        if self.dataset == 'sysu':
            samples = SYSUData(self.sysu_data_path, transform1=self.transform_color1, transform2=self.transform_color2,
                               transform3=self.transform_thermal)
            self.color_pos, self.thermal_pos = GenIdx(samples.train_color_label, samples.train_thermal_label)
            self.samples = samples

            rgb_samples = SYSUDataRGBNormalSamples(self.sysu_data_path)
            ir_samples = SYSUDataIRNormalSamples(self.sysu_data_path)

            self.stage1_rgb_loader = self.get_stage1_rgb_loader(rgb_samples)
            self.stage1_ir_loader = self.get_stage1_ir_loader(ir_samples)

            normal_samples = SYSUDataNormalSamples(self.sysu_data_path, transform1=self.transform_test,
                                      transform2=self.transform_test)
            self.normal_color_pos, self.normal_thermal_pos = GenIdx(normal_samples.train_color_label,
                                                                    normal_samples.train_thermal_label)
            self.normal_samples = normal_samples

            query_samples, gallery_samples_list = self._get_test_samples(self.dataset)
            query_loader = data.DataLoader(query_samples, batch_size=128, shuffle=False, drop_last=False,
                                                num_workers=self.num_workers)
            gallery_loaders = []
            for i in range(10):
                gallery_loader = data.DataLoader(gallery_samples_list[i], batch_size=128, shuffle=False,
                                                 drop_last=False, num_workers=self.num_workers)
                gallery_loaders.append(gallery_loader)
            self.query_loader = query_loader
            self.gallery_loaders = gallery_loaders

        elif self.dataset == 'regdb':
            samples = RegDBData(self.regdb_data_path, self.trial, transform1=self.transform_color1,
                                transform2=self.transform_color2, transform3=self.transform_thermal)
            self.color_pos, self.thermal_pos = GenIdx(samples.train_color_label, samples.train_thermal_label)
            self.samples = samples

            rgb_samples = RegDBDataRGBSamples(self.regdb_data_path, self.trial)
            ir_samples = RegDBDataIRSamples(self.regdb_data_path, self.trial)

            self.stage1_rgb_loader = self.get_stage1_rgb_loader(rgb_samples)
            self.stage1_ir_loader = self.get_stage1_ir_loader(ir_samples)

            normal_samples = RegDBDataNormalSamples(self.regdb_data_path, self.trial, transform1=self.transform_test,
                                                   transform2=self.transform_test)
            self.normal_color_pos, self.normal_thermal_pos = GenIdx(normal_samples.train_color_label,
                                                                    normal_samples.train_thermal_label)
            self.normal_samples = normal_samples

            query_samples, gallery_samples = self._get_test_samples(self.dataset)
            self.query_loader = data.DataLoader(query_samples, batch_size=128, shuffle=False, drop_last=False,
                                                num_workers=self.num_workers)
            gallery_loader = data.DataLoader(gallery_samples, batch_size=128, shuffle=False, drop_last=False,
                                             num_workers=self.num_workers)
            self.gallery_loaders = gallery_loader

    def _get_test_samples(self, dataset):
        if dataset == 'sysu':
            query_img, query_label, query_cam = process_query_sysu(self.sysu_data_path, mode=self.test_mode)
            query_samples = TestData(query_img, query_label, transform=self.transform_test,
                                     img_size=(self.img_w, self.img_h))
            self.query_label = query_label
            self.query_cam = query_cam

            self.n_query = len(query_label)

            gallery_samples_list = []
            for i in range(10):
                gall_img, gall_label, gall_cam = process_gallery_sysu(self.sysu_data_path, mode=self.test_mode, trial=i,
                                                                      gall_mode=self.gall_mode)
                self.gall_cam = gall_cam
                self.gall_label = gall_label
                self.n_gallery = len(gall_label)

                gallery_samples = TestData(gall_img, gall_label, transform=self.transform_test,
                                       img_size=(self.img_w, self.img_h))
                gallery_samples_list.append(gallery_samples)
            return query_samples, gallery_samples_list
        elif self.dataset == 'regdb':
            query_img, query_label = process_test_regdb(self.regdb_data_path, trial=self.trial, modal='thermal')
            query_samples = TestData(query_img, query_label, transform=self.transform_test,
                                     img_size=(self.img_w, self.img_h))
            self.query_label = query_label

            self.n_query = len(query_label)
            gall_img, gall_label = process_test_regdb(self.regdb_data_path, trial=self.trial, modal='visible')
            gallery_samples = TestData(gall_img, gall_label, transform=self.transform_test,
                                       img_size=(self.img_w, self.img_h))
            self.gall_label = gall_label
            self.n_gallery = len(gall_label)
            return query_samples, gallery_samples

    def get_train_loader(self):
        sampler = IdentitySampler(self.samples.train_color_label, self.samples.train_thermal_label, self.color_pos,
                                  self.thermal_pos, self.num_pos, int(self.batch_size / self.num_pos))
        self.samples.cIndex = sampler.index1
        self.samples.tIndex = sampler.index2
        train_loader = data.DataLoader(self.samples, batch_size=self.batch_size,
                                       sampler=sampler, num_workers=self.num_workers, drop_last=True)
        return train_loader

    def get_train_normal_loader(self):
        normal_sampler = IdentitySampler(self.normal_samples.train_color_label, self.normal_samples.train_thermal_label,
                                         self.normal_color_pos, self.normal_thermal_pos, self.num_pos,
                                         int(self.batch_size / self.num_pos))
        self.normal_samples.cIndex = normal_sampler.index1
        self.normal_samples.tIndex = normal_sampler.index2
        normal_train_loader = data.DataLoader(self.normal_samples, batch_size=self.stage1_batch_size,
                                       sampler=normal_sampler, num_workers=self.num_workers, drop_last=True)
        return normal_train_loader

    def get_stage1_rgb_loader(self, rgb_samples):
        datset = Dataset(rgb_samples.samples, transform=self.transform_test)
        train_loader = data.DataLoader(datset, batch_size=self.stage1_batch_size, num_workers=self.num_workers,
                                       shuffle=True, drop_last=True)
        return train_loader

    def get_stage1_ir_loader(self, ir_samples):
        datset = Dataset(ir_samples.samples, transform=self.transform_test)
        train_loader = data.DataLoader(datset, batch_size=self.stage1_batch_size, num_workers=self.num_workers,
                                       shuffle=True, drop_last=True)
        return train_loader