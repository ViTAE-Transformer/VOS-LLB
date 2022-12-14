import torch.optim as optim
from ltr.dataset import YouTubeVOS, Davis
from ltr.data import processing, sampler, LTRLoader
import ltr.models.joint.joint_net as joint_networks
import ltr.actors.segmentation as segm_actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
from ltr.models.loss.segmentation import LovaszSegLoss, DiffLoss
import ltr.admin.loading as network_loading
import os


def run(settings):
    settings.description = 'Default train settings for training full network'
    settings.batch_size = 32
    settings.num_workers = 8
    settings.multi_gpu = True
    settings.print_interval = 1
    settings.normalize_mean = [102.9801, 115.9465, 122.7717]
    settings.normalize_std = [1.0, 1.0, 1.0]

    settings.feature_sz = (52, 30)

    # Settings used for generating the image crop input to the network. See documentation of SegProcessing class in
    # ltr/data/processing.py for details.
    settings.output_sz = (settings.feature_sz[0] * 16, settings.feature_sz[1] * 16)     # Size of input image crop
    settings.search_area_factor = 5.0
    settings.crop_type = 'inside_major'
    settings.max_scale_change = None

    settings.center_jitter_factor = {'train': 3, 'test': (5.5, 4.5)}
    settings.scale_jitter_factor = {'train': 0.25, 'test': 0.5}

    # Datasets
    ytvos_train = YouTubeVOS(version="2018", multiobj=False, split='train')
    davis_train = Davis(version='2017', multiobj=False, split='train')

    ytvos_val = YouTubeVOS(version="2018", multiobj=False, split='jjvalid')

    # Data transform
    transform_joint = tfm.Transform(tfm.ToBGR(),
                                    tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.RandomAffine(p_flip=0.0, max_rotation=15.0,
                                                     max_shear=0.0, max_ar_factor=0.0,
                                                     max_scale=0.2, pad_amount=0),
                                    tfm.ToTensorAndJitter(0.2, normalize=False),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    transform_val = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=False),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    data_processing_train = processing.SegProcessing(search_area_factor=settings.search_area_factor,
                                                     output_sz=settings.output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor,
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     mode='sequence',
                                                     crop_type=settings.crop_type,
                                                     max_scale_change=settings.max_scale_change,
                                                     transform=transform_train,
                                                     joint_transform=transform_joint,
                                                     new_roll=True)

    data_processing_val = processing.SegProcessing(search_area_factor=settings.search_area_factor,
                                                   output_sz=settings.output_sz,
                                                   center_jitter_factor=settings.center_jitter_factor,
                                                   scale_jitter_factor=settings.scale_jitter_factor,
                                                   mode='sequence',
                                                   crop_type=settings.crop_type,
                                                   max_scale_change=settings.max_scale_change,
                                                   transform=transform_val,
                                                   joint_transform=transform_joint,
                                                   new_roll=True)

    # Train sampler and loader
    dataset_train = sampler.SegSampler([ytvos_train, davis_train], [6, 1],
                                       samples_per_epoch=settings.batch_size * 1000, max_gap=100, num_test_frames=3,
                                       num_train_frames=1,
                                       processing=data_processing_train)
    dataset_val = sampler.SegSampler([ytvos_val], [1],
                                     samples_per_epoch=settings.batch_size * 100, max_gap=100,
                                     num_test_frames=3,
                                     num_train_frames=1,
                                     processing=data_processing_val)

    loader_train = LTRLoader('train', dataset_train, training=True, num_workers=settings.num_workers,
                             stack_dim=1, batch_size=settings.batch_size)

    loader_val = LTRLoader('val', dataset_val, training=False, num_workers=settings.num_workers,
                           epoch_interval=2, stack_dim=1, batch_size=settings.batch_size)

    # Network
    net = joint_networks.steepest_descent_resnet50(filter_size=3, num_filters=32, optim_iter=5,
                                                   backbone_pretrained=True,
                                                   out_feature_dim=512,
                                                   frozen_stages=-1,
                                                   label_encoder_dims=(16, 32, 64),
                                                   use_bn_in_label_enc=False,
                                                   clf_feat_blocks=0,
                                                   final_conv=True,
                                                   backbone_type='mrcnn')

    base_net = network_loading.load_trained_network(settings.env.workspace_dir,
                                                    'ltr/joint/joint_stage1/JOINTNet_ep0050.pth.tar')

    net.load_state_dict(base_net.state_dict())

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)

    # Loss function
    objective = {
        'segm': LovaszSegLoss(per_image=False),
        'diff': DiffLoss()
    }

    loss_weight = {
        'segm': 100.0,
        'diff': 1.0
    }

    actor = segm_actors.SegActor(net=net, objective=objective, loss_weight=loss_weight,
                                 num_refinement_iter=2, disable_all_bn=True)

    # Optimizer
    optimizer = optim.Adam([{'params': actor.net.target_model.filter_initializer.parameters(), 'lr': 5e-5},
                            {'params': actor.net.target_model.filter_optimizer.parameters(), 'lr': 2e-5},
                            {'params': actor.net.target_model.feature_extractor.parameters(), 'lr': 2e-5},
                            {'params': actor.net.decoder.parameters(), 'lr': 2e-5},
                            {'params': actor.net.label_encoder.parameters(), 'lr': 2e-5},
                            {'params': actor.net.transformer.parameters(), 'lr': 2e-5},
                            {'params': actor.net.combiner.parameters(), 'lr': 2e-5},
                            {'params': actor.net.feature_extractor.layer2.parameters(), 'lr': 2e-5},
                            {'params': actor.net.feature_extractor.layer3.parameters(), 'lr': 2e-5},
                            {'params': actor.net.feature_extractor.layer4.parameters(), 'lr': 2e-5}],
                           lr=2e-4)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 75], gamma=0.2)

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(90, load_latest=True, fail_safe=False)
