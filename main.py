"""
Script to train/test a model for absolute ego pose regression using Transformers from camera images.
"""
import argparse
import torch
import numpy as np
import json
import logging
from util import utils
import time
from datasets.DeepLoc import DeepLocDataset
from datasets.RobotCar import RobotCar
from datasets.APRSegDataset import APRSegDataset
from datasets.APRDataset import APRDataset
from models.pose_losses import CameraPoseLoss, cross_entropy2d
from models.pose_regressors import get_model
from os.path import join
import wandb


def visualize_pose(preds,targets):
    # plot the predicted and target pose coordinates
    import matplotlib.pyplot as plt
    # scatter plot preds[(x,y)] and targets[(x,y)]
    tp_x = [t[0][0] for t in targets]
    tp_y = [t[0][1] for t in targets]
    pp_x = [p[0][0] for p in preds]
    pp_y = [p[0][1] for p in preds]
    plt.figure(figsize=(10,10))
    plt.scatter(tp_x, tp_y, c='r', label='targets')
    plt.scatter(pp_x, pp_y, c='b', label='preds')
    plt.xlabel('position-x[m]')
    plt.ylabel('position-y[m]')
    plt.legend()
    wandb.log({"pose predictions plot": wandb.Image(plt)})


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_name",
                            help="name of model to create (e.g. posenet, apr-transformer, dino-transformer")
    arg_parser.add_argument("--mode", help="train or test")
    arg_parser.add_argument("--config_file", help="path to configuration file", default="config/LocationRetrival_config_transposenet.json")
    arg_parser.add_argument("--checkpoint_path",
                            help="path to a pre-trained model")
    arg_parser.add_argument("--experiment", help="a short string to describe the experiment")
    arg_parser.add_argument('--entity', default=None, help='Username of your weights and biases account')
    args = arg_parser.parse_args()
    utils.init_logger()
    # Record execution details
    logging.info("Start {} with {}".format(args.model_name, args.mode))
    if args.experiment is not None:
        logging.info("Experiment details: {}".format(args.experiment))
    # Read configuration
    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)
    general_params = config['general']
    model_params = config[args.model_name]
    dataset_parmas = config['dataset']
    config = {**model_params, **general_params, **dataset_parmas}
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))
    # Initialize wandb
    wandb.init(project='location_retrieval', entity=args.entity, name=args.experiment, config=config)

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Create the model
    model = get_model(args.model_name, config).to(device)
    # Load the checkpoint if needed
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    if args.mode == 'train':
        # Set to train mode
        model.train()

        # Freeze parts of the model if indicated
        freeze = config.get("freeze")
        freeze_exclude_phrase = config.get("freeze_exclude_phrase")
        if isinstance(freeze_exclude_phrase, str):
            freeze_exclude_phrase = [freeze_exclude_phrase]
        if freeze:
            for name, parameter in model.named_parameters():
                freeze_param = True
                for phrase in freeze_exclude_phrase:
                    if phrase in name:
                        freeze_param = False
                        break
                if freeze_param:
                        parameter.requires_grad_(False)

        # Set the loss
        pose_loss = CameraPoseLoss(config).to(device)

        # Set the optimizer and scheduler
        params = list(model.parameters()) + list(pose_loss.parameters())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        transform = utils.train_transforms.get('baseline')
        mask_transform = utils.train_transforms.get('mask_transforms')
        logging.info(f'transforms: {transform}')
        
        if config.get('dataset') == 'APR_Beintelli':
            dataset = APRDataset(config, mode=args.mode, transforms=transform)
            logging.info("[Using camera type -{}], with total samples {}".format(dataset.cam_type, len(dataset)))
        elif config.get('dataset') == 'APR_Seg_Beintelli':
            dataset = APRSegDataset(config, mode=args.mode, transforms=transform, mask_transforms=mask_transform)
            logging.info("[Using camera type -{}], with total samples {}".format(dataset.cam_type, len(dataset)))
        elif config.get('dataset') == 'DeepLoc':
            dataset = DeepLocDataset(config,mode=args.mode,transforms=transform)
        elif config.get('dataset') == 'RobotCar':
            dataset = RobotCar(scene='loop', data_path='./data', train=True, 
                       config={'cropsize': 128, 'random_crop': True, 'subseq_length': 1, 'skip': 10},
                       transform=utils.train_transforms.get('robotcar'), target_transform=utils.train_transforms.get('target_transforms'), real=False)
    
        loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")
        # Train
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        for epoch in range(n_epochs):
            # Resetting temporal loss used for logging
            running_loss = 0.0
            n_samples = 0
            for batch_idx, minibatch in enumerate(dataloader):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_pose = minibatch.get('pose').to(dtype=torch.float32)
                img = minibatch.get('img')
                if img.size(1) == 1:
                    img = img.squeeze(1)
                    minibatch['img'] = img
                if gt_pose.size(1) == 1:
                    gt_pose = gt_pose.squeeze(1)
                    minibatch['pose'] = gt_pose
                gt_scene = None
                batch_size = gt_pose.shape[0]
                n_samples += batch_size
                n_total_samples += batch_size
                if freeze: # freeze the transformer layers
                    model.eval()
                    with torch.no_grad():
                        transformers_res = model.forward_transformers(minibatch)
                    model.train()
                # Zero the gradients
                optim.zero_grad()
                # Forward pass to estimate the pose
                if freeze:
                    res = model.forward_heads(transformers_res)
                else:
                    res = model(minibatch)
                est_pose = res.get('pose')
                criterion = pose_loss(est_pose, gt_pose)
                # Collect for recoding and plotting
                running_loss += criterion.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)
                # Back prop
                criterion.backward()
                optim.step()
                wandb.log({'lr': optim.param_groups[0]["lr"]})

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    if config.get('dataset') == 'APR_Beintelli' or config.get('dataset') == 'DeepLoc':
                        min_pose = dataloader.dataset.min_pose
                        max_pose = dataloader.dataset.max_pose
                        est_pose = est_pose.detach().cpu().numpy()
                        gt_pose = gt_pose.detach().cpu().numpy()
                        est_pose[:,:3] = utils.denormalize_pose(est_pose[:,:3],min_pose,max_pose, type='minmax')
                        gt_pose[:,:3] = utils.denormalize_pose(gt_pose[:,:3],min_pose,max_pose, type='minmax')
                    else:
                        pose_mean = dataloader.dataset.pose_mean
                        pose_std = dataloader.dataset.pose_std
                        est_pose = est_pose.detach().cpu().numpy()
                        gt_pose = gt_pose.detach().cpu().numpy()
                        est_pose[:,:3] = utils.denormalize_pose(est_pose[:,:3],pose_mean,pose_std, type='meanstd')
                        gt_pose[:,:3] = utils.denormalize_pose(gt_pose[:,:3],pose_mean,pose_std, type='meanstd')
                                        
                    posit_err, orient_err = utils.pose_err(torch.from_numpy(est_pose), torch.from_numpy(gt_pose))
                    logging.info("[Batch-{}/Epoch-{}] running loss: {:.3f},"
                                 "pose error: {:.2f}[m], {:.2f}[deg]".format(
                                                                        batch_idx+1, epoch+1, (running_loss/n_samples),
                                                                        posit_err.mean().item(),
                                                                        orient_err.mean().item()))
                    wandb.log({'train_loss':running_loss/n_samples})
                    wandb.log({'pose error': posit_err.mean().item()})
                    wandb.log({'orient_error': orient_err.mean().item()})

            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')
        torch.save(model.state_dict(), checkpoint_prefix + '_final.pth'.format(epoch))

    else: # Test
        # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        transform = utils.test_transforms.get('baseline')
        mask_transform = None
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        if config.get('dataset') == 'APR_Beintelli':
            dataset = APRDataset(config, mode=args.mode, transforms=transform)
            logging.info("[Using camera type -{}], with total samples {}".format(dataset.cam_type, len(dataset)))
        elif config.get('dataset') == 'APR_Seg_Beintelli':
            dataset = APRSegDataset(config, mode=args.mode, transforms=transform, mask_transforms=mask_transform)
            logging.info("[Using camera type -{}], with total samples {}".format(dataset.cam_type, len(dataset)))
        elif config.get('dataset') == 'DeepLoc':
            dataset = DeepLocDataset(config,mode=args.mode,transforms=transform)
        elif config.get('dataset') == 'RobotCar':
           dataset = RobotCar(scene='loop', data_path='./data', train=False, 
                       config={'cropsize': 128, 'random_crop': False, 'subseq_length': 1, 'skip': 10},
                       transform=utils.train_transforms.get('robotcar'), target_transform=utils.train_transforms.get('target_transforms'), real=False)      
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)
        
        stats = np.zeros((len(dataloader.dataset), 3))
        preds = []
        targets = []
        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_pose = minibatch.get('pose').to(dtype=torch.float32)
                img = minibatch.get('img')
                if img.size(1) == 1:
                    img = img.squeeze(1)
                    minibatch['img'] = img
                if gt_pose.size(1) == 1:
                    gt_pose = gt_pose.squeeze(1)
                    minibatch['pose'] = gt_pose
                # Forward pass to predict the pose
                tic = time.time()
                outs = model(minibatch)
                est_pose = outs.get('pose')
                image = minibatch.get('img')
                toc = time.time()
                if config.get('dataset') == 'APR_Beintelli' or config.get('dataset') == 'DeepLoc':
                    min_pose = dataloader.dataset.min_pose
                    max_pose = dataloader.dataset.max_pose
                    est_pose = est_pose.detach().cpu().numpy()
                    gt_pose = gt_pose.detach().cpu().numpy()
                    est_pose[:,:3] = utils.denormalize_pose(est_pose[:,:3],min_pose,max_pose, type='minmax')
                    gt_pose[:,:3] = utils.denormalize_pose(gt_pose[:,:3],min_pose,max_pose, type='minmax')
                else:
                    pose_mean = dataloader.dataset.pose_mean
                    pose_std = dataloader.dataset.pose_std
                    est_pose = est_pose.detach().cpu().numpy()
                    gt_pose = gt_pose.detach().cpu().numpy()
                    est_pose[:,:3] = utils.denormalize_pose(est_pose[:,:3],pose_mean,pose_std, type='meanstd')
                    gt_pose[:,:3] = utils.denormalize_pose(gt_pose[:,:3],pose_mean,pose_std, type='meanstd')

                preds.append(est_pose[:,:2])
                targets.append(gt_pose[:,:2])
                # Evaluate error
                posit_err, orient_err = utils.pose_err(torch.from_numpy(est_pose), torch.from_numpy(gt_pose))
                # Collect statistics
                stats[i, 0] = posit_err.item()
                stats[i, 1] = orient_err.item()
                stats[i, 2] = (toc - tic)*1000
                logging.info("Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                    stats[i, 0],  stats[i, 1],  stats[i, 2]))
            wandb.log({'Median pose error [m]': np.nanmedian(stats[:, 0])})
            wandb.log({'Median orient error [deg]': np.nanmedian(stats[:, 1])})
            wandb.log({'inference time': np.mean(stats[:, 2])})
        # plot targets and predictions
        visualize_pose(preds,targets)
        # write predictions to file for plotting / animation
        for p in preds:
            # log preds to a txt file
            with open('preds.txt', 'a') as f:
                f.write(str(p[0][0]) + ',' + str(p[0][1]) + '\n')
        
        for t in targets:
            # log targets to a txt file
            with open('targets.txt', 'a') as f:
                f.write(str(t[0][0]) + ',' + str(t[0][1]) + '\n')

        # Record overall statistics
        logging.info("Performance of {}".format(args.checkpoint_path))
        logging.info("Median pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(stats[:, 0]), np.nanmedian(stats[:, 1])))
        logging.info("Mean inference time:{:.2f}[ms]".format(np.mean(stats[:, 2])))

        
        



