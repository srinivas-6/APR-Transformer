"""
Script to train/test a model for absolute ego pose regression using Transformers from camera images.
"""

# ...
from os.path import join
import argparse
import logging

from tqdm import tqdm

# ...
import json
import time

# ...
import numpy as np
import torch

import matplotlib.pyplot as plt
import wandb

# ...
from util import utils

from datasets.DeepLoc import DeepLocDataset
from datasets.RobotCar import RobotCar
from datasets.RadarRobotCar import RadarRobotCar
from datasets.APRSegDataset import APRSegDataset
from datasets.APRDataset import APRDataset

from models.pose_losses import CameraPoseLoss, cross_entropy2d
from models.pose_regressors import get_model

# Plot the predicted and target Pose Coordinates:
def visualize_pose(preds, targets):
    
    # Scatter Plot preds[(x,y)] and targets[(x,y)]:
    tp_x = [t[0][0] for t in targets]
    tp_y = [t[0][1] for t in targets]
    
    pp_x = [p[0][0] for p in preds]
    pp_y = [p[0][1] for p in preds]
    
    # ...
    plt.figure(figsize=(10,10))
    plt.scatter(tp_x, tp_y, c='r', label='targets')
    plt.scatter(pp_x, pp_y, c='b', label='preds')
    
    plt.xlabel('position-x[m]')
    plt.ylabel('position-y[m]')
    plt.legend()
    
    # ...
    wandb.log({"pose predictions plot": wandb.Image(plt)})


# ...
def parse_config():

    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument("--model_name", help="name of model to create (e.g. posenet, apr-transformer, dino-transformer")
    arg_parser.add_argument("--mode", help="train or test")
    
    arg_parser.add_argument("--config_file", help="path to configuration file", default="config/LocationRetrival_config_transposenet.json")
    arg_parser.add_argument("--checkpoint_path", help="path to a pre-trained model")
    
    arg_parser.add_argument("--experiment", help="a short string to describe the experiment")
    arg_parser.add_argument('--entity', default=None, help='Username of your weights and biases account')
    
    args = arg_parser.parse_args()
    
    return args


# ...
def denormalize_gt_and_est_poses(config, dataloader, est_pose, gt_pose):
    
    # ...
    if config.get('dataset') == 'APR_Beintelli' or config.get('dataset') == 'DeepLoc':
                        
        min_pose = dataloader.dataset.min_pose
        max_pose = dataloader.dataset.max_pose
        
        est_pose = est_pose.detach().cpu().numpy()
        est_pose[:,:3] = utils.denormalize_pose(est_pose[:,:3],min_pose,max_pose, type='minmax')
            
        gt_pose = gt_pose.detach().cpu().numpy()
        gt_pose[:,:3] = utils.denormalize_pose(gt_pose[:,:3],min_pose,max_pose, type='minmax')
        
    # ...
    else:
                        
        pose_mean = dataloader.dataset.pose_mean
        pose_std = dataloader.dataset.pose_std
                        
        est_pose = est_pose.detach().cpu().numpy()
        est_pose[:,:3] = utils.denormalize_pose(est_pose[:,:3],pose_mean,pose_std, type='meanstd')
        
        gt_pose = gt_pose.detach().cpu().numpy()
        gt_pose[:,:3] = utils.denormalize_pose(gt_pose[:,:3],pose_mean,pose_std, type='meanstd')
    
    return gt_pose, est_pose


# ... 
def setup_training_dataloader(args, config):
    
    if config.get('dataset') == 'APR_Beintelli':
        
        if config.get('modality') == 'image':
            transform = utils.train_transforms.get('baseline')
            dataset = APRDataset(config, mode=args.mode, img_transforms=transform)
                
            logging.info("[Using camera type -{}], with total samples {}".format(dataset.cam_type, len(dataset)))
        
        elif config.get('modality') == 'lidar':
            transform = None
            dataset = APRDataset(config, mode=args.mode, points_transforms=transform)
            
            logging.info("[Using modality type -{}], with total samples {}".format(dataset.modality, len(dataset)))
        
    elif config.get('dataset') == 'APR_Seg_Beintelli':
        transform = utils.train_transforms.get('baseline')
        mask_transform = utils.train_transforms.get('mask_transforms')
        dataset = APRSegDataset(config, mode=args.mode, transforms=transform, mask_transforms=mask_transform)
            
        logging.info("[Using camera type -{}], with total samples {}".format(dataset.cam_type, len(dataset)))
        
    elif config.get('dataset') == 'DeepLoc':
        transform = utils.train_transforms.get('baseline')
        dataset = DeepLocDataset(config,mode=args.mode,transforms=transform)
            
        logging.info(f'transforms: {transform}')
        
    elif config.get('dataset') == 'RobotCar':
        dataset = RobotCar(scene='loop', data_path='./data', train=True, 
                           config={'cropsize': 128, 'random_crop': True, 'subseq_length': 1, 'skip': 10},
                           transform=utils.train_transforms.get('robotcar'), target_transform=utils.train_transforms.get('target_transforms'), real=False)
        
    elif config.get('dataset') == 'RadarRobotCar':
        dataset = RadarRobotCar(config, mode=args.mode)
        
    else:
        raise NotImplementedError(f"Handling for the Dataset: '{config.get('dataset')}' is not implemented.")
        
    # ...
    loader_params = {'batch_size': config.get('batch_size'), 'shuffle': True, 'drop_last': True, 'num_workers': config.get('n_workers')}
    dataloader = torch.utils.data.DataLoader(dataset, **loader_params)
    
    return dataloader

def setup_training_model_loss_optimizer(model, config):

    # Set Model to 'Train' Mode:
    model.train()
    
    # Freeze Parts of the Model, if indicated:
    freeze_exclude_phrase = config.get("freeze_exclude_phrase")
    
    if isinstance(freeze_exclude_phrase, str):
        freeze_exclude_phrase = [freeze_exclude_phrase]
    
    freeze = config.get("freeze")
    if freeze:
        
        for name, parameter in model.named_parameters():
            freeze_param = True
            
            for phrase in freeze_exclude_phrase:
                
                if phrase in name:
                    freeze_param = False
                    break
            
            if freeze_param:
                parameter.requires_grad_(False)
    
    # Set the Loss Function: 
    pose_loss = CameraPoseLoss(config).to(device)
    
    # Set the Optimizer and Scheduler:
    params = list(model.parameters()) + list(pose_loss.parameters())
        
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                             lr = config.get('lr'),
                             eps = config.get('eps'),
                             weight_decay = config.get('weight_decay'))
        
    scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                step_size = config.get('lr_scheduler_step_size'),
                                                gamma = config.get('lr_scheduler_gamma'))
    
    # ...
    return model, pose_loss, optim, scheduler, device, freeze
    
def compute_posit_and_orient_error(config, dataloader, est_pose, gt_pose):

    # ...
    gt_pose, est_pose = denormalize_gt_and_est_poses(config, dataloader, est_pose, gt_pose)
    
    # ...
    posit_err, orient_err = utils.pose_err(torch.from_numpy(est_pose), torch.from_numpy(gt_pose))           
    return posit_err, orient_err
    
    
def train_model(model, config, args):
    
    # ...
    model, pose_loss, optim, scheduler, device, freeze = setup_training_model_loss_optimizer(model, config)
    
    # ...
    dataloader = setup_training_dataloader(args, config)  

    if "validate_model" in config.keys() and config.get("validate_model"):

        val_dataloader = setup_test_dataloader(config, mode="test")
        best_pose_error_in_m = np.inf

    # Extract Training Details from Config:
    n_freq_print = config.get("n_freq_print")
    n_freq_checkpoint = config.get("n_freq_checkpoint")
    n_epochs = config.get("n_epochs")
        
    checkpoint_prefix = join(utils.create_output_dir('out'), utils.get_stamp_from_log())
        
    # Training Loop:
    n_total_samples = 0.0
        
    loss_vals = []
    sample_count = []
    
    for epoch in range(n_epochs):
        
        # Resetting temporal Loss (used for logging):
        running_loss = 0.0
        n_samples = 0
        
        for batch_idx, minibatch in enumerate(dataloader):

            for k, v in minibatch.items():
                minibatch[k] = v.to(device)
            
            # ...
            if 'img' in minibatch.keys():
                
                img = minibatch.get('img')
            
                if img.size(1) == 1:
                    img = img.squeeze(1)
                    minibatch['img'] = img
                
            elif 'points' in minibatch.keys():
                
                points = minibatch.get('points')
            
                if points.size(1) == 1:
                    points = points.squeeze(1)
                    minibatch['points'] = points
                
            else:
                raise NotImplementedError(f"Either 'img' or 'points' values need to be provided in data entries. Currently only {list(minibatch.keys())} are contained.")
                
            # ...
            gt_pose = minibatch.get('pose').to(dtype=torch.float32)
                
            if gt_pose.size(1) == 1:
                gt_pose = gt_pose.squeeze(1)
                minibatch['pose'] = gt_pose
                
            # ...
            gt_scene = None
            
            batch_size = gt_pose.shape[0]
            n_samples += batch_size
            n_total_samples += batch_size
            
            # Freeze the Transformers Layers:
            if freeze: 
                model.eval()
                
                with torch.no_grad():
                    transformers_res = model.forward_transformers(minibatch)
                
                model.train()
            
            # Zero the Gradients:
            optim.zero_grad()
                
            # Forward Pass to estimate the Pose:
            if freeze:
                res = model.forward_heads(transformers_res)
            else:
                res = model(minibatch)
                
            # Compute Pose Loss:
            est_pose = res.get('pose')
            criterion = pose_loss(est_pose, gt_pose)
            
            # Collect for Recoding and Plotting:
            running_loss += criterion.item()
                
            loss_vals.append(criterion.item())
            sample_count.append(n_total_samples)
            
            # Back Propagation (update Model weights):
            criterion.backward()
            optim.step()
            
            wandb.log({'lr': optim.param_groups[0]["lr"]})

            # Record Loss and Performance on Train Set:
            if batch_idx % n_freq_print == 0:
                    
                # ...
                posit_err, orient_err = compute_posit_and_orient_error(config, dataloader, est_pose, gt_pose)
                   
                # ...
                logging.info("[Batch-{}/Epoch-{}] running loss: {:.3f},"
                             "pose error: {:.2f}[m], {:.2f}[deg]".format(batch_idx+1, epoch+1, (running_loss/n_samples), posit_err.mean().item(), orient_err.mean().item()))
                
                wandb.log({'train_loss':running_loss/n_samples})
                wandb.log({'pose error': posit_err.mean().item()})
                wandb.log({'orient_error': orient_err.mean().item()})

        # Eval the performance of the model after each epoch (, if set in config):
        if "validate_model" in config.keys() and config.get("validate_model"):
        
            print(f"\n--- EVAL MODEL AFTER ITERATION: {epoch} ---\n")
            median_pose_error_in_m, median_pose_error_in_deg = eval_model(model, config, dataloader = val_dataloader, mode = "test") 
            model.train() 

            # ...
            print("\n--- SAVE MODEL: ", checkpoint_prefix + '_checkpoint-{}.pth'.format(epoch), " ---")
            torch.save(model.state_dict(), checkpoint_prefix + '_checkpoint-{}.pth'.format(epoch))

        else:
            # Save Checkpoint:
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_checkpoint-{}.pth'.format(epoch))

        # Scheduler Update:
        scheduler.step()

    # ...
    logging.info('Training completed')
    torch.save(model.state_dict(), checkpoint_prefix + '_final.pth'.format(epoch))

# ... 
def setup_test_dataloader(config, mode: str = "test"):
    
    # ...
    if config.get('dataset') == 'APR_Beintelli':
            
        if config.get('modality') == 'image':
            transform = utils.test_transforms.get('baseline')
            dataset = APRDataset(config, mode=mode, transforms=transform)
                
            logging.info("[Using camera type -{}], with total samples {}".format(dataset.cam_type, len(dataset)))
        
        elif config.get('modality') == 'lidar':
            transform = None
            dataset = APRDataset(config, mode=mode, transforms=transform)
            
            logging.info("[Using modality type -{}], with total samples {}".format(dataset.modality, len(dataset)))
        
    elif config.get('dataset') == 'APR_Seg_Beintelli':
        transform = utils.test_transforms.get('baseline')
        mask_transform = None
        dataset = APRSegDataset(config, mode=mode, transforms=transform, mask_transforms=mask_transform)
            
        logging.info("[Using camera type -{}], with total samples {}".format(dataset.cam_type, len(dataset)))
        
    elif config.get('dataset') == 'DeepLoc':
        transform = utils.test_transforms.get('baseline')
        dataset = DeepLocDataset(config,mode=mode,transforms=transform)
        
    elif config.get('dataset') == 'RobotCar':
        dataset = RobotCar(scene='loop', data_path='./data', train=False, 
                           config={'cropsize': 128, 'random_crop': False, 'subseq_length': 1, 'skip': 10},
                           transform=utils.train_transforms.get('robotcar'), target_transform=utils.train_transforms.get('target_transforms'), real=False)     
        
    elif config.get('dataset') == 'RadarRobotCar':
        dataset = RadarRobotCar(config, mode=mode)
            
    else:
        raise NotImplementedError(f"Handling for the Dataset: '{config.get('dataset')}' is not implemented.")
            
    # ...
    loader_params = {'batch_size': config.get('val_batch_size'), 'shuffle': False, 'num_workers': config.get('n_workers')}
    dataloader = torch.utils.data.DataLoader(dataset, **loader_params)
    
    return dataloader
    
def eval_model(model, config, dataloader = None, mode = "test"):
    
    # Set Model to Eval Mode:
    model.eval()

    # ...
    if dataloader is None:
        dataloader = setup_test_dataloader(config, mode=mode)
    
    # ...
    stats = np.zeros((len(dataloader.dataset), 3))
    
    preds = []
    targets = []
    
    # Eval Loop:
    with torch.no_grad():
        
        for i, minibatch in tqdm(enumerate(dataloader, 0), total=len(dataloader)):

            for k, v in minibatch.items():
                minibatch[k] = v.to(device)
 
            # ...
            if 'img' in minibatch.keys():
                
                img = minibatch.get('img')
            
                if img.size(1) == 1:
                    img = img.squeeze(1)
                    minibatch['img'] = img
                
            elif 'points' in minibatch.keys():
                
                points = minibatch.get('points')
            
                if points.size(1) == 1:
                    points = points.squeeze(1)
                    minibatch['points'] = points
                
            else:
                raise NotImplementedError(f"Either 'img' or 'points' values need to be provided in data entries. Currently only {list(minibatch.keys())} are contained.")
                    
            # ...
            gt_pose = minibatch.get('pose').to(dtype=torch.float32)
            if gt_pose.size(1) == 1:
                    
                gt_pose = gt_pose.squeeze(1)
                minibatch['pose'] = gt_pose
                
            # Forward Pass to predict the Pose:  
            tic = time.time()
            
            outs = model(minibatch)
            est_pose = outs.get('pose')
            # image = minibatch.get('img')
            
            toc = time.time()
            
            # ...
            batch_size = gt_pose.shape[0]
            for idx in range(batch_size):

                gt_pose_tmp, est_pose_tmp = denormalize_gt_and_est_poses(config, dataloader, est_pose[idx].view((1, 7)), gt_pose[idx].view((1, 7)))
                    
                preds.append(est_pose_tmp[:,:2])
                targets.append(gt_pose_tmp[:,:2])
                    
                # Evaluate Error:
                posit_err, orient_err = utils.pose_err(torch.from_numpy(est_pose_tmp), torch.from_numpy(gt_pose_tmp))
                        
                # Collect Statistics:
                stats[i*batch_size+idx, 0] = posit_err.item()
                stats[i*batch_size+idx, 1] = orient_err.item()
                stats[i*batch_size+idx, 2] = (toc - tic) * 1000
            
            # ...
            # logging.info("Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(stats[i, 0],  stats[i, 1],  stats[i, 2]))
            
        # ...
        wandb.log({'Median pose error [m]': np.nanmedian(stats[:, 0])})
        wandb.log({'Median orient error [deg]': np.nanmedian(stats[:, 1])})
        wandb.log({'inference time': np.mean(stats[:, 2])})

    # Plot Targets and Predictions:
    visualize_pose(preds, targets)
    
    # Write Predictions to File for Plotting/Animation:
    for p in preds:
            
        # log Preds to a .txt File:
        with open('preds.txt', 'a') as f:
            f.write(str(p[0][0]) + ',' + str(p[0][1]) + '\n')
        
    for t in targets:
        
        # Log Targets to a .txt File:
        with open('targets.txt', 'a') as f:
            f.write(str(t[0][0]) + ',' + str(t[0][1]) + '\n')

    # Record overall Statistics:
    median_pose_error_in_m = np.nanmedian(stats[:, 0])
    median_pose_error_in_deg = np.nanmedian(stats[:, 1])

    logging.info("Performance of {}".format(args.checkpoint_path))
    logging.info("Median pose error: {:.3f}[m], {:.3f}[deg]".format(median_pose_error_in_m, np.nanmedian(stats[:, 1])))
    logging.info("Mean inference time:{:.2f}[ms]".format(np.mean(stats[:, 2])))
    logging.info("Mean pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmean(stats[:, 0]),  np.nanmean(stats[:, 1])))
    
    return median_pose_error_in_m, median_pose_error_in_deg
    
# ...
if __name__ == "__main__":

    # ...
    numpy_seed = 2
    torch_seed = 0

    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    
    # Parse Arguments & Read Config:
    args = parse_config()
    
    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)
        
    general_params = config['general']
    model_params = config[args.model_name]
    dataset_parmas = config['dataset']
    
    config = {**model_params, **general_params, **dataset_parmas}
    
    # ...
    utils.init_logger()
    
    logging.info("Start {} with {}".format(args.model_name, args.mode))
    
    if args.experiment is not None:
        logging.info("Experiment details: {}".format(args.experiment))
        
    logging.info("Running with configuration:\n{}".format('\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))
    
    # Initialize wandb:
    wandb.init(project='location_retrieval', entity=args.entity, name=args.experiment, config=config)

    # Set the Torch Device:
    if torch.cuda.is_available():
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
        
    else:
        device_id = 'cpu'
    
    device = torch.device(device_id)
    logging.info("Using Device: {}".format(device_id))
    
    # Create the Model (& load checkpoint, if specified):
    model = get_model(args.model_name, config).to(device)
    
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    # ...
    if args.mode == 'train':
        train_model(model, config, args)

    else:
        eval_model(model, config, args)
        

# Train model with APRBeIntelli dataset:
# python main.py --model_name apr-transformer-pointnet --mode train --config_file config/APRBeintelli_lidar_points_config_aprtransformer.json --experiment <experiment-name> --entity <wb-username>

# Train model with RadarRobotCar dataset:
# python main.py --model_name apr-transformer-pointnet --mode train --config_file config/RadarRobotCar_lidar_points_config_aprtransformer.json --experiment <experiment-name> --entity <wb-username>
