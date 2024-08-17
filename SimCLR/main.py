import os
import random
import time
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

from args import conf
from transforms import RandomGaussianBlur
from Model import *


if __name__ == "__main__":
    # Option
    args = conf()
    print(args)

    if args.lmdb:
        from DataLoaderLMDB import Dataset_
    else:
        from DataLoader import Dataset_
    
    # Processing time
    starttime = time.time()
    today = datetime.now()
    weight_folder = "/" + today.strftime('%Y%m%d') + str(today.hour) + str(today.minute)

    # GPUs
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # to deterministic
    cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Training settings

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        RandomGaussianBlur(probability=0, radius=torch.rand(1).item()*2+2), #Random Gaussian blur with radius between 2 and 4
        transforms.ToTensor()
    ])
    
    train_portion = 1

    if args.val:
        val_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            RandomGaussianBlur(probability=0, radius=torch.rand(1).item()*2+2), #Random Gaussian blur with radius between 2 and 4
            transforms.ToTensor()
        ])
        if args.path2valdb == args.path2traindb:
            train_portion = 0.7
        val_dataset = Dataset_(args.path2valdb, transform=val_transform, train_portion=train_portion, shuffle=True ,val=True, seed=args.seed)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    train_dataset = Dataset_(args.path2traindb, transform=train_transform, train_portion=train_portion, shuffle=True ,val=False, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    #NT-Xent loss function as shown in "A Simple Framework for Contrastive Learning of Visual Representations" by Ting Chen et al.
    def nt_xent_loss(projections, temperature):
        batch_size = projections.shape[0] // 2
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(projections.device)
        
        similarity_matrix = torch.matmul(projections, projections.T)
        
        # Masking to remove similarity of identical samples
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(projections.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        logits /= temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(projections.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss

    # Model & optimizer
    model = Network(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # criterion = nn.MSELoss().to(device)
    scheduler = MultiStepLR(optimizer, milestones=[10, 18, 24], gamma=0.1)
    
    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        # Load the checkpoint
        checkpoint = torch.load(args.resume)

        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '')
            new_state_dict[new_key] = v

        # Load state_dict into the model
        model.load_state_dict(new_state_dict)
        args.start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])
        batch_losses = checkpoint.get('batch_losses', [])
        val_batch_losses = checkpoint.get('val_batch_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    if not args.no_multigpu:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    if not 'train_losses' in locals():  # Resume on existing data if available
        train_losses = []
        batch_losses = []
        val_batch_losses = []
        val_losses = []

    # Training
    num_epochs = args.epochs
    model.to(device)

    # Using mixed precision training for less memory consumption
    scaler = torch.cuda.amp.GradScaler()

    # Save checkpoint function
    def save_checkpoint(state, filename):
        torch.save(state, filename)
        print(f"Checkpoint saved to {filename}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for imgs in tqdm(iterable=train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs = torch.cat([imgs, imgs], dim=0)  # SimCLR requires two augmentations of the same batch
            imgs = imgs.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = nt_xent_loss(outputs, args.temperature)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            batch_losses.append(loss.item() * imgs.size(0))

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        if args.val:
            model.eval()
            running_loss = 0.0
            for imgs in tqdm(iterable=val_loader, desc=f"Validation of epoch {epoch+1}/{num_epochs}"):
                imgs = torch.cat([imgs, imgs], dim=0) 
                imgs = imgs.to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = nt_xent_loss(outputs, args.temperature)

                running_loss += loss.item() * imgs.size(0)
                val_batch_losses.append(loss.item() * imgs.size(0))

            epoch_loss = running_loss / len(val_loader.dataset)
            val_losses.append(epoch_loss)

        scheduler.step()

        if (epoch + 1) % args.save_interval == 0:
            os.makedirs(f"./data/weight/{args.usenet}/{args.dataset}" + weight_folder, exist_ok=True)
            checkpoint_filename = f"./data/weight/{args.usenet}/{args.dataset}{weight_folder}/checkpoint_epoch_{epoch+args.start_epoch}.pth.tar"
            save_checkpoint({
                'epoch': epoch + args.start_epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_losses': train_losses,
                'batch_losses': batch_losses,
                'val_batch_losses': val_batch_losses,
                'val_losses': val_losses,
            }, checkpoint_filename)
            print(f"Model checkpoint saved at epoch {epoch+1}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]}")
        # Processing time
        endtime = time.time()
        interval = endtime - starttime
        print("Elapsed time = {0:d}h {1:d}m {2:d}s".format(int(interval / 3600), int((interval % 3600) / 60), int((interval % 3600) % 60)))
        # Clear unused memory
        torch.cuda.empty_cache()

    print("Training completed.")
    # Processing time
    endtime = time.time()
    interval = endtime - starttime
    print("elapsed time = {0:d}h {1:d}m {2:d}s".format(int(interval / 3600), int((interval % 3600) / 60), int((interval % 3600) % 60)))
