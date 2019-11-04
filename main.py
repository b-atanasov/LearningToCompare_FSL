import argparse
from functools import partial
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np

import task_generator as tg
import relation_network as rln
from utils import Checkpoint, Accuracy


def train_episode(model, sample_dataloader, batch_dataloader, device, criterion, optimizer,
                  lr_scheduler, episode, args):
    samples, sample_labels = sample_dataloader.__iter__().next()
    batches, batch_labels = batch_dataloader.__iter__().next()

    samples = samples.to(device)
    batches = batches.to(device)
    batch_labels = batch_labels.to(device)

    model.train()
    relations = model(samples, batches)

    if args.loss_type == 'mse':
        one_hot_labels = (torch.zeros(batch_labels.size()[0], batch_labels.unique().size()[0])
                          .to(device)
                          .scatter_(1, batch_labels.view(-1, 1), 1))
        loss = criterion(relations, one_hot_labels)
    else:
        loss = criterion(relations, batch_labels)

    model.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    lr_scheduler.step(episode)

    accuracy = Accuracy()
    accuracy.add_batch(relations, batch_labels)
    return loss, accuracy.calculate()


def test_episode(model, sample_dataloader, test_dataloader, device):
    sample_images, sample_labels = sample_dataloader.__iter__().next()

    sample_images = sample_images.to(device)

    total_rewards = 0
    total_predictions = 0
    accuracy = Accuracy()
    for test_images, test_labels in test_dataloader:
        test_images, test_labels = test_images.to(device), test_labels.to(device)

        predicted_labels = model(sample_images, test_images)
        accuracy.add_batch(predicted_labels, test_labels)

    return accuracy.calculate()


def meta_test(model, task_sampler, device, episodes):
    print("Testing...")
    accuracies = []
    model.eval()
    with torch.no_grad():
        for i in range(episodes):
            sample_dataloader, test_dataloader = task_sampler.sample_task_data()
            accuracy = test_episode(model, sample_dataloader, test_dataloader, device)
            accuracies.append(accuracy)

    test_accuracy = np.array(accuracies).mean()
    print("test accuracy:", test_accuracy)
    return test_accuracy


def meta_train(model, train_task_sampler, eval_task_sampler, device, criterion, optimizer,
               lr_scheduler, args, checkpoint):
    print(f"Training on {device}...")

    if args.resume:
        best_accuracy = checkpoint.get_saved_accuracy()
    else:
        best_accuracy = 0.0

    for episode in range(args.start_episode, args.train_episodes):
        sample_dataloader, batch_dataloader = train_task_sampler.sample_task_data()
        loss, accuracy = train_episode(model, sample_dataloader, batch_dataloader, device,
                                       criterion, optimizer, lr_scheduler, episode, args)

        if (episode % 100 == 0) and (episode > 0):
            checkpoint.save(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                            episode=episode)
            print(f'episode {episode}: loss {loss.data.item()}, accuracy {accuracy}')

        if (episode % 5000 == 0) and (episode > 0):
            eval_accuracy = meta_test(model, eval_task_sampler, device, args.test_episodes)

            if eval_accuracy > best_accuracy:
                checkpoint.save_best(model=model, accuracy=eval_accuracy, episode=episode)
                best_accuracy = eval_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Few Shot Image Recognition')
    parser.add_argument('--backbone', type=str, choices=['Conv4', 'ResNet18'], default='ResNet18')
    parser.add_argument('--img_size', type=int, choices=[84, 224], default=84,
                        help='input images will be cropped to this size')
    parser.add_argument('--loss_type', type=str, choices=['mse', 'cross-entropy'], default='mse')
    parser.add_argument('--class_num', type=int, default=5)
    parser.add_argument('--sample_num_per_class', type=int, default=5)
    parser.add_argument('--batch_num_per_class', type=int, default=10)
    parser.add_argument('--test_batch_num_per_class', type=int, default=15)
    parser.add_argument('--train_episodes', type=int, default=500000)
    parser.add_argument('--test_episodes', type=int, default=600)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('--disable_cuda', action='store_true')
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--model_name', type=str, default='default')
    parser.add_argument('--train_folder', type=str)
    parser.add_argument('--test_folder', type=str)
    parser.add_argument('--resume', action='store_true', help='resume training from checkpoint')
    parser.add_argument('--start_episode', type=int, default=0)
    parser.add_argument('--checkpoints_folder', type=str, default='checkpoints')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    args.best_model_file = os.path.join(args.checkpoints_folder, f'best_{args.model_name}_{args.class_num}way_{args.sample_num_per_class}shot.pt')
    return args


def main(args):
    device = torch.device("cuda") if not args.disable_cuda and torch.cuda.is_available() else torch.device("cpu")

    relation_network = rln.RelationNetwork(args)
    relation_network = relation_network.to(device)

    checkpoint = Checkpoint(args)

    if not args.test:
        optimizer = torch.optim.Adam(relation_network.parameters(), lr=args.learning_rate)
        lr_scheduler = StepLR(optimizer, step_size=100000, gamma=0.5)

        if args.resume:
            checkpoint.load(model=relation_network, optimizer=optimizer, lr_scheduler=lr_scheduler)

        if args.loss_type == 'mse':
            criterion = nn.MSELoss().to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)

        train_task_sampler = tg.TaskSampler(args, train=True)
        test_task_sampler = tg.TaskSampler(args, train=False)

        meta_train(relation_network, train_task_sampler, test_task_sampler, device, criterion, optimizer, lr_scheduler, args, checkpoint)
    else:
        checkpoint.load_best(model=relation_network)
        test_task_sampler = tg.TaskSampler(args, train=False)
        meta_test(relation_network, test_task_sampler, device, args.test_episodes)


if __name__ == '__main__':
    args = parse_args()
    main(args)
