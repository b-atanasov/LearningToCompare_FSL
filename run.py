import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np

import task_generator as tg
import relation_network as rln


def train_episode(model, sample_dataloader, batch_dataloader, device, criterion, optimizer, lr_scheduler, episode):
    samples, sample_labels = sample_dataloader.__iter__().next()
    batches, batch_labels = batch_dataloader.__iter__().next()

    samples = samples.to(device)
    batches = batches.to(device)

    model.train()
    relations = model(samples, batches)

    one_hot_labels = torch.zeros(batch_labels.size()[0], batch_labels.unique().size()[0]).scatter_(1, batch_labels.view(-1,1), 1).to(device)
    loss = criterion(relations, one_hot_labels)

    model.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()    
    lr_scheduler.step(episode)
    return loss


def test_episode(model, sample_dataloader, test_dataloader, device):
    sample_images, sample_labels = sample_dataloader.__iter__().next()

    sample_images = sample_images.to(device)

    total_rewards = 0
    total_predictions = 0
    for test_images, test_labels in test_dataloader:
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        
        predicted_labels = model.forward(sample_images, test_images)
        predicted_labels = predicted_labels.argmax(1)

        rewards = (predicted_labels == test_labels).sum().data.item()
        total_rewards += rewards
        total_predictions += predicted_labels.size()[0]

    accuracy = total_rewards/1.0/total_predictions
    return accuracy


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


def meta_train(model, train_task_sampler, eval_task_sampler, device, criterion, optimizer, lr_scheduler, train_episodes, eval_episodes, model_state_file):
    print(f"Training on {device}...")

    last_accuracy = 0.0

    for episode in range(train_episodes):        
        sample_dataloader, batch_dataloader = train_task_sampler.sample_task_data()
        loss = train_episode(model, sample_dataloader, batch_dataloader, device, criterion, optimizer, lr_scheduler, episode)

        if (episode + 1)%100 == 0:
            print(f'episode: {episode + 1} loss {loss.data.item()}')

        if (episode + 1)%5000 == 0:
            eval_accuracy = meta_test(model, eval_task_sampler, device, eval_episodes)

            if eval_accuracy > last_accuracy:
                torch.save(model.state_dict(), model_state_file)
                print("save networks for episode:", episode)
                last_accuracy = eval_accuracy


def main():
    parser = argparse.ArgumentParser(description='Few Shot Image Recognition')
    parser.add_argument('--feature_dim', type=int, default=64)
    parser.add_argument('--relation_dim', type=int, default=8)
    parser.add_argument('--class_num', type=int, default=5)
    parser.add_argument('--sample_num_per_class', type=int, default=5)
    parser.add_argument('--batch_num_per_class', type=int, default=10)
    parser.add_argument('--test_batch_num_per_class', type=int, default=15)
    parser.add_argument('--episode', type=int, default=500000)
    parser.add_argument('--test_episode', type=int, default=600)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('--disable_cuda', action='store_true')
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--model_name', type=str, default='default')
    parser.add_argument('--train_folder', type=str)
    parser.add_argument('--test_folder', type=str)
    args = parser.parse_args()

    FEATURE_DIM = args.feature_dim
    RELATION_DIM = args.relation_dim
    CLASS_NUM = args.class_num
    SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
    BATCH_NUM_PER_CLASS = args.batch_num_per_class
    TEST_BATCH_NUM_PER_CLASS = args.test_batch_num_per_class
    EPISODE = args.episode
    TEST_EPISODE = args.test_episode
    LEARNING_RATE = args.learning_rate
    DISABLE_CUDA = args.disable_cuda
    INPUT_CHANNELS = args.input_channels
    MODEL_NAME = args.model_name
    STATE_FILE = os.path.join("models", f"{MODEL_NAME}_{CLASS_NUM}way_{SAMPLE_NUM_PER_CLASS}shot.pkl")
    TRAIN_FOLDER = args.train_folder
    TEST_FOLDER = args.test_folder

    device = torch.device("cuda") if not DISABLE_CUDA and torch.cuda.is_available() else torch.device("cpu")

    feature_encoder = rln.CNNEncoder(INPUT_CHANNELS, FEATURE_DIM)
    relation_module = rln.RelationModule(RELATION_DIM, FEATURE_DIM)

    relation_network = rln.RelationNetwork(feature_encoder, relation_module, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
    relation_network = relation_network.to(device)

    try:
        relation_network.load_state_dict(torch.load(STATE_FILE))
        print("load state success")
    except OSError:
        pass

    is_train_mode = TRAIN_FOLDER is not None and TEST_FOLDER is not None
    is_test_mode = TRAIN_FOLDER is None and TEST_FOLDER is not None

    if is_train_mode:
        mse = nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
        lr_scheduler = StepLR(optimizer, step_size=100000, gamma=0.5)

        train_task_sampler = tg.TaskSampler(TRAIN_FOLDER, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
        test_task_sampler = tg.TaskSampler(TEST_FOLDER, CLASS_NUM, SAMPLE_NUM_PER_CLASS, TEST_BATCH_NUM_PER_CLASS)

        meta_train(relation_network, train_task_sampler, test_task_sampler, device, mse, optimizer, lr_scheduler, EPISODE, TEST_EPISODE, STATE_FILE)
    elif is_test_mode:
        test_task_sampler = tg.TaskSampler(TEST_FOLDER, CLASS_NUM, SAMPLE_NUM_PER_CLASS, TEST_BATCH_NUM_PER_CLASS)
        meta_test(relation_network, test_task_sampler, device, TEST_EPISODE)


if __name__ == '__main__':
    main()
