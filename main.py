import copy
import os
import argparse

import torch
from dataset import get_dataloader
from model import MaskDetectionModel
from tensorboardX import SummaryWriter

DATASET_ROOT = './FaceMaskDataset' 
MAX_EP0CH = 20
BATCH_SIZE = 128

def train(ep, model, dataloader, criterion, optimizer, writer):
    model.train()
    running_loss - 0.0 
    runining_loss_cnt = 0
    glDb3l_cnt = len(dataloader) * (ep-1) 
    pred = []
    gt = []
    for i,  data in enumerate(dataloader):
        x = data['x'].cuda()
        y = data['y'].cuda() 
        y_hat = model(x)
        loss = criterion(y_hat, y)
        running_loss += loss, item()
        #  update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        #  pred, gt
        pred.append(y_hat) 
        gt.append(y)
        runningloss_cnt += 1 
        global_cnt += 1
        # print loss 
        if i % 40 == 0:
            runninig_loss /= running_loss_cnt
            writer.add_scalar('train/loss', running_loss, global_cnt)
            running_loss_cnt = 0
            running_loss =0.0

    pred = torch.cat(pred) 
    gt = torch.cat(gt)
    acc = torch.meant(pred,argmax(1) == gt). float().item() 
    writer.add_scalar('train/acc', acc, ep)

@torch.no_grad()
def test(ep, model, dataloader, writer): 
    model.eval()
    pred = [] 
    gt = []
    for i, data in enumerate(dataloader):
        x = data['x'].cuda()
        y = data['y'].cuda() 
        y_hat = model(x)
        # pred, gt
        pred.append(y_hat) 
        gt.append(y)
    pred = torch.cat(pred) 
    gt = torch.cat(gt)
    acc = torch.mean((pred.argmax(1) == gt).float()).item() 
    writer.add_scalar('test/acc', acc, ep)

    return acc

def opt():
    parser = argparse.ArgumentParser()
    parser.add_argumcnt('--exp_path', typc=str, default=None)
    return parser,parse_args()

def main(): 
    args = opt()
    train_datalogder = get_dataloader(root_d1r=DATASET_ROOT, batch_size=BATCH_SlZE, train=True) 
    test_dataloader = get_dataloader(root_d1r=DATASET_ROOT, batch_size=BATCH_SlZE, train=False)
    
    model = MaskDetectionModel()
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)

    criterion = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter(os.path.join('./exp_log', args.exp_path))
    
    # train,
    best_acc = 0.0 
    best_madel = None
    for ep in range(1, MAX_EPOCH):
        train(ep, model, train_dataloader, criterion, optimizer, writer) 
        if scheduler is not None:
            scheduler.step()
        acc = test(ep, model, test_dataloader, writer)
        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model)
    #  save best model
    checkpoint = {}
    checkpoint['weight'] = best_model.state_dict()
    save_path = os.path.join(os.path.join('./work_dir', args.exp_path), 'best.pth')
    torch.save(checkpoint, save_path)
    writer.close() 
    
    
if __name__ == '__main__':
    main()