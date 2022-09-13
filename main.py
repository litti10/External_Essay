import copy
import os
import argparse

import torch
from dataset import get_dataloader
from model import MaskDetectionModel
from tensorboardX import SummaryWriter

DATASET_ROOT = './FaceMaskDataset' 
MAX_EPOCH = 20
BATCH_SIZE = 128

def train(ep, model, dataloader, criterion, optimizer, writer):
    model.train() # 상태 전화
    running_loss = 0.0 
    running_loss_cnt = 0 # cnt (concount)
    global_cnt = len(dataloader) * (ep-1) 
    pred = []
    gt = []
    for i,  data in enumerate(dataloader):
        x = data['x'].cuda() # input image pixel # torch.randn(128,3,224,224)
        y = data['y'].cuda() # answer
        y_hat = model(x)
        loss = criterion(y_hat, y)
        running_loss += loss.item()
        #  update
        optimizer.zero_grad() # reset
        loss.backward()
        optimizer.step() 
        #  pred, gt
        pred.append(y_hat) 
        gt.append(y)
        running_loss_cnt += 1 
        global_cnt += 1
        # print loss 
        if i % 40 == 0:
            running_loss /= running_loss_cnt
            writer.add_scalar('train/loss', running_loss, global_cnt) # 그래프 만들기
            running_loss_cnt = 0
            running_loss =0.0

    pred = torch.cat(pred) #cat: 하나의 텐서로 묶는 명령어
    gt = torch.cat(gt)
    acc = torch.mean(pred.argmax(1) == gt).float().item() # pred.argmax--> 값 중 max값 찾기 / (1): row 단위 ((0): column 단위)
    writer.add_scalar('train/acc', acc, ep)

@torch.no_grad() # gradient 계산하는데 필요한 computational graph를 그리지 않겠다
def test(ep, model, dataloader, writer): 
    model.eval() #evaluation mode로 변환
    pred = [] 
    gt = []
    for i, data in enumerate(dataloader):
        x = data['x'].cuda() # dictionary에서 'x'를 key로 가지고 있는 값을 가져옴
        y = data['y'].cuda() # dictionary에서 'y'를 key로 가지고 있는 값을 가져옴
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
    return parser.parse_args()

def main(): 
    args = opt()
    train_dataloader = get_dataloader(root_dir=DATASET_ROOT, batch_size=BATCH_SIZE, train=True) 
    test_dataloader = get_dataloader(root_dir=DATASET_ROOT, batch_size=BATCH_SIZE, train=False)
    
    model = MaskDetectionModel()
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)

    criterion = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter(os.path.join('./exp_log', args.exp_path))
    
    # train,
    best_acc = 0.0 
    best_model = None
    for ep in range(1, MAX_EPOCH+1):
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