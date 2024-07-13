from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from utils import *

def train_one_epoch(
        train_loader,
        model,
        cls_weights,
        optimizer,
        scheduler,
        epoch,
        step,
        logger,
        config,
        writer
):
    '''
    train model for one epoch
    '''
    model.train()
    loss_list = []
    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        img, msk, label = data
        img, msk, label = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float(), \
                          label.cuda(non_blocking=True).float()

        gt_pre, out = model(img)
        loss = Loss(gt_pre, out, msk.long(), label, cls_weights)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()
    return step

def val_one_epoch(
        val_loader,
        model,
        cls_weights,
        epoch,
        logger,
):
    model.eval()
    loss_list = []
    with torch.no_grad():
        for data in tqdm(val_loader):
            img, msk, label = data
            img, msk, label = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float(), \
                              label.cuda(non_blocking=True).float()

            gt_pre, out = model(img)
            loss = Loss(gt_pre, out, msk.long(), label, cls_weights)
            loss_list.append(loss.item())

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)

def test_one_epoch(
        test_loader,
        model,
        config,
        logger,
        standard,
):
    model.eval()
    labels = list(range(config.num_classes))
    T = 0
    F1 = 0
    F2 = 0
    with torch.no_grad():
        for name, data in enumerate(tqdm(test_loader)):
            img, msk, label = data
            img, msk, label = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float(), \
                              label.cuda(non_blocking=True).float()
            gt_pre, out = model(img)
            mask = msk.squeeze(0).cpu().detach().numpy()
            out = torch.argmax(out, dim=1)
            pred = out.cpu().detach().numpy()
            save_imgs(img, msk, out, name, config.work_dir + 'outputs/')

            y_pre = np.array(pred).reshape(-1)
            y_true = np.array(mask).reshape(-1)
            confusion = confusion_matrix(y_true, y_pre, labels=labels)
            s1 = 0
            s2 = confusion.sum() - confusion[0, 0]
            for i in range(1, config.num_classes):
                s1 += confusion[i, i]
            if s2 != 0:
                I = s1 / s2
                if I >= standard:
                    T += 1
                else:
                    if (confusion.sum() - sum(confusion[:, 0])) == 0:
                        F1 += 1
                    else:
                        F2 += 1
        Accuracy = T * 100 / (T + F1 + F2)
        FNR = F1 * 100 / (T + F1 + F2)
        FPR = F2 * 100 / (T + F1 + F2)
        log_info = f'T: {T}, F1: {F1}, F2: {F2}'
        print(log_info)
        logger.info(log_info)
        log_info = f'Accuracy: {Accuracy}, FNR: {FNR}, FPR: {FPR}'
        print(log_info)
        logger.info(log_info)
