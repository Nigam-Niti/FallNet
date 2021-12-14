def train(config, model, loader, optimizer, epoch, checkpoint_path, best_model_path):
    print('#'*60)
    print('Epoch {}. Starting with training phase.'.format(epoch))
    model.train()
    config = {}
    config['log_interval'] = 100
    correct = 0
    total_loss = 0.0
    flag = 0
    Loss, Acc = AverageMeter(), AverageMeter()
    start = time.time()
    for batch_id, data in enumerate(loader):
        data, target = data[0], data[-1]
        # print("here")

        if torch.cuda.is_available():
           data = data.cuda()
           target = target.cuda()

        optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        loss = nn.CrossEntropyLoss()(output, target)
        #loss = WFocalLoss()(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        Loss.update(loss.item(), data.size(0))

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        num_corrects = pred.eq(target.view_as(pred)).sum().item()
        correct += num_corrects

        Acc.update_acc(num_corrects, data.size(0))

        if flag!= 0 and batch_id%config['log_interval'] == 0:
           print('Train Epoch: {} Batch [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {}/{} ({:.0f})%'.format(
                epoch, batch_id * len(data), len(loader.dataset),
                100. * batch_id / len(loader), Loss.avg, correct, Acc.count, 100. * Acc.avg))
        flag = 1

        
        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch,
            'loss_min': Loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        
        # save checkpoint
        save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        

    #total_loss /= len(loader.dataset) 
    print('Train Epoch: {} Average Loss: {:.6f} Average Accuracy: {}/{} ({:.0f})%'.format(
         epoch, Loss.avg, correct, Acc.count, 100. * Acc.avg ))
    print(f"Takes {time.time() - start}")
