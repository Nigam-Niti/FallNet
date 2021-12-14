def test(config, model, loader, text='Validation'):
    model.eval()
    correct = 0
    total_loss = 0.0
    Loss, Acc = AverageMeter(), AverageMeter()
    with torch.no_grad():
         for batch_id, data in enumerate(loader):
             data, target = data[0], data[-1]

             if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

             output = model(data)
             #loss = F.nll_loss(output, target)
             loss = nn.CrossEntropyLoss()(output, target)
             #loss = WFocalLoss()(output, target)

             total_loss += loss.item()

             Loss.update(loss.item(), data.size(0))

             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
             num_corrects = pred.eq(target.view_as(pred)).sum().item()
             correct += num_corrects

             Acc.update_acc(num_corrects, data.size(0))
           
    total_loss /= len(loader.dataset)
    print(text + ' Average Loss: {:.6f} Average Accuracy: {}/{} ({:.0f})%'.format(
         Loss.avg, correct, Acc.count , 100. * Acc.avg ))
