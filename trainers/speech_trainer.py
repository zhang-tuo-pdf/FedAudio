import logging
from math import ceil
import torch
from torch import nn

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super(MyModelTrainer, self, ).__init__(*args, **kwargs)
        self.param_size = sum(p.numel() for p in self.model.parameters())

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, client_idx, train_data, device, args):
        model = self.model

        freeze_param_size = self.param_size * (1 - args.partial_ratio_list[client_idx])
        param_size = 0
        for p in model.parameters():
            param_size += p.numel()
            if param_size < freeze_param_size:
                p.requires_grad = False
            else:
                p.requires_grad = True

        layer_trained = [1 if p.requires_grad else 0 for p in model.parameters()]
        # print(layer_trained)

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        epoch_loss = []
        for epoch in range(args.epochs_list[client_idx]):
            batch_loss = []
            for batch_idx, (_, data, target) in enumerate(train_data):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

                logging.info('Client Index = {}\tEpoch: {}\tBatch Loss: {:.6f}\tBatch Number: {}'.format(
                    client_idx, epoch, loss, batch_idx))

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                #            100. * (batch_idx + 1) / len(train_data), loss.item()))
                batch_loss.append(loss.item())
                # break
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}\tLayer Trained: {}'.format(
                client_idx, epoch, sum(epoch_loss) / len(epoch_loss), sum(layer_trained)))

        return layer_trained

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss(reduction='sum').to(device)

        with torch.no_grad():
            for keys, data, target in test_data:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                loss = criterion(output, target).data.item()
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct = pred.eq(target.data.view_as(pred)).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False