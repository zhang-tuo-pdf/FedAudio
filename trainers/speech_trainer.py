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

    def train(self, train_data, device, args, round_idx, client_idx):
        model = self.model
        model.to(device)
        model.train()
        logging.info(" Client ID "+str(client_idx) + " round Idx "+str(round_idx))
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (_, data, target) in enumerate(train_data):
                if args.model == 'BC_ResNet':
                    data = data.view(16, 1, 40, 99)
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

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None, round_idx = None) -> bool:
        return False