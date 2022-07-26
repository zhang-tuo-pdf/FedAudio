import logging, pdb
from math import ceil
import torch
from torch import nn
from tqdm import tqdm

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class FedProxModelTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super(
            FedProxModelTrainer,
            self,
        ).__init__(*args, **kwargs)
        self.param_size = sum(p.numel() for p in self.model.parameters())

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args, round_idx, client_idx):
        model = self.model
        model.to(device)
        model.train()
        logging.info(" Client ID " + str(client_idx) + " round Idx " + str(round_idx))
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            #  data, labels, lens
            for batch_idx, (data, labels, lens) in enumerate(train_data):
                # data = torch.squeeze(data, 1)
                last_global_parameter = model.parameters()
                data, labels, lens = data.to(device), labels.to(device), lens.to(device)
                optimizer.zero_grad()
                output = model(data, lens)
                loss = criterion(output, labels)
                # compute proximal_term
                proximal_term = 0.0
                for w, w_t in zip(model.parameters(), last_global_parameter):
                    proximal_term += (w - w_t).norm(2)
                loss = loss + (args.mu / 2) * proximal_term
                loss.backward()

                logging.info(
                    "Client Index = {}\tEpoch: {}\tBatch Loss: {:.6f}\tBatch Number: {}".format(
                        client_idx, epoch, loss, batch_idx
                    )
                )

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

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

        criterion = nn.CrossEntropyLoss(reduction="sum").to(device)

        with torch.no_grad():
            for batch_idx, (data, labels, lens) in enumerate(tqdm(test_data)):
                # for data, labels, lens in test_data:
                data, labels, lens = data.to(device), labels.to(device), lens.to(device)
                output = model(data, lens)
                loss = criterion(output, labels).data.item()
                pred = output.data.max(1, keepdim=True)[
                    1
                ]  # get the index of the max log-probability
                correct = pred.eq(labels.data.view_as(pred)).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss * labels.size(0)
                metrics["test_total"] += labels.size(0)
        return metrics

    def test_on_the_server(
        self,
        train_data_local_dict,
        test_data_local_dict,
        device,
        args=None,
        round_idx=None,
    ) -> bool:
        return False