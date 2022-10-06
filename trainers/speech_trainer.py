import logging, pdb
from math import ceil
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super(
            MyModelTrainer,
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
        # if args.setup == "centralized":
        #    criterion = nn.CrossEntropyLoss(args.weights.to(device)).to(device)
        # else:
        criterion = nn.CrossEntropyLoss().to(device)
        
        if args.client_optimizer == "sgd":
            if args.setup == "federated":
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
            elif args.setup == "centralized":
                lr_scaler = np.power(2, int(round_idx/5))
                lr = 0.0005 if float(args.lr / lr_scaler) < 0.0005 else float(args.lr / lr_scaler)
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            #  data, labels, lens
            for batch_idx, (data, labels, lens) in enumerate(train_data):
                # data = torch.squeeze(data, 1)
                data, labels, lens = data.to(device), labels.to(device), lens.to(device)
                optimizer.zero_grad()
                output = model(data, lens)
                loss = criterion(output, labels)
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

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0, "test_f1": 0}
        criterion = nn.CrossEntropyLoss(reduction="sum").to(device)
        
        # we need to create aggregate predictions for urbansound
        if args.dataset == "urban_sound":
            with torch.no_grad():
                label_list, pred_list = list(), list()
                logits_dict = dict()
                for _, (data, labels, lens, keys) in enumerate(tqdm(test_data)):
                    # for data, labels, lens in test_data:
                    data, labels, lens = data.to(device), labels.to(device), lens.to(device)
                    output = model(data, lens)
                    loss = criterion(output, labels).data.item()
                    logits_output = nn.Softmax(dim=1)(output)
                    
                    for idx in range(len(labels)):
                        if keys[idx] not in logits_dict:
                            logits_dict[keys[idx]] = dict()
                            logits_dict[keys[idx]]["logits"] = list()
                            logits_dict[keys[idx]]["label"] = labels.detach().cpu().numpy()[idx]
                        logits_dict[keys[idx]]["logits"].append(logits_output.detach().cpu().numpy()[idx])
                    metrics["test_loss"] += loss * labels.size(0)
                    
                for key in logits_dict:
                    label_list.append(logits_dict[key]["label"])
                    pred = np.argmax(np.mean(np.array(logits_dict[key]["logits"]), axis=0))
                    pred_list.append(pred)
                metrics["test_correct"] = np.sum(np.array(pred_list) == np.array(label_list))
                metrics["test_total"] = len(pred_list)
            metrics["test_f1"] = f1_score(label_list, pred_list, average='macro')
        else:
            with torch.no_grad():
                label_list, pred_list = list(), list()
                for _, (data, labels, lens) in enumerate(tqdm(test_data)):
                    # for data, labels, lens in test_data:
                    data, labels, lens = data.to(device), labels.to(device), lens.to(device)
                    output = model(data, lens)
                    loss = criterion(output, labels).data.item()
                    pred = output.data.max(1, keepdim=True)[
                        1
                    ]  # get the index of the max log-probability
                    correct = pred.eq(labels.data.view_as(pred)).sum()
                    for idx in range(len(labels)):
                        label_list.append(labels.detach().cpu().numpy()[idx])
                        pred_list.append(pred.detach().cpu().numpy()[idx][0])
                    metrics["test_correct"] += correct.item()
                    metrics["test_loss"] += loss * labels.size(0)
                    metrics["test_total"] += labels.size(0)
            metrics["test_f1"] = f1_score(label_list, pred_list, average='macro')
        
        if args.best_metric <= metrics["test_f1"]:
            args.best_metric = metrics["test_f1"]
            result_df = pd.DataFrame(index=["result"])
            result_df["best_f1"] = args.best_metric
            result_df.to_csv(args.csv_result_path)
        
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
