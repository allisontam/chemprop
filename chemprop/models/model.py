from argparse import Namespace

import torch
import torch.nn as nn
import numpy as np

from .mpn import MPN
from chemprop.nn_utils import get_activation_function, initialize_weights


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, output_raw: bool, use_cuda: bool):
        """
        Initializes the MoleculeModel.

        :param raw_score: Whether the model should apply activation to output.
        :param cuda: Whether or not to use cuda.
        """
        super(MoleculeModel, self).__init__()

        self.activation = nn.Identity()
        if output_raw:
            self.activation = nn.Sigmoid()
        self.use_cuda = use_cuda

    def create_encoder(self, args: Namespace):
        """
        Creates the paired message passing encoders for the model.

        :param args: Arguments.
        """
        self.drug_encoder = MPN(args, attn_readout=False)
        self.cmpd_encoder = MPN(args, attn_readout=True)

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        self.ops = args.ops

        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size  # To account for 2 molecules
            if args.use_input_features:
                first_linear_dim += args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # self.gamma = nn.Linear(args.hidden_size, args.hidden_size)
        # self.beta = nn.Linear(args.hidden_size, args.hidden_size)
        self.ffn = nn.ModuleList()

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
            self.ffn.append(nn.Sequential(*ffn))
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            self.ffn.append(nn.Sequential(*ffn))
            for _ in range(args.ffn_num_layers - 2):
                ffn = [
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ]
                self.ffn.append(nn.Sequential(*ffn))
            ffn = [
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size),
            ]
            self.ffn.append(nn.Sequential(*ffn))

        # Create FFN model
        # self.ffn = nn.Sequential(*ffn)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        smiles, feats = input  # TODO: in future, move drug/cmpd feats out of MPN

        learned_drug, _ = self.drug_encoder(batch=[x[0] for x in smiles],
                features_batch=[x[0] for x in feats])
        learned_cmpd, entropy = self.cmpd_encoder(batch=[x[1] for x in smiles],
                features_batch=[x[1] for x in feats],
                readout_embed=learned_drug)

        # gamma, beta = self.gamma(learned_drug), self.beta(learned_drug)

        # Incorporate pair features when available
        if feats[0][2] is not None:
            features_batch = torch.from_numpy(np.stack([x[2] for x in feats])).float()
            if self.use_cuda:
                features_batch = features_batch.cuda()
            learned_cmpd = torch.cat((learned_cmpd, features_batch), dim=1)
            raise ValueError("Won't work bc gamma doesn't have features")  # TODO

        output = learned_cmpd
        for module in self.ffn:
            # newInput = gamma * output + beta
            # output = module(newInput)
            output = module(output)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if not self.training:
            output = self.activation(output)

        return output, entropy


def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        raise NotImplementedError

    model = MoleculeModel(output_raw=args.output_raw, use_cuda=args.cuda)
    model.create_encoder(args)
    model.create_ffn(args)

    initialize_weights(model)

    return model
