import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

############ create Loss Function #########
class MyLoss(nn.Module):

    def __init__(self, eta=7):
        super(MyLoss, self).__init__()
        self.eta = eta
        # self.acc_factor_prob = 0.1
    pass


    '''
    weights has shape (n), multiply loss of point i with weights[i]
    '''
    def forward(self, outputs, y, weights, calculate_add = True, calculate_booster_weights=False, n_bins=-1):

        nns = torch.trunc(y)
        nns = nns.long()
        # nns: (n, k)

        k = nns.shape[1]
        n = nns.shape[0]

        n = outputs.shape[0]

        batch_size = y.shape[0] # figuring out batch size from size of knn matrix

        outputs = outputs.to(utils.primary_device)

        ################ one hot encoding #############
        if calculate_booster_weights:
            outputs = F.one_hot(outputs, n_bins)
        else:
            n_bins = outputs.shape[1]

        diff = 0
        booster_weights = 0

        # Loss term 1: Accuracy-related loss
        if calculate_add or calculate_booster_weights:
            reshaped_nns = torch.movedim(nns, 1, 0)
            # reshaped_nns: (k, n)
            del nns
            reshaped_nns = torch.unsqueeze(reshaped_nns, 2)
            # reshaped_nns: (k, n, 1)
            reshaped_nns = torch.movedim(reshaped_nns, 1, 2)
            # reshaped_nns: (k, 1, n)
            reshaped_nns = reshaped_nns.repeat(1, n_bins, 1)
            # reshaped_nns: (k, n_bins, n)


            refactored_outputs = torch.unsqueeze(outputs, 0)
            refactored_outputs = torch.movedim(refactored_outputs, 1, 2)
            refactored_outputs = refactored_outputs.repeat(k, 1, 1)

            cost_tensor_new = torch.gather(refactored_outputs, 2, reshaped_nns)
            del reshaped_nns
            del refactored_outputs

            reshaped_outputs = torch.transpose(outputs, 0, 1)
            reshaped_outputs = torch.reshape(reshaped_outputs, (1, n_bins, n))

            reshaped_outputs = reshaped_outputs[:, :, :batch_size]
            # reshaped outputs shape: (1, n_bins, n)
            # cost_tensor_new shape: (k, n_bins, n)

            batch_outputs = outputs[:batch_size, :]


            # find bin in which majority of points are
            bins = torch.argmax(cost_tensor_new, 1)
            # bins shape: (k, n)

            del cost_tensor_new

            # Find bin in which majority of points are
            majority_bins, _ = torch.mode(bins, dim=0)
            majority_bins_ohe = F.one_hot(majority_bins, n_bins)
            majority_bins_ohe = majority_bins_ohe.T
            majority_bins_ohe = torch.unsqueeze(majority_bins_ohe, 0)

            # Compute cross-entropy loss
            cross_entropy_loss = -majority_bins_ohe * torch.log(reshaped_outputs + 1e-9)
            # shape: (1, n_bins, n)
            
            # Weighted cross-entropy loss
            weighted_cross_entropy_loss = weights * cross_entropy_loss

            # Calculate the sum of differences 
            diff_sum = torch.sum(weighted_cross_entropy_loss) / (batch_size)

            # Compute booster weights
            errors = torch.sum(cross_entropy_loss, dim=1).detach()
            errors = torch.sum(errors, dim=0)
            booster_weights = errors.detach()

            del cross_entropy_loss
            del reshaped_outputs

        pass

        # Loss term 2: Bins distribution-related loss
        batch_outputs = outputs[:batch_size, :]
        b_top = torch.topk(batch_outputs, k=int(batch_size/n_bins), dim=0)[0]
        b = -torch.sum(b_top) / batch_size

        # Loss term 3: Geometric consistency loss between outputs and input batch
        #pairwise_distances_output = torch.cdist(outputs, outputs)
        #pairwise_distances_input = torch.cdist(inputs, inputs)
        #g = F.mse_loss(pairwise_distances_output, pairwise_distances_input)

        # Combine loss terms
        loss = self.eta * b + diff_sum

        del batch_outputs

        return (loss, diff_sum, b, 0, booster_weights)
