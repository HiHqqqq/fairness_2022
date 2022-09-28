import torch
def cal_rate(y, prediction, z,isabs=1):
    acc=torch.true_divide(torch.sum(prediction==y),len(y))
    # print('output:',prediction)
    # print('labels:',y)
    #print('test acc:{:.3f}'.format(acc))
    z_0_mask = (z == 0.0)
    z_1_mask = (z == 1.0)
    z_0 = int(torch.sum(z_0_mask)) + 1e-6
    z_1 = int(torch.sum(z_1_mask)) + 1e-6

    y_0_mask = (y == 0.0)
    y_1_mask = (y == 1.0)
    y_0 = int(torch.sum(y_0_mask)) + 1e-6
    y_1 = int(torch.sum(y_1_mask)) + 1e-6

    Pr_y_hat_1 = float(torch.sum((prediction == 1))) / (z_0 + z_1)

    Pr_y_hat_1_z_0 = float(torch.sum((prediction == 1)[z_0_mask])) / z_0
    Pr_y_hat_1_z_1 = float(torch.sum((prediction == 1)[z_1_mask])) / z_1

    y_1_z_0_mask = (y == 1.0) & (z == 0.0)
    y_1_z_1_mask = (y == 1.0) & (z == 1.0)
    y_1_z_0 = int(torch.sum(y_1_z_0_mask)) + 1e-6
    y_1_z_1 = int(torch.sum(y_1_z_1_mask)) + 1e-6



    Pr_y_hat_1_y_0 = float(torch.sum((prediction == 1)[y_0_mask])) / y_0
    Pr_y_hat_1_y_1 = float(torch.sum((prediction == 1)[y_1_mask])) / y_1
    Pr_y_hat_0_y_0 = float(torch.sum((prediction == 0)[y_0_mask])) / y_0
    Pr_y_hat_0_y_1 = float(torch.sum((prediction == 0)[y_1_mask])) / y_1

    Pr_y_hat_1_y_1_z_0 = float(torch.sum((prediction == 1)[y_1_z_0_mask])) / y_1_z_0  # P(y_hat=1|y=1,z=0)
    Pr_y_hat_1_y_1_z_1 = float(torch.sum((prediction == 1)[y_1_z_1_mask])) / y_1_z_1  # P(y_hat=1|y=1,z=1)
    Pr_y_hat_0_y_1_z_0 = float(torch.sum((prediction == 0)[y_1_z_0_mask])) / y_1_z_0  # P(y_hat=0|y=1,z=0)
    Pr_y_hat_0_y_1_z_1 = float(torch.sum((prediction == 0)[y_1_z_1_mask])) / y_1_z_1  # P(y_hat=0|y=1,z=1)

    y_0_z_0_mask = (y == 0.0) & (z == 0.0)
    y_0_z_1_mask = (y == 0.0) & (z == 1.0)
    y_0_z_0 = int(torch.sum(y_0_z_0_mask)) + 1e-6
    y_0_z_1 = int(torch.sum(y_0_z_1_mask)) + 1e-6

    Pr_y_hat_1_y_0_z_0 = float(torch.sum((prediction == 1)[y_0_z_0_mask])) / y_0_z_0  # P(y_hat=1|y=0,z=0)
    Pr_y_hat_1_y_0_z_1 = float(torch.sum((prediction == 1)[y_0_z_1_mask])) / y_0_z_1  # P(y_hat=1|y=0,z=1)
    Pr_y_hat_0_y_0_z_0 = float(torch.sum((prediction == 0)[y_0_z_0_mask])) / y_0_z_0  # P(y_hat=0|y=0,z=0)
    Pr_y_hat_0_y_0_z_1 = float(torch.sum((prediction == 0)[y_0_z_1_mask])) / y_0_z_1  # P(y_hat=0|y=0,z=1)

    # --------------------
    # 公平指标
    TPR_z0, TPR_z1 = Pr_y_hat_1_y_1_z_0, Pr_y_hat_1_y_1_z_1
    TNR_z0, TNR_z1 = Pr_y_hat_0_y_0_z_0, Pr_y_hat_0_y_0_z_1
    FPR_z0, FPR_z1 = Pr_y_hat_1_y_0_z_0, Pr_y_hat_1_y_0_z_1
    FNR_z0, FNR_z1 = Pr_y_hat_0_y_1_z_0, Pr_y_hat_0_y_1_z_1

    # print('TPR:', TPR_z0, TPR_z1, TPR_z0 - TPR_z1)
    # print('TNR:', TNR_z0, TNR_z1, TNR_z0 - TNR_z1)
    # print('FPR:', FPR_z0, FPR_z1, FPR_z0 - FPR_z1)
    # print('FNR:', FNR_z0, FNR_z1, FNR_z0 - FNR_z1)
    if isabs==0:
        return acc,TPR_z0 - TPR_z1,TNR_z0 - TNR_z1

    DP=abs(Pr_y_hat_1_z_0-Pr_y_hat_1_z_1)
    TPR_abs=abs(TPR_z0 - TPR_z1)
    TNR_abs=abs(TNR_z0 - TNR_z1)

    return acc,TPR_abs,TNR_abs,DP