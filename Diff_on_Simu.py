import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from src.data_processing import processingData4
from src.diffusion_model import UnetVector, GaussianDiffusionCondition
from src.utils import MyDataset, set_seed, evaluate, ecrps

torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)


set_seed(0)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

HYPERPARAMETERS = {
    'window_size': 3,
    'lr': 0.00095,
    'wd': 0.0001,
    'epochs': 500,
    'batch_size': 20,
    'sample_size': 20,
    'time_steps': 100,
    'beta_schedule': 'cosine',
    'use_selfcond': True, 
}


data_directory = './Datasets/'
model_directory = './runs/model/'

env_num = 1


results = []

for table_num in range(1, 51): 
    print(f"Processing environment {env_num}, table {table_num}")
    
    (LossMin, LossMax, trainValid_input, trainValid_output, test_input, test_output) = processingData4(
        data_directory, env_num, table_num, HYPERPARAMETERS['window_size']
    )


    trainValid_output = trainValid_output[:, np.newaxis]
    test_input = torch.tensor(test_input, device=device)
    test_output = test_output[:, np.newaxis]
    
    dataset = MyDataset(trainValid_input, trainValid_output)
    dataloader = DataLoader(dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=True)

    model = UnetVector(dim=64, dim_mults=(2, 4, 8, 16), channels=1, self_condition=HYPERPARAMETERS['use_selfcond'])
    model.to(device)

    diffusion = GaussianDiffusionCondition(
        model,
        device,
        timesteps=HYPERPARAMETERS['time_steps'],
        sampling_timesteps=HYPERPARAMETERS['time_steps'],
        objective='pred_x0',
        beta_schedule=HYPERPARAMETERS['beta_schedule']
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=HYPERPARAMETERS['lr'],
        weight_decay=HYPERPARAMETERS['wd']
    )

    
    for epoch in range(HYPERPARAMETERS['epochs']):
        for step, batch in enumerate(dataloader):
            x_input, x_target = batch
            x_input, x_target = x_input.to(device), x_target.to(device)
            optimizer.zero_grad()
            loss = diffusion(x_target, x_input)
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    sampled_pts = diffusion.sample(test_input, HYPERPARAMETERS['sample_size'])


    result = sampled_pts.reshape(-1, HYPERPARAMETERS['sample_size']).cpu().numpy()


    mae_result, rmse_result, mape_result = evaluate(test_output, np.mean(result, axis=-1))
    ecrps_result = ecrps(test_output, result)


    print(f'Performance Metrics for Table {table_num}:\n MAE: {mae_result}, RMSE: {rmse_result}, MAPE: {mape_result},ECRPS: {ecrps_result}')

    results.append({
        'env_num': env_num,
        'table_num': table_num,
        'MAE': mae_result,
        'RMSE': rmse_result,
        'MAPE': mape_result,
        'ECRPS': ecrps_result,
    })

results_df = pd.DataFrame(results)
results_df.to_csv('Best_Performance_1.csv', index=False)
print("All tables completed, best performance metrics are saved in 'Best_Performance_1.csv'.")
