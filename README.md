**Probabilistic Loss Reserving Prediction via Denoising Diffusion Model**

This project implements a conditional denoising diffusion model to handle and predict loss reserving from both real and simulated sources. It includes modules for data processing, model training, and evaluation of prediction results. The project enables users to apply a diffusion model to various data types and get predictive insights from the model.

## File Structure

- **src/diffusion_model.py**: Core implementation of the diffusion model, including training and prediction.
- **src/data_processing_real.py**: Data processing module for real ScheduleP data.
- **src/data_processing.py**: Data processing module for simulated data.
- **Diff_on_Real.py**: Implements the diffusion model on real ScheduleP data.
- **Diff_on_Simu.py**: Implements the diffusion model on simulated data.
- **src/utils.py**: Contains evaluation metrics and utility functions.

## Installation

First, clone this repository:

```bash
git clone https://github.com/shiyinggao/DiffLR.git
```

Then install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. To process real data and train the model:

   ```bash
   python Diff_on_Real.py
   ```

2. To process simulated data and train the model:

   ```bash
   python Diff_on_Simu.py
   ```

3. You can find evaluation metrics such as `mae`, `rmse`, `mape`, and `ecrps` in `utils.py` to evaluate the model’s predictions.

## Dependencies

This project requires the following Python libraries:

- `numpy`
- `torch`
- `matplotlib`
- `pandas`

To install these dependencies, run the following command:

```bash
pip install numpy torch matplotlib pandas
```

## License

This project is licensed under the [MIT License](./LICENSE).
