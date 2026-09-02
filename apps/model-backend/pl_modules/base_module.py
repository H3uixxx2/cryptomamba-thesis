import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.regression import MeanAbsolutePercentageError as MAPE
    

class BaseModule(pl.LightningModule):

    def __init__(
        self,
        lr=0.0002, 
        lr_step_size=50,
        lr_gamma=0.1,
        weight_decay=0.0, 
        logger_type=None,
        window_size=14,
        y_key='Close',
        optimizer='adam',
        mode='default',
        loss='rmse',
        # Temperature of the smooth position in trade_pnl, which is logged as
        # val/neg_pnl for every run so a checkpoint can be selected on the objective
        # the trading test actually uses.
        madl_temp=0.005,
    ):
        super().__init__()

        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.logger_type = logger_type
        self.y_key = y_key
        self.optimizer = optimizer
        self.batch_size = None
        self.mode = mode
        self.window_size = window_size
        self.loss = loss
        self.madl_temp = madl_temp

        # self.loss = lambda x, y: torch.sqrt(tmp(x, y))
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.mape = MAPE()
        self.normalization_coeffs = None

    def trade_pnl(self, y, y_hat, y_old):
        """Differentiable stand-in for what the paper's strategies actually pay.

        Reading utils/trade.py, none of the three strategies looks at squared error.
        ``vanilla`` acts on the sign of (pred - today) past a 1% deadband; ``smart``
        and ``smart_w_short`` size the position by how far the prediction sits from
        today. So profit comes from being on the right side of the *big* days, and a
        confident wrong call on a 6% day costs far more than a hesitant right one on a
        0.2% day — a distinction RMSE cannot express.

        This is Mean Absolute Directional Loss written as money: take a position
        tanh(r_hat / temp) — a smooth stand-in for the sign, so gradients exist — and
        collect that fraction of the realised move. The result is USD of profit per
        unit of capital, which puts it in the same units as the RMSE term it is
        blended with, so the weight does not need rescaling by hand.

        Returned positive-is-good; callers negate it to make a loss.
        """
        realised = (y - y_old) / y_old.clamp(min=1e-8)
        predicted = (y_hat - y_old) / y_old.clamp(min=1e-8)
        position = torch.tanh(predicted / self.madl_temp)
        return (y_old * realised * position).mean()














    def forward(self, x, y_old=None):
        if self.mode == 'default':
            return self.model(x).reshape(-1)
        elif self.mode == 'diff':
            return self.model(x).reshape(-1) + y_old
        elif self.mode == 'ret':
            # model outputs a relative return r_hat; reconstruct price around the
            # last observed close so the network never has to model the price level
            return y_old * (1.0 + self.model(x).reshape(-1))


        
    def set_normalization_coeffs(self, factors):
        if factors is None:
            return
        scale = factors.get(self.y_key).get('max') - factors.get(self.y_key).get('min')
        shift = factors.get(self.y_key).get('min')
        self.normalization_coeffs = (scale, shift)

    def denormalize(self, y, y_hat):
        if self.normalization_coeffs is not None:
            scale, shift = self.normalization_coeffs
            y = y * scale + shift
            y_hat = y_hat * scale + shift
        return y, y_hat

    def training_step(self, batch, batch_idx):
        x = batch['features']
        y = batch[self.y_key]
        y_old = batch[f'{self.y_key}_old']
        if self.batch_size is None:
            self.batch_size = x.shape[0]

        y_hat = self.forward(x, y_old).reshape(-1)
        y, y_hat = self.denormalize(y, y_hat)
        mse = self.mse(y_hat, y)
        rmse = torch.sqrt(mse)
        mape = self.mape(y_hat, y)
        l1 = self.l1(y_hat, y)

        self.log("train/mse", mse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        self.log("train/rmse", rmse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("train/mape", mape.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("train/mae", l1.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)

        if self.loss == 'mse':
            total = mse
        elif self.loss == 'rmse':
            total = rmse
        elif self.loss == 'mae':
            total = l1
        elif self.loss == 'mape':
            total = mape

        return total
        
    
    def validation_step(self, batch, batch_idx):
        x = batch['features']
        y = batch[self.y_key]
        y_old = batch[f'{self.y_key}_old']
        if self.batch_size is None:
            self.batch_size = x.shape[0]

        y_hat = self.forward(x, y_old).reshape(-1)
        y, y_hat = self.denormalize(y, y_hat)
        mse = self.mse(y_hat, y)
        rmse = torch.sqrt(mse)
        mape = self.mape(y_hat, y)
        l1 = self.l1(y_hat, y)

        self.log("val/mse", mse.detach(), sync_dist=True, batch_size=self.batch_size, prog_bar=False)
        self.log("val/rmse", rmse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("val/mape", mape.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("val/mae", l1.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)

        # Logged for every point run, not just profit-trained ones, so a run can be
        # selected on the objective that actually decides the trading test. Negated
        # because checkpointing and early stopping both minimise.
        y_old_denorm, _ = self.denormalize(y_old, y_old)
        pnl = self.trade_pnl(y, y_hat, y_old_denorm)
        self.log("val/neg_pnl", (-pnl).detach(), batch_size=self.batch_size,
                 sync_dist=True, prog_bar=True)
        return {
            "val_loss": mse,
        }
    
    def test_step(self, batch, batch_idx):
        x = batch['features']
        y = batch[self.y_key]
        y_old = batch[f'{self.y_key}_old']
        if self.batch_size is None:
            self.batch_size = x.shape[0]
        y_hat = self.forward(x, y_old).reshape(-1)
        y, y_hat = self.denormalize(y, y_hat)
        mse = self.mse(y_hat, y)
        rmse = torch.sqrt(mse)
        mape = self.mape(y_hat, y)
        l1 = self.l1(y_hat, y)

        self.log("test/mse", mse.detach(), sync_dist=True, batch_size=self.batch_size, prog_bar=False)
        self.log("test/rmse", rmse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("test/mape", mape.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("test/mae", l1.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        return {
            "test_loss": mse,
        }
    
    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optim = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == 'sgd':
            optim = torch.optim.SGD(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f'Unimplemented optimizer {self.optimizer}')
        scheduler = torch.optim.lr_scheduler.StepLR(optim, 
                                                    self.lr_step_size, 
                                                    self.lr_gamma
                                                    )
        return [optim], [scheduler]

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        scheduler.step()
