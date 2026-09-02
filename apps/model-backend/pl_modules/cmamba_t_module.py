from models.cmamba_t import CMambaT
from .base_module import BaseModule


class CryptoMambaTModule(BaseModule):

    def __init__(
        self,
        num_features=5,
        window_size=14,
        d_model=32,
        n_blocks=4,
        d_state=16,
        d_conv=4,
        expand=2,
        mlp_ratio=2,
        drop=0.1,
        lr=0.0002,
        lr_step_size=50,
        lr_gamma=0.1,
        weight_decay=0.0,
        logger_type=None,
        y_key='Close',
        optimizer='adam',
        mode='ret',
        loss='rmse',
        madl_temp=0.005,
        **kwargs
    ):
        super().__init__(lr=lr,
                         lr_step_size=lr_step_size,
                         lr_gamma=lr_gamma,
                         weight_decay=weight_decay,
                         logger_type=logger_type,
                         y_key=y_key,
                         optimizer=optimizer,
                         mode=mode,
                         window_size=window_size,
                         loss=loss,
                         madl_temp=madl_temp,
                         )

        self.model = CMambaT(
            num_features=num_features,
            window_size=window_size,
            d_model=d_model,
            n_blocks=n_blocks,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            mlp_ratio=mlp_ratio,
            drop=drop,
            **kwargs
        )
