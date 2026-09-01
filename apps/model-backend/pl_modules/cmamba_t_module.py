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
        dir_loss_lambda=0.0,
        dir_loss_scale=100.0,
        distributional=False,
        madl_lambda=0.0,
        madl_temp=0.005,
        heavy_tail=False,
        nu_init=2.0,
        selective_lambda=0.0,
        selective_kappa=1.0,
        mixture=False,
        mixture_max_tail=0.25,
        mixture_tail_ratio_init=4.0,
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
                         dir_loss_lambda=dir_loss_lambda,
                         dir_loss_scale=dir_loss_scale,
                         distributional=distributional,
                         madl_lambda=madl_lambda,
                         madl_temp=madl_temp,
                         heavy_tail=heavy_tail,
                         nu_init=nu_init,
                         selective_lambda=selective_lambda,
                         selective_kappa=selective_kappa,
                         mixture=mixture,
                         mixture_max_tail=mixture_max_tail,
                         mixture_tail_ratio_init=mixture_tail_ratio_init,
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
            distributional=distributional,
            # The mixture needs a third output channel for the per-day tail logit.
            head_outputs=3 if mixture else (2 if distributional else 1),
            **kwargs
        )
