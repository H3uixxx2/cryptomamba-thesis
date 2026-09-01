import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
try:  # CMamba pulls native mamba_ssm/causal_conv1d; keep optional so the LSTM/GRU/iTransformer
    from models.cmamba import CMamba  # noqa: F401  baselines can import BaseModule without them.
except ImportError:  # pragma: no cover - native deps absent (e.g. baseline-only Colab / macOS)
    CMamba = None
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
        return_center=0.0,
        return_scale=1.0,
        return_clip=0.5,
        dir_loss_lambda=0.0,
        dir_loss_scale=100.0,
        distributional=False,
        sigma_init=0.03,
        sigma_floor=1e-6,
        madl_lambda=0.0,
        madl_temp=0.005,
        heavy_tail=False,
        nu_init=2.0,
        selective_lambda=0.0,
        selective_kappa=1.0,
        mixture=False,
        mixture_max_tail=0.25,
        mixture_tail_ratio_init=4.0,
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
        self.return_center = float(return_center)
        self.return_scale = float(return_scale)
        self.return_clip = float(return_clip)
        if self.mode == 'scaled_log_return':
            if not math.isfinite(self.return_center):
                raise ValueError('return_center must be finite')
            if not math.isfinite(self.return_scale) or self.return_scale <= 0.0:
                raise ValueError('return_scale must be finite and positive')
            if not math.isfinite(self.return_clip) or self.return_clip <= 0.0:
                raise ValueError('return_clip must be finite and positive')
            if self.loss not in {'return_mse', 'return_huber'}:
                raise ValueError(
                    'scaled_log_return mode requires return_mse or return_huber loss'
                )
        # optional direction-aware auxiliary loss (0.0 = off, paper behavior unchanged):
        # BCE on the sign of the predicted move; the relative move r_hat is scaled by
        # dir_loss_scale so a 1% daily move maps to a logit of ~1.
        self.dir_loss_lambda = dir_loss_lambda
        self.dir_loss_scale = dir_loss_scale
        # Distributional head (default off — the point-forecast path is untouched).
        # Trained and selected on CRPS, which is the metric that can actually see the
        # one signal this series has: the conditional variance.
        self.distributional = distributional
        self.log_sigma_init = math.log(sigma_init)
        self.sigma_floor = sigma_floor
        # Profit-shaped auxiliary term (0.0 = off, every earlier run unchanged).
        # temp sets how fast the position saturates: typical daily moves are ~2%, so
        # 0.005 means an ordinary day already asks for a near-full position.
        self.madl_lambda = madl_lambda
        self.madl_temp = madl_temp
        # Heavy-tailed predictive distribution (default off — Gaussian, as recorded).
        # One shared learnable parameter, not a head output: BTC's tails are a property
        # of the series, not of the day, and a per-day nu would be three extra ways to
        # overfit 1,446 training days.
        self.heavy_tail = heavy_tail
        if heavy_tail:
            self.log_nu_m1 = nn.Parameter(torch.tensor(float(nu_init)))
        # Selective-profit term (default off). See .ai/phase-selective-prediction.md:
        # sigma is stop-gradiented inside the position, because measured against the
        # bounded CRPS scale force this term wins above lambda ~ 0.22 and collapses it.
        self.selective_lambda = selective_lambda
        self.selective_kappa = selective_kappa
        # Centred two-scale Gaussian mixture (blueprint §3.2). Both components share mu,
        # so tail modelling cannot move the point forecast by relabelling components, and
        # the mean stays directly comparable with persistence. One global tail ratio plus a
        # per-day tail probability: d + 2 parameters over a Gaussian head.
        self.mixture = mixture
        self.mixture_max_tail = mixture_max_tail
        if mixture:
            if heavy_tail:
                raise ValueError('mixture and heavy_tail are two different heads')
            if not 0.0 < mixture_max_tail < 0.5:
                raise ValueError('mixture_max_tail must be in (0, 0.5)')
            if mixture_tail_ratio_init <= 1.0:
                raise ValueError('mixture_tail_ratio_init must exceed 1')
            self.raw_tail_ratio = nn.Parameter(torch.tensor(
                math.log(math.expm1(mixture_tail_ratio_init - 1.0))))

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

    def forward_dist(self, x, y_old):
        """Return (mu, sigma) in price units for a distributional head.

        The point-forecast path can only ever tie the persistence baseline when the
        conditional mean is unpredictable, because today's price *is* the optimal
        estimate of tomorrow's. The spread is a different story: conditional variance
        is predictable on this series, so a model that forecasts the whole
        distribution has something real to be right about even when the mean does not.

        ``mu`` reuses the return reconstruction unchanged, so this cannot quietly
        degrade the point forecast — it is the same arithmetic with a second output
        bolted on. ``sigma`` is produced as a fraction of the last close and then
        scaled to price, since volatility is naturally relative.
        """
        mu, sigma, _, _ = self.forward_dist_full(x, y_old)
        return mu, sigma

    def forward_dist_full(self, x, y_old):
        """``(mu, sigma, component_scales, weights)``; the last two are None off-mixture.

        ``sigma`` is the parameter the head's own CRPS takes: the Gaussian standard
        deviation, the Student-t *scale*, or — for the mixture, which has no single scale
        parameter — the predictive standard deviation. Callers that need a prediction
        interval must go through :meth:`interval_halfwidth`, never assume 1.645 * sigma.
        """
        width = 3 if self.mixture else 2
        out = self.model(x)
        if out.shape[-1] != width:
            raise ValueError(f'distributional mode needs a {width}-wide head, '
                             f'got width {out.shape[-1]}')
        r_hat, log_s = out[..., 0].reshape(-1), out[..., 1].reshape(-1)
        if self.mode == 'ret':
            mu = y_old * (1.0 + r_hat)
        elif self.mode == 'diff':
            mu = y_old + r_hat
        else:
            mu = r_hat
        # Offset so an untrained head starts at a plausible daily move rather than
        # sigma = 100% of price, which would swamp the first epochs.
        sigma_rel = torch.exp(log_s.clamp(-8.0, 4.0) + self.log_sigma_init)
        scale = (y_old.abs() * sigma_rel).clamp_min(self.sigma_floor)
        if not self.mixture:
            return mu, scale, None, None

        # Core scale is the head's own; the tail component is a fixed global multiple of
        # it, so the two can never cross and there is no label switching to worry about.
        tail_weight = self.mixture_max_tail * torch.sigmoid(out[..., 2].reshape(-1))
        scales = torch.stack((scale, scale * (1.0 + F.softplus(self.raw_tail_ratio))),
                             dim=-1)
        weights = torch.stack((1.0 - tail_weight, tail_weight), dim=-1)
        sd = torch.sqrt((weights * scales.square()).sum(-1).clamp_min(self.sigma_floor ** 2))
        return mu, sd, scales, weights

    @staticmethod
    def _expected_abs_normal(delta, scale):
        """E|N(delta, scale^2)| — the one primitive the mixture CRPS is built from."""
        scale = scale.clamp_min(1e-12)
        z = delta / scale
        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        return 2.0 * scale * pdf + delta * (2.0 * cdf - 1.0)

    @classmethod
    def mixture_crps(cls, y, mu, scales, weights):
        """Analytic CRPS of a finite Gaussian mixture (scoringRules ``crps_mixnorm``).

        CRPS = sum_i w_i A(y - mu_i, s_i) - 1/2 sum_ij w_i w_j A(mu_i - mu_j, sqrt(si^2+sj^2))
        with A = E|N(.,.)|. Exact and differentiable, so no Monte Carlo score noise is
        added on top of the gradient. Verified against a 4-million-draw simulation to 6e-4
        relative before this was wired in.
        """
        means = mu.unsqueeze(-1).expand_as(scales)
        first = (weights * cls._expected_abs_normal(
            y.unsqueeze(-1) - means, scales)).sum(-1)
        gap = means.unsqueeze(-1) - means.unsqueeze(-2)
        pair_scale = torch.sqrt(scales.square().unsqueeze(-1)
                                + scales.square().unsqueeze(-2))
        pair_weight = weights.unsqueeze(-1) * weights.unsqueeze(-2)
        second = 0.5 * (pair_weight
                        * cls._expected_abs_normal(gap, pair_scale)).sum((-1, -2))
        return first - second

    @torch.no_grad()
    def interval_halfwidth(self, sigma, scales=None, weights=None, p=0.95,
                           iterations=48):
        """Half-width of the central 2p-1 interval, under the distribution actually issued.

        Assuming 1.645 * sigma for everything is the bug that already cost this project
        once: for a Student-t, sigma is the scale and the right multiplier is t(p, nu); for
        the mixture there is no closed form at all, so the CDF is bisected.
        """
        if self.mixture:
            if scales is None or weights is None:
                raise ValueError('mixture intervals need component scales and weights')
            lo = torch.zeros_like(sigma)
            hi = 16.0 * scales.max(dim=-1).values
            for _ in range(iterations):
                mid = 0.5 * (lo + hi)
                z = mid.unsqueeze(-1) / scales.clamp_min(1e-12)
                covered = (weights * torch.erf(z / math.sqrt(2.0))).sum(-1)
                lo = torch.where(covered < 2.0 * p - 1.0, mid, lo)
                hi = torch.where(covered >= 2.0 * p - 1.0, mid, hi)
            return 0.5 * (lo + hi)
        if self.heavy_tail:
            from scipy.stats import t as _student_t
            return float(_student_t.ppf(p, float(self.nu))) * sigma
        return math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * p - 1.0)).item() * sigma

    @staticmethod
    def gaussian_crps(y, mu, sigma):
        """Closed-form CRPS of N(mu, sigma) against a scalar outcome.

        CRPS is a *proper* scoring rule: it is minimised only by the true predictive
        distribution, so a model cannot win it by inflating or shrinking the spread.
        That is what makes it safe to optimise directly, unlike a bare variance loss.
        Reduces to MAE when sigma -> 0, so it stays comparable in price units.
        """
        z = (y - mu) / sigma
        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))

    @property
    def nu(self):
        """Degrees of freedom, clamped away from both ends.

        The Student-t CRPS has a 1/(nu - 1) factor and diverges at nu = 1; past about
        50 the distribution is Gaussian to well inside float precision and the gradient
        on nu vanishes, so there is nothing above that worth searching.

        The floor is 2.05, not 2. At exactly 2 the variance is infinite, so the predictive
        standard deviation the selective term and the gate divide by does not exist — and
        clamp passes zero gradient at its bound, so a model that reached 2 would stay
        there. At 2.05 the SD is 6.40x the scale: finite, and large enough that an arm
        parked on the floor shrinks its positions until the registered
        mean(abs(position)) >= 0.05 disqualifier catches it, rather than failing quietly.
        """
        if not self.heavy_tail:
            return None
        return (1.0 + F.softplus(self.log_nu_m1)).clamp(2.05, 50.0)

    def predictive_sd(self, sigma):
        """Standard deviation of the predictive distribution, which is not its scale.

        ``forward_dist`` emits the *scale*, because that is what the location-scale CRPS
        takes. For a Gaussian the two coincide; for a Student-t
        SD = scale * sqrt(nu / (nu - 2)), which is 1.73x at the initial nu = 3.13. Sizing a
        position by the scale would therefore make a heavy-tailed arm act as though it were
        1.7x more certain than it claims, at an effective kappa its Gaussian counterpart
        never sees — and the 2x2 would be comparing two things at once.

        ``nu`` is detached for the same reason A1 detaches ``sigma``: attached, the profit
        term gets a gradient path into the tail parameter and can reshape the predictive
        distribution to suit its own position sizing. The spread is shaped by the proper
        scoring rule alone.
        """
        if not self.heavy_tail:
            return sigma
        nu = self.nu.detach()
        return sigma * torch.sqrt(nu / (nu - 2.0))

    @staticmethod
    def _betainc(x, a, b, iters=180):
        """Regularized incomplete beta I_x(a, b), differentiable in every argument.

        torch 2.8 ships neither ``betainc`` nor ``StudentT.cdf``, so this is the
        continued fraction of Numerical Recipes' ``betacf`` written in torch ops. The
        iteration count is fixed rather than convergence-tested: data-dependent control
        flow would make the backward pass depend on the values, and 180 terms is well
        past convergence over the range this model can reach. Verified against SciPy to
        nine decimals in tests.
        """
        x, a, b = torch.broadcast_tensors(x, a, b)
        # The fraction only converges on one side of this boundary; the reflection
        # I_x(a,b) = 1 - I_{1-x}(b,a) covers the other. Both branches are evaluated on
        # clamped inputs so the discarded one can never emit a NaN into the gradient.
        swap = x >= (a + 1.0) / (a + b + 2.0)
        xs = torch.where(swap, 1.0 - x, x)
        as_, bs = torch.where(swap, b, a), torch.where(swap, a, b)
        # Bounds must come from the tensor's dtype, not from float64 literals. Training
        # runs in float32, where 1e-300 underflows to 0 and 1 - 1e-16 rounds to exactly
        # 1, so log(0) and log1p(-1) appear. Their values survive — exp(-inf) is 0 — but
        # the gradient is inf * 0 = NaN, it reaches the shared nu, and one optimiser step
        # turns the entire model to NaN. t = 0 alone is enough: it puts x at exactly 1.
        # The lower bound is the smallest normal, not eps: with a = 1/2 the incomplete
        # beta behaves like sqrt(x) near zero, so clamping at eps would leave an error of
        # sqrt(eps)/2 — 7.5e-9 in float64, enough to miss SciPy at nine decimals. What
        # matters for the gradient is only that the bound be strictly positive, since
        # clamp passes zero gradient for inputs outside it.
        eps, tiny = torch.finfo(xs.dtype).eps, torch.finfo(xs.dtype).tiny
        xs = xs.clamp(tiny, 1.0 - eps)

        qab, qap, qam = as_ + bs, as_ + 1.0, as_ - 1.0

        def guard(t):
            return torch.where(t.abs() < tiny, torch.full_like(t, tiny), t)

        c = torch.ones_like(xs)
        d = 1.0 / guard(1.0 - qab * xs / qap)
        h = d
        for m in range(1, iters + 1):
            m2 = 2.0 * m
            num = m * (bs - m) * xs / ((qam + m2) * (as_ + m2))
            d = 1.0 / guard(1.0 + num * d)
            c = guard(1.0 + num / c)
            h = h * d * c
            num = -(as_ + m) * (qab + m) * xs / ((as_ + m2) * (qap + m2))
            d = 1.0 / guard(1.0 + num * d)
            c = guard(1.0 + num / c)
            h = h * d * c

        log_front = (as_ * torch.log(xs) + bs * torch.log1p(-xs)
                     + torch.lgamma(qab) - torch.lgamma(as_) - torch.lgamma(bs))
        front = torch.exp(log_front) * h / as_
        return torch.where(swap, 1.0 - front, front)

    @classmethod
    def student_t_cdf(cls, t, nu):
        """CDF of the standard Student-t, via the incomplete beta identity.

        F(t) = 1 - I_z(nu/2, 1/2)/2 for t >= 0 and I_z(nu/2, 1/2)/2 for t < 0, with
        z = nu / (nu + t^2). Written symmetrically so the branch costs no accuracy at
        t = 0, where both halves agree at exactly 1/2.
        """
        t, nu = torch.broadcast_tensors(torch.as_tensor(t), torch.as_tensor(nu))
        half = cls._betainc(nu / (nu + t * t), nu / 2.0,
                            torch.full_like(nu, 0.5)) / 2.0
        return torch.where(t >= 0, 1.0 - half, half)

    @classmethod
    def student_t_crps(cls, y, mu, sigma, nu):
        """Closed-form CRPS of a location-scale Student-t (scoringRules ``crps_t``).

        Still a proper scoring rule, so optimising it directly cannot be gamed by
        widening or narrowing the interval, and still analytic — no sampling, so no
        extra gradient variance over the Gaussian head it replaces. Gaussian tails
        systematically under-price the days that move BTC most; this does not.
        """
        w = (y - mu) / sigma
        cdf = cls.student_t_cdf(w, nu)
        log_pdf = (torch.lgamma((nu + 1.0) / 2.0) - torch.lgamma(nu / 2.0)
                   - 0.5 * torch.log(nu * math.pi)
                   - (nu + 1.0) / 2.0 * torch.log1p(w * w / nu))
        pdf = torch.exp(log_pdf)

        def log_beta(p, q):
            return torch.lgamma(p) + torch.lgamma(q) - torch.lgamma(p + q)

        half = torch.full_like(nu, 0.5)
        log_const = (0.5 * torch.log(nu) + log_beta(half, nu - 0.5)
                     - 2.0 * log_beta(half, nu / 2.0))
        const = 2.0 * torch.exp(log_const) / (nu - 1.0)
        return sigma * (w * (2.0 * cdf - 1.0)
                        + 2.0 * pdf * (nu + w * w) / (nu - 1.0) - const)

    def selective_pnl(self, y, mu, sigma, y_old):
        """Profit from a position sized by signal-to-noise, in USD per unit of capital.

        The divisor is the model's own sigma, so a day it is unsure about takes a small
        position on its own — that is the abstention this round is testing, learned
        rather than applied afterwards as a rule.

        The divisor is the predictive *standard deviation*, not the head's scale — for a
        Student-t those differ by sqrt(nu/(nu-2)), and dividing by the scale would run a
        heavy-tailed arm at an effective kappa its Gaussian counterpart never sees (A7).

        ``sigma`` is detached. Left attached, this term's gradient pushes sigma down on
        every day the model called correctly, and measured against the CRPS scale force
        (bounded at 0.2337) it wins at any lambda above about 0.22 — well below the one
        this round uses. Detaching makes that exactly zero rather than merely small, so
        sigma is shaped only by the proper scoring rule. Abstention is unaffected: sigma
        still sets position size, it just stops being trainable through this path.
        """
        denominator = self.predictive_sd(sigma.detach())
        position = torch.tanh((mu - y_old) / (self.selective_kappa * denominator))
        return ((y - y_old) * position).mean()

    def _dist_metrics(self, batch):
        """Shared by train and val so the two can never drift apart."""
        y = batch[self.y_key]
        y_old = batch[f'{self.y_key}_old']
        mu, sigma, scales, weights = self.forward_dist_full(batch['features'], y_old)
        y, mu = self.denormalize(y, mu)
        if self.mixture:
            per_day = self.mixture_crps(y, mu, scales, weights)
        elif self.heavy_tail:
            per_day = self.student_t_crps(y, mu, sigma, self.nu)
        else:
            per_day = self.gaussian_crps(y, mu, sigma)
        return y, mu, sigma, per_day.mean()

    def forward(self, x, y_old=None):
        if self.mode == 'default':
            return self.model(x).reshape(-1)
        elif self.mode == 'diff':
            return self.model(x).reshape(-1) + y_old
        elif self.mode == 'ret':
            # model outputs a relative return r_hat; reconstruct price around the
            # last observed close so the network never has to model the price level
            return y_old * (1.0 + self.model(x).reshape(-1))
        elif self.mode == 'scaled_log_return':
            raw_standardized = self.model(x).reshape(-1)
            predicted_log_return = (
                self.return_center + self.return_scale * raw_standardized
            ).clamp(-self.return_clip, self.return_clip)
            return y_old * torch.exp(predicted_log_return)

    def return_space_metrics(self, y, y_old, raw_standardized):
        """Score a scalar output against train-standardized next-close log return.

        ``return_center`` and ``return_scale`` are frozen train-only statistics. The
        model emits the standardized value directly, so the loss is independent of
        Bitcoin's absolute dollar price. Price reconstruction remains available for
        the existing prediction and next-open trading boundary.
        """
        if (y <= 0.0).any() or (y_old <= 0.0).any():
            raise ValueError('return-space targets and previous closes must be positive')
        target_log_return = torch.log(y / y_old)
        target_standardized = (
            target_log_return - self.return_center
        ) / self.return_scale
        return_mse = F.mse_loss(raw_standardized, target_standardized)
        return_huber = F.huber_loss(
            raw_standardized, target_standardized, delta=1.0
        )
        predicted_log_return = (
            self.return_center + self.return_scale * raw_standardized
        ).clamp(-self.return_clip, self.return_clip)
        predicted_close = y_old * torch.exp(predicted_log_return)
        loss = return_mse if self.loss == 'return_mse' else return_huber
        return {
            'loss': loss,
            'return_mse': return_mse,
            'return_huber': return_huber,
            'predicted_close': predicted_close,
            'predicted_log_return': predicted_log_return,
            'target_log_return': target_log_return,
        }

    def _scaled_log_return_step(self, batch, prefix):
        x = batch['features']
        y = batch[self.y_key]
        y_old = batch[f'{self.y_key}_old']
        if self.batch_size is None:
            self.batch_size = x.shape[0]
        y, y_old = self.denormalize(y, y_old)
        raw_standardized = self.model(x).reshape(-1)
        metrics = self.return_space_metrics(y, y_old, raw_standardized)
        predicted_close = metrics['predicted_close']
        price_rmse = torch.sqrt(self.mse(predicted_close, y))
        direction = (
            (metrics['predicted_log_return'] > 0.0)
            == (metrics['target_log_return'] > 0.0)
        ).float().mean()
        predicted_up = (metrics['predicted_log_return'] > 0.0).float().mean()
        self.log(f'{prefix}/return_mse', metrics['return_mse'].detach(),
                 batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        self.log(f'{prefix}/return_huber', metrics['return_huber'].detach(),
                 batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log(f'{prefix}/rmse', price_rmse.detach(),
                 batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log(f'{prefix}/direction', direction.detach(),
                 batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        self.log(f'{prefix}/predicted_up', predicted_up.detach(),
                 batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        return metrics
        
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

        if self.mode == 'scaled_log_return':
            return self._scaled_log_return_step(batch, 'train')['loss']

        if self.distributional:
            y, mu, sigma, crps = self._dist_metrics(batch)
            self.log("train/crps", crps.detach(), batch_size=self.batch_size,
                     sync_dist=True, prog_bar=True)
            self.log("train/rmse", torch.sqrt(self.mse(mu, y)).detach(),
                     batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            self.log("train/sigma", sigma.mean().detach(), batch_size=self.batch_size,
                     sync_dist=True, prog_bar=False)
            if self.heavy_tail:
                self.log("train/nu", self.nu.detach(), batch_size=self.batch_size,
                         sync_dist=True, prog_bar=False)
            if self.selective_lambda > 0:
                y_old_denorm, _ = self.denormalize(y_old, y_old)
                selective = self.selective_pnl(y, mu, sigma, y_old_denorm)
                position = torch.tanh((mu - y_old_denorm)
                                      / (self.selective_kappa * sigma.detach()))
                self.log("train/selective", selective.detach(),
                         batch_size=self.batch_size, sync_dist=True, prog_bar=True)
                # The registered disqualifier: a saturated or collapsed position means
                # the gate has stopped discriminating, whatever CRPS says.
                self.log("train/abs_position", position.abs().mean().detach(),
                         batch_size=self.batch_size, sync_dist=True, prog_bar=False)
                return crps - self.selective_lambda * selective
            return crps

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

        if self.madl_lambda > 0:
            y_old_denorm, _ = self.denormalize(y_old, y_old)
            pnl = self.trade_pnl(y, y_hat, y_old_denorm)
            self.log("train/pnl", pnl.detach(), batch_size=self.batch_size,
                     sync_dist=True, prog_bar=True)
            total = total + self.madl_lambda * (-pnl)

        if self.dir_loss_lambda > 0:
            y_old_denorm, _ = self.denormalize(y_old, y_old)
            r_hat = (y_hat - y_old_denorm) / y_old_denorm.clamp(min=1e-8)
            target_up = (y > y_old_denorm).float()
            dir_loss = F.binary_cross_entropy_with_logits(self.dir_loss_scale * r_hat, target_up)
            self.log("train/dir_loss", dir_loss.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            total = total + self.dir_loss_lambda * dir_loss
        return total
        
    
    def validation_step(self, batch, batch_idx):
        x = batch['features']
        y = batch[self.y_key]
        y_old = batch[f'{self.y_key}_old']
        if self.batch_size is None:
            self.batch_size = x.shape[0]

        if self.mode == 'scaled_log_return':
            metrics = self._scaled_log_return_step(batch, 'val')
            return {"val_loss": metrics['loss']}

        if self.distributional:
            y, mu, sigma, crps = self._dist_metrics(batch)
            rmse = torch.sqrt(self.mse(mu, y))
            # Fraction of outcomes inside the 90% interval — a calibrated forecast
            # sits near 0.90. Far below means the spread is too tight to trust, far
            # above means it is padded and the CRPS gain is not real skill.
            coverage = ((y - mu).abs() <= 1.6448536 * sigma).float().mean()
            self.log("val/crps", crps.detach(), batch_size=self.batch_size,
                     sync_dist=True, prog_bar=True)
            self.log("val/rmse", rmse.detach(), batch_size=self.batch_size,
                     sync_dist=True, prog_bar=True)
            self.log("val/coverage90", coverage.detach(), batch_size=self.batch_size,
                     sync_dist=True, prog_bar=False)
            self.log("val/sigma", sigma.mean().detach(), batch_size=self.batch_size,
                     sync_dist=True, prog_bar=False)
            return {"val_loss": crps}

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
        if self.mode == 'scaled_log_return':
            metrics = self._scaled_log_return_step(batch, 'test')
            return {"test_loss": metrics['loss']}
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
