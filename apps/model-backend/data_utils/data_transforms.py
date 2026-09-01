import torch

from data_utils.exogenous import validate_features as validate_exogenous_features

PRICE_KEYS = ('Open', 'High', 'Low', 'Close')
DERIVED_FEATURES = ('ret1', 'hl_range', 'co_gap')


class DataTransform:
    def __init__(self, is_train, use_volume=False, additional_features=[],
                 drop_timestamp=False, log_volume=False, window_norm=False,
                 derived_features=[], exogenous_features=None):
        """Defaults reproduce the paper pipeline exactly.

        Hygiene flags (all default False/empty, opt-in via training config `feature_flags`):
          drop_timestamp: keep Timestamp in the output dict but exclude the raw epoch
                          seconds (~1.6e9) from the feature tensor.
          log_volume:     Volume feature becomes log1p(Volume / 1e9).
          window_norm:    price channels become x / last_observed_close - 1 (per-window
                          anchor); targets stay in raw price space (pair with mode='ret').
          derived_features: extra stationary channels computed INSIDE the window (no new
                          CSV columns, no look-ahead): 'ret1' daily Close return (first
                          day padded 0), 'hl_range' (High-Low)/Close, 'co_gap' Close/Open-1.
          exogenous_features: Track-3 channels already merged into the split dataframe
                          (data_utils/exogenous.py; statically transformed + t-1 shifted).
                          Under window_norm each becomes x - x[last observed day], the
                          anchored analogue of the price scheme (safe when values near 0).
        """
        self.is_train = is_train
        self.keys = ['Timestamp', 'Open', 'High', 'Low', 'Close']
        if use_volume:
            self.keys.append('Volume')
        self.keys += additional_features
        self.drop_timestamp = drop_timestamp
        self.log_volume = log_volume
        self.window_norm = window_norm
        unknown = set(derived_features) - set(DERIVED_FEATURES)
        if unknown:
            raise ValueError(f'Unknown derived_features: {sorted(unknown)}')
        self.derived_features = list(derived_features)
        self.exogenous_features = list(exogenous_features or [])
        validate_exogenous_features(self.exogenous_features)
        print(self.keys + self.derived_features + self.exogenous_features)


    def __call__(self, window):
        data_list = []
        output = {}
        if 'Timestamp_orig' in window.keys():
            self.keys.append('Timestamp_orig')
        anchor = None
        if self.window_norm:
            close = torch.tensor(window.get('Close').tolist())
            anchor = close[-2]  # last close the model is allowed to see
        for key in self.keys:
            data = torch.tensor(window.get(key).tolist())
            if key == 'Volume':
                data /= 1e9
            output[key] = data[-1]
            output[f'{key}_old'] = data[-2]
            if key == 'Timestamp_orig':
                continue
            feat = data[:-1]
            if key == 'Timestamp' and self.drop_timestamp:
                continue
            if key == 'Volume' and self.log_volume:
                feat = torch.log1p(feat)
            if key in PRICE_KEYS and self.window_norm:
                feat = feat / anchor - 1.0
            data_list.append(feat.reshape(1, -1))
        if self.derived_features:
            raw = {k: torch.tensor(window.get(k).tolist()) for k in PRICE_KEYS}
            c, o, h, l = raw['Close'], raw['Open'], raw['High'], raw['Low']
            w = c.shape[0] - 1  # feature days are rows 0..w-1; row w is the target day
            ret1 = torch.zeros(w)
            ret1[1:] = c[1:w] / c[:w - 1] - 1.0
            derived = {
                'ret1': ret1,
                'hl_range': (h[:w] - l[:w]) / c[:w],
                'co_gap': c[:w] / o[:w] - 1.0,
            }
            for name in self.derived_features:
                data_list.append(derived[name].reshape(1, -1))
        for name in self.exogenous_features:
            data = torch.tensor(window.get(name).tolist())
            feat = data[:-1]  # rows 0..w-1; the target day never enters features
            if self.window_norm:
                feat = feat - feat[-1]  # anchor = last observed day, like prices
            data_list.append(feat.reshape(1, -1))
        features = torch.cat(data_list, 0)
        output['features'] = features
        return output

    def set_initial_seed(self, seed):
        self.rng.seed(seed)
