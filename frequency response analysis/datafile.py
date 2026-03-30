import numpy as np
import pandas as pd

try:
    from lmfit import Parameters
    HAS_LMFIT = True
except ImportError:
    HAS_LMFIT = False


STATE_NAMES = [
    "S", "Sp", "R", "Rp", "Ac", "M", "C_tic", "P", "Pm", "C", "I"
]


DEFAULT_PARAM_ORDER = [
    "k_green", "k_red", "b_green", "b_red",
    "k_sp_b", "k_sp_u", "k_rp_b", "k_rp_u",
    "beta", "l0", "Kc", "d_m", "k_tl",
    "k_tli_b", "k_tli_u", "d_p", "k_fold", "b_fold",
    "n_gamma", "R_max",
    "S_0", "R_0", "Sp_0", "Rp_0", "mRNA_0",
    "P_0", "Pm_0", "k_gr", "C_max", "C_0", "n_tcs"
]


class SimpleParams:
    def __init__(self):
        self._vals = {}

    def add(self, name, value):
        self._vals[name] = float(value)

    def valuesdict(self):
        return dict(self._vals)

    def copy(self):
        out = SimpleParams()
        out._vals = self._vals.copy()
        return out


def _make_params():
    if HAS_LMFIT:
        return Parameters()
    return SimpleParams()


def load_guess_file(guess_file="guess.txt", param_names=None):
    """
    Loads parameter guesses from a plain text file like MATLAB dlmread('guess.txt','\\t').

    If the file contains a single column, values are matched to param_names in order.
    If it contains at least two columns and the first column is text, you should instead
    use a CSV loader tailored to your file format. This function assumes numeric values only.
    """
    if param_names is None:
        param_names = DEFAULT_PARAM_ORDER

    raw = np.loadtxt(guess_file, delimiter=None)
    raw = np.asarray(raw).reshape(-1)

    if len(raw) < len(param_names):
        raise ValueError(
            f"guess.txt has {len(raw)} values but {len(param_names)} parameter names are required."
        )

    params = _make_params()
    for name, val in zip(param_names, raw[:len(param_names)]):
        params.add(name, value=float(val))

    return params


def load_parameter_csv(csv_file):
    """
    Flexible CSV loader for a file with columns like:
      parameter,value
    or
      name,value
    """
    df = pd.read_csv(csv_file)

    possible_name_cols = ["parameter", "param", "name", "Parameter", "Name"]
    possible_value_cols = ["value", "Value", "guess", "Guess"]

    name_col = None
    value_col = None

    for c in possible_name_cols:
        if c in df.columns:
            name_col = c
            break

    for c in possible_value_cols:
        if c in df.columns:
            value_col = c
            break

    if name_col is None or value_col is None:
        raise ValueError(
            "Could not detect parameter-name and value columns in CSV."
        )

    params = _make_params()
    for _, row in df.iterrows():
        params.add(str(row[name_col]), value=float(row[value_col]))

    return params


def build_x0_from_params(params):
    p = params.valuesdict()

    x0 = np.array([
        p["S_0"],
        p["Sp_0"],
        p["R_0"],
        p["Rp_0"],
        0.0,
        p["mRNA_0"],
        0.0,
        p["P_0"],
        p["Pm_0"],
        p["C_0"],
        0.0
    ], dtype=float)

    return x0


def datafile(
    guess_file="guess.txt",
    param_csv=None,
    t_i=0.0,
    t_f=960.0,
    t_inc=1.0,
    input_mode="dark",
    setpoint=0.0
):
    """
    MATLAB-style data container for the nonlinear TCS model.
    """
    tspan = np.arange(t_i, t_f + t_inc, t_inc)

    if param_csv is not None:
        params = load_parameter_csv(param_csv)
    else:
        params = load_guess_file(guess_file=guess_file)

    x0 = build_x0_from_params(params)

    DF = {
        "Initial_Parameters": params,
        "Initial_Conditions": x0,
        "t_i": t_i,
        "t_f": t_f,
        "t_inc": t_inc,
        "nstep": len(tspan) - 1,
        "tspan": tspan,
        "ODE_size": len(STATE_NAMES),
        "State_Names": STATE_NAMES,
        "input_mode": input_mode,
        "setpoint": setpoint,
    }

    return DF