"""
HyperParameter: Named Hyper-Parameters with Default Values

The Python counterpart of ScalaTion's `scalation.HyperParameter`: a map from a hyper-parameter
name to its VALUE that also remembers each registered parameter's DEFAULT (ScalaTion's
`hp += ("p", 1, 1)` registers p with value 1 and default 1; `hp("p") = 6` changes the value only).
It prints as ScalaTion prints it in a REPORT block,

    HyperParameter(p -> (6,1), d -> (1,1), q -> (1,1), P -> (1,1), D -> (1,1), Q -> (1,1), s -> (7,7),
                   px -> (4,1), spec -> (1,1), crx -> (0,0), nneg -> (1,1))

listing the registered parameters only (value, default).  It is a `dict` of the values, so
`hp["p"]`, `hp.get("crx", 0)`, `{**hp}` and `dict(hp)` work as for any dict; entries set without
registration (the Python-only "skip", or "lambda" of the regression) are values without a default
and are not printed, as in ScalaTion, where they live in other objects.

`freeze` makes one read-only (Python only): `AR.hp`, the defaults of every forecaster, is frozen, so
no model or script can change the defaults of the models made after it -- the values of a model are
set in its `Args` (modeling/forecasting/args.py); `copy` gives a writable copy.

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.5
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional


def _num_str(v: Any, like: Any = None) -> str:
    """A value as ScalaTion prints it in a HyperParameter: an integer (or a bool) as an integer, a
    double as Java's `Double.toString` (an integral double registered as an integer stays one)."""
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v.is_integer() and (like is None or isinstance(like, (int, bool))):
            return str(int(v))
        return repr(v)
    return str(v)


def _rebuild(values: Dict[str, Any], defaults: Dict[str, Any], frozen: Optional[str]) -> "HyperParameter":
    """Unpickles a HyperParameter (its values, defaults and read-only state)."""
    hp = HyperParameter(values, defaults)
    hp._frozen = frozen
    return hp


class HyperParameter(dict):
    """
    A dict of hyper-parameter values that also keeps the default of each registered parameter.

    Examples
    --------
    hp = HyperParameter()
    hp.add("p", 1, 1)          # ScalaTion: hp += ("p", 1, 1)
    hp["p"] = 6                # ScalaTion: hp("p") = 6
    print(hp)                  # HyperParameter(p -> (6,1))
    """

    def __init__(self, values: Optional[Mapping[str, Any]] = None,
                 defaults: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(values or {})
        self._defaults: Dict[str, Any] = dict(defaults or {})
        self._frozen: Optional[str] = None                             # the message when read-only

    # -- read-only (Python only) -------------------------------------------------------------

    def freeze(self, message: str = "this HyperParameter is read-only") -> "HyperParameter":
        """Makes this object read-only: every change raises TypeError(message); `copy` is writable."""
        self._frozen = message
        return self

    @property
    def frozen(self) -> bool:
        """Whether this object is read-only."""
        return self._frozen is not None

    def _check(self) -> None:
        msg = getattr(self, "_frozen", None)                         # (absent while unpickling)
        if msg is not None:
            raise TypeError(msg)

    def __reduce__(self):
        return _rebuild, (dict(self), dict(self._defaults), self._frozen)

    def __setitem__(self, key: str, value: Any) -> None:
        self._check()
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        self._check()
        super().__delitem__(key)

    def update(self, *args, **kwargs) -> None:
        self._check()
        super().update(*args, **kwargs)

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self._check()
        return super().setdefault(key, default)

    def pop(self, key: str, *default) -> Any:
        self._check()
        return super().pop(key, *default)

    def popitem(self):
        self._check()
        return super().popitem()

    def clear(self) -> None:
        self._check()
        super().clear()

    def __ior__(self, other):
        self._check()
        return super().__ior__(other)

    # -- ScalaTion's API ---------------------------------------------------------------------

    def add(self, name: str, value: Any, default: Any = None) -> "HyperParameter":
        """Registers a hyper-parameter with its value and default (ScalaTion's `hp += (name, value, default)`)."""
        self[name] = value
        self._defaults[name] = value if default is None else default
        return self

    def default(self, name: str) -> Any:
        """The default of a registered hyper-parameter (KeyError for an unregistered one)."""
        return self._defaults[name]

    def registered(self) -> Iterable[str]:
        """The names of the registered hyper-parameters, in registration order."""
        return self._defaults.keys()

    def copy(self) -> "HyperParameter":
        """A (writable) copy with the same values and defaults."""
        return HyperParameter(self, self._defaults)

    def updateReturn(self, *pairs, **nvs) -> "HyperParameter":
        """A copy with the given values changed (ScalaTion's `updateReturn (("p", 0), ("px", 0))`)."""
        hp = self.copy()
        for k, v in pairs:
            hp[k] = v
        hp.update(nvs)
        return hp

    def __repr__(self) -> str:
        items = (f"{k} -> ({_num_str(self.get(k, d), d)},{_num_str(d)})" for k, d in self._defaults.items())
        return "HyperParameter(" + ", ".join(items) + ")"

    __str__ = __repr__
