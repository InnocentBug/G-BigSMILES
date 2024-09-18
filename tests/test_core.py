import pytest

from gbigsmiles import Atom, GBigSMILESInitNotEnoughError, GBigSMILESInitTooMuchError


def test_core_errors():
    Atom("C")
    with pytest.raises(GBigSMILESInitTooMuchError):
        Atom("C", ["C"])
    with pytest.raises(GBigSMILESInitNotEnoughError):
        Atom()
