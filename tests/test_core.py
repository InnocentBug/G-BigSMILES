import pytest

from gbigsmiles import Atom, GBigSMILESInitNotEnoughError, GBigSMILESInitTooMuchError


def test_core_errors():
    Atom("C")
    with pytest.raises(GBigSMILESInitTooMuchError):
        try:
            Atom("C", ["C"])
        except Exception as exc:
            print(exc)
            raise exc

    with pytest.raises(GBigSMILESInitNotEnoughError):
        try:
            Atom()
        except Exception as exc:
            print(exc)
            raise exc
