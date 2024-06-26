import warnings
import hashlib
from numba.core.registry import CPUDispatcher, cpu_target
from numba.core import types, dispatcher
from numba.core.caching import FunctionCache
from numba.core.serialize import dumps
from numba.core.target_extension import (
    dispatcher_registry,
    target_registry,
    CPU,
)


class ZFunctionCache(FunctionCache):
    """
    How to Use Z Function Caching in Numba is a hackish way to cache Higher
    Order Functions (HOPs). Internally this will change how Numba compute the
    caching key when a function is involved in an overload signature. For more
    info, see how the `_index_key` is implemented.

    ### How to use it

    The main change involves importing the `custom_dispatcher.py` file and
    modifying the `@jit` decorator to include `_target="Z"`.

    1. Import `custom_dispatcher.py` to import the modified dispatcher to cache
    higher order string functions.

    2. Apply the `@jit` decorator to your function, adding the `_target="Z"`
    parameter to enable Z Function caching. Here is an example:

    ```python
    import custom_dispatcher

    @jit(nopython=True)
    def pow(x):
        return x ** 2

    @jit(_target="Z", cache=True, nopython=True)
    def my_function(f, x):
        return f(x) + 2 * x + 1
    ```

    Targetting the "Z" dispatcher, "my_function" will automatically be able
    to cache not only string functions but any HOP. Use with
    caution!!!
    """

    def _index_key(self, sig, codegen):
        def _is_first_class_function(obj):
            if isinstance(obj, (tuple, list)):
                return any(map(_is_first_class_function, obj))
            elif isinstance(obj, dispatcher.Dispatcher):
                return _is_first_class_function(obj._type)
            return isinstance(obj, types.Dispatcher)

        def _compute_custom_key(typ):
            if hasattr(typ, "py_func"):
                py_func = typ.py_func
            else:
                py_func = typ.key().py_func
            return (py_func.__module__, py_func.__qualname__)

        def map_only(types, func, iterable):
            return tuple(
                [
                    func(i)
                    if isinstance(i, types)
                    else (
                        map_only(types, func, i) if isinstance(i, (tuple, list)) else i
                    )
                    for i in iterable
                ]
            )

        key = super()._index_key(sig, codegen)
        if any(map(_is_first_class_function, sig)):
            sig = map_only(types.Dispatcher, _compute_custom_key, sig)
            key = (sig,) + key[1:]

        cvars = ()
        if self._py_func.__closure__ is not None:
            cvars = tuple([x.cell_contents for x in self._py_func.__closure__])
            if any(map(_is_first_class_function, cvars)):
                cvars = map_only(dispatcher.Dispatcher, _compute_custom_key, cvars)
                cvarbytes = dumps(cvars)
                hasher = lambda x: hashlib.sha256(x).hexdigest()
                key = key[:-1] + (key[-1][0], hasher(cvarbytes))
        return key


class Z(CPU): ...


class ZDispatcher(CPUDispatcher):
    targetdescr = cpu_target

    def enable_caching(self):
        self._cache = ZFunctionCache(self.py_func)


target_registry["Z"] = Z
dispatcher_registry[target_registry["Z"]] = ZDispatcher
