import numpy as np
from typing import Optional, Dict, Union, Set, Iterator, Callable, Any, Tuple
from autograd.tensor import Tensor
from autograd.parameter import Parameter
from collections import OrderedDict
# import autograd.functional as F
# @TODO Module.zero_grad DOES NOT WORK AT ALL
def _forward_unimplmented(self, *input: Any) -> None:
    r'''This gets called as a placeholder if the programmer forgets
         to implement a forward method (it is required!)'''
    raise NotImplementedError('forward was not implemented >:(')

class Module:
    def __init__(self) -> None:
        self.training = True
        self._parameters: Dict[str, Optional[Parameter]] = OrderedDict()
        self._modules: Dict[str, Optional['Module']] = OrderedDict()

    def __call__(self, x):
        self.forward(x)

    forward: Callable[..., Any] = _forward_unimplmented
    data:np.ndarray
    
    # scalped pytorch code :)

    def _call_impl(self, *input, **kwargs):
        forward_call = self.forward
        # assuming we have no forwards or backwards hooks bc i'm to lazy to figure out how to implement them
        return forward_call(*input, **kwargs)

    __call__ : Callable[..., Any] = _call_impl

    def __repr__(self) -> str:
        if len(list(self.named_modules())) != 1: return f"Module(training={self.training}, \
                                                    modules={repr([x for x, n in self.named_modules() if x != ''])}, \
                                                    parameters={repr([x for x, n in self.named_parameters() if x != ''])})"
        else: return f"Module(training={self.training}, \
                                                    parameters={repr([x for x, n in self.named_parameters() if x != ''])})"
    def register_parameter(self, name, param:Optional[Parameter]) -> None:
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")

        elif not isinstance(name, str):
            raise TypeError("parameter name should be a string. "
                            "Got {}".format(type(name)))
        elif '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError("cannot assign '{}' object to parameter '{}' "
                            "(torch.nn.Parameter or None required)"
                            .format(type(param), name))
        else:
            self._parameters[name] = param
        '''
        elif param.grad_fn:
            raise ValueError(
                "Cannot assign non-leaf Tensor to parameter '{0}'. Model "
                "parameters must be created explicitly. To express '{0}' "
                "as a function of another Tensor, compute the value in "
                "the forward() method.".format(name))
        '''

    def add_module(self, name: str, module) -> None:
        r"""Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(
                type(module)))
        elif not isinstance(name, str):
            raise TypeError("module name should be a string. Got {}".format(
                type(name)))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\", got: {}".format(name))
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        self._modules[name] = module

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '', remove_duplicate: bool = True):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.
        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
            or not
        Yields:
            (string, Module): Tuple of name and module
        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.
        """
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix, remove_duplicate):
                    yield m

    def modules(self) -> Iterator['Module']:
        r"""Returns an iterator over all modules in the network.
        Yields:
            Module: a module in the network
        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.
        """

        for _, module in self.named_modules():
            yield module

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        r"""Returns an iterator over module parameters.

        This is typically passed to an optimizer.

        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            Parameter: module parameter

        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        r"""Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            (string, Parameter): Tuple containing the name and parameter

        Example::

            >>> for name, param in self.named_parameters():
            >>>    if name in ['bias']:
            >>>        print(param.size())

        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem


    def _named_members(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getattr__(self, name: str) -> Union[Tensor, 'Module']:
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__,  self._modules)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(Parameter or None expected)"
                                .format(type(value), name))
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters )
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(Module or None expected)"
                                    .format(type(value), name))
                modules[name] = value
            else:
                object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        keys = module_attrs + attrs + parameters + modules

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def zero_grad(self, set_to_none:bool = False) -> None:
        '''Zero out every parameter's gradient that is stored in this optimizer'''
        for name, parameter in self.named_parameters():
            if set_to_none:
                parameter.grad = None
            else:
                # print(f'set parameter \"{name}  .grad.zero_grad()')
                parameter.zero_grad()
        # recursivly navigtate and zero out sub module gradient parameters as wwell
        for module_name, module in self.named_modules():
            if module_name == '': continue # skip the self-reference
            module.zero_grad(set_to_none=set_to_none)

class Linear(Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert isinstance(self.in_features, int) and isinstance(self.out_features, int), "features count must be an integer"
        self.w = Parameter(self.in_features, self.out_features)
        self.b = Parameter(self.out_features)

    def forward(self, x):
        #self.w.shape == (batch_size*data or 'in_features', batch_size*out_features)
        #x.shape == (batch_size, in_fetures)
        out = x @ self.w + self.b
        #assert out.shape == (32, 10), out.shape
        return out

class Dropout(Module):
    def __init__(self, discard_rate:float) -> None:
        super().__init__()
        self.discard_rate = discard_rate
    
    def forward(self, x):
        ...