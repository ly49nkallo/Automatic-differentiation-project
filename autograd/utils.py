from .tensor import Array_like
from .module import Module
from typing import Optional, Union
from pathlib import Path
import warnings
import os
import pickle
import numpy as np

def moving_average(a:Array_like, n=3) :
    #if not isinstance(a, Array_like): raise TypeError("Argument must be an Array_like")
    if not isinstance(a, np.ndarray): a = np.array(a)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def serialize_model(model: Module, file_path:Optional[Union[str, Path]] = None) -> tuple:

    '''Turn model into serializable data and store in filesystem
        Args:
            model: Model instance to be stored
            file_path (Optional[str]): file path to store model
                Defaults to current directory in a folder called \"saved_models\"
                
        Yields:
            Code (bool, size): Returns whether the process was sucessful
                                Returns size of serialized file in bytes'''
    
    
    if not isinstance(model, Module): raise TypeError(f"model must be an instance of autograd.Module, got {type(model)}")
    if file_path is None: file_path = f"{os.getcwd()}/saved_models/{model.__class__.__name__}.pkl"
    file = Path(file_path)
    if file.is_file(): warnings.warn(f"Overwriting file at {str(file)}")
    if os.path.isdir(Path(f"{os.getcwd()}/saved_models")) == False: 
        os.mkdir(Path(f"{os.getcwd()}/saved_models"))
    open(file,'w')
    with open(file, 'wb') as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)
    return True

def load_model(file_path:Optional[Union[str, Path]] = None, name:Optional[str] = None) -> Module:
    if file_path is None and name is None: raise NameError("Cannot load model without either name or filepath")
    if file_path is not None:
        with open(Path(file_path), 'rb') as inp:
            return pickle.load(inp)
    elif name is not None:
        with open(Path(f"{os.getcwd()}/saved_models/{name}.pkl"), 'rb') as inp:
            return pickle.load(inp)
    
    

