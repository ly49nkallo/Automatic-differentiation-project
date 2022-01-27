import numpy as np

class Tensor():
    def __init__(self, data):
        if isinstance(data, list):
            try:
                self.data:np.ndarray = np.asarray(data)
            except:
                raise NameError('data must be castable into numpy.ndarray')
        elif isinstance(data, np.ndarray):
            self.data:np.ndarray = data
        else:
            raise AttributeError('data must be either a ndarray or list, got {0}'.format(type(data)))
        self.dtype = self.data.dtype
        

    def is_parameter(self):
        return False

    def shape(self):
        return self.data.shape
    
    def size(self):
        return self.data.size

    def __repr__(self):
        return '<<' + repr(self.shape()) + f' dtype = {self.data.dtype}' + repr(self.__class__) + '>>'
        
class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data)
    
    def is_parameter(self):
        return True


