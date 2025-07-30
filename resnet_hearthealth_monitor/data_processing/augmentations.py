import numpy as np
import random
from scipy.signal import resample

def gaussian_noise(x, noise_factor=0.05):    
    noise = np.random.normal(0, noise_factor, x.shape)
    return x + noise

def baseline_wander(x, wander_factor=0.05, frequency=0.5):
    """
    Adds a low-frequency sinusoidal drift to the input signal.
    
    Parameters:
        x (np.array): 1D array representing the ECG signal.
        wander_factor (float): The amplitude of the drift.
        frequency (float): The frequency (in Hz) of the drift. 
                           Lower values yield a slowly varying drift.
    
    Returns:
        np.array: The signal with added baseline wander.
    """
    n = x.shape[0]
    # Create a time vector spanning 0 to 1 second (assuming x covers 1 second).
    t = np.linspace(0, 1, n)
    # Generate a low-frequency sinusoidal drift.
    drift = wander_factor * np.sin(2 * np.pi * frequency * t)
    return x + drift

def stretch(x):
    # Determine the series length:
    # If x is 1D, use x.shape[0]; if it's 2D, assume the series length is in axis 1.
    n = x.shape[0] if len(x.shape) == 1 else x.shape[1]
    
    # Generate a random multiplier between 0.9 and 1.1.
    multiplier = random.uniform(0.9, 1.1)
    l = int(n * multiplier)
    
    # Resample x to length l. (Make sure `resample` is imported from scipy.signal or defined elsewhere.)
    x1 = resample(x, l)
    
    # Adjust y to have exactly n elements:
    if l < n:
        x1_ = np.zeros(shape=(n,))
        x1_[:l] = x1
    else:
        x1_ = x1[:n]
    return x1_

def amplify(x, factor=None):
    """
    Amplifies the input signal x.
    
    If no factor is provided, a random factor is computed based on x and a random offset.
    Otherwise, the provided factor is used.
    
    Parameters:
        x (array-like): The input signal.
        factor (float, optional): The amplification factor. Defaults to None.
    
    Returns:
        array-like: The amplified signal.
    """
    alpha = random.random() - 0.5
    if factor is None:
        # Compute a random amplification factor; note that the formula below is applied elementwise
        factor = -alpha * x + (1 + alpha)
    return x * factor

class Gaussian_Noise:
    def __init__(self, noise_factor=0.05):
        self.noise_factor = noise_factor
    
    def __call__(self, x):
        return gaussian_noise(x, self.noise_factor)
    
    def __repr__(self):
        return f'Gaussian_Noise'

class Baseline_Wander:
    def __init__(self, wander_factor=0.05, frequency=0.5):
        self.wander_factor = wander_factor
        self.frequency = frequency
    
    def __call__(self, x):
        return baseline_wander(x, self.wander_factor, self.frequency)
    
    def __repr__(self):
        return f"Baseline_Wander"
    
class Stretch:
    def __init__(self):
        pass
    
    def __call__(self,x):
        return stretch(x)
    
    def __repr__(self):
        return 'Stretch'

class Amplify:
    def __init__(self, factor=None):
        """
        Initialize the Amplify augmentation.
        
        Parameters:
            factor (float, optional): Default amplification factor to use. If None,
                                      a random factor is used during each call.
        """
        self.factor = factor
    
    def __call__(self, x, factor=None):
        """
        Amplify the input signal x.
        
        Parameters:
            x (array-like): The input signal.
            factor (float, optional): An optional amplification factor. If not provided,
                                      self.factor is used.
        
        Returns:
            array-like: The amplified signal.
        """
        if factor is None:
            factor = self.factor
        return amplify(x, factor)
    
    def __repr__(self):
        return f"Amplify"

class Amplify:
    def __init__(self, amplify_factor=1.0):
        self.amplify_factor = amplify_factor
    
    def __call__(self, x):
        return amplify(x, self.amplify_factor)
    
    def __repr__(self):
        return f'Amplify'


class Augment:
    def __init__(self, augmentation_list, return_prints = False):
        self.augmentation_list = augmentation_list
        self.return_prints = return_prints
        
    def __call__(self, x):
        augmentations_performed = ''
        
        for augmentation in self.augmentation_list:
            if np.random.binomial(1, 0.5) == 1:
                x = augmentation(x)
                augmentations_performed += f'{augmentation} '
                
        if not self.return_prints:
            return x
        return x, augmentations_performed
    
def generate_holdout_using_augmentations(x, shift_config):
    """
    Applies a sequence of data shifts/augmentations to a single time series sample.
    
    Parameters:
        x (array-like): 1D array of features (e.g., time series data).
        shift_config (dict): Dictionary containing augmentation parameters:
            - wander_factor: amplitude for baseline wander.
            - frequency: frequency for baseline wander.
            - noise_factor: standard deviation for Gaussian noise.
            - amplify_factor: factor to scale the signal.
    
    Returns:
        np.array: The augmented time series.
    """
    # Ensure x is a numpy array of type float32.
    x = np.array(x, dtype=np.float32)
    
    # Apply baseline wander augmentation.
    x = baseline_wander(x, wander_factor=shift_config.get("wander_factor", 0.05),
                        frequency=shift_config.get("frequency", 0.5))
    
    # Apply Gaussian noise augmentation.
    x = gaussian_noise(x, noise_factor=shift_config.get("noise_factor", 0.05))
    
    # Apply amplitude amplification.
    x = amplify(x, factor=shift_config.get("amplify_factor", 1.2))
    
    # Apply time stretching/shrinking.
    x = stretch(x)
    
    return x