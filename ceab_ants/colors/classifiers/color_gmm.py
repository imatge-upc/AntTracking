
import dill
from pathlib import Path
from sklearn import mixture


class ColorGMM():

    def __init__(self, color=None, num_gauss=4, cov_type="full", max_iter=1000, n_init=10, verbose=False, verbose_interval=10):
        self.color = color

        self.model = mixture.GaussianMixture(
            # means_init=[[30, 30, 30], [34, 34, 34], [126, 126, 126], [130, 130, 130]]
            n_components=num_gauss, 
            covariance_type=cov_type, 
            max_iter=max_iter, 
            n_init=n_init, 
            verbose=verbose,
            verbose_interval=verbose_interval
        )
        self.model.n_features_in_ = 3 
        self.model.feature_names_in_ = ["red", "green", "blue"]
    
    def train(self, data):
        self.model.fit(data) # np.concatenate(data) # data = img.reshape(-1, 3)
    
    def score(self, data):
        return self.model.score(data) # data = img.reshape(-1, 3)

    def save(self, basename, color=None):
        if not (color or self.color): raise Exception("Missing color")
        with open(f"{basename}_{color or self.color}.dill", 'wb') as f:
            dill.dump(self.model, f)

    def load(self, filename):
        self.color = self.color or Path(filename).stem.split("_")[-1]
        with open(filename, 'rb') as f:
            self.model = dill.load(f)
