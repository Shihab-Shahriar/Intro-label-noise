import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin, clone, BaseEstimator, clone
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle, check_random_state


def read_uci(dataset,stats=False):
    from sklearn.utils import shuffle
    from sklearn.preprocessing import MinMaxScaler
    path = f'datasets/{dataset}.txt'
    df = pd.read_csv(path,delim_whitespace=True,header=None)
    df = df.astype('float64')
    data = df.values
    X,Y = data[:,1:],data[:,0]
    Y = LabelEncoder().fit_transform(Y)
    X = MinMaxScaler().fit_transform(X)
    if stats:
        labels,freq = np.unique(Y,return_counts=True)
        print(dataset,X.shape,len(labels),freq.min()/freq.max(),freq)
    return shuffle(X,Y,random_state=42)

def linearly_sep2D(n_samples=800,dist=.25,random_state=None):
    rns = check_random_state(random_state)
    lr = LogisticRegression(penalty='none',random_state=rns)
    lr.coef_ = np.array([[-1.0,1.0]])
    lr.intercept_ = np.array([0.0])
    lr.classes_ = np.array([0,1])
    Xp = rns.uniform(size=(n_samples,2))
    yp = lr.predict(Xp)
    d = lr.decision_function(Xp)
    idx = ((d>dist) | (d<-dist))
    return Xp[idx],yp[idx]


# def corrupt_label(y,cm):
#     if not isinstance(cm,np.ndarray):
#         noise = cm
#         L = len(np.unique(y))
#         cm = np.full((L,L),noise/(L-1))
#         cm[np.diag_indices(L)] = 1 - noise
#     print(cm)
#     a = cm[y]
#     s = a.cumsum(axis=1)
#     r = np.random.rand(a.shape[0])[:,None]
#     yn = (s > r).argmax(axis=1)
#     return yn

def noisify(Y:np.ndarray,frac:float,random_state=None,sample_weight=None,index=False): 
    random_state = check_random_state(random_state)
    nns = int(len(Y)*frac)
    labels = np.unique(Y)
    target_idx = random_state.choice(len(Y),size=nns,replace=False,p=sample_weight)
    #print(target_idx[:3])
    target_mask = np.full(Y.shape,0,dtype=np.bool)
    target_mask[target_idx] = 1
    w = Y.copy()
    mask = target_mask.copy()
    while True:
        left = mask.sum()
        #print(left)
        if left==0:break
        new_labels =  random_state.choice(labels,size=left)
        w[mask] = new_labels
        mask = mask & (w==Y)
    assert (w[target_idx]==Y[target_idx]).sum()==0
    assert (w[~target_mask]==Y[~target_mask]).sum()==len(Y)-nns
    if index: return w,target_mask
    return w

class AdaBoost(BaseEstimator, ClassifierMixin):
    def __init__(self,base,T=20):
        self.base = base
        self.T = T

    def fit(self,X,Y):
        self.models_ = []
        self.wts = []
        N = len(X)
        Wt = np.full((N,),1/N)
        for _ in range(self.T):
            self.wts.append(Wt.copy()*N)
            clf = clone(self.base).fit(X,Y,sample_weight=Wt)
            preds = clf.predict(X)
            e = 1 - accuracy_score(Y,preds)
            if e==0:
                continue
            alpha = .5*np.log((1-e)/e)
            match = preds==Y
            Wt[match] /= np.exp(alpha)
            Wt[~match] *= np.exp(alpha)
            Wt /= Wt.sum()
            self.models_.append((alpha,clf))
        return self
    
    def predict(self,X):
        N = len(X)
        Ypred = np.zeros((N,2)) #2 since binary classification
        for a,clf in self.models_:
            yp = clf.predict(X)
            Ypred[range(N),yp] += a
        return np.argmax(Ypred,axis=1)


class NoisyEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, noise_level):
        self.estimator = estimator
        self.noise_level = noise_level

    def fit(self, X, Y):
        Yn = noisify(Y, self.noise_level)
        self.estimator.fit(X, Yn)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


















