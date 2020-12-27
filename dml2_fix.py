import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from scipy.stats import norm


# DML1 Algorithm for Partial Linear Model in Chernozhukov et al. (2018).
# Single treatment version.  D is 1-d array.

# Model :  Y = D * theta  + g(X) + U ;
#          D = m(X) + V .


class dml2(object):

    def __init__(self, method='Lasso', folds=2):
        self.method = method
        self.folds = folds
        self.theta = None

    # Alternative methods : 'Lasso', 'RandomForest',
    # fold_size : number of folds in sample splitting procedure. Default = 2 , 50-50 sample splitting

    def get_theta_score(self, Y_main, Y_aux, X_main, X_aux, D_main, D_aux):
        # get a single estimate of theta

        if self.method == 'Lasso':
            g = linear_model.LassoCV(alphas=np.linspace(0.1, 1, 10), fit_intercept=False, max_iter=10000)
            m = linear_model.LassoCV(alphas=np.linspace(0.1, 1, 10), fit_intercept=False, max_iter=10000)

        elif self.method == "RandomForest":
            g = RandomForestRegressor(n_estimators=100, bootstrap=True, random_state=None)
            m = RandomForestRegressor(n_estimators=100, bootstrap=True, random_state=None)

        elif self.method == "NeuralNet":
            g = MLPRegressor(max_iter=10000,hidden_layer_sizes=(10, ),activation ='relu')
            m = MLPRegressor(max_iter=10000,hidden_layer_sizes=(10, ),activation ='relu')
        else:
            print("invalid method")
            pass

        """
        First Stage : Use auxiliary parts to estimate nuisance estimator g and m.
        """
        # Estimate m,g
        m.fit(X_aux, D_aux)
        fitX = np.empty([X_aux.shape[0], X_aux.shape[1] + 1])
        for i in range(X_aux.shape[0]):
            fitX[i] = np.append(D_aux[i], X_aux[i])
        g.fit(fitX, Y_aux)

        """
        Second Stage : Use main parts to calculate empirical expectation of score.
        """
        m_hat = m.predict(X_main)
        preX = np.empty([X_main.shape[0], X_main.shape[1] + 1])
        for i in range(X_main.shape[0]):
            preX[i] = np.append(0, X_main[i])
        g_hat = g.predict(preX)

        # Calculate the score function for further estimating the variance.
        # Math : sigma_hat^2 = J_0_hat^{-1}* mean(E_nk[score * score']) * (J_0_hat^{-1})'
        # First wo need to compute J_0_hat
        # Math : J_0_hat = mean(E_nk[psi^a]),
        # where the linear score function can be written as : psi = psi^a * theta + psi^b
        # The score function for PLM is : psi = (Y - D*theta - g(X))(D - m(X))
        # Hence, psi^a for PLM is : psi^a = -D(D - m(X))
        psi_a = []
        psi_b = []
        for i in range(Y_main.shape[0]):
            psi_a.append(-D_main[i] * (D_main[i] - m_hat[i]))
            psi_b.append((Y_main[i] - g_hat[i]) * (D_main[i] - m_hat[i]))
        Enk_psi_a = np.mean(psi_a)
        Enk_psi_b = np.mean(psi_b)
        return Enk_psi_b, Enk_psi_a, psi_a, psi_b

    def fit(self, outcome, treatment, other_variables):

        # Input data :
        Y = outcome
        D = treatment
        X = other_variables
        self.sample_size = Y.shape[0]
        self.num_of_variables = X.shape[1]
        # Sample Splitting :
        kf = KFold(n_splits=self.folds, shuffle=True, random_state=10)
        # Initialization :
        score_a =[]
        score_b =[]
        E_psi_b = []
        E_psi_a = []
        for train_index, test_index in kf.split(Y):
            Y_aux, X_aux, D_aux = Y[train_index], X[train_index], D[train_index]
            Y_main, X_main, D_main = Y[test_index], X[test_index], D[test_index]
            # Iterate : calculate theta for every splitting ways.
            Enk_psi_b, Enk_psi_a, psi_a, psi_b = self.get_theta_score(Y_main, Y_aux, X_main, X_aux, D_main, D_aux)
            E_psi_b.append(Enk_psi_b)
            E_psi_a.append(Enk_psi_a)
            score_a.append(psi_a)
            score_b.append(psi_b)

        # Value of estimator
        J0_hat = np.mean(E_psi_a)
        theta_tilde = -np.power(J0_hat, -1) * np.mean(E_psi_b)
        self.theta = theta_tilde

        # Use empirical score function to calculate variance estimator
        score_a =np.array(score_a).reshape(self.sample_size)
        score_b =np.array(score_b).reshape(self.sample_size)
        psi_2 = np.empty(self.sample_size)
        for i in range(self.sample_size):
            psi= score_a[i] * self.theta + score_b[i]
            psi_2[i] = psi**2
        E_psi_2 = np.mean(psi_2)

        var_est = np.power(J0_hat, -1) * np.mean(E_psi_2) * np.power(J0_hat, -1).T
        self.var_est = var_est


    def result(self):
        print('-------------------------------------------------------------------')
        print('Received', self.sample_size, 'samples, with', 1, 'treatment and', self.num_of_variables,
              'other variables.')
        print('The method used to estimate nuisance parameter is', self.method, '.')
        print('The sample is splitted into', self.folds, 'parts.')
        print('The result is theta_tilde =', self.theta)
        print('The value variance estimator is', self.var_est)
        print('-------------------------------------------------------------------')


    def inference(self, theta_0=0, alpha=0.05):
        # Use the fact that sqrt(N) * sigma^{-1} * (theta_tilde - theta_0) is asymptotic gaussian.

        # Calculate the test static.
        W = np.sqrt(self.sample_size) * np.power(self.var_est, -1 / 2) * (self.theta - theta_0)
        print('-------------------------------------------------------------------')
        print('The value of test static is', W)
        # Calculate the p-value
        p_value = (1 - norm.cdf(np.abs(W))) * 2
        print('The p-value is', p_value)
        self.p_value = p_value
        self.static = W
        if p_value <= alpha:
            print('Reject null hypothesis beta_0_{j} =', theta_0, 'under', alpha, 'significance level')
            return False
        else:
            print('Accept null hypothesis beta_0_{j} =', theta_0, 'under', alpha, 'significance level')
            return True


    def confidence_interval(self, alpha=0.05):
        c = norm.ppf(1 - alpha / 2) * np.sqrt(self.var_est / self.sample_size)
        confidence_interval = [self.theta - c, self.theta + c]

        return confidence_interval

# Test for dml2
def data_generate(samples=100, features=500, active=4):
    # the model is : Y = X_1 * beta_1 + ... + X_active * beta_active + err , X ~ N(0,1), err ~ N(0,1)
    # coefficient is generated from uniform distribution taking value from [0,2]/[0,4].

    X = np.random.multivariate_normal(np.zeros(features), np.eye(features), samples)
    err = np.random.standard_normal(samples)
    err.reshape(samples, 1)
    # coef = np.random.uniform(0,4,active)
    # coef = np.array([1,2,-2,-1.5,1,1.5,0.5,-0.5,1.2,-0.2])
    coef = np.array([1, 2, -2, -1.5])
    coef.reshape(active, 1)
    Y = np.dot(X[:, :active], coef) + err
    # Y = np.dot(X[:,0],2) + np.sum(np.sin(X[:,1:active])) + err
    return X, Y, coef

# TEST:
#
# model = dml2(method='NeuralNet')
# X, Y, coef = data_generate()
# model.fit(Y, X[:, 1], np.delete(X, 1, 1))
# model.result()
# model.inference()
# print(model.confidence_interval())

