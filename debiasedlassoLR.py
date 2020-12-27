import numpy as np
from sklearn import linear_model
from scipy.stats import norm


class DebiasedLassoLR():
    # Debiased Lasso for Logistic Regression

    def fit(self, X, y):
        self.data = X
        self.n = X.shape[0]
        self.p = X.shape[1]
        # Step 1 : calculate the initial value, Lasso estimator beta_hat
        beta = linear_model.LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000, fit_intercept=False,
                                               C=1)
        beta.fit(X, y)
        beta_hat = beta.coef_.reshape(self.p, 1)
        # Step 2 : calculate inverse matrix
        self.Theta_hat = self.find_inverse_matrix(X, beta_hat)
        # Step 3 : calculate debiased lasso estimator
        # First calculate dRou_beta_hat
        fun = np.zeros(self.p)
        for i in range(self.n):
            fun += X[i, :] * (1 / (1 + np.exp(-np.dot(X[i, :], beta_hat))) - y[i])
        dRou_beta_hat = fun / self.n
        dRou_beta_hat = dRou_beta_hat.reshape(self.p, 1)
        # Use result above to obtain b_hat :
        b_hat = beta_hat - np.dot(self.Theta_hat, dRou_beta_hat)
        self.b_hat = b_hat
        # Finally, calculate the estimated variance of b_hat for inferences
        var = np.diag(np.dot(np.dot(self.Theta_hat, dRou_beta_hat), np.dot(self.Theta_hat, dRou_beta_hat).T)) * self.n
        self.var = var

    def find_inverse_matrix(self, X, beta_hat, tuning_parameter=None):
        # We can use the basic nodewise lasso based on X_beta_hat instead of X.
        # Calculate X_beta_hat
        w = np.zeros(self.n)
        for i in range(self.n):
            w[i] = -np.exp(-np.dot(X[i, :], beta_hat) / 2) / (1 + np.exp(-np.dot(X[i, :], beta_hat)))
        W_beta_hat = np.diag(w)
        X_beta_hat = np.dot(W_beta_hat, X)
        # Initailization :
        C_hat = np.empty([self.p, self.p])
        tau_hatsquare = np.empty(self.p)
        Theta_hat = np.empty([self.p, self.p])
        if tuning_parameter is None:
            tuning_parameter = np.full(self.p, 0.2)
        # The nodewise lasso regression
        for j in range(self.p):
            # Regress X_{-j} on X_{j} to obtain the j-th row of matrix C_hat (index j is from 0 to p-1).
            gamma = linear_model.Lasso(alpha=0.1, fit_intercept=False, max_iter=10000)
            gamma.fit(np.delete(X_beta_hat, j, 1), X_beta_hat[:, j])
            C_hat[j, :] = np.insert(gamma.coef_, j, 1)
            # Calculate tau_hat_[j]^2.
            tau_hatsquare[j] = np.linalg.norm((X_beta_hat[:, j] - gamma.predict(np.delete(X_beta_hat, j, 1))),
                                              ord=2) ** 2 / self.n + tuning_parameter[
                                   j] * np.linalg.norm(gamma.coef_, ord=1)
            # Calculate the j-the row of the inverse matrix Theta_hat at last.
            Theta_hat[j, :] = C_hat[j, :] / tau_hatsquare[j]

        return Theta_hat

    def confidence_regions(self, alpha=0.05):
        # Consturct (1 - alpha/2) confidence region for beta_0. Default : 95%
        X = self.data
        confidence_regions = []
        for j in range(self.p):
            # An asymptotic pointwise confidence interval for beta_0_{j}
            Phi_minus1 = norm.ppf(1 - alpha / 2)
            c = Phi_minus1 * np.sqrt(self.var[j]) * np.sqrt(1 / self.n)
            confidence_regions.append([self.b_hat[j] - c, self.b_hat[j] + c])

        return confidence_regions

    def inference(self, j, beta_0=0, alpha=0.05):

        # Calculate the test static V, which is asymptotic gaussian.
        V = np.sqrt(self.n) * (self.b_hat[j] - beta_0) / (np.sqrt(self.var[j]))
        print('-------------------------------------------------------------------')
        print('The value of test static is', V)
        # Calculate the p-value
        p_value = (1 - norm.cdf(np.abs(V))) * 2
        print('The p-value is', p_value)
        self.p_value = p_value
        self.static = V
        if p_value <= alpha:
            print('Reject null hypothesis beta_0_{j} =', beta_0, 'under', alpha, 'significance level')
            return False
        else:
            print('Accept null hypothesis beta_0_{j} =', beta_0, 'under', alpha, 'significance level')
            return True

    def results(self, j):
        print('-------------------------------------------------------------------')
        print('Received', self.n, 'samples and', self.p, 'features.')
        print('The result is b_hat_j =', self.b_hat[j])
        print('The estimated variance is', self.var[j])
        print('-------------------------------------------------------------------')


# TEST:
# def logistic_data_generate(sample=100, features=500, active=4):
#     X = np.random.multivariate_normal(np.zeros(features), np.eye(features), sample)
#     coef = np.array([1, 2, -2, -1.5])
#     # coef = np.array([1, 2, -2, -1.5, 1, 1.5, 0.5, -0.5, 1.2, -0.2])
#     # coef = coef = np.random.uniform(0,4,active)
#     coef.reshape(active, 1)
#     Y = np.empty(sample)
#     for i in range(sample):
#         pr = 1 / (1 + np.exp(-np.dot(X[i, :active], coef)))
#         Y[i] = np.random.binomial(1, pr)
#     return X, Y, coef
#
#
# X, Y, coef = logistic_data_generate()
#
# a = DebiasedLassoLR()
# a.fit(X, Y)
# a.results(0)
# print(a.confidence_regions()[0])
# a.inference(j=0)
