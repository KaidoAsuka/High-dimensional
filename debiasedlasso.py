import numpy as np
from sklearn import linear_model
from scipy.stats import norm
from sklearn.datasets import make_sparse_uncorrelated


# Algorithm introduced in van de geer et al. (2014)
class DebiasedLasso() :

    def __init__(self):
        pass

    def find_inverse_matrix(self,X,tuning_parameter = None) :
    # Calculate the relaxed inverse matrix Theta_hat.
        #Initailization :
        C_hat = np.empty([self.p,self.p])
        tau_hatsquare = np.empty(self.p)
        Theta_hat =np.empty([self.p,self.p])
        if tuning_parameter == None:
            tuning_parameter = np.full(self.p, 0.1)

        # The lasso for nodewise regression
        for j in range(self.p) :
            # Regress X_{-j} on X_{j} to obtain the j-th row of matrix C_hat (index j is from 0 to p-1).
            gamma = linear_model.Lasso(alpha= 0.1 , fit_intercept=False, max_iter=10000)
            gamma.fit(np.delete(X,j,1),X[:,j])
            C_hat[j,:] =np.insert(gamma.coef_,j,1)
            # Calculate tau_hat_[j]^2.
            # tau_hatsquare[j] = np.linalg.norm((X[:,j] - gamma.predict(np.delete(X,j,1))), ord=2)**2/self.n + tuning_parameter[j] * np.linalg.norm(gamma.coef_,ord=1)
            # Alternative:
            tau_hatsquare[j] =np.dot((X[:,j] - gamma.predict(np.delete(X,j,1))).T , X[:,j]) / self.n
            # Calculate the j-the row of the inverse matrix Theta_hat at last.
            Theta_hat[j,:] = C_hat[j,:]/ tau_hatsquare[j]

        self.Theta_hat = Theta_hat


    def fit(self,X,y) :
    # Calculate the Debiased Lasso estimator
        self.data = X
        self.n = X.shape[0]
        self.p = X.shape[1]
    # Step 1 : calculate the initial value, Lasso estimator beta_hat
        beta = linear_model.Lasso(alpha = 0.5 , fit_intercept= False, max_iter=10000)
        beta.fit(X,y)
        beta_hat = beta.coef_.reshape(self.p,1)

    # Step 2 : calculate lambda * kappa_hat
    # The default tuning parameter of lasso is set to be 1.
        lambda_mul_kappa = 0.5 * np.sign(beta_hat)

    # Alternative : Use the representation lambda * kappa_hat = X.T * (Y - X*beta_hat)/n
    # Results using this method is inaccurate for unknown cause.
    #     res = (y.reshape(self.n,1) - np.dot(X,beta_hat))
    #     res.reshape(self.n,1)
    #     lambda_mul_kappa = np.dot(X.T, res)/self.n

    # Step 3 : Use the results above and the inverse matrix Theta_hat to obtain the Debiased Lasso estimator b_hat
        self.find_inverse_matrix(X)
        Theta_hat = self.Theta_hat
        b_hat = beta_hat + np.dot(Theta_hat,lambda_mul_kappa)
        self.b_hat = b_hat

    # **Calculate a consistent error variance estimator
        count = 0
        for element in beta_hat:
            if element != 0:
                count += 1

        var_est = np.sum(np.power(y - beta.predict(X),2))/(self.n - count)
        self.var_est = var_est


    def results(self):
        print('-------------------------------------------------------------------')
        print('Recieved',self.n,'samples and',self.p,'features.')
        print('The result is b_hat =',self.b_hat)
        print('-------------------------------------------------------------------')


    def confidence_regions(self,alpha = 0.05,sigma = None):
    # Consturct (1 - alpha/2) confidence region for beta_0. Default : 95%. 

        if sigma is None:
        # Calculate a consistent estimator of sigma
            sigma = np.power(self.var_est, 1/2)

        X = self.data
        Sigma_hat = np.dot(X.T,X)/self.n

        confidence_regions = []
        for j in range(self.p):
        #An asymptotic pointwise confidence interval for beta_0_{j}
            Phi_minus1 = norm.ppf(1- alpha/2)
            Omega_hat = np.dot(np.dot(self.Theta_hat,Sigma_hat),self.Theta_hat.T)
            c = Phi_minus1 * sigma * np.sqrt(Omega_hat[j,j]/self.n)
            confidence_regions.append([self.b_hat[j] - c,self.b_hat[j] + c])

        return confidence_regions


    def inference(self,j,beta_0 = 0,alpha = 0.05,sigma = None):

        if sigma is None :
        # Calculate a consistent estimator of sigma
            sigma = np.power(self.var_est,1/2)
        # Calculate the test static V, which is asymptotic gaussian.
        X = self.data
        Sigma_hat = np.dot(X.T, X) / self.n
        Omega_hat = np.dot(np.dot(self.Theta_hat,Sigma_hat),self.Theta_hat.T)
        V = np.sqrt(self.n)*(self.b_hat[j] - beta_0) / (np.sqrt(Omega_hat[j,j]) * sigma)
        print('-------------------------------------------------------------------')
        print('The value of test static is',V)

        # Calculate the p-value
        p_value =  (1-norm.cdf(np.abs(V))) * 2
        print('The p-value is',p_value)
        self.p_value = p_value
        self.static = V
        if p_value <= alpha :
            print('Reject null hypothesis beta_0_{j} =',beta_0,'under', alpha,'significance level')
            return False
        else:
            print('Accept null hypothesis beta_0_{j} =',beta_0,'under', alpha, 'significance level')
            return True

