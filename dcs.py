import numpy as np
from sklearn import linear_model
from scipy.stats import norm
from scipy.optimize import minimize
import cvxpy as cp

class DecorrelatedScore():
    # Deccorrelated Score Method for Logistic Regression

    def fit(self, outcome, treatment, other_variables):
        # Input data :
        X = other_variables
        Y = outcome
        Z = np.array(treatment).reshape(X.shape[0],1)
        Q = np.append(Z, X, axis=1)
        self.sample_size = X.shape[0]
        # Step 1 : Initial estimation :
        clf = linear_model.LogisticRegression(penalty='l1',solver='liblinear',max_iter=10000,fit_intercept=False,C= 1)
        clf.fit(Q,Y)
        beta_hat = clf.coef_.reshape([Q.shape[1],1])
        # print('initial estimator is',beta_hat)
        #Step 2 : Estimate w
        self.w_hat = self.findw(X,Z,beta_hat)

        #Step 3 : Calculate score function :
        func = np.empty(X.shape[0])
        gamma = beta_hat[1:]
        for i in range(X.shape[0]):
            func[i] = (Y[i] - (1 / (1 + np.exp(-np.dot(Q[i, :], beta_hat)))))* (
                      Z[i] - np.dot(X[i, :], self.w_hat))
        S = -np.mean(func)

        func1 = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            func1[i] = (Y[i]  - 1/ (1 + np.exp(-np.dot(X[i, :], gamma)))) * (
                      Z[i] - np.dot(X[i, :], self.w_hat))
        S0 = -np.mean(func1)
        self.score = S0

        # Calculate one step estimator
        self.pfim = self.findpfim(X,Z,beta_hat)
        theta_tilde = beta_hat[0] - S/self.pfim
        self.theta = theta_tilde

    def findw(self,X,Z,beta_hat):
        # Calculate the estimator w_tilde using L1-penalty.
        Q = np.append(Z, X, axis=1)
        def bound(w):
            # w.reshape(X.shape[1],1)
            func = np.zeros(X.shape[1])
            for i in range(X.shape[0]):
                func += X[i,:] * ( np.exp(np.dot(Q[i,:],beta_hat))/ ((1 + np.exp(np.dot(Q[i,:],beta_hat)) ) ** 2) ) * (Z[i] - np.dot(X[i,:],w))
            func = func/X.shape[0]
            return (np.max(func) - 0.5)
        def l1(w) :
            return np.linalg.norm(w,1)

        def fun(w):
            res = 0
            for i in range(X.shape[0]):
                res += ( np.exp(np.dot(Q[i,:],beta_hat))/ ((1 + np.exp(np.dot(Q[i,:],beta_hat)) ) ** 2) ) * ((Z[i] - np.dot(X[i,:],w)) ** 2) - 0.1 * np.linalg.norm(w,1)
            res = res/X.shape[0]
            return res
        # Solve l1 minimization using cvxpy : unavailable
        # w = cp.Variable(shape=X.shape[1])
        # boundary = 0.1
        # cons = [bound(w,X,Z,beta_hat) <= boundary]
        # obj = cp.Minimize(cp.norm(w, 1))
        # prob = cp.Problem(obj, cons)
        # prob.solve()
        # print("status: {}".format(prob.status))
        # return w.value
        # Solve l1 minimization using scipy
        cons = {'type':"ineq","fun":bound}
        w_tilde = minimize(l1,x0= np.zeros(X.shape[1]),constraints=cons)
        # w_check = minimize(fun,x0 = np.zeros(X.shape[1]),method = 'trust-constr')
        return  w_tilde.x


    def findpfim(self,X,Z,beta_hat):
        # Calculate the partial fisher information matrix.
        Q = np.append(Z, X, axis=1)
        func = 0
        for i in range(X.shape[0]):
            func += Z[i] * ( np.exp(np.dot(Q[i, :], beta_hat)) / ((1 + np.exp(np.dot(Q[i, :], beta_hat))) ** 2))*(
                Z[i] - np.dot(X[i, :], self.w_hat))
        pfim = func/self.sample_size
        return  pfim

    def inference(self,alpha = 0.05):
        # Calculate test static
        n = self.sample_size
        U_hat_n = np.sqrt(n) * self.score * np.power(self.pfim,-1/2)
        print('-------------------------------------------------------------------')
        print('The value of test static is', U_hat_n)

        # Calculate the p-value
        p_value = (1 - norm.cdf(np.abs(U_hat_n))) * 2
        print('The p-value is', p_value)
        self.p_value = p_value
        self.static = U_hat_n
        if p_value <= alpha:
            print('Reject null hypothesis theta_0 =', 0, 'under', alpha, 'significance level')
            return False
        else:
            print('Accept null hypothesis theta_0 =', 0, 'under', alpha, 'significance level')
            return True

    def confidence_interval(self,alpha = 0.05):
        c = np.power(self.sample_size,-1/2) * norm.ppf(1 - alpha / 2) * np.power(self.pfim,-1/2)
        confidence_interval = [self.theta - c, self.theta + c]
        print('the ci of theta is', confidence_interval)
        return confidence_interval

    def result(self):

        print('one step estimator =',self.theta )

        self.inference()



