import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class FinancialPlanningAI:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = ['age', 'income', 'expenses', 'debt', 'risk_tolerance', 'financial_goals']
        self.models = {'Random Forest': {}, 'Linear Regression': {}, 'SVR': {}}
        self.scalers = {}

    def generate_synthetic_data(self, n_samples=1000):
        np.random.seed(42)
        data = {
            'age': np.random.randint(18, 70, n_samples),
            'income': np.random.normal(60000, 20000, n_samples),
            'expenses': np.random.normal(40000, 15000, n_samples),
            'debt': np.random.exponential(10000, n_samples),
            'risk_tolerance': np.random.randint(1, 6, n_samples),
            'financial_goals': np.random.randint(1, 4, n_samples),
        }
        for i in range(n_samples):
            if data['expenses'][i] > data['income'][i]:
                data['expenses'][i] = data['income'][i] * 0.8
            if data['income'][i] < 20000:
                data['income'][i] = 20000
            if data['expenses'][i] < 15000:
                data['expenses'][i] = 15000
        targets = {
            'emergency_fund': [],
            'investment_allocation': [],
            'savings_rate': [],
            'debt_payoff_months': []
        }
        for i in range(n_samples):
            emergency_months = max(3, min(12, 6 + (data['risk_tolerance'][i] - 3)))
            targets['emergency_fund'].append(data['expenses'][i] * emergency_months / 12)
            stock_allocation = min(100, max(0, (100 - data['age'][i]) * 0.5 + data['risk_tolerance'][i] * 10))
            targets['investment_allocation'].append(stock_allocation)
            savings_capacity = (data['income'][i] - data['expenses'][i]) / data['income'][i]
            savings_rate = max(0.05, min(0.5, savings_capacity * 0.8))
            targets['savings_rate'].append(savings_rate)
            if data['debt'][i] > 1000:
                monthly_payment = max(100, (data['income'][i] - data['expenses'][i]) * 0.3 / 12)
                payoff_months = min(120, data['debt'][i] / monthly_payment)
            else:
                payoff_months = 0
            targets['debt_payoff_months'].append(payoff_months)
        df = pd.DataFrame(data)
        for key, values in targets.items():
            df[key] = values
        return df

    def train_models(self):
        df = self.generate_synthetic_data()
        X = df[self.feature_columns]
        targets = ['emergency_fund', 'investment_allocation', 'savings_rate', 'debt_payoff_months']
        for target in targets:
            y = df[target]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_scaled, y)
            lr = LinearRegression()
            lr.fit(X_scaled, y)
            svr = SVR()
            svr.fit(X_scaled, y)
            self.models['Random Forest'][target] = rf
            self.models['Linear Regression'][target] = lr
            self.models['SVR'][target] = svr
            self.scalers[target] = scaler

    def evaluate_models(self):
        df = self.generate_synthetic_data()
        X = df[self.feature_columns]
        targets = ['emergency_fund', 'investment_allocation', 'savings_rate', 'debt_payoff_months']
        results = {}
        for target in targets:
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train_scaled, y_train)
            rf_pred = rf.predict(X_test_scaled)
            rf_r2 = r2_score(y_test, rf_pred)
            svr = SVR()
            svr.fit(X_train_scaled, y_train)
            svr_pred = svr.predict(X_test_scaled)
            svr_r2 = r2_score(y_test, svr_pred)
            lr = LinearRegression()
            lr.fit(X_train_scaled, y_train)
            lr_pred = lr.predict(X_test_scaled)
            lr_r2 = r2_score(y_test, lr_pred)
            max_r2 = max(rf_r2, svr_r2, lr_r2)
            rf_diff = (rf_r2 - max_r2) * 100
            svr_diff = (svr_r2 - max_r2) * 100
            lr_diff = (lr_r2 - max_r2) * 100
            results[target] = {
                'Random Forest R2': rf_r2,
                'SVR R2': svr_r2,
                'Linear Regression R2': lr_r2,
                'Random Forest Diff (%)': rf_diff,
                'SVR Diff (%)': svr_diff,
                'Linear Regression Diff (%)': lr_diff
            }
        return results

    def predict(self, user_data, model_choice='Random Forest'):
        if not self.models['Random Forest']:
            self.train_models()
        predictions = {}
        for target in self.models[model_choice]:
            scaler = self.scalers[target]
            model = self.models[model_choice][target]
            from app.utils import get_prediction_and_scaling
            pred = get_prediction_and_scaling(model, scaler, user_data, self.feature_columns)
            predictions[target] = pred
        return predictions
