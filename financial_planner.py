import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="AI Financial Planner",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .recommendation-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_prediction_and_scaling(model, scaler, user_data, feature_columns):
    X_scaled = scaler.transform(pd.DataFrame([user_data])[feature_columns])
    return model.predict(X_scaled)[0]

class FinancialPlanningAI:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = ['age', 'income', 'expenses', 'debt', 'risk_tolerance', 'financial_goals']
        self.models = {}  # Ensure models dict is always initialized
        self.scalers = {} # Ensure scalers dict is always initialized
        
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic financial data for training"""
        np.random.seed(42)
        
        data = {
            'age': np.random.randint(18, 70, n_samples),
            'income': np.random.normal(60000, 20000, n_samples),
            'expenses': np.random.normal(40000, 15000, n_samples),
            'debt': np.random.exponential(10000, n_samples),
            'risk_tolerance': np.random.randint(1, 6, n_samples),  # 1-5 scale
            'financial_goals': np.random.randint(1, 4, n_samples),  # 1: retirement, 2: home, 3: education
        }
        
        # Ensure realistic relationships
        for i in range(n_samples):
            if data['expenses'][i] > data['income'][i]:
                data['expenses'][i] = data['income'][i] * 0.8
            if data['income'][i] < 20000:
                data['income'][i] = 20000
            if data['expenses'][i] < 15000:
                data['expenses'][i] = 15000
        
        # Target variables (what we want to predict)
        targets = {
            'emergency_fund': [],
            'investment_allocation': [],
            'savings_rate': [],
            'debt_payoff_months': []
        }
        
        for i in range(n_samples):
            # Emergency fund (3-12 months of expenses)
            emergency_months = max(3, min(12, 6 + (data['risk_tolerance'][i] - 3)))
            targets['emergency_fund'].append(data['expenses'][i] * emergency_months / 12)
            
            # Investment allocation (0-100% stocks based on age and risk tolerance)
            stock_allocation = min(100, max(0, 
                (100 - data['age'][i]) * 0.5 + data['risk_tolerance'][i] * 10))
            targets['investment_allocation'].append(stock_allocation)
            
            # Savings rate (% of income)
            savings_capacity = (data['income'][i] - data['expenses'][i]) / data['income'][i]
            savings_rate = max(0.05, min(0.5, savings_capacity * 0.8))
            targets['savings_rate'].append(savings_rate)
            
            # Debt payoff time (months)
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
        """Train ML models for financial recommendations"""
        df = self.generate_synthetic_data()
        X = df[self.feature_columns]
        self.models = {'Random Forest': {}, 'Linear Regression': {}, 'SVR': {}}
        self.scalers = {}
        targets = ['emergency_fund', 'investment_allocation', 'savings_rate', 'debt_payoff_months']
        for target in targets:
            y = df[target]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            # Train all models
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
        """Evaluate and compare model accuracy for each target using Random Forest, SVR, and Linear Regression."""
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
            # Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train_scaled, y_train)
            rf_pred = rf.predict(X_test_scaled)
            rf_r2 = r2_score(y_test, rf_pred)
            # SVR
            svr = SVR()
            svr.fit(X_train_scaled, y_train)
            svr_pred = svr.predict(X_test_scaled)
            svr_r2 = r2_score(y_test, svr_pred)
            # Linear Regression
            lr = LinearRegression()
            lr.fit(X_train_scaled, y_train)
            lr_pred = lr.predict(X_test_scaled)
            lr_r2 = r2_score(y_test, lr_pred)
            # Percentage differences
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
        """Make predictions for user using selected model"""
        if not self.models:
            self.train_models()
        predictions = {}
        for target in self.models[model_choice]:
            scaler = self.scalers[target]
            model = self.models[model_choice][target]
            pred = get_prediction_and_scaling(model, scaler, user_data, self.feature_columns)
            predictions[target] = pred
        return predictions

def main():
    st.markdown('<h1 class="main-header">🤖 AI Personal Financial Planner</h1>', unsafe_allow_html=True)
    
    # Initialize AI model
    if 'ai_model' not in st.session_state:
        st.session_state.ai_model = FinancialPlanningAI()
    
    # Sidebar for user input
    st.sidebar.header("📊 Your Financial Profile")
    
    with st.sidebar:
        age = st.slider("Age", 18, 70, 30, help="Your current age.")
        income = st.number_input("Annual Income (INR ₹)", min_value=100000, max_value=41500000, value=100000, step=100000, help="Total yearly income before tax.")
        expenses = st.number_input("Annual Expenses (INR ₹)", min_value=10000, max_value=int(income*0.95), value=min(10000, int(income*0.75)), step=10000, help="Total yearly expenses.")
        debt = st.number_input("Total Debt (INR ₹)", min_value=0, max_value=100000000, value=0, step=10000, help="Total outstanding debt.")
        model_choice = st.selectbox("Choose model type", ["Random Forest", "Linear Regression", "SVR"], help="Select the ML model for analysis.")
        st.subheader("Risk & Goals")
        risk_tolerance = st.select_slider(
            "Risk Tolerance", 
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: {1: "Very Conservative", 2: "Conservative", 3: "Moderate", 4: "Aggressive", 5: "Very Aggressive"}[x],
            help="How much risk are you willing to take with investments?"
        )
        financial_goal = st.selectbox(
            "Primary Financial Goal",
            options=[1, 2, 3],
            format_func=lambda x: {1: "Retirement Planning", 2: "Home Purchase", 3: "Education Fund"}[x],
            index=0,
            help="Select your main financial goal."
        )
        analyze_button = st.button("🔍 Analyze My Finances")
        evaluate_button = st.button("📊 Evaluate Model Accuracy")
    
    # Main content area
    if analyze_button:
        # Prepare user data
        user_data = {
            'age': age,
            'income': income,
            'expenses': expenses,
            'debt': debt,
            'risk_tolerance': risk_tolerance,
            'financial_goals': financial_goal
        }
        
        # Get predictions
        with st.spinner("Analyzing your financial profile..."):
            predictions = st.session_state.ai_model.predict(user_data, model_choice)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💡 AI Recommendations")
            
            # Emergency Fund
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            st.write("**Emergency Fund Target**")
            emergency_amount = predictions['emergency_fund']
            st.metric("Recommended Amount", f"₹{emergency_amount:,.0f}")
            st.progress(min(1.0, 10000/emergency_amount), text="Your current emergency fund progress")
            st.write(f"This covers approximately {emergency_amount/(expenses/12):.1f} months of expenses")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Investment Allocation
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            st.write("**Investment Portfolio Allocation**")
            stock_pct = predictions['investment_allocation']
            bond_pct = 100 - stock_pct
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Stocks", f"{stock_pct:.0f}%")
            with col_b:
                st.metric("Bonds", f"{bond_pct:.0f}%")
            
            # Portfolio pie chart
            fig_pie = px.pie(
                values=[stock_pct, bond_pct],
                names=['Stocks', 'Bonds'],
                title="Recommended Portfolio Allocation"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Savings Rate
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            st.write("**Savings Strategy**")
            savings_rate = predictions['savings_rate']
            monthly_savings = (income * savings_rate) / 12
            st.metric("Recommended Savings Rate", f"{savings_rate*100:.1f}%")
            st.metric("Monthly Savings Amount", f"₹{monthly_savings:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Debt Payoff
            if debt > 1000:
                st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                st.write("**Debt Payoff Plan**")
                payoff_months = predictions['debt_payoff_months']
                monthly_payment = debt / payoff_months if payoff_months > 0 else 0
                st.metric("Payoff Timeline", f"{payoff_months:.0f} months")
                st.metric("Monthly Payment", f"₹{monthly_payment:,.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                payoff_months = 0
        
        # Financial Health Dashboard
        with st.expander("📈 Financial Health Dashboard", expanded=True):
            debt_to_income = debt / income
            savings_potential = (income - expenses) / income
            emergency_coverage = emergency_amount / (expenses / 12)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Debt/Income Ratio", f"{debt_to_income:.2f}")
            with col2:
                st.metric("Savings Potential", f"{savings_potential*100:.1f}%")
            with col3:
                st.metric("Emergency Coverage (months)", f"{emergency_coverage:.1f}")
            with col4:
                st.metric("Age", f"{age}")
        
        # Progress visualization
        with st.expander("🎯 Financial Goals Progress", expanded=True):
            progress_data = {
                'Category': ['Emergency Fund', 'Debt Payoff', 'Investment', 'Savings Rate'],
                'Current': [
                    min(100, (10000/emergency_amount)*100),
                    max(0, 100 - payoff_months) if debt > 1000 else 100,
                    min(100, stock_pct),
                    min(100, savings_rate*100)
                ],
                'Target': [100, 100, 100, 100]
            }
            progress_df = pd.DataFrame(progress_data)
            fig_bar = px.bar(
                progress_df, 
                x='Category', 
                y=['Current', 'Target'],
                title="Financial Goals Progress",
                barmode='group'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Personalized advice
        with st.expander("💭 Personalized Financial Advice", expanded=True):
            advice = []
            if debt_to_income > 0.36:
                st.warning("Your debt-to-income ratio is high. Consider reducing debt for better financial health.")
            if savings_potential < 0.1:
                st.info("Your savings potential is low. Try to reduce expenses or increase income.")
            if emergency_coverage < 3:
                st.warning("Your emergency fund covers less than 3 months of expenses. Aim for at least 3-6 months.")
            if age < 30 and stock_pct < 70:
                st.info("As a young investor, you can consider a higher allocation to stocks for long-term growth.")
            if age > 55 and stock_pct > 60:
                st.info("Consider reducing stock exposure as you approach retirement.")
            if not advice:
                advice.append("✅ **Great Job**: Your financial profile looks healthy! Keep up the good work.")
            if financial_goal == 1:
                if age < 40:
                    advice.append("🕒 **Retirement Planning**: Start investing early to maximize compounding for retirement.")
                elif age >= 40 and age < 55:
                    advice.append("🕒 **Retirement Planning**: Review your retirement corpus and consider increasing your savings rate if needed.")
                else:
                    advice.append("🕒 **Retirement Planning**: Focus on capital preservation and review your withdrawal strategy.")
            elif financial_goal == 2:
                advice.append("🏠 **Home Purchase**: Consider building a larger down payment to reduce loan burden and interest costs.")
                if debt > 0.3 * income:
                    advice.append("🏠 **Home Purchase**: Try to reduce existing debt before taking on a home loan.")
            elif financial_goal == 3:
                if age < 35:
                    advice.append("🎓 **Education Fund**: Start a dedicated investment plan for education goals early to benefit from long-term growth.")
                else:
                    advice.append("🎓 **Education Fund**: Consider safer investment options as the education goal approaches.")
            for tip in advice:
                st.write(tip)
    
    elif evaluate_button:
        with st.expander("📊 Model Evaluation Results", expanded=True):
            results = st.session_state.ai_model.evaluate_models()
            eval_df = pd.DataFrame(results).T
            eval_df["Random Forest R2"] = eval_df["Random Forest R2"] * 100
            eval_df["SVR R2"] = eval_df["SVR R2"] * 100
            eval_df["Linear Regression R2"] = eval_df["Linear Regression R2"] * 100
            st.dataframe(eval_df.style.format({
                "Random Forest R2": "{:.2f}",
                "SVR R2": "{:.2f}",
                "Linear Regression R2": "{:.2f}",
                "Random Forest Diff (%)": "{:.2f}",
                "SVR Diff (%)": "{:.2f}",
                "Linear Regression Diff (%)": "{:.2f}"
            }))
        
    else:
        # Welcome screen
        st.write("## Welcome to Your AI Financial Planner!")
        st.write("This intelligent financial planning tool uses machine learning to provide personalized recommendations based on your financial profile.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("### 🎯 Personalized Goals")
            st.write("Get customized financial targets based on your age, income, and risk tolerance.")
        
        with col2:
            st.write("### 📊 Smart Analytics")
            st.write("AI-powered analysis of your financial health and optimization opportunities.")
        
        with col3:
            st.write("### 💡 Actionable Advice")
            st.write("Receive specific, actionable recommendations to improve your financial future.")
        
        st.info("👈 Fill out your financial profile in the sidebar and click 'Analyze My Finances' to get started!")

if __name__ == "__main__":
    main()