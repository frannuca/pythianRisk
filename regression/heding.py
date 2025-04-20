import pandas as pd
from sklearn.linear_model import LassoCV

def select_top_factors(y: pd.Series, X: pd.DataFrame, n_factors: int = 5):
    model = LassoCV(cv=5).fit(X, y)
    coeffs = pd.Series(model.coef_, index=X.columns)
    top_features = coeffs.abs().nlargest(n_factors).index
    return top_features


if __name__ == "__main__":
    # Example usage
    # Assuming df_example is a DataFrame with your data and target_series is your target variable
    # df_example = pd.DataFrame(...)
    # target_series = pd.Series(...)

    # Example DataFrame and Series for demonstration
    df_example = pd.DataFrame({
        "col1": [1, 2, 3, 4],
        "col2": [5, 6, 7, 8],
        "col3": [2, 4, 6, 8]
    })
    target_series = pd.Series([10, 20, 30, 40])

    selected = select_top_factors(target_series, df_example, 2)
    print(selected)