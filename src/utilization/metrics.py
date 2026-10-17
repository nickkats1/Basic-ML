from sklearn.metrics import mean_squared_error, r2_score


def eval_metrics(y_test, y_pred) -> dict[str, float]:
    return {
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred)
    }