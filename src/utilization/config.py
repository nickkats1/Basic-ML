class Cfg:
    """utilization config"""

    # Raw data sources.
    credit_url: str = (
        "https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/"
        "refs/heads/main/final_exam_2021/credit_bureau.csv"
    )
    app_url: str = (
        "https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/"
        "refs/heads/main/final_exam_2021/applications.csv"
    )
    demo_url: str = (
        "https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/"
        "refs/heads/main/final_exam_2021/demographic.csv"
    )
    
    # feature from credit dataset
    
    credit_features: list[str] = [
        "ssn",
        "zip_code",
        "fico",
        "num_late",
        "past_def",
        "num_bankruptcy"
    ]
    
    # feature from "app" dataset
    
    app_features: list[str] = [
        "app_id",
        "ssn",
        "zip_code",
        "income",
        "homeownership",
        "purchases",
        "credit_limit"
    ]
    
    demographic_features: list[str] = [
        "zip_code",
        "avg_income",
        "density"
    ]
    
    
    test_size: float = 0.20
    
    seed: int = 42
    

    # main features
    
    features: tuple[str, ...] = (
        "income",
        "homeownership",
        "credit_limit",
        "fico",
        "num_late",
        "past_def",
        "num_bankruptcy",
        "avg_income",
        "density",
    )

    target: str = "log_odds_util"