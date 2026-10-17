class Cfg:
    """Airplane configuration."""

    # Raw data sources
    sales_url: str = (
        "https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/"
        "refs/heads/main/final_exam_2020/airplane_sales.csv"
    )
    specs_url: str = (
        "https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/"
        "refs/heads/main/final_exam_2020/airplane_specs.csv"
    )
    performance_url: str = (
        "https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/"
        "refs/heads/main/final_exam_2020/airplane_perf.csv"
    )

    # Training
    test_size: float = 0.20
    seed: int = 42
    cv_folds: int = 4

    # Model features
    features: tuple[str, ...] = (
        "age",
        "pass",
        "fixgear",
        "tdrag",
        "horse",
        "fuel",
        "ceiling",
        "cruise",
    )

    log_features: tuple[str, ...] = (
        "horse",
        "fuel",
        "ceiling",
        "cruise",
    )

    target: str = "price"