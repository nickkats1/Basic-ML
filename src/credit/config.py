class Cfg:
    """Credit-risk configuration."""

    # Raw data source
    url: str = (
        "https://raw.githubusercontent.com/LeeMorinUCF/"
        "QMB6358S22/main/demo_24_Classification/credit.csv"
    )

    # Training
    test_size: float = 0.20
    seed: int = 42
    cv_folds: int = 4

    # Model features
    features: tuple[str, ...] = (
        "checkingstatus1",
        "duration",
        "history",
        "purpose",
        "amount",
        "savings",
        "employ",
        "installment",
        "status",
        "others",
        "residence",
        "property",
        "age",
        "otherplans",
        "housing",
        "cards",
        "job",
        "liable",
        "tele",
        "foreign",
    )

    target: str = "Default"