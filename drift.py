from evidently import Report
from evidently.presets import (
    DataDriftPreset
)

def gerar_drift(df):

    train = df[
        df["ano"] <= 2023
    ]

    test = df[
        df["ano"] >= 2024
    ]

    report = Report(

        metrics=[
            DataDriftPreset()
        ]
    )

    snapshot = report.run(
        reference_data=train,
        current_data=test
    )

    snapshot.save_html(
        "drift_report.html"
    )

    print(
        "\nDrift salvo!"
    )