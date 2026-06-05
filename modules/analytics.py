import pandas as pd

def topic_counts():

    try:

        df = pd.read_csv(
            "data/topic_assignments.csv"
        )

        if "topic" in df.columns:

            return (
                df["topic"]
                .value_counts()
                .to_dict()
            )

        return {}

    except Exception as e:

        print(e)

        return {}